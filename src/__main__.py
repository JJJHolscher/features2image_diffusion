"""
The training and evaluation of a DDPM conditioned on the features from a CNN
trained on MNIST.
The features are extracted by taking the hidden state of a sparse autoencoder
trained on the activations on some hidden layer of the MNIST CNN.

The original code is cloned in mid December 2023 from
https://github.com/TeaPearce/Conditional_Diffusion_MNIST
Most of that script is in unet.py, this file is the train function from there.
That repo trained a DDPM conditioned on the MNIST class labels.

I (Jochem Hölscher) have edited the code such that the same DDPM is conditioned
on the features instead.
"""

import argtoml
from jaxtyping import Float
from jo3mnist.vis import to_img
from jo3util.root import run_dir as jo3run_dir
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .unet import DDPM, ContextUnet
from .data import load_mnist_with_features

O = argtoml.parse_args()
DEVICE = O.device


def evaluate(
    ddpm: DDPM,
    f: Float[torch.Tensor, "n_example feature"],
    x: Float[torch.Tensor, "n_example 1 28 28"],
    n_sample: int,
    result_dir: Path,
    ep: int,
    gif: bool = False
):
    """Compare conditioned ddpm generations with images from the data set.

    Args:
        ddpm: the diffusion model that has been trained for generating images
        f: features to condition the model's image generation with
        x: images corresponding to f. These are shown together with the image
            generations for visual comparison.
        n_sample: the number of images to generate per comparison image.
        ws_test: how much the features will guide the image generetation
        result_dir: the directory in which to store the generated plots
        ep: the epoch after which this function was called. This is only used
            for naming the output files.
        gif: whether to create a gif showing the image generation process
    """
    n_example = len(f)
    x_gen, x_gen_store = ddpm.sample(
        f, n_sample, (1, 28, 28), DEVICE
    )

    fig, axs = plt.subplots(n_sample + 1, n_example, facecolor='gray')
    fig.tight_layout()
    for c in range(n_example):
        for r in range(n_sample):
            i = r * n_example + c
            axs[r, c].imshow(to_img(x_gen[i].to("cpu")))
            axs[r, c].set_axis_off()

        axs[n_sample, c].imshow(to_img(x[c].to("cpu")))
        axs[n_sample, c].set_axis_off()

    plt.savefig(result_dir / f"image_ep{ep}.png")
    print(f"saved image at {result_dir}/image_ep{ep}.png")

    if gif:
        # gif of images evolving over time, based on x_gen_store
        fig, axs = plt.subplots(
            nrows=n_sample,
            ncols=n_example,
            sharex=True,
            sharey=True,
            figsize=(8, 3)
        )

        def animate_diff(i, x_gen_store):
            print(
                f"gif animating frame {i} of",
                x_gen_store.shape[0],
                end='\r'
            )
            plots = []
            for row in range(int(n_sample)):
                for col in range(n_example):
                    axs[row, col].clear()
                    axs[row, col].set_xticks([])
                    axs[row, col].set_yticks([])
                    # plots.append(axs[row, col].imshow(
                    #     x_gen_store[i,(row*n_classes)+col,0],
                    #     cmap='gray')
                    # )
                    plots.append(axs[row, col].imshow(
                        -x_gen_store[i, (row*n_example)+col, 0],
                        cmap='gray',
                        vmin=(-x_gen_store[i]).min(),
                        vmax=(-x_gen_store[i]).max())
                    )
            return plots
        ani = FuncAnimation(
            fig,
            animate_diff,
            fargs=[x_gen_store],
            interval=200,
            blit=False,
            repeat=True,
            frames=x_gen_store.shape[0]
        )
        ani.save(
            result_dir / f"gif_ep{ep}.gif",
            dpi=100,
            writer=PillowWriter(fps=5)
        )
        print(f"saved image at {result_dir}/gif_ep{ep}.gif")


def train_epoch(
    ddpm: DDPM,
    train_loader: DataLoader,
    optim: Optimizer,
    lr: float,
    ep: int,
    n_epoch: int
):
    """A single epoch of training the DDPM.

    The DDPM is trained to generate MNIST images while conditioned on features.
    These features were extracted from a convolutional neural network by a
    sparse autoencoder.

    The aim is to use the trained DDPM to interpret the features by lettinng
    the DDPM generate images while conditioned on edited versions of the
    features.

    Args:
        ddpm: The diffusion model that is trained and mutated.
        train_loader: This yields (features, image, class) in batches.
            It should terminate once all batches have been iterated over once.
        optim: The optimizer, which is probably Adam.
        lr: Learning rate
        ep: The epoch number keeps track of how many previous training epochs
            have occured.
        n_epoch: The total amount of epochs for training.
    """
    ddpm.train()

    # linear learning rate decay
    optim.param_groups[0]['lr'] = lr * (1 - ep / n_epoch)

    pbar = tqdm(train_loader)
    loss_ema = None
    for f, x, _ in pbar:
        optim.zero_grad()
        x = x.to(DEVICE)
        f = f.to(DEVICE)
        loss = ddpm(x, f)
        loss.backward()
        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
        pbar.set_description(f"loss: {loss_ema:.4f}")
        optim.step()


def train(
    features_path: Path,
    mnist_path: Path,
    run_dir: Path,
    n_epoch: int = 20,
    batch_size: int = 64,
    n_T: int = 400,  # 500
    hidden_size: int = 128,  # 128 ok, 256 better (but slower)
    lr: float = 1e-4,
    drop_prob: float = 0.0,
    n_example: int = 5,  # Amount of test set images to use
    n_sample: int = 4,  # Amount of generations per test set image
):
    """Create and train a DDPM on MNIST conditioned on features from a CNN.

    After each epoch, the DDPM parameters are stored in
    $run_dir/model/epoch-$epoch_number.pth
    Load these with:
        ddpm = DDPM(*foo, **bar)
        ddpm.load_state_dict(torch.load($path))

    Args:
        features_path: the directory containing a train and test subdirectory.
            Those subdirectories contain numbered `.npy` files. Each file
            contains a feature array corresponding to an MNIST image.
        mnist_path: the directory at which MNIST resides or should be
            downloaded.
        run_dir: a directory unique to this run. It will be populated with
            subdirectories for the DDPM models, logs and images generated
            during evaluation. This may already be prepopulated, in which case
            parts or all aspects of training and evaluation will be skipped.
        n_epoch: the number of times the DDPM will be fed the entire MNIST
            dataset. Every epoch, the model will be evaluated and stored.
        batch_size: the number of data points to feed into the DDPM at once. At
            the end of every batch, gradients are computed and backpropagated
            to adjust the DDPM's weights.
        n_T: the number of steps taken during the image generation process.
        hidden_size: the size of the activations at the slimmest part of the
            U-net in the DDPM.
        lr: the learning rate decreases the size of the weight updates. The
            learning rate decreases linearly per epoch.
        n_example: the number of test images that the DDPM will try to
            approximate during evaluation.
        n_sample: the number of generations per example test image the DDPM
            creates during evaluation.
    """
    # mnist is already normalised 0 to 1
    # _ = transforms.Compose([transforms.ToTensor()])
    train_loader, test_loader = load_mnist_with_features(
        features_path,
        mnist_path,
        batch_size
    )
    test_loader = iter(test_loader)
    n_classes = next(iter(train_loader))[0].shape[-1]

    ddpm = DDPM(
        nn_model=ContextUnet(
            in_channels=1,
            hidden_size=hidden_size,
            n_classes=n_classes
        ),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=DEVICE,
        drop_prob=drop_prob
    )
    ddpm.to(DEVICE)
    ddpm_loaded = False

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    model_dir = run_dir / "model"
    result_dir = run_dir / "result"
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    for ep in range(n_epoch):

        model_path = model_dir / f"epoch-{ep}.pth"
        if not model_path.exists():
            print(f'epoch {ep}')
            train_epoch(ddpm, train_loader, optim, lr, ep, n_epoch)
            torch.save(ddpm.state_dict(), model_path)
            ddpm_loaded = True

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            if (result_dir / f"image_ep{ep}.png").exists():
                continue
            if not ddpm_loaded:
                ddpm.load_state_dict(torch.load(model_path))
                ddpm_loaded = True

            f, x, _ = next(test_loader)
            evaluate(
                ddpm,
                f[:n_example],
                x[:n_example],
                n_sample,
                result_dir,
                ep=ep,
                gif=ep % 5 == 0 or ep == int(n_epoch-1)
            )


if __name__ == "__main__":
    for RUN in O.run:
        RUN_DIR = jo3run_dir(RUN)
        if not RUN_DIR.exists():
            RUN_DIR.mkdir()
            argtoml.save(O, RUN_DIR / "config.toml")
            argtoml.save(RUN, RUN_DIR / "hyparam.toml")
        train(
            RUN.feature_dir,
            O.mnist_dir,
            run_dir=RUN_DIR,
            n_epoch=RUN.epochs,
            hidden_size=RUN.hidden_size,
            batch_size=RUN.batch_size,
            drop_prob=RUN.drop_prob
        )
