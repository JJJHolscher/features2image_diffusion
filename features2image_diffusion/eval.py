from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from jaxtyping import Float
from jo3util.debug import breakpoint as jo3breakpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MnistFeaturesSet, loader_to_dataframe
from .unet import DDPM, load_ddpm
from .vis import tabulate_generations


def generate_on_edits(
    feature_ids,
    centile_edits,
    data_ids,
    model_path,
    data_set,
    out_dir: Path,
    num_generations,
    device,
    n_channels,
    **kwargs,
):
    num_features = data_set[0][0].shape[-1]
    img_len = data_set[0][1].shape[-1]
    feature_ids = feature_ids if feature_ids else list(range(num_features))

    ddpm = load_ddpm(
        model_path,
        num_features,
        device=device,
        n_channels=n_channels,
        img_len=img_len,
        **kwargs,
    ).eval()

    centiles = calculate_centiles(data_set, centile_edits)

    for d, data_id in enumerate(data_ids):
        print(f"\n[{datetime.now()}] [{d}/{len(data_ids)}] {data_id=}")

        features, original_image, _ = data_set[data_id]
        features = torch.tensor(features)

        # Generate images without edited features for comparison.
        generate_and_save(
            f"{out_dir}/{data_id}/unedited",
            original_image,
            features,
            ddpm,
            num_generations,
            device,
        )

        for feature_id in tqdm(feature_ids):
            for e, edit in enumerate(centiles[:, feature_id]):
                # Edit a single feature and then generate images.
                edited_features = features[:]
                edited_features[feature_id] = edit
                generate_and_save(
                    f"{out_dir}/{data_id}/{feature_id}/{e}",
                    original_image,
                    edited_features,
                    ddpm,
                    num_generations,
                    device,
                )


def generate_and_save(
    path: str,
    original_image,
    features: Float[torch.Tensor, "feature"],
    *args,
    **kwargs,
):
    generations = generate(features.unsqueeze(0), *args, **kwargs)
    tabulate_generations(generations, original_image)
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    np.save(path + "-features.npy", features.numpy())
    np.save(path + "-generations.npy", generations)
    plt.savefig(path + "-images.png")
    plt.close()


def generate(
    features: Float[torch.Tensor, "batch feature"],
    ddpm: DDPM,
    num_generations: int,
    device,
    shape: tuple = (3, 64, 64),
) -> Float[np.ndarray, "num_generations shape"]:
    # A single call with 256 features, 8 generations and a batch size of 1,
    # takes about 15 seconds.
    # Going over all features for a single data point, each with 4 edits, takes
    # a bit more than 1 hour.
    with torch.no_grad():
        generations, _ = ddpm.sample(
            features,
            num_generations,
            shape,
            device,
            verbose=False,
            store=False,
        )
    return generations.cpu().numpy()


def calculate_centiles(data_set, centiles):
    df = loader_to_dataframe(DataLoader(data_set, batch_size=64))
    description = df.describe(centiles)
    out = []
    for c in centiles:
        out.append(description.loc[f"{int(c * 100)}%"].values[1:])
    out = np.stack(out)
    del df
    del description
    return out
