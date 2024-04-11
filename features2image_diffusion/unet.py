""" 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

This script was copied by Jochem Hölscher in mid December 2023.
The main edits I made are in ContextUnet.forward and DDPM.sample.
Those changes have to do with making the context no longer one-hot.
Also the training function is moved to __main__.py.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        """
        standard ResNet style convolutional block
        """
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        """
        process and downscale the image feature maps
        """
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        """
        process and upscale the image feature maps
        """
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        generic one layer FC NN for embedding things  
        """
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, hidden_size=256, n_classes=10, img_len=28):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(
            in_channels, hidden_size, is_res=True
        )

        self.down1 = UnetDown(hidden_size, hidden_size)
        self.down2 = UnetDown(hidden_size, 2 * hidden_size)

        self.to_vec = nn.Sequential(nn.AvgPool2d(img_len // 4), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * hidden_size)
        self.timeembed2 = EmbedFC(1, 1 * hidden_size)
        self.contextembed1 = EmbedFC(n_classes, 2 * hidden_size)
        self.contextembed2 = EmbedFC(n_classes, 1 * hidden_size)

        self.up0 = nn.Sequential(
            # when concat temb and cemb end up w 6*hidden_size
            # nn.ConvTranspose2d(6 * hidden_size, 2 * hidden_size, 7, 7),
            # otherwise just have 2*hidden_size
            nn.ConvTranspose2d(
                2 * hidden_size,
                2 * hidden_size,
                img_len // 4,
                img_len // 4
            ),
            nn.GroupNorm(8, 2 * hidden_size),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * hidden_size, hidden_size)
        self.up2 = UnetUp(2 * hidden_size, hidden_size)
        self.out = nn.Sequential(
            nn.Conv2d(2 * hidden_size, hidden_size, 3, 1, 1),
            nn.GroupNorm(8, hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # mask out context if context_mask == 1
        c = c * (-1 * (1 - context_mask))  # need to flip 0 <-> 1

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.hidden_size * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.hidden_size * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.hidden_size, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.hidden_size, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # if want to avoid add and multiply embeddings
        # up2 = self.up1(up1, down2)
        # add and multiply embeddings
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(
        0, T + 1, dtype=torch.float32
    ) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(
        self,
        x: Float[torch.Tensor, "batch color width height"],
        c: Float[torch.Tensor, "batch feature"],
    ):
        """
        this method is used in training, so samples t and noise randomly
        """

        # t ~ Uniform(0, n_T)
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )
        # We should predict the "error term" from this x_t. We return loss.

        # dropout context with some probability
        context_mask = torch.bernoulli(
            torch.zeros_like(c) + self.drop_prob
        ).to(self.device)

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(
            noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask)
        )

    def sample(
        self,
        context: Float[torch.Tensor, "batch feature"],
        n_sample: int,
        size: tuple,  # usually (1, 28, 28)
        device: str,
        verbose: bool = True,
        store: bool = True
    ):
        """Generate images guided by the context.

        we follow the guidance sampling scheme described in
        'Classifier-Free Diffusion Guidance'
        to make the fwd passes efficient,
        we concat two versions of the dataset,
        one with context_mask=0 and the other context_mask=1
        we then mix the outputs with the guidance scale, w
        where w>0 means more guidance

        Args:
            context: features that guide the model during generation
            n_sample: the number of generations per feature
            size: shape of a single data point, in the case of mnist this
                is (1, 28, 28).
            device: the device (cpu, gpu, tpu) to run the matrix operations on
            guide_w: how much the context influences the generations
        """

        batch_size = n_sample * len(context)
        # sample initial noise x_T ~ N(0, 1)
        x_i = torch.randn(batch_size, *size).to(device)
        # context for us just cycles throught the mnist labels
        c_i = context.to(device)
        c_i = c_i.repeat((n_sample, 1))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        # c_i = c_i.repeat((2, 1))
        # context_mask = context_mask.repeat((2, 1))
        # context_mask[batch_size:] = 1.  # makes second half of batch context free

        x_i_store = []  # keep track of generated steps for plotting
        if verbose:
            print()
        for i in range(self.n_T, 0, -1):
            if verbose:
                print(f"sampling timestep {i}", end="\r")
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(batch_size, 1, 1, 1)

            # double batch
            # x_i = x_i.repeat(2, 1, 1, 1)
            # t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(batch_size, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            # eps1 = eps[:batch_size]
            # eps2 = eps[batch_size:]
            # eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:batch_size]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if store and (i % 20 == 0 or i == self.n_T or i < 8):
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


def load_ddpm(
    path: Path,
    n_classes: int,
    hidden_size: int,
    diffusion_steps: int,
    device: str,
):
    ddpm = DDPM(
        nn_model=ContextUnet(
            in_channels=1, hidden_size=hidden_size, n_classes=n_classes
        ),
        betas=(1e-4, 0.02),
        n_T=diffusion_steps,
        device=device,
        drop_prob=0.0,
    )
    ddpm.to(device)
    ddpm.load_state_dict(
        torch.load(path)
    )  # "./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))
    ddpm.to(device)
    return ddpm
