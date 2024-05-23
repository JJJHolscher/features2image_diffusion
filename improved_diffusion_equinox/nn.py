"""
Various utilities for neural networks.
"""

import math

# import torch as th
# import torch.nn as nn
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.nn as jnn
import jax.tree_util as jtu


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
# class SiLU(eqx.Module):
    # def __call__(self, x):
        # return x * jnn.sigmoid(x)


# class GroupNorm32(nn.GroupNorm):
    # def forward(self, x):
        # return super().forward(x.float()).type(x.dtype)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module: eqx.Module):
    """
    Zero out the parameters of a module and return it.
    """
    eqx.tree_at(lambda m: m.weight, module, jnp.zeros_like(module.weight))  #type:ignore
    eqx.tree_at(lambda m: m.bias, module, jnp.zeros_like(module.bias))  #type:ignore
    return module


# def scale_module(module, scale):
    # """
    # Scale the parameters of a module and return it.
    # """
    # for p in module.parameters():
        # p.detach().mul_(scale)
    # return module


def mean_flat(array: jnp.ndarray):
    """
    Take the mean over all non-batch dimensions.
    """
    return array.mean(axis=list(range(0, len(array.shape))))


def timestep_embedding(timesteps: int, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
    )
    args = timesteps.astype(jnp.float32) * freqs
    embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        # embedding = jnp.concat([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        embedding = jnp.concat([embedding, jnp.zeros_like(embedding[0])], axis=-1)
    return embedding
