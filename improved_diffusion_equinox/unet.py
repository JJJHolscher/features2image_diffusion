#! /usr/bin/env python3
# vim:fenc=utf-8

from abc import abstractmethod
from typing import List, Callable, Union, Set, Tuple, Optional

import math

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Float, Array, Int


# from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    # SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class TimestepBlock(eqx.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def __call__(self, x: Float[Array, "N C ..."], emb, state: nn.State):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        return x, state


class TimestepEmbedSequential(TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    layers: List[Callable]

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x: Float[Array, "N C ..."], emb, state: nn.State):
        for layer in self.layers:
            if isinstance(layer, TimestepBlock):
                x, state = layer(x, emb, state)
            else:
                x, state = layer(x, state)
        return x, state


class Upsample(eqx.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    channels: int
    dims: int
    conv: Optional[nn.Conv]

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)
        else:
            self.conv = None

    def __call__(self, x, state):
        assert x.shape[1] == self.channels
        # if self.dims == 3:
            # x = F.interpolate(
                # x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            # )
        # else:
            # x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = jnp.repeat(x, 2, axis=-1)
        if self.dims > 1:
            x = jnp.repeat(x, 2, axis=-2)
        if self.conv:
            x = self.conv(x)
        return x, state


class Downsample(eqx.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def __call__(self, x, state):
        assert x.shape[1] == self.channels
        return self.op(x), state


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential([
            normalization(channels),
            jnn.silu,
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        ])
        self.emb_layers = nn.Sequential([
            jnn.silu,
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        ])
        self.out_layers = nn.Sequential([
            normalization(self.out_channels),
            jnn.silu,
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        ])

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def __call__(self, x, emb, state):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h, state = self.in_layers(x)
        emb_out = self.emb_layers(emb) # .type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = jnp.array_split(emb_out, 2, axis=1)
            h, state = out_norm(h, state) * (1 + scale) + shift
            h, state = out_rest(h, state)
        else:
            h = h + emb_out
            h, state = self.out_layers(h, state)
        return self.skip_connection(x) + h, state


class QKVAttention(eqx.Module):
    """
    A module which performs QKV attention.
    """

    def __call__(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = jnp.split(qkv, ch, axis=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jnp.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = jax.nn.softmax(weight, axis=-1)  # weight.float(), dim=-1).type(weight.dtype)
        return jnp.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(jnp.prod(jnp.array(spatial)))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += jnp.array([matmul_ops]) # used to be th.DoubleTensor


class AttentionBlock(nn.StatefulLayer):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    channels: int
    num_heads: int
    norm: nn.GroupNorm
    qkv: nn.Conv
    attention: QKVAttention
    proj_out: nn.Conv

    def __init__(self, channels, num_heads=1):
        self.channels = channels
        self.num_heads = num_heads

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def __call__(self, x, state):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv, state = self.norm(x, state)
        qkv = self.qkv(qkv)
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial), state


class UNetModel(eqx.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    """
    in_channels: int
    model_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: Union[Set[int], List[int], Tuple[int]]
    dropout: float
    channel_mult: Union[Set[int], List[int], Tuple[int]]
    conv_resample: bool
    dims: int
    num_classes: Optional[int]
    num_heads: int
    num_heads_upsample: int
    time_embed: nn.Sequential
    label_emb: Optional[nn.Embedding]
    input_blocks: List[TimestepEmbedSequential]
    middle_block: TimestepEmbedSequential
    output_blocks: List[TimestepEmbedSequential]
    out: nn.Sequential

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential([
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        ])

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.label_emb = None

        self.input_blocks = [
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ]
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = []
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers: List[eqx.Module] = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential([
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        ])

    # def convert_to_fp16(self):
        # """
        # Convert the torso of the model to float16.
        # """
        # self.input_blocks.apply(convert_module_to_f16)
        # self.middle_block.apply(convert_module_to_f16)
        # self.output_blocks.apply(convert_module_to_f16)

    # def convert_to_fp32(self):
        # """
        # Convert the torso of the model to float32.
        # """
        # self.input_blocks.apply(convert_module_to_f32)
        # self.middle_block.apply(convert_module_to_f32)
        # self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        leaves = jtu.tree_leaves(self, is_leaf=eqx.is_array)
        return leaves[0].dtype

    def __call__(self, x: Float[Array, "N C ..."], timesteps: Int[Array, "N"], y: Optional[Float[Array, "N C ..."]], state: nn.State):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if y is not None and self.label_emb is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # h = x.type(self.inner_dtype)
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = jnp.concat([h, hs.pop()], axis=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = jnp.concat([h, hs.pop()], axis=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result
