import jax
import jax.dlpack
import jax.random as jrd
import jax.numpy as jnp
import jax.scipy as jsp
import torch
from torch import Tensor
import torch.nn.functional as F

key = jrd.PRNGKey(0)
key, subkey = jrd.split(key)

# H and W need to be upscaled
# jax's x = C×H×W
x = jnp.arange(3 * 4 * 4, dtype=jnp.float32).reshape(3, 4, 4)
x_jax = jnp.repeat(x, 2, axis=-1)
x_jax = jnp.repeat(x_jax, 2, axis=-2)
print(x_jax.shape)

# torch's x = B×C×H×W
x_torch = jnp.stack((x, x))
x_torch = torch.from_dlpack(jax.dlpack.to_dlpack(x_torch))
x_torch = F.interpolate(x_torch, scale_factor=2, mode="nearest")
print(x_torch.shape)

# final comparison
x_torch_arr = jax.dlpack.from_dlpack(torch.to_dlpack(x_torch))[0]
assert jnp.array_equal(x_torch_arr, x_jax)

