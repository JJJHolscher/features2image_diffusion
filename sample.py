# import debugpy
import jax

# debugpy.listen(5678)
# Make variables visible to the debugger inside jitted fuctions.
# jax.config.update("jax_disable_jit", True)
# On my debugputer the gpu is not powerful enought.
# jax.config.update("jax_platform_name", "cpu")
print("starting")

import argtoml
from jo3util.eqx import load as jo3load
from improved_diffusion_equinox import script_util
from pathlib import Path

args = argtoml.parse_args(["improved-diffusion.toml", "imagenet64_cond_270M_250K.toml"], grandparent=False)
# args["diffusion"]["steps"] = 2

model = jo3load(args["model_path"], script_util.create_model)
diffusion = script_util.create_gaussian_diffusion(**args["diffusion"])

import jax.random as jrd
import jax.numpy as jnp

key = jrd.PRNGKey(args["seed"])
key, k1, k2 = jrd.split(key, 3)


import pickle
with open("./sample.pickle", 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(key, outp, pickle.HIGHEST_PROTOCOL)



batch_size = 16 # 16 # args["batch_size"] # change this to 1 for debugging
model_kwargs = {"y": jrd.randint(
    key=k1,
    minval=0,
    maxval=1000, # 1000, change this to 0 for debugging
    shape=(batch_size,)
)}
sample_fn = (
    diffusion.p_sample_loop if not args["use_ddim"] else diffusion.ddim_sample_loop
)

# print("debugpy is waiting for a client...")
# debugpy.wait_for_client()
# print("... client connected!")


sample = sample_fn(
    model,
    (batch_size, 3, args["unet"]["image_size"], args["unet"]["image_size"]),
    clip_denoised=args["clip_denoised"],
    model_kwargs=model_kwargs,
    noise=jnp.zeros((batch_size, 3, 64, 64)) + 0.5,
    key=k2
)
print(sample)

import numpy as np

sample = jnp.clip((sample + 1) * 127.5, 0, 255)
# sample = jnp.clip(sample * 255, 0, 255)
sample = jnp.transpose(sample, (0, 2, 3, 1))
# sample = np.ascontiguousarray(sample, dtype=int)
sample = np.array(sample)
sample = sample.astype(np.uint8)


print(np.sum((sample[0] - sample[3])**2))

from PIL import Image

for i, s in enumerate(sample):
    img = Image.fromarray(s, "RGB")
    img.save(f"../out/sample/{i}.png")


# ```
# sample = sample.permute(0, 2, 3, 1)
# sample = sample.contiguous()
# ```

