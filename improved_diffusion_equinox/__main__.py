#! /usr/bin/env python3
# vim:fenc=utf-8

import datetime
from pathlib import Path

import argtoml
import jax
import jax.random as jrd
import jax.numpy as jnp

from . import logger
from .image_datasets import load_data
from .resample import create_named_schedule_sampler
from .script_util import create_model, create_gaussian_diffusion

from .train import TrainLoop


def train_main(args):
    logger.configure(dir=str(Path("../log") / datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")))
    key = jrd.PRNGKey(args["seed"])
    keys = jrd.split(key, 2)

    logger.log("creating model and diffusion...")
    model = create_model(**args["unet"])
    diffusion = create_gaussian_diffusion(**args["diffusion"])
    # model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args["schedule_sampler"], diffusion)

    logger.log("creating data loader...")
    data = load_data(
        feature_dir=args["feature_dir"],
        image_dir=args["image_dir"],
        batch_size=args["batch_size"],
        image_size=args["unet"]["image_size"],
    )

    logger.log("training...")

    loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args["batch_size"],
        microbatch=args["microbatch"],
        lr=args["lr"],
        ema_rate=args["ema_rate"],
        log_interval=args["log_interval"],
        save_interval=args["save_interval"],
        resume_checkpoint=args["resume_checkpoint"],
        use_fp16=args["use_fp16"],
        fp16_scale_growth=args["fp16_scale_growth"],
        schedule_sampler=schedule_sampler,
        weight_decay=args["weight_decay"],
        lr_anneal_steps=args["lr_anneal_steps"],
        model_init=args["unet"],
    )

    loop.run_loop(key=keys[1])


if __name__ == "__main__":
    O = argtoml.parse_args(Path("improved-diffusion.toml"))
    # jax.config.update("jax_numpy_dtype_promotion", "strict")
    # jax.config.update("jax_enable_x64", True)
    if O["debug"]:
        # jax.config.update("jax_disable_jit", True)
        import debugpy
        debugpy.listen(5678)
        print("debugpy is waiting for a client...")
        debugpy.wait_for_client()
        print("client connected")

    if O["train"]:
        train_main(O)
    # if "sample" in O and O["sample"]:
        # sample_main(O)
    
