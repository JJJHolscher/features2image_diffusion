#! /usr/bin/env python3
# vim:fenc=utf-8

from pathlib import Path

import argtoml
import jax
import jax.random as jrd
import jax.numpy as jnp

from . import logger
from .image_datasets import load_data
from .resample import create_named_schedule_sampler
from .script_util import create_model_and_diffusion

from .train import TrainLoop


def train_main(args):
    logger.configure()
    key = jrd.PRNGKey(args["seed"])
    keys = jrd.split(key, 2)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args["architecture"], key=keys[0])
    # model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args["schedule_sampler"], diffusion)

    logger.log("creating data loader...")
    data = load_data(
        feature_dir=args["feature_dir"],
        image_dir=args["image_dir"],
        batch_size=args["batch_size"],
        image_size=args["architecture"]["image_size"],
    )

    logger.log("training...")
    TrainLoop(
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
    ).run_loop(key=keys[1])


def sample_main(args):
    logger.configure()

    key = jax.random.PRNGKey(args["seed"])

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args)
    model.load_state_dict(
        dist_util.load_state_dict(args["model_path"], map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args["batch_size"] < args["num_samples"]:
        model_kwargs = {}
        if args["class_cond"]:
            classes = jrd.randint(
                key=key, minval=0, maxval=NUM_CLASSES, shape=(args["batch_size"],)) #, device=dist_util.dev())
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args["use_ddim"] else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args["batch_size"], 3, args["image_size"], args["image_size"]),
            clip_denoised=args["clip_denoised"],
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(jnp.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [jnp.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args["class_cond"]:
            gathered_labels = [
                jnp.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args["batch_size"]} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args["num_samples"]]
    if args["class_cond"]:
        label_arr = jnp.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args["num_samples"]]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args["class_cond"]:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


if __name__ == "__main__":
    O = argtoml.parse_args(Path("improved-diffusion.toml"))
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_enable_x64", True)
    if "train" in O and O["train"]:
        train_main(O)
    if "sample" in O and O["sample"]:
        sample_main(O)
    
