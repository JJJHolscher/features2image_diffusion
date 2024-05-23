#! /usr/bin/env python3
# vim:fenc=utf-8

import copy
import functools
import os

import blobfile as bf
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
from jo3util.eqx import save as jo3save
from jo3util.eqx import load as jo3load
import optax

from . import logger #, dist_util
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossSecondMomentResampler, UniformSampler
from .script_util import create_model

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        model_init={},
    ):
        self.model = model
        self.model_init = model_init
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()

        # self.model_params = list(self.model.parameters())
        self.master_params = copy.deepcopy(model) # self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        self._load_and_sync_parameters()
        # if self.use_fp16:
            # self._setup_fp16()

        # self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.opt = optax.adamw(self.lr, weight_decay=self.weight_decay)
        self.opt_state = self.opt.init(eqx.filter(model, eqx.is_array))
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model = jo3load(resume_checkpoint, create_model)
            # self.model.load_state_dict(resume_checkpoint)
            # (dist_util.load_state_dict(
                    # resume_checkpoint, map_location=dist_util.dev()
            # ))

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            ema_params = jo3load(ema_checkpoint, create_model)

        return ema_params


    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.eqx"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.opt_state = eqx.tree_deserialise_leaves(opt_checkpoint, self.opt_state)


    def run_loop(self, key):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            batch, cond = batch.numpy(), cond.numpy()
            key, subkey = jrd.split(key)
            self.run_step(batch, cond, subkey)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, key):
        self.forward_backward(batch, cond, key)
        # if self.use_fp16:
            # self.optimize_fp16()
        # else:
        self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond, key):
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch]  # .to(dist_util.dev())
            micro_cond = cond[i : i + self.microbatch]  # to(dist_util.dev())
            key, subkey = jrd.split(key)
            t, weights = self.schedule_sampler.sample(micro.shape[0], key=subkey) # , dist_util.dev())

            (loss, terms), grads = jax.vmap(
                self.diffusion.training_losses, 
                (None, 0, 0, 0, None))(
                    self.ddp_model,
                    micro,
                    t,
                    micro_cond,
                    subkey
                )

            # if isinstance(self.schedule_sampler, LossSecondMomentResampler):
                # self.schedule_sampler.update_with_local_losses(
                    # t, losses["loss"]
                # )

            loss = (loss * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in terms.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                loss *= loss_scale

            updates, self.opt_state = self.opt.update(grads, self.opt_state, self.model)
            self.model = optax.apply_updates(self.model, updates)

    # def optimize_fp16(self):
        # if any(not jnp.isfinite(p.grad).all() for p in self.model_params):
            # self.lg_loss_scale -= 1
            # logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            # return
# 
        # model_grads_to_master_grads(self.model_params, self.master_params)
        # self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        # self._log_grad_norm()
        # self._anneal_lr()
        # self.opt.step()
        # for rate, params in zip(self.ema_rate, self.ema_params):
            # update_ema(params, self.master_params, rate=rate)
        # master_params_to_model_params(self.model_params, self.master_params)
        # self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        # self._anneal_lr()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", jnp.sqrt(sqsum))

    # def _anneal_lr(self):
        # if not self.lr_anneal_steps:
            # return
        # frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        # lr = self.lr * (1 - frac_done)
        # for param_group in self.opt.param_groups:
            # param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.eqx"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.eqx"
            bf.join(get_blob_logdir(), filename)
            jo3save(bf.join(get_blob_logdir(), filename), params, self.model_init)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        filename = bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.eqx")
        eqx.tree_serialise_leaves(filename, self.opt_state)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = str(filename).split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.eqx"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts, values):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
