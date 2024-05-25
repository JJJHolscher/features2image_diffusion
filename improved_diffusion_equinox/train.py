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
import jax.tree_util as jtu
from jax.sharding import PositionalSharding
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

        self._load_and_sync_parameters()
        if self.use_fp16:
            self.model = self.model.convert_to_fp16()

        # self.model_params = list(self.model.parameters())
        # self.master_params = copy.deepcopy(model) # self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

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
                copy.deepcopy(self.model) for _ in range(len(self.ema_rate))
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
        ema_params = copy.deepcopy(self.model)

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
            batch, cond = jnp.array(batch.numpy()), jnp.array(cond.numpy())
            if self.use_fp16:
                batch = batch.astype(jax.dtypes.bfloat16)

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
        grads = self.forward_backward(batch, cond, key)
        # if self.use_fp16:
            # self.optimize_fp16()
        # else:

        # average grads and apply updates
        grads = TrainLoop.avg_pytree(grads)
        updates, self.opt_state = self.opt.update(grads, self.opt_state, self.model)
        params, static = eqx.partition(self.model, eqx.is_array)
        params = optax.apply_updates(params, updates)
        self.model = eqx.combine(params, static)

        self.optimize_normal(grads)
        self.log_step()

    # @eqx.filter_jit
    # @eqx.filter_grad(has_aux=True)
    # @staticmethod



    def forward_backward(self, batch, cond, key):
        all_grads = []
        # sharding = PositionalSharding(jax.devices()).reshape(8)
        # model = jax.device_put(self.ddp_model, sharding.replicate())
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch]  # .to(dist_util.dev())
            micro_cond = cond[i : i + self.microbatch]  # to(dist_util.dev())
            key, k1, k2 = jrd.split(key, 3)
            t, weights = self.schedule_sampler.sample(micro.shape[0], key=k1) # , dist_util.dev())

            # put on devices
            # t = jax.device_put(t, sharding)
            # weights = jax.device_put(weights, sharding)
            # micro = jax.device_put(micro, sharding)
            # micro_cond = jax.device_put(micro_cond, sharding)

            params, static = eqx.partition(self.model, eqx.is_array)
            # params, static = eqx.partition(self.ddp_model, eqx.is_array)

            def single_step(params, timesteps, micro_batch, micro_cond, key):
                model = eqx.combine(params, static)
                loss, mse, vb = self.diffusion.training_losses(
                    model,
                    micro_batch,
                    timesteps,
                    micro_cond,
                    key,
                )
                return loss, mse, vb

            single_step_1 = jax.pmap(single_step, in_axes=(None, 0, 0, 0, None))

            @eqx.filter_grad(has_aux=True)
            def single_step_2(params, timesteps, weights, micro_batch, micro_cond, key):
                loss, mse, vb = single_step_1(params, timesteps, micro_batch, micro_cond, key)
                out = (loss * weights).mean()
                return out, {"loss": loss, "mse": mse, "vb": vb}


            grads, terms = single_step_2(
                params,
                t,
                weights,
                micro,
                micro_cond,
                k2
            )

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in terms.items()}
            )
            all_grads.append(grads)
            # if self.use_fp16:
                # loss_scale = 2 ** self.lg_loss_scale
                # loss *= loss_scale

            print("loss:", terms["loss"].mean())
        return all_grads

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

    @staticmethod
    def avg_pytree(pytrees):
        # a list where each element is a list of leaves
        all_leaves = [jtu.tree_flatten(tree)[0] for tree in pytrees]

        avg = []
        for leaf in all_leaves[0]:
            if eqx.is_array(leaf):
                avg.append(jnp.zeros_like(leaf))
            else:
                avg.append(leaf)

        num_leaves = len(avg)
        for leaves in all_leaves:
            for i, leaf in enumerate(leaves):
                if eqx.is_array(leaf):
                    avg[i] += leaf / num_leaves

        # reconstruct the average leaves into a pytree
        treedef = jtu.tree_structure(pytrees[0])
        return jtu.tree_unflatten(treedef, avg)

    def optimize_normal(self, grads):
        self._log_grad_norm(grads)
        # self._anneal_lr()
        for i in range(len(self.ema_rate)):
            params, static = eqx.partition(self.ema_params[i], eqx.is_array)
            params = optax.incremental_update(params, grads, self.ema_rate[i])
            self.ema_params[i] = eqx.combine(params, static)
        # for rate, params in zip(self.ema_rate, self.ema_params):
            # update_ema(params, self.model, rate=rate)

    @staticmethod
    def _log_grad_norm(grads):
        sqsum = 0.0
        for p in jtu.tree_leaves(grads)[0]:
            if eqx.is_array(p):
                sqsum += (p ** 2).sum().item()
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

        save_checkpoint(0, self.model)
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
