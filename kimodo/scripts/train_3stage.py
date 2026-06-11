# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train the 3-stage text-to-motion model on the bones_seed (SOMA) data.

This is the SOMA counterpart to ``train_3stage_hml3d.py``. The 3-stage
architecture, training step, checkpointing, and viz are identical — only
the dataset side differs:

  * ``train_3stage_hml3d.py`` uses :class:`HumanML3DTextMotionDataset` with
    HML3D-specific knobs (window_size, unit_length, min_motion_len) and
    ``KimodoLoss(fk_kind="chainreset_hml3d")`` (chain-reset FK).
  * This file uses :func:`build_soma_dataset` (dispatches between
    :class:`SOMATextMotionDataset` and :class:`SOMABonesSeedDataset` based
    on ``cfg.data.kind``) with SOMA knobs (max_frames, pad_to_max_frames,
    prefetch_factor) and the default standard parent-relative FK loss.

Pipeline (forward) — same as the HML3D version::

    text_feat       = LLM2Vec(text)                          # frozen
    mask_logits     = MaskGenerator(text_feat, pad_mask)
    mask_probs      = sigmoid(mask_logits)                   # continuous in [0, 1]
    values          = ConstraintGenerator(text_feat, mask_probs, pad_mask)
    observed_motion = mask_probs * values                    # gated values
    motion_t        = q_sample(motion, t)
    pred_clean      = Denoiser(motion_t, ..., motion_mask=mask_probs,
                                              observed_motion=observed_motion)

Run::

    python -m kimodo.scripts.train_3stage \\
        --config configs/training/bones_seed_3stage_small.yaml

Multi-GPU::

    KIMODO_TEXT_ENCODER_LOCAL_DIR=/weka/jungbin/hf_models \\
    torchrun --standalone --nproc_per_node=8 --redirects 3 --tee 3 \\
        -m kimodo.scripts.train_3stage \\
        --config configs/training/bones_seed_3stage_small.yaml \\
        output_dir=/weka/jungbin/kimodo_runs/bones_seed_3stage_v1
"""

from __future__ import annotations

import os as _os  # noqa: E402
if int(_os.environ.get("WORLD_SIZE", "1")) > 1:
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.cuda.amp import GradScaler  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from kimodo.data import build_collate_fn  # noqa: E402
from kimodo.model.common import instantiate_from_dict  # noqa: E402
from kimodo.model.constraint_generator import ConstraintGenerator  # noqa: E402
from kimodo.model.diffusion import Diffusion  # noqa: E402
from kimodo.model.mask_generator import MaskGenerator  # noqa: E402
from kimodo.model.three_stage import ThreeStageKimodo  # noqa: E402
from kimodo.scripts.train import (  # noqa: E402
    KimodoLoss,
    ModelEMA,
    build_soma_dataset,
    build_text_encoder,
    load_config,
    maybe_barrier,
    setup_distributed,
)
from kimodo.scripts.train_3stage_hml3d import (  # noqa: E402
    load_3stage_checkpoint,
    save_3stage_checkpoint,
    train_one_step_3stage,
    viz_3stage_samples,
)

log = logging.getLogger("kimodo.train_3stage")


# -----------------------------------------------------------------------------
# Model build (same shape as the HML3D version; only the dataset differs).
# -----------------------------------------------------------------------------
def _build_3stage_model(model_cfg_path: str, stats_path: str, fps_override: Optional[int]):
    """Instantiate Stage 1 / 2 / 3 sub-modules with a SHARED motion_rep.

    Returns ``(three_stage_module, diffusion)``.
    """
    raw = OmegaConf.to_container(OmegaConf.load(model_cfg_path), resolve=False)

    denoiser_cfg = raw["motion_denoiser"]
    denoiser_cfg["ckpt_path"] = None
    mr_cfg = denoiser_cfg["motion_rep"]
    mr_cfg["stats_path"] = stats_path
    if fps_override is not None:
        mr_cfg["fps"] = int(fps_override)
    denoiser = instantiate_from_dict(denoiser_cfg)
    motion_rep = denoiser.motion_rep

    mg_cfg = dict(raw["mask_generator"])
    cg_cfg = dict(raw["constraint_generator"])
    mg_cfg.pop("_target_", None)
    cg_cfg.pop("_target_", None)
    mask_generator = MaskGenerator(motion_rep=motion_rep, **mg_cfg)
    constraint_generator = ConstraintGenerator(motion_rep=motion_rep, **cg_cfg)

    three_stage = ThreeStageKimodo(mask_generator, constraint_generator, denoiser)

    diff_cfg = raw.get("diffusion", {})
    num_base_steps = int(diff_cfg.get("num_base_steps", 1000))
    diffusion = Diffusion(num_base_steps=num_base_steps)

    return three_stage, diffusion


def _build_cpu_motion_rep_for_dataset(model_cfg_path: str, stats_path: str, fps_override: Optional[int]):
    raw = OmegaConf.to_container(OmegaConf.load(model_cfg_path), resolve=False)
    mr_cfg = dict(raw["motion_denoiser"]["motion_rep"])
    mr_cfg["stats_path"] = stats_path
    if fps_override is not None:
        mr_cfg["fps"] = int(fps_override)
    return instantiate_from_dict(mr_cfg)


# -----------------------------------------------------------------------------
# Warm-start helper — handles both .safetensors (raw state_dict) and .pt
# (Kimodo training checkpoint with top-level 'denoiser' key whose weights
# live under ``root_model.*``).
# -----------------------------------------------------------------------------
def _maybe_load_subset(target_module: nn.Module, ckpt_path: Optional[str], prefix_options) -> None:
    """Load a state_dict subset into ``target_module``, auto-handling wrappers.

    Args:
        target_module: nn.Module to load into (e.g. ``three_stage.denoiser``).
        ckpt_path: path to ``.pt`` or ``.safetensors`` — None to skip.
        prefix_options: tuple of prefix strings to try on the unwrapped dict.
            First prefix with any matching key wins (its prefix is stripped
            before loading). If no prefix matches, the unwrapped state_dict
            is loaded directly with ``strict=False`` — relies on key shapes
            matching as-is (correct for SOMA single-stage ``.pt`` whose
            inner keys are already in the shape ``three_stage.denoiser``
            expects, namely ``root_model.*``).
    """
    if not ckpt_path:
        return
    # Don't use ``kimodo.model.loading.load_checkpoint_state_dict`` — it
    # eagerly calls ``.detach()`` on every value, which crashes on Kimodo
    # ``.pt`` training checkpoints whose top-level dict contains non-tensor
    # values like ``step: int`` / ``optimizer: dict``. Load raw and find the
    # tensor sub-dict ourselves.
    ckpt_path_str = str(ckpt_path)
    if ckpt_path_str.endswith(".safetensors"):
        from safetensors.torch import load_file as load_safetensors
        sd_full = load_safetensors(ckpt_path_str)
    else:
        sd_full = torch.load(ckpt_path_str, map_location="cpu", weights_only=False)

    def _is_tensor_dict(d):
        return isinstance(d, dict) and len(d) > 0 and all(torch.is_tensor(v) for v in d.values())

    sd = None
    if _is_tensor_dict(sd_full):
        sd = sd_full
    elif isinstance(sd_full, dict):
        # Look for the model sub-dict under common wrapper keys.
        for top_key in ("denoiser", "three_stage", "model", "state_dict"):
            if top_key in sd_full and _is_tensor_dict(sd_full[top_key]):
                sd = sd_full[top_key]
                log.info("Unwrapping checkpoint top-level key %r (%d tensors)", top_key, len(sd))
                break
    if sd is None:
        raise ValueError(
            f"Could not find a tensor sub-dict in checkpoint {ckpt_path_str}. "
            f"Top-level keys: {list(sd_full.keys()) if isinstance(sd_full, dict) else type(sd_full)}"
        )

    chosen_prefix = None
    for prefix in prefix_options:
        if any(k.startswith(f"{prefix}.") for k in sd.keys()):
            chosen_prefix = prefix
            break
    if chosen_prefix is None:
        log.info(
            "No prefix in %s matched; loading raw state_dict (%d tensors) into %s.",
            list(prefix_options), len(sd), type(target_module).__name__,
        )
        info = target_module.load_state_dict(sd, strict=False)
    else:
        sd_stripped = {
            k.replace(f"{chosen_prefix}.", "", 1): v
            for k, v in sd.items() if k.startswith(f"{chosen_prefix}.")
        }
        log.info(
            "Stripping prefix %r — loading %d/%d tensors into %s.",
            chosen_prefix, len(sd_stripped), len(sd), type(target_module).__name__,
        )
        info = target_module.load_state_dict(sd_stripped, strict=False)
    if info.missing_keys:
        log.info("  missing keys: %d (first 3: %s)", len(info.missing_keys), info.missing_keys[:3])
    if info.unexpected_keys:
        log.info("  unexpected keys: %d (first 3: %s)", len(info.unexpected_keys), info.unexpected_keys[:3])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    OmegaConf.resolve(cfg)

    env = setup_distributed()
    seed = int(cfg.trainer.seed) + env.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda", env.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    output_dir = Path(cfg.output_dir)
    if env.is_main:
        from kimodo.scripts._config_snapshot import snapshot_configs
        snapshot_configs(cfg, output_dir)
    maybe_barrier()

    log_path = output_dir / f"train_rank{env.rank}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )

    def _excepthook(exc_type, exc, tb):
        log.error("Uncaught exception on rank %d", env.rank, exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _excepthook

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("Distributed: world_size=%d rank=%d local_rank=%d", env.world_size, env.rank, env.local_rank)

    # ----- Build all three trainable stages + diffusion -----
    three_stage, diffusion = _build_3stage_model(
        cfg.model_config_path, cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    )
    three_stage = three_stage.to(device)
    diffusion = diffusion.to(device)
    motion_rep = three_stage.motion_rep
    log.info("motion_rep_dim=%d, nbjoints=%d, fps=%d",
             motion_rep.motion_rep_dim, motion_rep.nbjoints, motion_rep.fps)

    # Optional warm-starts. For Stage 3 from a SOMA single-stage ``.pt``,
    # the weights are wrapped under top-level ``denoiser`` (we unwrap) and
    # then live as ``root_model.*`` inside — which is exactly the layout
    # ``three_stage.denoiser`` (a Kimodo wrapper) expects, so no further
    # prefix strip is needed.
    _maybe_load_subset(
        three_stage.denoiser,
        cfg.get("init_motion_denoiser_safetensors"),
        ("motion_denoiser",),  # only used if the file actually has this prefix
    )
    _maybe_load_subset(
        three_stage.mask_generator,
        cfg.get("init_mask_generator_safetensors"),
        ("mask_generator",),
    )
    _maybe_load_subset(
        three_stage.constraint_generator,
        cfg.get("init_constraint_generator_safetensors"),
        ("constraint_generator",),
    )

    # ----- Dataset -----
    dataset_motion_rep = _build_cpu_motion_rep_for_dataset(
        cfg.model_config_path, cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    )
    # Stats for viz (used to unnormalize features before dumping). KimodoMotionRep
    # stores mean/std under .stats (a Stats wrapper that registers the tensors as
    # buffers); the HumanML3D native rep stores them directly on the module.
    if hasattr(dataset_motion_rep, "stats"):
        _mean_t = dataset_motion_rep.stats.mean
        _std_t = dataset_motion_rep.stats.std
    else:
        _mean_t = dataset_motion_rep.mean
        _std_t = dataset_motion_rep.std
    mean = _mean_t.detach().cpu().numpy().astype(np.float32)
    std = _std_t.detach().cpu().numpy().astype(np.float32)

    def _build_dataset():
        return build_soma_dataset(cfg, dataset_motion_rep, seed=seed)

    if env.is_main:
        dataset = _build_dataset()
    maybe_barrier()
    if not env.is_main:
        dataset = _build_dataset()
    maybe_barrier()

    sampler = None
    if env.is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=int(cfg.trainer.seed))
    pad_to = (
        int(cfg.data.get("pad_to_max_frames", cfg.data.max_frames))
        if cfg.data.get("pad_to_max_frames", True) else None
    )
    num_workers = int(cfg.data.num_workers)
    prefetch_factor = int(cfg.data.get("prefetch_factor", 4)) if num_workers > 0 else None
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=int(cfg.trainer.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=True,
        collate_fn=build_collate_fn(pad_to=pad_to),
        persistent_workers=bool(num_workers) and True,
    )
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**loader_kwargs)

    # ----- Optimizer / scheduler -----
    optimizer = torch.optim.AdamW(
        three_stage.parameters(),
        lr=float(cfg.trainer.lr),
        betas=tuple(cfg.trainer.betas),
        weight_decay=float(cfg.trainer.weight_decay),
    )
    warmup_steps = int(cfg.trainer.warmup_steps)
    num_steps = int(cfg.trainer.num_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- AMP / scaler -----
    mp = str(cfg.trainer.mixed_precision).lower()
    autocast_dtype: Optional[torch.dtype]
    if mp == "fp16":
        autocast_dtype = torch.float16
        scaler = GradScaler()
    elif mp == "bf16":
        autocast_dtype = torch.bfloat16
        scaler = None
    else:
        autocast_dtype = None
        scaler = None

    ema: Optional[ModelEMA] = None
    if float(cfg.trainer.ema_decay) > 0:
        ema = ModelEMA(three_stage, decay=float(cfg.trainer.ema_decay))

    # ----- Text encoder (frozen, serialised across ranks) -----
    log.info(
        "Building text encoder (HF_HUB_OFFLINE=%s, KIMODO_TEXT_ENCODER_LOCAL_DIR=%s) ...",
        os.environ.get("HF_HUB_OFFLINE", "0"),
        os.environ.get("KIMODO_TEXT_ENCODER_LOCAL_DIR", "(unset)"),
    )
    if env.is_distributed:
        for r in range(env.world_size):
            if env.rank == r:
                log.info("Rank %d building text encoder ...", r)
                text_encoder = build_text_encoder(cfg.text_encoder, device=device)
                log.info("Rank %d text encoder built.", r)
            maybe_barrier()
    else:
        text_encoder = build_text_encoder(cfg.text_encoder, device=device)

    # ----- DDP wrap -----
    if env.is_distributed:
        three_stage = DDP(three_stage, device_ids=[env.local_rank], find_unused_parameters=False)

    # ----- Loss (Stage 3 kimodo loss with DEFAULT 'standard' parent-relative FK) -----
    loss_fn = KimodoLoss(
        motion_rep,
        dict(cfg.trainer.loss_weights),
        smooth_l1_beta=float(cfg.trainer.get("smooth_l1_beta", 1.0)),
    ).to(device)

    # ----- Resume -----
    start_step = 0
    if cfg.trainer.get("resume_from"):
        start_step = load_3stage_checkpoint(
            Path(cfg.trainer.resume_from),
            three_stage, optimizer, scheduler, scaler, ema, device,
        )

    # ----- Logging sinks -----
    tb_writer = None
    if env.is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(output_dir / "tb"))
        except Exception as e:
            log.warning("TensorBoard not available: %s", e)

    log.info("Starting training from step %d for %d steps", start_step, num_steps)
    three_stage.train()

    grad_accum = int(cfg.trainer.grad_accum)
    log_every = int(cfg.trainer.log_every)
    ckpt_every = int(cfg.trainer.ckpt_every)
    viz_every = int(cfg.trainer.viz_every)
    grad_clip = float(cfg.trainer.grad_clip)
    ema_every = int(cfg.trainer.ema_every)
    text_drop_prob = float(cfg.trainer.text_drop_prob)
    mask_drop_prob = float(cfg.trainer.mask_drop_prob)
    w_motion = float(cfg.trainer.w_motion)
    w_stage2_aux = float(cfg.trainer.w_stage2_aux)
    w_sparsity = float(cfg.trainer.w_sparsity)
    sparsity_target = float(cfg.trainer.sparsity_target)
    smooth_l1_beta = float(cfg.trainer.get("smooth_l1_beta", 1.0))

    step = start_step
    epoch = 0
    while step < num_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            step_t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            agg_loss: Dict[str, float] = {}
            for _accum in range(grad_accum):
                losses = train_one_step_3stage(
                    three_stage, diffusion, loss_fn, batch, text_encoder, device,
                    text_drop_prob, mask_drop_prob, autocast_dtype,
                    w_motion=w_motion, w_stage2_aux=w_stage2_aux,
                    w_sparsity=w_sparsity, sparsity_target=sparsity_target,
                    smooth_l1_beta=smooth_l1_beta,
                )
                loss = losses["loss"] / grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                for k, v in losses.items():
                    agg_loss[k] = agg_loss.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v) / grad_accum

            if scaler is not None:
                scaler.unscale_(optimizer)
            params_for_clip = (
                three_stage.parameters() if not hasattr(three_stage, "module") else three_stage.module.parameters()
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params_for_clip, grad_clip if grad_clip > 0 else float("inf"),
            )
            agg_loss["grad_norm"] = float(grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            if ema is not None and step % ema_every == 0:
                ema.update(three_stage.module if hasattr(three_stage, "module") else three_stage)

            step += 1
            step_dt = time.time() - step_t0

            if env.is_main and step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                line = (
                    f"step={step} loss={agg_loss['loss']:.4f} "
                    f"grad_norm={agg_loss.get('grad_norm', 0.0):.2f} "
                    f"lr={lr_now:.2e} dt={step_dt:.2f}s | "
                    f"motion={agg_loss.get('l_motion_total',0):.3f} "
                    f"stage2_aux={agg_loss.get('l_stage2_aux',0):.3f} "
                    f"sparsity={agg_loss.get('l_sparsity',0):.4f} "
                    f"mask_density={agg_loss.get('mask_density',0):.4f}"
                )
                log.info(line)
                if tb_writer is not None:
                    for k, v in agg_loss.items():
                        tb_writer.add_scalar(f"train/{k}", v, step)
                    tb_writer.add_scalar("train/lr", lr_now, step)
                    tb_writer.add_scalar("train/step_time", step_dt, step)
                    if torch.cuda.is_available():
                        dev_idx = device.index if device.index is not None else 0
                        tb_writer.add_scalar("gpu/mem_allocated_GB", torch.cuda.max_memory_allocated(dev_idx) / 1e9, step)
                        tb_writer.add_scalar("gpu/mem_reserved_GB", torch.cuda.max_memory_reserved(dev_idx) / 1e9, step)
                        torch.cuda.reset_peak_memory_stats(dev_idx)

            if env.is_main and ckpt_every and step % ckpt_every == 0:
                save_3stage_checkpoint(
                    output_dir / f"ckpt_step{step:07d}.pt",
                    three_stage, optimizer, scheduler, scaler, ema, step, cfg,
                )
                latest = output_dir / "latest.pt"
                try:
                    if latest.exists() or latest.is_symlink():
                        latest.unlink()
                    latest.symlink_to(f"ckpt_step{step:07d}.pt")
                except OSError:
                    shutil.copy2(output_dir / f"ckpt_step{step:07d}.pt", latest)

            if env.is_main and viz_every and step % viz_every == 0:
                try:
                    viz_3stage_samples(
                        three_stage, diffusion, text_encoder, device, cfg, step,
                        output_dir / "viz", mean=mean, std=std,
                    )
                except Exception as e:
                    log.warning("Viz failed at step %d: %s", step, e)

            if step >= num_steps:
                break
        epoch += 1

    if env.is_main:
        save_3stage_checkpoint(
            output_dir / "ckpt_final.pt",
            three_stage, optimizer, scheduler, scaler, ema, step, cfg,
        )
    from kimodo.scripts.train import cleanup_distributed
    cleanup_distributed(env)


if __name__ == "__main__":
    main()
