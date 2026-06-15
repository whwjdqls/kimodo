# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape-aware Kimodo text-to-motion training script.

Sibling of ``train.py`` that additionally conditions the denoiser on each
actor's per-sample rest-pose joints (``neutral_joints`` (B, 30, 3)). The body
shape enters the model as a single prefix token alongside text/timestep/heading
via :class:`kimodo.model.shape_encoder.ShapeEncoder`; the FK loss runs against
the actor's actual bone lengths.

Backward-compatible by construction: every kimodo-module change Phase 1 + 4
introduced defaults the new kwargs to ``None``, so ``train.py``'s behavior is
bit-identical to its pre-change form (verified by ``Phase 7`` snapshot diff).

Usage (single GPU):
    python -m kimodo.scripts.train_skel_aware \\
        --config configs/training/bones_seed_skel_aware.yaml

Multi-GPU (single node):
    torchrun --standalone --nproc_per_node=8 \\
        -m kimodo.scripts.train_skel_aware \\
        --config configs/training/bones_seed_skel_aware.yaml
"""
from __future__ import annotations

import os as _os  # noqa: E402
if int(_os.environ.get("WORLD_SIZE", "1")) > 1:
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from kimodo.data import SOMABonesSeedDatasetShapeAware, build_collate_fn
from kimodo.model.diffusion import Diffusion
from kimodo.model.shape_encoder import ShapeEncoder

from kimodo.scripts.train import (
    # infra
    setup_distributed, cleanup_distributed, maybe_barrier,
    load_config,
    # builders
    build_denoiser_from_model_config, build_text_encoder, encode_texts,
    _build_cpu_motion_rep, _resolve_num_steps,
    # training pieces
    KimodoLoss, ModelEMA, ConstraintSampler,
    ensure_full_training_schedule,
    save_checkpoint, load_checkpoint,
    _NullCtx,
)

log = logging.getLogger("kimodo.train_skel_aware")


# -----------------------------------------------------------------------------
# Training step (shape-aware variant)
# -----------------------------------------------------------------------------
def train_one_step_skel_aware(
    denoiser: nn.Module,
    shape_encoder: nn.Module,
    diffusion: Diffusion,
    loss_fn: KimodoLoss,
    batch: Dict[str, Any],
    text_encoder,
    device: torch.device,
    text_drop_prob: float,
    shape_drop_prob: float,
    autocast_dtype: Optional[torch.dtype],
    constraint_sampler: Optional[ConstraintSampler] = None,
) -> Dict[str, torch.Tensor]:
    """Mirrors :func:`kimodo.scripts.train.train_one_step` with two additions:

      1. ``neutral_joints`` (B, 30, 3) is pulled from the batch, optionally
         dropped per-sample (``shape_drop_prob``), encoded by ``shape_encoder``,
         and fed to the denoiser as ``shape_feat``.
      2. ``neutral_joints`` is also forwarded to ``loss_fn`` so the FK position
         + velocity terms FK against the actor's actual rest pose.
    """
    motion: torch.Tensor = batch["motion"].to(device, non_blocking=True)
    pad_mask: torch.Tensor = batch["pad_mask"].to(device, non_blocking=True)
    first_heading: torch.Tensor = batch["first_heading_angle"].to(device, non_blocking=True)
    neutral_joints: torch.Tensor = batch["neutral_joints"].to(device, non_blocking=True)
    texts: List[str] = batch["text"]
    B = motion.shape[0]

    # Text drop for CFG.
    text_keep = torch.rand(B) > text_drop_prob
    if not text_keep.all():
        texts = [t if k else "" for t, k in zip(texts, text_keep.tolist())]
    text_feat, text_pad_mask = encode_texts(text_encoder, texts, device)
    if not text_keep.all():
        text_keep_mask = text_keep.to(device=device, dtype=text_feat.dtype).view(-1, 1, 1)
        text_feat = text_feat * text_keep_mask

    # Shape encoding. Per-sample drop replaces the encoded token with zeros so
    # the model learns to handle "unknown body". neutral_joints itself is NOT
    # zeroed — the FK loss still uses the actor's true neutrals so the loss
    # target is faithful; only the conditioning signal is dropped.
    shape_emb = shape_encoder(neutral_joints)                              # (B, 1, D_shape)
    if shape_drop_prob > 0.0:
        shape_keep = (torch.rand(B, device=device) > shape_drop_prob).to(shape_emb.dtype)
        shape_emb = shape_emb * shape_keep.view(-1, 1, 1)

    ensure_full_training_schedule(diffusion)
    t = torch.randint(0, diffusion.num_base_steps, (B,), device=device)
    noise = torch.randn_like(motion)
    motion_t = diffusion.q_sample(motion, t, noise=noise)

    if constraint_sampler is None:
        motion_mask = torch.zeros_like(motion)
        observed_motion = torch.zeros_like(motion)
    else:
        lengths = batch["lengths"].to(device, non_blocking=True)
        observed_motion, motion_mask = constraint_sampler(motion, lengths)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if (autocast_dtype is not None and device.type == "cuda")
        else _NullCtx()
    )

    with autocast_ctx:
        pred_clean = denoiser(
            motion_t,
            pad_mask,
            text_feat,
            text_pad_mask,
            t,
            first_heading_angle=first_heading,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            shape_feat=shape_emb,
        )
        pred_clean = pred_clean.float()
        losses = loss_fn(pred_clean, motion, pad_mask, neutral_joints=neutral_joints)
    losses["data_absmax"] = motion.detach().abs().max()
    losses["timestep_mean"] = t.float().mean()
    return losses


# -----------------------------------------------------------------------------
# Dataset / shape encoder / checkpoint helpers
# -----------------------------------------------------------------------------
def build_shape_aware_dataset(cfg: DictConfig, dataset_motion_rep, seed: int):
    return SOMABonesSeedDatasetShapeAware(
        data_root=cfg.data.data_root,
        natural_csv_path=cfg.data.natural_csv_path,
        temporal_labels_path=cfg.data.temporal_labels_path,
        multi_timeline_path=cfg.data.multi_timeline_path,
        train_split_path=cfg.data.get("train_split_path"),
        nan_audit_path=cfg.data.get("nan_audit_path"),
        fps=cfg.data.fps,
        max_clip_sec=float(cfg.data.get("max_clip_sec", 10.0)),
        max_segment_sec=float(cfg.data.get("max_segment_sec", 15.0)),
        rand_offset_max_sec=float(cfg.data.get("rand_offset_max_sec", 2.0)),
        min_frames=cfg.data.min_frames,
        include_mirrored=bool(cfg.data.include_mirrored),
        skeleton=dataset_motion_rep.skeleton,
        motion_rep=dataset_motion_rep,
        normalize=True,
        random_heading_aug=bool(cfg.data.random_heading_aug),
        cache_index=cfg.data.get("cache_index"),
        seed=seed,
        packed_motions_path=cfg.data.get("packed_proportional_motions_path"),
    )


def build_shape_encoder(cfg: DictConfig) -> ShapeEncoder:
    se_cfg = cfg.get("shape_encoder", {})
    return ShapeEncoder(
        n_joints=int(se_cfg.get("n_joints", 30)),
        hidden_dim=int(se_cfg.get("hidden_dim", 256)),
        output_dim=int(se_cfg.get("output_dim", 512)),
        num_layers=int(se_cfg.get("num_layers", 3)),
        normalize_output=bool(se_cfg.get("normalize_output", True)),
    )


def save_checkpoint_with_shape(
    out_path: Path,
    denoiser: nn.Module,
    shape_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[GradScaler],
    ema: Optional[ModelEMA],
    step: int,
    cfg: DictConfig,
) -> None:
    """Wrap ``save_checkpoint`` to also persist the shape encoder weights."""
    save_checkpoint(out_path, denoiser, optimizer, scheduler, scaler, ema, step, cfg)
    # Tack the shape encoder weights into the saved dict.
    blob = torch.load(out_path, map_location="cpu", weights_only=False)
    blob["shape_encoder"] = (
        shape_encoder.module.state_dict() if hasattr(shape_encoder, "module")
        else shape_encoder.state_dict()
    )
    torch.save(blob, out_path)


def load_shape_from_checkpoint(ckpt_path: Path, shape_encoder: nn.Module) -> None:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "shape_encoder" in blob:
        target = shape_encoder.module if hasattr(shape_encoder, "module") else shape_encoder
        target.load_state_dict(blob["shape_encoder"])
        log.info("Loaded shape_encoder weights from %s", ckpt_path)


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

    device = (
        torch.device("cuda", env.local_rank) if torch.cuda.is_available()
        else torch.device("cpu")
    )

    output_dir = Path(cfg.output_dir)
    if env.is_main:
        from kimodo.scripts._config_snapshot import snapshot_configs
        snapshot_configs(cfg, output_dir)
    maybe_barrier()

    log_path = output_dir / f"train_skel_aware_rank{env.rank}.log"
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

    # ----- denoiser & diffusion -----
    denoiser = build_denoiser_from_model_config(
        cfg.model_config_path,
        cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    ).to(device)
    if cfg.get("init_from_safetensors"):
        from kimodo.model.loading import load_checkpoint_state_dict
        sd = load_checkpoint_state_dict(cfg.init_from_safetensors)
        sd = {k.replace("denoiser.backbone.", ""): v for k, v in sd.items()}
        denoiser.load_state_dict(sd, strict=False)
        log.info("Initialized denoiser weights from %s", cfg.init_from_safetensors)

    diffusion = Diffusion(num_base_steps=int(_resolve_num_steps(cfg))).to(device)
    motion_rep = denoiser.motion_rep

    # ----- shape encoder -----
    shape_encoder = build_shape_encoder(cfg).to(device)
    log.info(
        "Built ShapeEncoder: J=%d D=%d (params=%d)",
        shape_encoder.n_joints, shape_encoder.output_dim,
        sum(p.numel() for p in shape_encoder.parameters()),
    )

    # ----- dataset / dataloader -----
    dataset_motion_rep = _build_cpu_motion_rep(
        cfg.model_config_path,
        cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    )

    def _build_dataset():
        return build_shape_aware_dataset(cfg, dataset_motion_rep, seed=seed)

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

    # ----- optimizer / scheduler (over BOTH denoiser AND shape_encoder) -----
    optimizer = torch.optim.AdamW(
        list(denoiser.parameters()) + list(shape_encoder.parameters()),
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
        ema = ModelEMA(denoiser, decay=float(cfg.trainer.ema_decay))

    # ----- text encoder -----
    log.info("Building text encoder ...")
    if env.is_distributed:
        for r in range(env.world_size):
            if env.rank == r:
                text_encoder = build_text_encoder(cfg.text_encoder, device=device)
            maybe_barrier()
    else:
        text_encoder = build_text_encoder(cfg.text_encoder, device=device)

    # ----- DDP wrap -----
    if env.is_distributed:
        denoiser = DDP(denoiser, device_ids=[env.local_rank], find_unused_parameters=False)
        shape_encoder = DDP(shape_encoder, device_ids=[env.local_rank], find_unused_parameters=False)

    # ----- loss -----
    loss_fn = KimodoLoss(
        motion_rep,
        cfg.trainer.loss_weights,
        smooth_l1_beta=float(cfg.trainer.get("smooth_l1_beta", 1.0)),
    )

    phase = str(cfg.trainer.get("phase", "text"))
    if phase not in ("text", "constraints"):
        raise ValueError(f"trainer.phase must be 'text' or 'constraints', got {phase!r}")
    constraint_sampler: Optional[ConstraintSampler] = None
    if phase == "constraints":
        cw = cfg.trainer.get("constraint_weights", None)
        constraint_sampler = ConstraintSampler(
            motion_rep,
            weights=OmegaConf.to_container(cw) if cw is not None else None,
            none_prob=float(cfg.trainer.get("constraint_none_prob", 0.10)),
            mix_prob=float(cfg.trainer.get("constraint_mix_prob", 0.25)),
            max_keyframes=int(cfg.trainer.get("constraint_max_keyframes", 20)),
            seed=int(cfg.trainer.seed) + 7 * env.rank,
        )

    start_step = 0
    if cfg.trainer.get("resume_from"):
        start_step = load_checkpoint(
            Path(cfg.trainer.resume_from),
            denoiser, optimizer, scheduler, scaler, ema, device,
        )
        load_shape_from_checkpoint(Path(cfg.trainer.resume_from), shape_encoder)

    tb_writer = None
    if env.is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(output_dir / "tb"))
        except Exception as e:
            log.warning("TensorBoard not available: %s", e)

    log.info("Starting training from step %d for %d steps", start_step, num_steps)
    denoiser.train()
    shape_encoder.train()

    grad_accum = int(cfg.trainer.grad_accum)
    log_every = int(cfg.trainer.log_every)
    ckpt_every = int(cfg.trainer.ckpt_every)
    grad_clip = float(cfg.trainer.grad_clip)
    ema_every = int(cfg.trainer.ema_every)
    text_drop_prob = float(cfg.trainer.text_drop_prob)
    shape_drop_prob = float(cfg.trainer.get("shape_drop_prob", 0.1))
    log.info("text_drop_prob=%.2f shape_drop_prob=%.2f", text_drop_prob, shape_drop_prob)

    step = start_step
    epoch = 0
    while step < num_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            step_t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            if constraint_sampler is not None:
                constraint_sampler.progress = step / max(1, num_steps)
            agg_loss: Dict[str, float] = {}
            for _accum in range(grad_accum):
                losses = train_one_step_skel_aware(
                    denoiser, shape_encoder, diffusion, loss_fn, batch,
                    text_encoder, device, text_drop_prob, shape_drop_prob,
                    autocast_dtype, constraint_sampler=constraint_sampler,
                )
                loss = losses["loss"] / grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                for k, v in losses.items():
                    agg_loss[k] = agg_loss.get(k, 0.0) + float(v.detach()) / grad_accum

            if scaler is not None:
                scaler.unscale_(optimizer)
            params_for_clip = (
                list(denoiser.parameters()) + list(shape_encoder.parameters())
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
                ema.update(denoiser.module if hasattr(denoiser, "module") else denoiser)

            step += 1
            step_dt = time.time() - step_t0

            if env.is_main and step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                line = (
                    f"step={step} loss={agg_loss['loss']:.4f} "
                    f"grad_norm={agg_loss.get('grad_norm', 0.0):.2f} "
                    f"lr={lr_now:.2e} dt={step_dt:.2f}s"
                )
                comp = " ".join(
                    f"{k.replace('l_',''):s}={v:.3f}"
                    for k, v in agg_loss.items() if k.startswith("l_")
                )
                if comp:
                    line += " | " + comp
                log.info(line)
                if tb_writer is not None:
                    for k, v in agg_loss.items():
                        tb_writer.add_scalar(f"train/{k}", v, step)
                    tb_writer.add_scalar("train/lr", lr_now, step)
                    tb_writer.add_scalar("train/step_time", step_dt, step)

            if env.is_main and ckpt_every and step % ckpt_every == 0:
                save_checkpoint_with_shape(
                    output_dir / f"ckpt_step{step:07d}.pt",
                    denoiser, shape_encoder, optimizer, scheduler, scaler, ema, step, cfg,
                )
                latest = output_dir / "latest.pt"
                try:
                    if latest.exists() or latest.is_symlink():
                        latest.unlink()
                    latest.symlink_to(f"ckpt_step{step:07d}.pt")
                except OSError:
                    shutil.copy2(output_dir / f"ckpt_step{step:07d}.pt", latest)

            if step >= num_steps:
                break
        epoch += 1

    if env.is_main:
        save_checkpoint_with_shape(
            output_dir / "ckpt_final.pt",
            denoiser, shape_encoder, optimizer, scheduler, scaler, ema, step, cfg,
        )
    cleanup_distributed(env)


if __name__ == "__main__":
    main()
