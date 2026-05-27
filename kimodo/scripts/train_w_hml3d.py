# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train a KIMODO-style text-to-motion model on HumanML3D (in Kimodo features).

This is the HumanML3D counterpart to ``kimodo.scripts.train``. Everything
heavy (denoiser build, diffusion, optimizer, EMA, AMP, DDP setup, text
encoder loading, train step, FK-free smooth-L1 loss, checkpointing) is
imported from ``kimodo.scripts.train`` — we only override the dataset
construction and visualisation.

Key differences from the SOMA pipeline:

* Data: ``HumanML3DTextMotionDataset`` reads pre-converted ``(T, 273)``
  features from per-id NPZs + multi-caption HumanML3D text files.
* Skeleton: ``SMPLXSkeleton22`` is used as a CONTAINER ONLY (gives
  ``nbjoints=22`` so the motion_rep produces 273-dim features). We never
  call ``skeleton.fk()`` because the rotations follow HumanML3D's
  chain-reset semantics, not standard parent-relative FK.
* γ₇ FK consistency loss is OFF (config default ``loss_weights.fk: 0.0``).
* Visualisation: predicted features are decoded via the HumanML3D
  conversion pipeline (``benchmark/humanml3d_to_kimodo.kimodo_to_humanml3d``
  + HumanML3D ``recover_from_ric``) instead of ``motion_rep.inverse``.

Usage (single GPU):
    python -m kimodo.scripts.train_w_hml3d --config configs/training/hml3d.yaml

Multi-GPU (single node):
    torchrun --standalone --nproc_per_node=8 -m kimodo.scripts.train_w_hml3d \\
        --config configs/training/hml3d.yaml
"""

from __future__ import annotations

# Force HF offline mode when DDP, BEFORE any kimodo imports. (Same reasoning
# as train.py — see that file's header comment.)
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
from typing import Dict, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.cuda.amp import GradScaler  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from kimodo.data import (  # noqa: E402
    HumanML3DTextMotionDataset,
    build_humanml3d_collate_fn,
)
from kimodo.model.diffusion import Diffusion  # noqa: E402
from kimodo.scripts.train import (  # noqa: E402
    KimodoLoss,
    ModelEMA,
    build_denoiser_from_model_config,
    build_text_encoder,
    encode_texts,
    load_checkpoint,
    load_config,
    maybe_barrier,
    save_checkpoint,
    setup_distributed,
    train_one_step,
    _build_cpu_motion_rep,
    _resolve_num_steps,
)

log = logging.getLogger("kimodo.train_hml3d")


# -----------------------------------------------------------------------------
# Viz — HumanML3D-aware (no skeleton.fk)
# -----------------------------------------------------------------------------
@torch.no_grad()
def viz_generate_samples_hml3d(
    denoiser: nn.Module,
    diffusion: Diffusion,
    text_encoder,
    device: torch.device,
    cfg: DictConfig,
    step: int,
    out_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    """Sample motions and save the unnormalized (T, 273) features per prompt.

    We DO NOT call ``motion_rep.inverse`` here because the data's
    ``global_rot_data`` follows HumanML3D's chain-reset semantics. The saved
    NPZ contains the raw kimodo features; convert to HumanML3D 263-D and
    render with ``benchmark/humanml3d_to_kimodo.kimodo_to_humanml3d`` +
    ``recover_from_ric`` in a separate post-processing step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = (denoiser.module if hasattr(denoiser, "module") else denoiser).eval()
    motion_rep = raw.motion_rep
    prompts = list(cfg.viz.prompts)
    num_frames = int(cfg.viz.num_frames)
    n_steps = int(cfg.viz.num_denoising_steps)

    text_feat = encode_texts(text_encoder, prompts, device)
    text_pad_mask = torch.ones(text_feat.shape[:2], dtype=torch.bool, device=device)

    B = len(prompts)
    pad_mask = torch.ones(B, num_frames, dtype=torch.bool, device=device)
    first_heading = torch.zeros(B, device=device)
    motion_mask = torch.zeros(B, num_frames, motion_rep.motion_rep_dim, device=device)
    observed = torch.zeros(B, num_frames, motion_rep.motion_rep_dim, device=device)

    use_timesteps, map_tensor = diffusion.space_timesteps(n_steps)
    diffusion.calc_diffusion_vars(use_timesteps)

    cur = torch.randn(B, num_frames, motion_rep.motion_rep_dim, device=device)
    for i in list(range(n_steps))[::-1]:
        t = torch.full((B,), i, device=device, dtype=torch.long)
        t_map = map_tensor[t]
        pred_clean = raw(
            cur, pad_mask, text_feat, text_pad_mask, t_map,
            first_heading_angle=first_heading,
            motion_mask=motion_mask, observed_motion=observed,
        )
        eps = (
            diffusion.sqrt_recip_alphas_cumprod[t, None, None] * cur - pred_clean
        ) / diffusion.sqrt_recipm1_alphas_cumprod[t, None, None]
        alpha_bar_prev = diffusion.alphas_cumprod_prev[t, None, None]
        cur = pred_clean * alpha_bar_prev.sqrt() + (1 - alpha_bar_prev).sqrt() * eps

    raw.train()

    # cur is normalized; unnormalize back to raw kimodo features.
    motion = cur.float().cpu().numpy()
    motion = motion * std + mean  # (B, T, 273)
    for i, prompt in enumerate(prompts):
        safe = prompt.lower().replace(" ", "_").replace(".", "")[:40]
        np.savez(
            out_dir / f"step{step:07d}_{i}_{safe}.npz",
            features=motion[i],
            prompt=prompt,
        )
    log.info("Wrote %d HumanML3D viz samples (raw kimodo features) to %s", B, out_dir)


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
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
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
    assert motion_rep.motion_rep_dim == 273, (
        f"Expected HumanML3D-Kimodo motion_rep_dim=273, got {motion_rep.motion_rep_dim}. "
        f"Check that the model config uses SMPLXSkeleton22."
    )

    # ----- dataset / dataloader -----
    dataset_motion_rep = _build_cpu_motion_rep(
        cfg.model_config_path,
        cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    )

    mean = np.load(cfg.data.mean_path).astype(np.float32)
    std = np.load(cfg.data.std_path).astype(np.float32)

    def _build_dataset():
        return HumanML3DTextMotionDataset(
            motion_dir=cfg.data.motion_dir,
            text_dir=cfg.data.text_dir,
            split_file=cfg.data.split_file,
            mean=mean,
            std=std,
            fps=int(cfg.data.fps),
            window_size=int(cfg.data.window_size),
            max_motion_length=int(cfg.data.max_motion_length),
            min_motion_len=int(cfg.data.min_motion_len),
            unit_length=int(cfg.data.unit_length),
            random_heading_aug=bool(cfg.data.random_heading_aug),
            skeleton=dataset_motion_rep.skeleton,
            motion_rep=dataset_motion_rep,
            seed=seed,
        )

    if env.is_main:
        dataset = _build_dataset()
    maybe_barrier()
    if not env.is_main:
        dataset = _build_dataset()
    maybe_barrier()

    sampler = None
    if env.is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=int(cfg.trainer.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.trainer.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=True,
        collate_fn=build_humanml3d_collate_fn(),
        persistent_workers=bool(cfg.data.num_workers) and True,
    )

    # ----- optimizer / scheduler -----
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
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
        ema = ModelEMA(denoiser, decay=float(cfg.trainer.ema_decay))

    # ----- text encoder (frozen, serialized across ranks) -----
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
        denoiser = DDP(denoiser, device_ids=[env.local_rank], find_unused_parameters=False)

    # ----- loss -----
    # For HumanML3D, γ₇ is computed with the **chain-reset** FK to match the
    # data's rotation convention. Running the standard parent-relative FK on
    # these rotations would give wrong joint positions, so we dispatch to the
    # chain-reset variant here. The skeleton's neutral_joints (SMPLXSkeleton22
    # T-pose) and HumanML3D's t2m_kinematic_chain provide the bone offsets.
    weights = dict(cfg.trainer.loss_weights)
    loss_fn = KimodoLoss(
        motion_rep,
        weights,
        smooth_l1_beta=float(cfg.trainer.get("smooth_l1_beta", 1.0)),
        fk_kind="chainreset_hml3d",
        # Paper-faithful: compare FK(pred_rot, GT_bone_lengths) against the GT
        # positions block. With the corrected raw_offsets-based FK, this is a
        # physically meaningful per-meter error in the actor's actual skeleton.
        fk_target="gt",
    ).to(device)

    # Phase 2 (constraints) is supported by ConstraintSampler but disabled by
    # default here. Reuse the SOMA pipeline's sampler if needed.
    constraint_sampler = None
    phase = str(cfg.trainer.get("phase", "text"))
    if phase == "constraints":
        from kimodo.scripts.train import ConstraintSampler
        cw = cfg.trainer.get("constraint_weights", None)
        constraint_sampler = ConstraintSampler(
            motion_rep,
            weights=OmegaConf.to_container(cw) if cw is not None else None,
            seed=int(cfg.trainer.seed) + 7 * env.rank,
        )
        log.info("Phase 2 (constraints) active with weights %s", constraint_sampler.weights)
    else:
        log.info("Phase 1: text-only training (no constraints).")

    # ----- resume -----
    start_step = 0
    if cfg.trainer.get("resume_from"):
        start_step = load_checkpoint(
            Path(cfg.trainer.resume_from),
            denoiser, optimizer, scheduler, scaler, ema, device,
        )

    # ----- logging sinks -----
    tb_writer = None
    if env.is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(output_dir / "tb"))
        except Exception as e:
            log.warning("TensorBoard not available: %s", e)

    log.info("Starting training from step %d for %d steps", start_step, num_steps)
    denoiser.train()

    grad_accum = int(cfg.trainer.grad_accum)
    log_every = int(cfg.trainer.log_every)
    print_every = int(cfg.trainer.get("print_every", 1))
    ckpt_every = int(cfg.trainer.ckpt_every)
    viz_every = int(cfg.trainer.viz_every)
    grad_clip = float(cfg.trainer.grad_clip)
    ema_every = int(cfg.trainer.ema_every)
    text_drop_prob = float(cfg.trainer.text_drop_prob)

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
                losses = train_one_step(
                    denoiser, diffusion, loss_fn, batch, text_encoder, device,
                    text_drop_prob, autocast_dtype, constraint_sampler=constraint_sampler,
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
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    denoiser.parameters() if not hasattr(denoiser, "module") else denoiser.module.parameters(),
                    grad_clip,
                )
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

            if env.is_main and print_every > 0 and step % print_every == 0:
                comp_print = " ".join(
                    f"{k.replace('l_', '')}={v:.4f}"
                    for k, v in agg_loss.items() if k.startswith("l_")
                )
                print(
                    f"[step {step:>7d}] loss={agg_loss['loss']:.4f}"
                    + (f" | {comp_print}" if comp_print else ""),
                    flush=True,
                )

            if env.is_main and step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                line = (
                    f"step={step} loss={agg_loss['loss']:.4f} "
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
                save_checkpoint(
                    output_dir / f"ckpt_step{step:07d}.pt",
                    denoiser, optimizer, scheduler, scaler, ema, step, cfg,
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
                    viz_generate_samples_hml3d(
                        denoiser, diffusion, text_encoder, device, cfg, step,
                        output_dir / "viz", mean=mean, std=std,
                    )
                except Exception as e:
                    log.warning("Viz failed at step %d: %s", step, e)

            if step >= num_steps:
                break
        epoch += 1

    if env.is_main:
        save_checkpoint(
            output_dir / "ckpt_final.pt",
            denoiser, optimizer, scheduler, scaler, ema, step, cfg,
        )
    from kimodo.scripts.train import cleanup_distributed
    cleanup_distributed(env)


if __name__ == "__main__":
    main()
