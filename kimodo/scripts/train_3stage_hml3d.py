# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train the 3-stage text-to-motion model on HumanML3D (in Kimodo features).

Pipeline (forward, training and inference)::

    text_feat       = LLM2Vec(text)                          # frozen
    mask_logits     = MaskGenerator(text_feat, pad_mask)
    mask_probs      = sigmoid(mask_logits)                   # continuous in [0, 1]
    values          = ConstraintGenerator(text_feat, mask_probs, pad_mask)
    observed_motion = mask_probs * values                    # gated values
    motion_t        = q_sample(motion, t)
    pred_clean      = Denoiser(motion_t, ..., motion_mask=mask_probs,
                                              observed_motion=observed_motion)

No ground-truth mask exists. The losses are:

  * ``L_motion``      — existing 7-component kimodo loss (smooth-L1 + chain-reset γ₇).
                        The ONLY loss that uses GT motion. Backprops through Stage 3
                        and (via ``observed_motion = mask * values``) through Stages
                        1 and 2.
  * ``L_sparsity``    — ``λ_s * mean(mask_probs)``. Prevents trivial mask=1 collapse.
                        If ``sparsity_target > 0``, hinge: ``max(0, mean - target)``.
  * ``L_stage2_aux``  — optional smooth-L1(values, GT_motion). Dense supervision for
                        Stage 2; mask is NOT used here so it does not leak any
                        "GT mask".

Almost all helpers (denoiser build, diffusion, KimodoLoss, EMA, AMP, DDP, text
encoder loading, ConstraintSampler, checkpoint utilities) are imported from
``kimodo.scripts.train`` / ``train_w_hml3d``; we only override training-step
construction and visualisation.

Run:
    python -m kimodo.scripts.train_3stage_hml3d \
        --config configs/training/hml3d_3stage.yaml

Multi-GPU:
    KIMODO_TEXT_ENCODER_LOCAL_DIR=/weka/jungbin/hf_models \
    torchrun --standalone --nproc_per_node=8 --redirects 3 --tee 3 \
        -m kimodo.scripts.train_3stage_hml3d \
        --config configs/training/hml3d_3stage.yaml \
        output_dir=/weka/jungbin/kimodo_runs/hml3d_3stage_v1
"""

from __future__ import annotations

# Force HF offline mode under DDP, BEFORE any kimodo imports below — same
# reasoning as ``train_w_hml3d.py``.
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
import torch.nn.functional as F  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.cuda.amp import GradScaler  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from kimodo.data import HumanML3DTextMotionDataset, build_humanml3d_collate_fn  # noqa: E402
from kimodo.model.common import instantiate_from_dict  # noqa: E402
from kimodo.model.constraint_generator import ConstraintGenerator  # noqa: E402
from kimodo.model.diffusion import Diffusion  # noqa: E402
from kimodo.model.mask_generator import MaskGenerator  # noqa: E402
from kimodo.model.three_stage import ThreeStageKimodo  # noqa: E402
from kimodo.scripts.train import (  # noqa: E402
    KimodoLoss,
    ModelEMA,
    _NullCtx,
    build_text_encoder,
    encode_texts,
    ensure_full_training_schedule,
    load_config,
    maybe_barrier,
    setup_distributed,
)

log = logging.getLogger("kimodo.train_3stage_hml3d")


# -----------------------------------------------------------------------------
# Model build
# -----------------------------------------------------------------------------
def _build_3stage_model(model_cfg_path: str, stats_path: str, fps_override: Optional[int]):
    """Instantiate Stage 1 / 2 / 3 sub-modules with a SHARED motion_rep.

    Returns ``(three_stage_module, diffusion)``. ``three_stage_module`` is the
    single ``nn.Module`` that wraps all three trainable stages — DDP-friendly.
    """
    raw = OmegaConf.to_container(OmegaConf.load(model_cfg_path), resolve=False)

    # Stage 3 (motion denoiser) — same shape as the existing 2-stage pipeline.
    denoiser_cfg = raw["motion_denoiser"]
    denoiser_cfg["ckpt_path"] = None  # always train from-scratch unless we explicitly load.
    mr_cfg = denoiser_cfg["motion_rep"]
    mr_cfg["stats_path"] = stats_path
    if fps_override is not None:
        mr_cfg["fps"] = int(fps_override)
    denoiser = instantiate_from_dict(denoiser_cfg)
    motion_rep = denoiser.motion_rep

    # Stage 1 & 2 — pass the shared motion_rep in explicitly.
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
    """Build a CPU-only KimodoMotionRep for the dataset workers (no GPU tensors)."""
    raw = OmegaConf.to_container(OmegaConf.load(model_cfg_path), resolve=False)
    mr_cfg = dict(raw["motion_denoiser"]["motion_rep"])
    mr_cfg["stats_path"] = stats_path
    if fps_override is not None:
        mr_cfg["fps"] = int(fps_override)
    return instantiate_from_dict(mr_cfg)


# -----------------------------------------------------------------------------
# 3-stage training step
# -----------------------------------------------------------------------------
def train_one_step_3stage(
    three_stage: nn.Module,                # may be DDP-wrapped
    diffusion: Diffusion,
    loss_fn: KimodoLoss,
    batch: Dict[str, Any],
    text_encoder,
    device: torch.device,
    text_drop_prob: float,
    mask_drop_prob: float,
    autocast_dtype: Optional[torch.dtype],
    w_motion: float,
    w_stage2_aux: float,
    w_sparsity: float,
    sparsity_target: float,
    smooth_l1_beta: float,
) -> Dict[str, torch.Tensor]:
    """Single end-to-end 3-stage training step.

    Returns a dict containing ``loss`` (the total backward target) and various
    detached per-component scalars for logging.
    """
    motion: torch.Tensor = batch["motion"].to(device, non_blocking=True)  # (B, T, D)
    pad_mask: torch.Tensor = batch["pad_mask"].to(device, non_blocking=True)
    first_heading: torch.Tensor = batch["first_heading_angle"].to(device, non_blocking=True)
    texts = batch["text"]
    B, T, D = motion.shape
    pad_mask_f = pad_mask.to(motion.dtype)

    # Text encoding + CFG drop (existing logic from train.py).
    keep_text = torch.rand(B) > text_drop_prob
    if not keep_text.all():
        texts = [t if k else "" for t, k in zip(texts, keep_text.tolist())]
    text_feat = encode_texts(text_encoder, texts, device)
    if not keep_text.all():
        keep_mask = keep_text.to(device=device, dtype=text_feat.dtype).view(-1, 1, 1)
        text_feat = text_feat * keep_mask
    text_pad_mask = torch.ones(text_feat.shape[:2], dtype=torch.bool, device=device)

    # Autocast context (bf16/fp16 on CUDA only).
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if (autocast_dtype is not None and device.type == "cuda")
        else _NullCtx()
    )

    with autocast_ctx:
        # ----- Stage 1 + Stage 2 -----
        # When DDP-wrapped, ``three_stage`` is the DDP object; underlying module
        # is at ``.module``. We call the wrapped object's forward via our small
        # internal interface methods, which DDP forwards through transparently.
        inner = three_stage.module if hasattr(three_stage, "module") else three_stage
        st12 = inner.predict_mask_and_values(text_feat, text_pad_mask, pad_mask)
        mask_probs: torch.Tensor = st12["mask_probs"]      # (B, T, D) in [0,1]
        values: torch.Tensor = st12["values"]               # (B, T, D)

        # Mask-CFG drop: zero out the whole mask for some samples so Stage 3
        # keeps working at text-only inference.
        if mask_drop_prob > 0:
            keep_mask_sample = (torch.rand(B, device=device) > mask_drop_prob).to(mask_probs.dtype)
            keep_mask_sample = keep_mask_sample.view(B, 1, 1)
            mask_probs_used = mask_probs * keep_mask_sample
        else:
            mask_probs_used = mask_probs

        observed_motion = mask_probs_used * values  # (B, T, D)

        # ----- Stage 3 (motion denoiser + diffusion) -----
        # Restore the full training schedule (viz/sampling mutates it in place).
        ensure_full_training_schedule(diffusion)
        t = torch.randint(0, diffusion.num_base_steps, (B,), device=device)
        noise = torch.randn_like(motion)
        motion_t = diffusion.q_sample(motion, t, noise=noise)

        pred_clean = inner.forward_denoiser(
            motion_t, pad_mask, text_feat, text_pad_mask, t,
            first_heading_angle=first_heading,
            mask_probs=mask_probs_used,
            observed_motion=observed_motion,
        )
        pred_clean = pred_clean.float()  # stabilise loss in fp32 even under bf16 forward.

        # Motion-reconstruction loss (existing 7-component kimodo loss).
        motion_losses = loss_fn(pred_clean, motion, pad_mask)
        l_motion = motion_losses["loss"]

    # The rest of the losses don't need autocast (they're cheap, fp32 is fine).
    # ----- Sparsity regulariser on mask_probs -----
    mask_denom = (pad_mask_f.sum() * D).clamp_min(1.0)
    mean_mask_density = (mask_probs * pad_mask_f.unsqueeze(-1)).sum() / mask_denom
    if sparsity_target > 0:
        l_sparsity = torch.clamp(mean_mask_density - float(sparsity_target), min=0.0)
    else:
        l_sparsity = mean_mask_density

    # ----- Stage 2 dense aux supervision (smooth-L1 vs GT motion) -----
    if w_stage2_aux > 0:
        st2_diff = F.smooth_l1_loss(values, motion, reduction="none", beta=float(smooth_l1_beta))
        st2_num = (st2_diff * pad_mask_f.unsqueeze(-1)).sum()
        st2_den = (pad_mask_f.sum() * D).clamp_min(1.0)
        l_stage2_aux = st2_num / st2_den
    else:
        l_stage2_aux = motion.new_zeros(())

    total = (
        float(w_motion) * l_motion
        + float(w_stage2_aux) * l_stage2_aux
        + float(w_sparsity) * l_sparsity
    )

    out = {k: v.detach() for k, v in motion_losses.items() if k.startswith("l_")}
    out["l_motion_total"] = l_motion.detach()
    out["l_stage2_aux"] = l_stage2_aux.detach()
    out["l_sparsity"] = l_sparsity.detach()
    out["mask_density"] = mean_mask_density.detach()
    out["loss"] = total
    return out


# -----------------------------------------------------------------------------
# Checkpointing (with extra weight keys)
# -----------------------------------------------------------------------------
def save_3stage_checkpoint(
    path: Path,
    three_stage: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    ema: Optional[ModelEMA],
    step: int,
    cfg: DictConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = three_stage.module if hasattr(three_stage, "module") else three_stage
    obj = {
        "step": step,
        "three_stage": inner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "cfg": OmegaConf.to_container(cfg, resolve=False),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    log.info("Saved checkpoint: %s (step=%d)", path, step)


def load_3stage_checkpoint(
    path: Path,
    three_stage: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    ema: Optional[ModelEMA],
    device: torch.device,
) -> int:
    log.info("Resuming from %s", path)
    obj = torch.load(path, map_location=device)
    inner = three_stage.module if hasattr(three_stage, "module") else three_stage
    inner.load_state_dict(obj["three_stage"])
    if obj.get("optimizer"):
        optimizer.load_state_dict(obj["optimizer"])
    if scheduler is not None and obj.get("scheduler"):
        scheduler.load_state_dict(obj["scheduler"])
    if scaler is not None and obj.get("scaler"):
        scaler.load_state_dict(obj["scaler"])
    if ema is not None and obj.get("ema"):
        ema.load_state_dict(obj["ema"])
    return int(obj.get("step", 0))


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------
@torch.no_grad()
def viz_3stage_samples(
    three_stage: nn.Module,
    diffusion: Diffusion,
    text_encoder,
    device: torch.device,
    cfg: DictConfig,
    step: int,
    out_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    """Run the full 3-stage pipeline on a few prompts and dump raw features."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inner = three_stage.module if hasattr(three_stage, "module") else three_stage
    inner.eval()
    motion_rep = inner.motion_rep
    prompts = list(cfg.viz.prompts)
    num_frames = int(cfg.viz.num_frames)
    n_steps = int(cfg.viz.num_denoising_steps)
    threshold = cfg.viz.get("mask_threshold")
    threshold = None if threshold is None else float(threshold)

    text_feat = encode_texts(text_encoder, prompts, device)
    text_pad_mask = torch.ones(text_feat.shape[:2], dtype=torch.bool, device=device)
    B = len(prompts)
    pad_mask = torch.ones(B, num_frames, dtype=torch.bool, device=device)
    first_heading = torch.zeros(B, device=device)

    pack = inner.sample_constraint_pack(
        text_feat, text_pad_mask, pad_mask, mask_threshold=threshold,
    )
    motion_mask = pack["mask"]
    observed = pack["observed_motion"]

    use_timesteps, map_tensor = diffusion.space_timesteps(n_steps)
    diffusion.calc_diffusion_vars(use_timesteps)
    cur = torch.randn(B, num_frames, motion_rep.motion_rep_dim, device=device)
    for i in list(range(n_steps))[::-1]:
        t = torch.full((B,), i, device=device, dtype=torch.long)
        t_map = map_tensor[t]
        pred_clean = inner.denoiser(
            cur, pad_mask, text_feat, text_pad_mask, t_map,
            first_heading_angle=first_heading,
            motion_mask=motion_mask, observed_motion=observed,
        )
        eps = (
            diffusion.sqrt_recip_alphas_cumprod[t, None, None] * cur - pred_clean
        ) / diffusion.sqrt_recipm1_alphas_cumprod[t, None, None]
        alpha_bar_prev = diffusion.alphas_cumprod_prev[t, None, None]
        cur = pred_clean * alpha_bar_prev.sqrt() + (1 - alpha_bar_prev).sqrt() * eps

    inner.train()
    motion = cur.float().cpu().numpy() * std + mean
    mask_np = motion_mask.float().cpu().numpy()
    for i, prompt in enumerate(prompts):
        safe = prompt.lower().replace(" ", "_").replace(".", "")[:40]
        np.savez(
            out_dir / f"step{step:07d}_{i}_{safe}.npz",
            features=motion[i],
            mask=mask_np[i],
            prompt=prompt,
        )
    log.info("Wrote %d 3-stage viz samples (features + mask) to %s", B, out_dir)


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

    # ----- Build all three trainable stages + diffusion -----
    three_stage, diffusion = _build_3stage_model(
        cfg.model_config_path, cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    )
    three_stage = three_stage.to(device)
    diffusion = diffusion.to(device)
    motion_rep = three_stage.motion_rep
    assert motion_rep.motion_rep_dim == 273, (
        f"Expected HumanML3D-Kimodo motion_rep_dim=273, got {motion_rep.motion_rep_dim}"
    )

    # Optional warm-starts.
    def _maybe_load_subset(target_module: nn.Module, ckpt_path: Optional[str], prefix: str):
        if not ckpt_path:
            return
        from kimodo.model.loading import load_checkpoint_state_dict
        sd = load_checkpoint_state_dict(ckpt_path)
        # Best-effort: strip any expected prefix.
        sd = {k.replace(f"{prefix}.", ""): v for k, v in sd.items() if k.startswith(prefix)} or sd
        target_module.load_state_dict(sd, strict=False)
        log.info("Warm-started %s from %s", prefix, ckpt_path)

    _maybe_load_subset(three_stage.denoiser, cfg.get("init_motion_denoiser_safetensors"), "motion_denoiser")
    _maybe_load_subset(three_stage.mask_generator, cfg.get("init_mask_generator_safetensors"), "mask_generator")
    _maybe_load_subset(three_stage.constraint_generator, cfg.get("init_constraint_generator_safetensors"), "constraint_generator")

    # ----- Dataset -----
    dataset_motion_rep = _build_cpu_motion_rep_for_dataset(
        cfg.model_config_path, cfg.stats_path,
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
            clip_normalized=cfg.data.get("clip_normalized"),
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

    # ----- Loss (Stage 3 kimodo loss with chain-reset FK γ_7) -----
    loss_fn = KimodoLoss(
        motion_rep,
        dict(cfg.trainer.loss_weights),
        smooth_l1_beta=float(cfg.trainer.get("smooth_l1_beta", 1.0)),
        fk_kind="chainreset_hml3d",
        fk_target="gt",
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
