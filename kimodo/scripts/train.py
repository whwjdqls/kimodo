# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo text-to-motion training script.

Usage (single GPU):
    python -m kimodo.scripts.train --config configs/training/default.yaml

Multi-GPU (single node):
    torchrun --standalone --nproc_per_node=4 -m kimodo.scripts.train \\
        --config configs/training/default.yaml

Resume:
    python -m kimodo.scripts.train --config configs/training/default.yaml \\
        trainer.resume_from=/path/to/ckpt.pt
"""

from __future__ import annotations

# Force HuggingFace offline mode under DDP, BEFORE any kimodo/transformers/HF
# imports below. With multiple ranks, concurrent HF cache validation races and
# fails with spurious "does not appear to have a file named model-XXXXX-of-
# XXXXX.safetensors" errors. The cache must already be warm (which it is after
# a single-GPU smoke run). This needs to happen before any import of
# huggingface_hub, since `HF_HUB_OFFLINE` is read at module import time.
import os as _os  # noqa: E402
if int(_os.environ.get("WORLD_SIZE", "1")) > 1:
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import copy
import logging
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from kimodo.data import SOMATextMotionDataset, build_collate_fn
from kimodo.model.common import instantiate_from_dict, materialize_value
from kimodo.model.diffusion import Diffusion
from kimodo.motion_rep.feature_utils import length_to_mask
from kimodo.skeleton.kinematics import fk

log = logging.getLogger("kimodo.train")


# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------
@dataclass
class DistEnv:
    world_size: int
    rank: int
    local_rank: int
    is_distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistEnv:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return DistEnv(world_size, rank, local_rank, is_distributed=True)
    return DistEnv(world_size=1, rank=0, local_rank=0, is_distributed=False)


def cleanup_distributed(env: DistEnv) -> None:
    if env.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        v = value.clone()
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        v /= dist.get_world_size()
        return v
    return value


def maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
def load_config(config_path: str, overrides: List[str]) -> DictConfig:
    """Load a YAML config with OmegaConf and apply CLI dotted overrides."""
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def build_denoiser_from_model_config(model_cfg_path: str, stats_path: str, fps_override: Optional[int]) -> nn.Module:
    """Instantiate the denoiser using the model schema config (omitting the diffusion wrapper).

    Returns the denoiser nn.Module with motion_rep attached. Does not load any
    checkpoint weights (denoiser ckpt_path is forced to None).
    """
    raw = OmegaConf.load(model_cfg_path)
    raw = OmegaConf.to_container(raw, resolve=False)
    denoiser_cfg = raw["denoiser"]
    # Force training from-scratch: drop ckpt_path.
    denoiser_cfg["ckpt_path"] = None
    # Inject stats path / fps override into motion_rep.
    mr = denoiser_cfg["motion_rep"]
    mr["stats_path"] = stats_path
    if fps_override is not None:
        mr["fps"] = int(fps_override)
    denoiser = instantiate_from_dict(denoiser_cfg)
    return denoiser


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
class KimodoLoss(nn.Module):
    """KIMODO-paper smooth-L1 loss per feature block + optional FK consistency.

    Component weights follow the paper:
        gamma_1 = smooth_root_pos       = 10.0
        gamma_2 = global_root_heading   =  2.0
        gamma_3 = local_joints_positions= 10.0
        gamma_4 = velocities            =  3.0
        gamma_5 = global_rot_data       = 10.0
        gamma_6 = foot_contacts         =  4.0
        gamma_7 = fk (FK consistency)   =  5.0
    """

    def __init__(
        self,
        motion_rep,
        weights: Dict[str, float],
        smooth_l1_beta: float = 1.0,
        fk_target: str = "gt",  # "gt": match GT positions; "pred": match pred positions block
    ):
        super().__init__()
        self.motion_rep = motion_rep
        self.weights = dict(weights)
        self.smooth_l1_beta = float(smooth_l1_beta)
        assert fk_target in ("gt", "pred")
        self.fk_target = fk_target

    def _smooth_l1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # element-wise smooth-L1 (no reduction); we apply our own pad-mask average.
        return torch.nn.functional.smooth_l1_loss(
            pred, target, reduction="none", beta=self.smooth_l1_beta,
        )

    def _fk_term(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """gamma_7: ||FK(predicted rotations) - GT joint positions||_1

        Both pred and target are normalized features. We unnormalize the slices
        we need locally so we don't mutate the inputs.
        """
        from kimodo.skeleton.transforms import global_rots_to_local_rots
        from kimodo.geometry import cont6d_to_matrix

        mr = self.motion_rep
        pred_un = mr.unnormalize(pred)
        # predicted global 6D rotations -> local rotation matrices
        pred_rot_data = pred_un[..., mr.slice_dict["global_rot_data"]]
        pred_rot_data = pred_rot_data.reshape(*pred_rot_data.shape[:2], mr.nbjoints, 6)
        pred_global_rot = cont6d_to_matrix(pred_rot_data)
        pred_local_rot = global_rots_to_local_rots(pred_global_rot, mr.skeleton)

        # Root positions to use as FK origin. Use the predicted smooth_root_pos so
        # the FK output sits in the predicted world frame.
        pred_smooth_root = pred_un[..., mr.slice_dict["smooth_root_pos"]]
        # smooth_root_pos has the planar component; recover the actual root xyz
        # from local_joints_positions if needed. Use smooth_root_pos directly
        # (consistent with motion_rep.inverse default behavior).
        _, pred_posed, _ = mr.skeleton.fk(pred_local_rot, pred_smooth_root)

        # Target joints in world space — read from the GT features.
        target_un = mr.unnormalize(target)
        if self.fk_target == "gt":
            tgt_smooth_root = target_un[..., mr.slice_dict["smooth_root_pos"]]
            tgt_local_jp = target_un[..., mr.slice_dict["local_joints_positions"]]
            tgt_local_jp = tgt_local_jp.reshape(*tgt_local_jp.shape[:2], mr.nbjoints, 3)
            # local_joints_positions has the per-joint offsets in pelvis-relative
            # frame with the hips xz-offset added; recovering world positions:
            tgt_world = tgt_local_jp.clone()
            tgt_world[..., 0] += tgt_smooth_root[..., None, 0]
            tgt_world[..., 2] += tgt_smooth_root[..., None, 2]
        else:
            # "pred": compare to predicted positions instead of GT
            pred_smooth_root_p = pred_un[..., mr.slice_dict["smooth_root_pos"]]
            pred_local_jp = pred_un[..., mr.slice_dict["local_joints_positions"]]
            pred_local_jp = pred_local_jp.reshape(*pred_local_jp.shape[:2], mr.nbjoints, 3)
            tgt_world = pred_local_jp.clone()
            tgt_world[..., 0] += pred_smooth_root_p[..., None, 0]
            tgt_world[..., 2] += pred_smooth_root_p[..., None, 2]

        diff = (pred_posed - tgt_world).abs()  # (B, T, J, 3)
        mask = pad_mask.float().unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        num = (diff * mask).sum()
        den = mask.sum() * float(mr.nbjoints * 3) + 1e-8
        return num / den

    def forward(
        self,
        pred: torch.Tensor,            # (B, T, D) predicted clean motion (normalized)
        target: torch.Tensor,          # (B, T, D) ground-truth clean motion (normalized)
        pad_mask: torch.Tensor,        # (B, T) True for valid frames
    ) -> Dict[str, torch.Tensor]:
        diff = self._smooth_l1(pred, target)  # (B, T, D)
        mask = pad_mask.float().unsqueeze(-1)  # (B, T, 1)

        losses: Dict[str, torch.Tensor] = {}
        total = pred.new_zeros(())
        denom = pred.new_tensor(0.0)

        for name, sl in self.motion_rep.slice_dict.items():
            w = float(self.weights.get(name, 0.0))
            if w == 0.0:
                continue
            d_block = diff[..., sl]  # (B, T, d_block)
            block_n = float(d_block.shape[-1])
            num = (d_block * mask).sum()
            den = mask.sum() * block_n + 1e-8
            block_loss = num / den
            losses[f"l_{name}"] = block_loss.detach()
            total = total + w * block_loss
            denom = denom + w

        # gamma_7: FK consistency between predicted rotations and joint positions.
        fk_w = float(self.weights.get("fk", 0.0))
        if fk_w > 0.0:
            fk_loss = self._fk_term(pred, target, pad_mask)
            losses["l_fk"] = fk_loss.detach()
            total = total + fk_w * fk_loss
            denom = denom + fk_w

        losses["loss_data"] = (total / denom.clamp_min(1e-8)).detach()
        losses["loss"] = total / denom.clamp_min(1e-8)
        return losses


# -----------------------------------------------------------------------------
# EMA
# -----------------------------------------------------------------------------
class ModelEMA:
    """Polyak averaging of parameters."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        full = model.state_dict()
        for k, v in self.shadow.items():
            full[k].copy_(v)
        model.load_state_dict(full)


# -----------------------------------------------------------------------------
# Text encoder wrapping
# -----------------------------------------------------------------------------
def build_text_encoder(text_cfg: DictConfig, device: torch.device):
    from kimodo.model.llm2vec import LLM2VecEncoder

    # Repo IDs by default; KIMODO_TEXT_ENCODER_LOCAL_DIR overrides with flat
    # local-dir mirrors to avoid HF cache/snapshot races under DDP.
    base = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    peft = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
    local_root = os.environ.get("KIMODO_TEXT_ENCODER_LOCAL_DIR")
    if local_root:
        base_local = os.path.join(local_root, "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
        peft_local = os.path.join(local_root, "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")
        if os.path.isdir(base_local) and os.path.isdir(peft_local):
            base = base_local
            peft = peft_local

    enc = LLM2VecEncoder(
        base_model_name_or_path=base,
        peft_model_name_or_path=peft,
        dtype="float32" if text_cfg.get("fp32", False) else "bfloat16",
        llm_dim=4096,
        device=str(device) if text_cfg.get("device", "auto") == "auto" else text_cfg.device,
    )
    return enc


@torch.no_grad()
def encode_texts(text_encoder, texts: List[str], device: torch.device) -> torch.Tensor:
    """Return (B, num_text_tokens=1, llm_dim) on device.

    LLM2VecEncoder returns (B, 1, D); we pass through unchanged. The denoiser
    pads to num_text_tokens_override (50) internally.
    """
    feats, _ = text_encoder(texts)
    return feats.to(device=device)


# -----------------------------------------------------------------------------
# Constraint sampler (Phase 2)
# -----------------------------------------------------------------------------
# End-effector joint indices in SOMASkeleton30 (verified at runtime).
_SOMA30_EE_JOINTS = {
    "LeftHand": 13, "RightHand": 19, "LeftFoot": 24, "RightFoot": 28,
}


class ConstraintSampler:
    """Sample diverse constraint masks for Phase-2 training.

    Operates directly on the normalized GT motion features so we don't need a
    full decode -> constraint-object -> re-encode roundtrip. Mirrors the
    constraint families used in the benchmark testsuite.
    """

    def __init__(
        self,
        motion_rep,
        weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        self.motion_rep = motion_rep
        self.skeleton = motion_rep.skeleton
        if weights is None:
            # Tunable distribution over constraint families.
            weights = {
                "none": 0.40,
                "fullbody_keyframes": 0.20,
                "fullbody_inbetween": 0.10,
                "root_waypoints": 0.10,
                "root_path": 0.10,
                "ee_keyframes": 0.10,
            }
        self.weights = weights
        names = list(weights.keys())
        self._names = names
        self._probs = np.asarray(
            [float(weights[n]) for n in names], dtype=np.float64,
        )
        self._probs = self._probs / self._probs.sum()
        self._rng = np.random.default_rng(seed)

    def _zero_pair(self, motion_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros_like(motion_norm),
            torch.zeros_like(motion_norm),
        )

    def __call__(
        self,
        motion_norm: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """motion_norm: (B, T, D) normalized GT motion; lengths: (B,).

        Returns observed_motion, motion_mask — both (B, T, D), normalized.
        """
        B, T, D = motion_norm.shape
        observed, mask = self._zero_pair(motion_norm)
        mr = self.motion_rep
        sr_sl = mr.slice_dict["smooth_root_pos"]
        head_sl = mr.slice_dict["global_root_heading"]
        pos_sl = mr.slice_dict["local_joints_positions"]
        rot_sl = mr.slice_dict["global_rot_data"]
        J = mr.nbjoints

        for b in range(B):
            L = int(lengths[b].item())
            if L <= 1:
                continue
            choice = self._names[self._rng.choice(len(self._names), p=self._probs)]

            if choice == "none":
                continue

            if choice == "fullbody_keyframes":
                n = int(self._rng.integers(1, min(5, L) + 1))
                idxs = sorted(self._rng.choice(L, size=n, replace=False).tolist())
                observed[b, idxs, :] = motion_norm[b, idxs, :]
                mask[b, idxs, :] = 1.0

            elif choice == "fullbody_inbetween":
                # Just the first and last valid frame, full pose.
                idxs = [0, L - 1] if L > 1 else [0]
                observed[b, idxs, :] = motion_norm[b, idxs, :]
                mask[b, idxs, :] = 1.0

            elif choice == "root_waypoints":
                n = int(self._rng.integers(2, min(8, L) + 1))
                idxs = sorted(self._rng.choice(L, size=n, replace=False).tolist())
                observed[b, idxs, sr_sl] = motion_norm[b, idxs, sr_sl]
                mask[b, idxs, sr_sl] = 1.0
                # Sometimes also constrain heading (path_2dposrot style).
                if self._rng.random() < 0.5:
                    observed[b, idxs, head_sl] = motion_norm[b, idxs, head_sl]
                    mask[b, idxs, head_sl] = 1.0

            elif choice == "root_path":
                observed[b, :L, sr_sl] = motion_norm[b, :L, sr_sl]
                mask[b, :L, sr_sl] = 1.0
                if self._rng.random() < 0.5:
                    observed[b, :L, head_sl] = motion_norm[b, :L, head_sl]
                    mask[b, :L, head_sl] = 1.0

            elif choice == "ee_keyframes":
                n = int(self._rng.integers(1, min(4, L) + 1))
                idxs = sorted(self._rng.choice(L, size=n, replace=False).tolist())
                # Random subset of EE joints.
                ee_names = list(_SOMA30_EE_JOINTS.keys())
                k = int(self._rng.integers(1, len(ee_names) + 1))
                chosen = self._rng.choice(ee_names, size=k, replace=False)
                # Also pin smooth_root_pos at those frames so EE positions have
                # a defined world frame (matches the constraint-object rule).
                observed[b, idxs, sr_sl] = motion_norm[b, idxs, sr_sl]
                mask[b, idxs, sr_sl] = 1.0
                for jn in chosen:
                    jidx = _SOMA30_EE_JOINTS[jn]
                    p_start, p_end = pos_sl.start + jidx * 3, pos_sl.start + (jidx + 1) * 3
                    r_start, r_end = rot_sl.start + jidx * 6, rot_sl.start + (jidx + 1) * 6
                    observed[b, idxs, p_start:p_end] = motion_norm[b, idxs, p_start:p_end]
                    mask[b, idxs, p_start:p_end] = 1.0
                    observed[b, idxs, r_start:r_end] = motion_norm[b, idxs, r_start:r_end]
                    mask[b, idxs, r_start:r_end] = 1.0

        return observed, mask


# -----------------------------------------------------------------------------
# Train step
# -----------------------------------------------------------------------------
def train_one_step(
    denoiser: nn.Module,
    diffusion: Diffusion,
    loss_fn: KimodoLoss,
    batch: Dict[str, Any],
    text_encoder,
    device: torch.device,
    text_drop_prob: float,
    autocast_dtype: Optional[torch.dtype],
    constraint_sampler: Optional["ConstraintSampler"] = None,
) -> Dict[str, torch.Tensor]:
    motion: torch.Tensor = batch["motion"].to(device, non_blocking=True)  # (B, T, D)
    pad_mask: torch.Tensor = batch["pad_mask"].to(device, non_blocking=True)
    first_heading: torch.Tensor = batch["first_heading_angle"].to(device, non_blocking=True)
    texts: List[str] = batch["text"]
    B = motion.shape[0]
    T = motion.shape[1]

    # Random text drop for CFG (replace dropped with empty string -> zeroed feats).
    keep = torch.rand(B) > text_drop_prob
    if not keep.all():
        texts = [t if k else "" for t, k in zip(texts, keep.tolist())]
    text_feat = encode_texts(text_encoder, texts, device)
    # zero out dropped ones (encode_texts already keeps shape; double-safety)
    if not keep.all():
        keep_mask = keep.to(device=device, dtype=text_feat.dtype).view(-1, 1, 1)
        text_feat = text_feat * keep_mask

    text_pad_mask = torch.ones(text_feat.shape[:2], dtype=torch.bool, device=device)

    # Sample diffusion timesteps uniformly.
    t = torch.randint(0, diffusion.num_base_steps, (B,), device=device)

    # q_sample (forward diffusion). We pre-cache the full schedule (already done in __init__).
    noise = torch.randn_like(motion)
    motion_t = diffusion.q_sample(motion, t, noise=noise)

    # Conditioning. Phase 1 (no constraint sampler): zero motion_mask + zero
    # observed_motion (the denoiser treats this as fully unconstrained).
    # Phase 2: sample diverse constraints from the GT motion.
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
        )
        # Cast to float32 for loss computation (stability with bf16 forward).
        pred_clean = pred_clean.float()
        losses = loss_fn(pred_clean, motion, pad_mask)
    return losses


class _NullCtx:
    def __enter__(self):  # noqa: D401
        return None

    def __exit__(self, *args):  # noqa: D401
        return False


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    denoiser: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    ema: Optional[ModelEMA],
    step: int,
    cfg: DictConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "denoiser": (denoiser.module if hasattr(denoiser, "module") else denoiser).state_dict(),
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


def load_checkpoint(
    path: Path,
    denoiser: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    ema: Optional[ModelEMA],
    device: torch.device,
) -> int:
    log.info("Resuming from %s", path)
    obj = torch.load(path, map_location=device)
    (denoiser.module if hasattr(denoiser, "module") else denoiser).load_state_dict(obj["denoiser"])
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
# Visualization
# -----------------------------------------------------------------------------
@torch.no_grad()
def viz_generate_samples(
    denoiser: nn.Module,
    diffusion: Diffusion,
    text_encoder,
    device: torch.device,
    cfg: DictConfig,
    step: int,
    out_dir: Path,
) -> None:
    """Generate a few sample motions and write them as NPZ files (and optionally MP4)."""
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

    # spaced timesteps
    use_timesteps, map_tensor = diffusion.space_timesteps(n_steps)
    diffusion.calc_diffusion_vars(use_timesteps)

    cur = torch.randn(B, num_frames, motion_rep.motion_rep_dim, device=device)
    indices = list(range(n_steps))[::-1]
    for i in indices:
        t = torch.full((B,), i, device=device, dtype=torch.long)
        t_map = map_tensor[t]
        pred_clean = raw(
            cur,
            pad_mask,
            text_feat,
            text_pad_mask,
            t_map,
            first_heading_angle=first_heading,
            motion_mask=motion_mask,
            observed_motion=observed,
        )
        # DDIM step
        eps = (
            diffusion.sqrt_recip_alphas_cumprod[t, None, None] * cur - pred_clean
        ) / diffusion.sqrt_recipm1_alphas_cumprod[t, None, None]
        alpha_bar_prev = diffusion.alphas_cumprod_prev[t, None, None]
        cur = pred_clean * alpha_bar_prev.sqrt() + (1 - alpha_bar_prev).sqrt() * eps

    motion = cur  # normalized
    output = motion_rep.inverse(motion, is_normalized=True, return_numpy=True)
    raw.train()
    for i, prompt in enumerate(prompts):
        out = {k: v[i] for k, v in output.items()}
        safe = prompt.lower().replace(" ", "_").replace(".", "")[:40]
        np.savez(out_dir / f"step{step:07d}_{i}_{safe}.npz", **out, prompt=prompt)
    log.info("Wrote %d viz samples to %s", B, out_dir)


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

    # Surface uncaught exceptions into the rank log file so DDP crashes are
    # diagnosable without staring at torchrun's truncated stderr.
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

    # ----- dataset / dataloader -----
    # The dataset runs in CPU workers; give it its own CPU-resident motion_rep
    # so the on-GPU denoiser.motion_rep stays GPU-only.
    dataset_motion_rep = _build_cpu_motion_rep(
        cfg.model_config_path,
        cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    )
    # Build the dataset index on rank 0 first so the other ranks just load the
    # cached JSON, instead of redundantly scanning the dataset 8 times.
    def _build_dataset():
        return SOMATextMotionDataset(
            data_root=cfg.data.data_root,
            temporal_labels_path=cfg.data.temporal_labels_path,
            fps=cfg.data.fps,
            min_frames=cfg.data.min_frames,
            max_frames=cfg.data.max_frames,
            skeleton=dataset_motion_rep.skeleton,
            motion_rep=dataset_motion_rep,
            normalize=True,
            random_heading_aug=bool(cfg.data.random_heading_aug),
            include_mirrored=bool(cfg.data.include_mirrored),
            cache_index=cfg.data.get("cache_index"),
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
    pad_to = int(cfg.data.get("pad_to_max_frames", cfg.data.max_frames)) if cfg.data.get("pad_to_max_frames", True) else None
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.trainer.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=True,
        collate_fn=build_collate_fn(pad_to=pad_to),
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

    # ----- text encoder (frozen) -----
    # Strategy:
    # 1. Rank 0 builds the text encoder normally (network allowed) so the HF
    #    cache is fully populated/validated.
    # 2. All other ranks then load with HF_HUB_OFFLINE=1 so they bypass the
    #    HF cache-validation API roundtrips that race when N>1 ranks load
    #    concurrently. Without this they get spurious
    #    "does not appear to have a file named model-XXXXX-of-XXXXX.safetensors"
    #    errors even though the file is present locally.
    log.info(
        "Building text encoder (HF_HUB_OFFLINE=%s, KIMODO_TEXT_ENCODER_LOCAL_DIR=%s) ...",
        os.environ.get("HF_HUB_OFFLINE", "0"),
        os.environ.get("KIMODO_TEXT_ENCODER_LOCAL_DIR", "(unset)"),
    )
    # Serialize across ranks: rank 0 first, then 1, ..., world_size-1.
    # Each rank loads while the others wait at a barrier. Total cost is
    # world_size * load_time, but loading from a flat local-dir on WekaFS is
    # ~30 s/rank, so 8 GPUs = ~4 minutes total — acceptable for a multi-hour
    # training run, and eliminates every form of HF cache race.
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
    loss_fn = KimodoLoss(
        motion_rep,
        cfg.trainer.loss_weights,
        smooth_l1_beta=float(cfg.trainer.get("smooth_l1_beta", 1.0)),
    )

    # ----- constraint sampler (Phase 2 only) -----
    phase = str(cfg.trainer.get("phase", "text"))
    if phase not in ("text", "constraints"):
        raise ValueError(f"trainer.phase must be 'text' or 'constraints', got {phase!r}")
    constraint_sampler: Optional[ConstraintSampler] = None
    if phase == "constraints":
        cw = cfg.trainer.get("constraint_weights", None)
        constraint_sampler = ConstraintSampler(
            motion_rep,
            weights=OmegaConf.to_container(cw) if cw is not None else None,
            seed=int(cfg.trainer.seed) + 7 * env.rank,
        )
        log.info("Phase 2: constraint sampler active with weights %s", constraint_sampler.weights)
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

            if env.is_main and step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                line = (
                    f"step={step} loss={agg_loss['loss']:.4f} "
                    f"lr={lr_now:.2e} dt={step_dt:.2f}s"
                )
                comp = " ".join(f"{k.replace('l_',''):s}={v:.3f}" for k, v in agg_loss.items() if k.startswith("l_"))
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
                # also keep "latest" symlink for easy resume
                latest = output_dir / "latest.pt"
                try:
                    if latest.exists() or latest.is_symlink():
                        latest.unlink()
                    latest.symlink_to(f"ckpt_step{step:07d}.pt")
                except OSError:
                    shutil.copy2(output_dir / f"ckpt_step{step:07d}.pt", latest)

            if env.is_main and viz_every and step % viz_every == 0:
                try:
                    viz_generate_samples(
                        denoiser, diffusion, text_encoder, device, cfg, step,
                        output_dir / "viz",
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
    cleanup_distributed(env)


def _build_cpu_motion_rep(model_cfg_path: str, stats_path: str, fps_override: Optional[int]):
    """Build a CPU-only KimodoMotionRep + Skeleton matching the model config.

    Used by the data loader workers (the on-GPU motion_rep from the denoiser
    cannot be shared across processes / CPU workers).
    """
    raw = OmegaConf.load(model_cfg_path)
    raw = OmegaConf.to_container(raw, resolve=False)
    mr_cfg = raw["denoiser"]["motion_rep"]
    mr_cfg["stats_path"] = stats_path
    if fps_override is not None:
        mr_cfg["fps"] = int(fps_override)
    return instantiate_from_dict(mr_cfg)


def _resolve_num_steps(cfg: DictConfig) -> int:
    """Resolve the diffusion num_base_steps from the model config (default 1000)."""
    try:
        raw = OmegaConf.load(cfg.model_config_path)
        return int(raw.get("num_base_steps", 1000))
    except Exception:
        return 1000


if __name__ == "__main__":
    main()
