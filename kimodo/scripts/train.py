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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from kimodo.data import SOMABonesSeedDataset, SOMATextMotionDataset, build_collate_fn
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
        gamma_8 = fk_v (FK velocity)    =  (off by default)

    The fk_v term compares the per-frame world-joint velocity of FK(pred_rot)
    against the per-frame world-joint velocity of the GT positions (built from
    the GT positions block exactly like fk_target='gt'). Same skeleton/bone
    lengths on both sides, L1 on the velocity difference.
    """

    def __init__(
        self,
        motion_rep,
        weights: Dict[str, float],
        smooth_l1_beta: float = 1.0,
        fk_target: str = "gt",
        # fk_target values:
        #   "gt"     -> compare FK(pred_rot) against the GT positions block.
        #               Paper-faithful but assumes the FK calibration (skeleton bone
        #               offsets) matches the data — true for SOMA/SMPL fixed skeletons
        #               but NOT for HumanML3D which uses per-actor bone lengths.
        #   "pred"   -> compare FK(pred_rot) against the predicted positions block.
        #               Self-consistency only (no GT supervision).
        #   "fk_gt"  -> compare FK(pred_rot) against FK(gt_rot) under the SAME
        #               canonical skeleton. The constant per-actor calibration error
        #               cancels on both sides, leaving a pure rotational-consistency
        #               loss. Use this with ``fk_kind="chainreset_hml3d"``.
        fk_kind: str = "standard",  # 'standard' | 'chainreset_hml3d' | 'hml3d_native'
        fk_neutral_joints: Optional[torch.Tensor] = None,
        fk_chains: Optional[List[List[int]]] = None,
    ):
        super().__init__()
        self.motion_rep = motion_rep
        self.weights = dict(weights)
        self.smooth_l1_beta = float(smooth_l1_beta)
        assert fk_target in ("gt", "pred", "fk_gt")
        self.fk_target = fk_target
        assert fk_kind in ("standard", "chainreset_hml3d", "hml3d_native")
        self.fk_kind = fk_kind
        if fk_kind == "hml3d_native":
            # HumanML3D-native FK uses parent-relative cont6d rotations and
            # the canonical bone offsets baked into ``motion_rep`` (derived
            # from motion 012314 at construction time).
            from kimodo.motion_rep.humanml3d_native import HML3D_KINEMATIC_CHAIN
            self.fk_chains = [list(c) for c in (fk_chains or HML3D_KINEMATIC_CHAIN)]
            self._fk_bone_lengths_cache: Optional[torch.Tensor] = None  # unused; kept for API parity
        elif fk_kind == "chainreset_hml3d":
            from kimodo.motion_rep.fk_hml3d import (
                HML3D_KINEMATIC_CHAIN,
                HML3D_RAW_OFFSETS,
            )
            self.fk_chains = [list(c) for c in (fk_chains or HML3D_KINEMATIC_CHAIN)]
            # Register HumanML3D's axis-aligned unit offsets as a buffer so the
            # FK runs on the model's device. NOTE: ``fk_neutral_joints`` is no
            # longer used — HumanML3D's FK requires raw axis offsets multiplied
            # by per-sample bone lengths, not the canonical T-pose bone vectors.
            self.register_buffer(
                "fk_raw_offsets", HML3D_RAW_OFFSETS.clone().float(), persistent=False,
            )
            # Lazy cache for per-joint bone lengths. HumanML3D's preprocessing
            # uniform_skeleton's every motion to a canonical SMPL-22 skeleton
            # (template = motion 012314 frame 0), so the per-batch derived
            # lengths are identical across samples and across steps. We
            # populate this on the first _fk_term call and reuse forever.
            self._fk_bone_lengths_cache: Optional[torch.Tensor] = None
        else:
            self.fk_chains = None

    def _smooth_l1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # element-wise smooth-L1 (no reduction); we apply our own pad-mask average.
        return torch.nn.functional.smooth_l1_loss(
            pred, target, reduction="none", beta=self.smooth_l1_beta,
        )

    def _fk_world_from_pred(
        self, pred_un: torch.Tensor,
        bone_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute (B, T, J, 3) world joint positions from unnormalized predictions.

        Branches on ``self.fk_kind``:
          - ``"standard"``: predicted 6D rotations interpreted as global
            rotations under parent-relative semantics; we convert to local
            rotations then run ``motion_rep.skeleton.fk``.
          - ``"chainreset_hml3d"``: predicted 6D rotations interpreted as
            HumanML3D chain-reset global rotations; we walk the kinematic
            chains and accumulate bone vectors of length ``bone_lengths[j]``
            in HumanML3D's canonical ``raw_offset`` direction. ``bone_lengths``
            must be supplied (typically derived from the GT batch).
        """
        from kimodo.geometry import cont6d_to_matrix

        mr = self.motion_rep

        # HumanML3D-native is the odd-one-out: rotations are (J-1) parent-
        # relative 6Ds with the root rotation living in the rot_velocity
        # block, so we dispatch here BEFORE the kimodo-style slice extraction.
        if self.fk_kind == "hml3d_native":
            from kimodo.motion_rep.humanml3d_native import hml3d_native_fk_world_joints
            B, T, _ = pred_un.shape
            sd = mr.slice_dict
            rot6d = pred_un[..., sd["rot_data"]].reshape(B, T, mr.nbjoints - 1, 6)
            rot_vel = pred_un[..., sd["rot_velocity"]].squeeze(-1)
            lin_vel = pred_un[..., sd["lin_velocity"]]
            root_h = pred_un[..., sd["root_height"]].squeeze(-1)
            if getattr(mr, "canonical_offsets", None) is None:
                raise ValueError(
                    "fk_kind='hml3d_native' requires motion_rep.canonical_offsets "
                    "(pass canonical_joints_path to HumanML3DNativeMotionRep)."
                )
            return hml3d_native_fk_world_joints(
                rot6d, rot_vel, lin_vel, root_h,
                canonical_offsets=mr.canonical_offsets,
                chains=self.fk_chains,
            )

        pred_rot_data = pred_un[..., mr.slice_dict["global_rot_data"]]
        pred_rot_data = pred_rot_data.reshape(*pred_rot_data.shape[:2], mr.nbjoints, 6)
        pred_global_rot = cont6d_to_matrix(pred_rot_data)  # (B, T, J, 3, 3)
        pred_smooth_root = pred_un[..., mr.slice_dict["smooth_root_pos"]]  # (B, T, 3)

        # Reconstruct the ACTUAL world root from features the same way
        # KimodoMotionRep.inverse does (kimodo_motionrep.py:195-198). The
        # encoder stores local_jp[root_idx] = (root.x - smooth_root.x, root.y,
        # root.z - smooth_root.z) — xz are smoothing residuals, y is absolute
        # world height. Anchoring FK at smooth_root_pos instead (the previous
        # behavior) put a uniform per-frame xz translation equal to the ADMM
        # smoother residual onto every joint (~2-3 cm typical, up to ~13 cm in
        # dynamic frames) — an irreducible floor the rotations could not fix.
        pred_local_jp = pred_un[..., mr.slice_dict["local_joints_positions"]]
        pred_local_jp = pred_local_jp.reshape(*pred_local_jp.shape[:2], mr.nbjoints, 3)
        root_idx = int(mr.skeleton.root_idx)
        actual_root = torch.stack(
            [
                pred_smooth_root[..., 0] + pred_local_jp[..., root_idx, 0],
                pred_local_jp[..., root_idx, 1],
                pred_smooth_root[..., 2] + pred_local_jp[..., root_idx, 2],
            ],
            dim=-1,
        )  # (B, T, 3)

        if self.fk_kind == "standard":
            from kimodo.skeleton.transforms import global_rots_to_local_rots
            pred_local_rot = global_rots_to_local_rots(pred_global_rot, mr.skeleton)
            _, pred_posed, _ = mr.skeleton.fk(pred_local_rot, actual_root)
            return pred_posed
        elif self.fk_kind == "chainreset_hml3d":
            # bone_lengths resolution:
            #   1) caller supplied -> use directly (training: _fk_positions uses the
            #      per-batch cache for speed and to keep gradient-free)
            #   2) loss has a populated cache -> use that (viz at step > 0)
            #   3) cold path: derive from the input's own positions block (viz at
            #      step 0, or any callsite that doesn't go through _fk_positions)
            if bone_lengths is None:
                if self._fk_bone_lengths_cache is not None:
                    bone_lengths = self._fk_bone_lengths_cache.to(
                        device=pred_un.device, dtype=pred_un.dtype,
                    )
                else:
                    from kimodo.motion_rep.fk_hml3d import (
                        derive_bone_lengths_from_world_joints,
                        world_joints_from_kimodo_features,
                    )
                    pos_world_for_bl = world_joints_from_kimodo_features(
                        pred_un, mr.slice_dict, n_joints=mr.nbjoints,
                    )
                    bone_lengths = derive_bone_lengths_from_world_joints(
                        pos_world_for_bl, chains=self.fk_chains,
                        n_joints=mr.nbjoints, reduce="median",
                    )
            from kimodo.motion_rep.fk_hml3d import chainreset_fk_world_joints
            return chainreset_fk_world_joints(
                pred_global_rot,
                actual_root,
                bone_lengths=bone_lengths,
                raw_offsets=self.fk_raw_offsets,
                chains=self.fk_chains,
            )
        else:
            raise ValueError(f"unknown fk_kind: {self.fk_kind}")

    def _fk_positions(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run FK on predicted rotations and build a matching GT positions tensor.

        Returns ``(pred_posed, tgt_world)``, both ``(B, T, J, 3)`` in world
        coordinates. Shared between the ``fk`` (position L1) and ``fk_v``
        (velocity L1) terms so the FK forward only runs once per step.

        ``fk_target='gt'`` builds the target from the GT positions block (paper),
        ``fk_target='pred'`` from the predicted positions block (pure
        self-consistency), and ``fk_target='fk_gt'`` runs FK on the GT rotations
        through the same skeleton so per-actor calibration cancels.

        Both pred and target are *normalized* features here; we unnormalize
        locally without mutating inputs.
        """
        mr = self.motion_rep
        pred_un = mr.unnormalize(pred)
        target_un = mr.unnormalize(target)

        # For chain-reset FK we need per-joint bone lengths. HumanML3D's
        # preprocessing already uniform-skeleton'd every motion to a canonical
        # SMPL-22, so the per-sample derived lengths are identical across the
        # whole dataset. We populate a (J,) cache on the first call and reuse
        # for every subsequent step — saves the per-step derivation cost and
        # makes the implicit "all samples share one skeleton" assumption
        # explicit. Use no_grad so this is purely a constant on the forward
        # pass — gradient still flows through pred_global_rot in _fk_world_from_pred.
        bone_lengths: Optional[torch.Tensor] = None
        if self.fk_kind == "chainreset_hml3d":
            from kimodo.motion_rep.fk_hml3d import (
                derive_bone_lengths_from_world_joints,
                world_joints_from_kimodo_features,
            )
            if self._fk_bone_lengths_cache is None:
                with torch.no_grad():
                    gt_world = world_joints_from_kimodo_features(
                        target_un, mr.slice_dict, n_joints=mr.nbjoints,
                    )  # (B, T, J, 3)
                    per_sample = derive_bone_lengths_from_world_joints(gt_world)  # (B, J)
                    # Median across the batch (any sample would do since they're
                    # uniform; median is a defensive choice against a freak NaN).
                    canonical = per_sample.median(dim=0).values.detach()  # (J,)
                self._fk_bone_lengths_cache = canonical
                log.info(
                    "KimodoLoss: cached canonical bone lengths (J=%d). Max sample-spread within "
                    "first batch: %.2e m (should be ~0 on HumanML3D).",
                    canonical.shape[0],
                    float((per_sample - canonical[None]).abs().max()),
                )
            bone_lengths = self._fk_bone_lengths_cache.to(
                device=target_un.device, dtype=target_un.dtype,
            )

        pred_posed = self._fk_world_from_pred(pred_un, bone_lengths=bone_lengths)

        if self.fk_target == "fk_gt":
            # Pure rotational-consistency target: run FK on the GT rotations
            # through the SAME skeleton (same bone_lengths) so any per-sample
            # calibration cancels and the loss measures only rotational + root
            # errors.
            with torch.no_grad():
                tgt_world = self._fk_world_from_pred(target_un, bone_lengths=bone_lengths)
        elif self.fk_kind == "hml3d_native":
            # 263-D layout: recover world joint positions from the ric block
            # plus integrated root state. Mirrors HumanML3D's recover_from_ric.
            from kimodo.motion_rep.humanml3d_native import (
                hml3d_native_world_joints_from_features,
            )
            src_un = target_un if self.fk_target == "gt" else pred_un
            tgt_world = hml3d_native_world_joints_from_features(
                src_un, n_joints=mr.nbjoints,
            )
        else:
            # Kimodo 273-D layout: read positions block of GT (paper) or of the
            # predicted features (pure self-consistency).
            src_un = target_un if self.fk_target == "gt" else pred_un
            src_smooth_root = src_un[..., mr.slice_dict["smooth_root_pos"]]
            src_local_jp = src_un[..., mr.slice_dict["local_joints_positions"]]
            src_local_jp = src_local_jp.reshape(*src_local_jp.shape[:2], mr.nbjoints, 3)
            tgt_world = src_local_jp.clone()
            # local_joints_positions stores world XYZ minus (smooth_root_x, 0, smooth_root_z).
            tgt_world[..., 0] = tgt_world[..., 0] + src_smooth_root[..., None, 0]
            tgt_world[..., 2] = tgt_world[..., 2] + src_smooth_root[..., None, 2]

        return pred_posed, tgt_world

    def _fk_term(
        self,
        pred_posed: torch.Tensor,
        tgt_world: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """gamma_7: mean |FK(pred_rot) - tgt_world|_1 over valid frames * J * 3."""
        mr = self.motion_rep
        diff = (pred_posed - tgt_world).abs()  # (B, T, J, 3)
        mask = pad_mask.float().unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        num = (diff * mask).sum()
        den = mask.sum() * float(mr.nbjoints * 3) + 1e-8
        return num / den

    def _fk_v_term(
        self,
        pred_posed: torch.Tensor,
        tgt_world: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """gamma_8: L1 on the first-order temporal difference of FK joint positions.

        Velocity at frame t is ``pos[t+1] - pos[t]`` (no fps scale — only the
        relative weight matters). A velocity sample is valid iff BOTH endpoint
        frames are unpadded.
        """
        mr = self.motion_rep
        pred_v = pred_posed[:, 1:] - pred_posed[:, :-1]  # (B, T-1, J, 3)
        tgt_v = tgt_world[:, 1:] - tgt_world[:, :-1]
        diff = (pred_v - tgt_v).abs()
        v_pad = pad_mask[:, 1:] & pad_mask[:, :-1]  # (B, T-1)
        mask = v_pad.float().unsqueeze(-1).unsqueeze(-1)  # (B, T-1, 1, 1)
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

        # gamma_7/8: FK position + velocity consistency. Only run FK once
        # and feed the resulting (pred_posed, tgt_world) into both terms.
        fk_w = float(self.weights.get("fk", 0.0))
        fk_v_w = float(self.weights.get("fk_v", 0.0))
        if fk_w > 0.0 or fk_v_w > 0.0:
            pred_posed, tgt_world = self._fk_positions(pred, target)
            if fk_w > 0.0:
                fk_loss = self._fk_term(pred_posed, tgt_world, pad_mask)
                losses["l_fk"] = fk_loss.detach()
                total = total + fk_w * fk_loss
                denom = denom + fk_w
            if fk_v_w > 0.0:
                fk_v_loss = self._fk_v_term(pred_posed, tgt_world, pad_mask)
                losses["l_fk_v"] = fk_v_loss.detach()
                total = total + fk_v_w * fk_v_loss
                denom = denom + fk_v_w

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
    """Build a frozen text encoder. Dispatches on ``text_cfg.type``.

    Supported types:
      - ``llm2vec`` (default) — LLM2Vec-Meta-Llama-3-8B (text_dim=4096, pooled).
                                Very slow per-step. For training runs,
                                **precompute embeddings offline** via
                                ``kimodo/scripts/precompute_text_embeddings.py``
                                and point ``text_encoder.cache_path`` at the
                                resulting .pt — the live encoder is then never
                                called during training.
      - ``clip``              — CLIP text tower (text_dim=512 for ViT-B,
                                768 for ViT-L). MDM uses ViT-B/32. Batched;
                                fast enough to run live.

    If ``text_cfg.cache_path`` is set, a ``CachedTextEncoder`` is returned
    that loads precomputed features from disk — the underlying live encoder
    is never built.
    """
    enc_type = str(text_cfg.get("type", "llm2vec")).lower()
    dev = str(device) if text_cfg.get("device", "auto") == "auto" else text_cfg.device

    cache_path = text_cfg.get("cache_path", None)
    if cache_path:
        from kimodo.model.cached_text import CachedTextEncoder
        return CachedTextEncoder(cache_path=str(cache_path), device=dev)

    if enc_type == "clip":
        from kimodo.model.clip_text import CLIPTextEncoder
        model_name = text_cfg.get("model_name", "openai/clip-vit-base-patch32")
        return CLIPTextEncoder(
            model_name_or_path=model_name,
            device=dev,
            dtype="float32" if text_cfg.get("fp32", False) else "bfloat16",
            pooled=bool(text_cfg.get("pooled", True)),
            max_length=int(text_cfg.get("max_length", 22)),
        )

    if enc_type != "llm2vec":
        raise ValueError(
            f"Unknown text_encoder.type={enc_type!r}; expected 'llm2vec' | 'clip'."
        )

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

    return LLM2VecEncoder(
        base_model_name_or_path=base,
        peft_model_name_or_path=peft,
        dtype="float32" if text_cfg.get("fp32", False) else "bfloat16",
        llm_dim=4096,
        device=dev,
    )


@torch.no_grad()
def encode_texts(
    text_encoder, texts: List[str], device: torch.device,
):
    """Return ``(feats, text_pad_mask)``.

    * ``feats``         : ``(B, L, D)`` text features on ``device``.
    * ``text_pad_mask`` : ``(B, L)`` bool — True = valid token, False = padding.

    For pooled encoders (LLM2Vec, CLIP-pooled) ``L=1`` and the mask is
    all-True. For per-token encoders (DistilBERT, CLIP-non-pooled) the mask
    reflects the encoder's attention mask so the denoiser only attends to
    real tokens.

    The denoiser pads ``L`` to ``num_text_tokens_override`` internally.
    """
    feats, lengths = text_encoder(texts)
    feats = feats.to(device=device)
    B, L = int(feats.shape[0]), int(feats.shape[1])
    if isinstance(lengths, int):
        lengths = [lengths] * B
    pad_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for i, n in enumerate(lengths):
        n = max(0, min(int(n), L))
        if n > 0:
            pad_mask[i, :n] = True
    return feats, pad_mask


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
# Diffusion schedule helper
# -----------------------------------------------------------------------------
def ensure_full_training_schedule(diffusion: Diffusion) -> None:
    """Reset the diffusion buffers to the full ``num_base_steps`` schedule.

    DDIM sampling (used in viz / inference) calls ``calc_diffusion_vars`` with a
    subsampled timestep set, mutating the schedule buffers in place. Training's
    ``q_sample`` must use the full contiguous schedule, so we restore it here.
    The full ``use_timesteps`` (an ``arange``) is cached on the diffusion object
    to avoid recomputing it every step.
    """
    cached = getattr(diffusion, "_full_use_timesteps", None)
    if cached is None or cached.device != diffusion.device:
        cached = diffusion.space_timesteps(diffusion.num_base_steps)[0]
        diffusion._full_use_timesteps = cached
    diffusion.calc_diffusion_vars(cached)


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
    text_feat, text_pad_mask = encode_texts(text_encoder, texts, device)
    # zero out dropped ones (encode_texts already keeps shape; double-safety)
    if not keep.all():
        keep_mask = keep.to(device=device, dtype=text_feat.dtype).view(-1, 1, 1)
        text_feat = text_feat * keep_mask

    # IMPORTANT: restore the FULL training diffusion schedule before q_sample.
    # Sampling/viz calls ``diffusion.calc_diffusion_vars(spaced_timesteps)`` which
    # mutates the schedule buffers (sqrt_alphas_cumprod, ...) IN PLACE for a
    # subsampled (e.g. 50-step) DDIM schedule. If we don't reset them here, the
    # next training step's q_sample indexes those subsampled buffers with the
    # contiguous timestep t in [0, num_base_steps) and adds a wrong amount of
    # noise — which permanently destabilizes training right after the first viz.
    ensure_full_training_schedule(diffusion)

    # Sample diffusion timesteps uniformly.
    t = torch.randint(0, diffusion.num_base_steps, (B,), device=device)

    # q_sample (forward diffusion) — now guaranteed to use the full schedule.
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
    # Cheap data-sanity signals for spotting representation/normalization issues.
    # (Logged to TB; a healthy normalized batch is usually within ~[-5, 5].)
    losses["data_absmax"] = motion.detach().abs().max()
    losses["timestep_mean"] = t.float().mean()
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
def _cfg_sample(
    raw: nn.Module,
    diffusion: Diffusion,
    *,
    text_feat: torch.Tensor,
    text_pad_mask: torch.Tensor,
    pad_mask: torch.Tensor,
    first_heading: torch.Tensor,
    motion_mask: torch.Tensor,
    observed: torch.Tensor,
    n_steps: int,
    cfg_scale: float,
    device: torch.device,
    sampler: str = "ddim",
) -> torch.Tensor:
    """Classifier-free-guided x0 sampler (SOMA twin of ``train_w_hml3d._cfg_sample``).

    A doubled-batch forward recovers the conditional + unconditional x0 in one
    pass; ``cfg_scale <= 1`` skips the second branch.

    ``sampler``: ``"ddim"`` (deterministic, default) or ``"ddpm"`` (stochastic
    ancestral sampling — adds ``sqrt(posterior_variance) * z`` at every step
    except the last). See the HML3D copy of this function for details.
    """
    if sampler not in ("ddim", "ddpm"):
        raise ValueError(f"sampler must be 'ddim' or 'ddpm'; got {sampler!r}")

    B, T = pad_mask.shape
    D = raw.motion_rep.motion_rep_dim

    use_timesteps, map_tensor = diffusion.space_timesteps(n_steps)
    diffusion.calc_diffusion_vars(use_timesteps)
    cur = torch.randn(B, T, D, device=device)

    do_cfg = cfg_scale is not None and float(cfg_scale) > 1.0
    if do_cfg:
        text_feat_2x = torch.cat([text_feat, torch.zeros_like(text_feat)], dim=0)
        text_pad_2x = torch.cat([text_pad_mask, text_pad_mask], dim=0)
        pad_2x = torch.cat([pad_mask, pad_mask], dim=0)
        fh_2x = torch.cat([first_heading, first_heading], dim=0)
        mm_2x = torch.cat([motion_mask, motion_mask], dim=0)
        obs_2x = torch.cat([observed, observed], dim=0)

    for i in list(range(n_steps))[::-1]:
        t = torch.full((B,), i, device=device, dtype=torch.long)
        t_map = map_tensor[t]
        if do_cfg:
            cur_2x = torch.cat([cur, cur], dim=0)
            t_2x = torch.cat([t_map, t_map], dim=0)
            pred_2x = raw(
                cur_2x, pad_2x, text_feat_2x, text_pad_2x, t_2x,
                first_heading_angle=fh_2x,
                motion_mask=mm_2x, observed_motion=obs_2x,
            )
            pred_cond, pred_uncond = pred_2x[:B], pred_2x[B:]
            pred_clean = pred_uncond + float(cfg_scale) * (pred_cond - pred_uncond)
        else:
            pred_clean = raw(
                cur, pad_mask, text_feat, text_pad_mask, t_map,
                first_heading_angle=first_heading,
                motion_mask=motion_mask, observed_motion=observed,
            )

        if sampler == "ddim":
            eps = (
                diffusion.sqrt_recip_alphas_cumprod[t, None, None] * cur - pred_clean
            ) / diffusion.sqrt_recipm1_alphas_cumprod[t, None, None]
            alpha_bar_prev = diffusion.alphas_cumprod_prev[t, None, None]
            cur = pred_clean * alpha_bar_prev.sqrt() + (1 - alpha_bar_prev).sqrt() * eps
        else:  # "ddpm"
            mean = (
                diffusion.posterior_mean_coef1[t, None, None] * pred_clean
                + diffusion.posterior_mean_coef2[t, None, None] * cur
            )
            if i > 0:
                noise = torch.randn_like(cur)
                log_var = diffusion.posterior_log_variance_clipped[t, None, None]
                cur = mean + torch.exp(0.5 * log_var) * noise
            else:
                cur = mean
    return cur


@torch.no_grad()
def viz_step(
    denoiser: nn.Module,
    diffusion: Diffusion,
    text_encoder,
    device: torch.device,
    cfg: DictConfig,
    step: int,
    out_dir: Path,
    test_examples: Optional[List[dict]] = None,
    extra_prompts: Optional[List[str]] = None,
    tb_writer=None,
) -> None:
    """Unified per-step viz: writes ``<out_dir>/<step:07d>/<i>_<id>.{npz,mp4}``.

    Each item is a single (caption, gen) sample rendered side-by-side with its
    GT motion when one is available, or with a blank ``(no GT)`` left panel
    otherwise — so the layout stays consistent across both kinds.

    Items come from two sources, batched into one diffusion forward:
      1. ``extra_prompts`` (default ``cfg.viz.prompts``) — no GT, length =
         ``cfg.viz.num_frames``, first_heading = 0.
      2. ``test_examples`` — held-out (id, caption, gt_features, length,
         first_heading) tuples from :py:func:`_load_test_examples_soma`.

    NPZ contains the decoded gen fields (``posed_joints``, root pos, etc.) +
    the prompt; MP4 is the side-by-side stick figure.
    """
    import imageio.v3 as iio
    from kimodo.scripts.render_soma import render_sidebyside

    step_dir = out_dir / f"{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    raw = (denoiser.module if hasattr(denoiser, "module") else denoiser).eval()
    motion_rep = raw.motion_rep
    n_steps = int(cfg.viz.num_denoising_steps)
    cfg_scale = float(cfg.viz.get("cfg_scale", 2.5))
    fps = int(cfg.data.get("fps", 20))
    num_frames_default = int(cfg.viz.num_frames)
    save_mp4 = bool(cfg.viz.get("save_videos", True))

    if extra_prompts is None:
        extra_prompts = [p for p in (cfg.viz.get("prompts", []) or []) if isinstance(p, str) and p]
    test_examples = test_examples or []

    items: List[dict] = []
    for i, p in enumerate(extra_prompts):
        safe = p.lower().replace(" ", "_").replace(".", "")[:40] or f"prompt_{i}"
        items.append({
            "id": f"prompt_{i:02d}_{safe}",
            "caption": p,
            "gt_features": None,
            "length": num_frames_default,
            "first_heading": 0.0,
        })
    for ex in test_examples:
        items.append({
            "id": ex["id"],
            "caption": ex["caption"],
            "gt_features": ex["gt_features"],
            "length": int(ex["length"]),
            "first_heading": float(ex["first_heading"]),
        })
    if not items:
        return

    captions = [it["caption"] for it in items]
    lengths = [it["length"] for it in items]
    max_T = int(max(lengths))
    B = len(items)

    text_feat, text_pad_mask = encode_texts(text_encoder, captions, device)
    pad_mask = torch.zeros(B, max_T, dtype=torch.bool, device=device)
    for i, L in enumerate(lengths):
        pad_mask[i, :L] = True
    first_heading = torch.tensor(
        [it["first_heading"] for it in items], dtype=torch.float32, device=device,
    )
    motion_mask = torch.zeros(B, max_T, motion_rep.motion_rep_dim, device=device)
    observed = torch.zeros(B, max_T, motion_rep.motion_rep_dim, device=device)

    cur = _cfg_sample(
        raw, diffusion,
        text_feat=text_feat, text_pad_mask=text_pad_mask,
        pad_mask=pad_mask, first_heading=first_heading,
        motion_mask=motion_mask, observed=observed,
        n_steps=n_steps, cfg_scale=cfg_scale, device=device,
    )  # (B, max_T, D) normalized
    raw.train()

    joint_parents = motion_rep.skeleton.joint_parents.cpu().numpy().tolist()
    video_stack: List[np.ndarray] = []
    for i, it in enumerate(items):
        L = it["length"]
        gen_feats = cur[i:i + 1, :L].float()
        gen_out = motion_rep.inverse(gen_feats, is_normalized=True, return_numpy=False)
        gen_joints = gen_out["posed_joints"][0].cpu().numpy()
        gen_dict = {
            k: (v[0].cpu().numpy() if torch.is_tensor(v) else v)
            for k, v in gen_out.items()
        }

        gt_joints: Optional[np.ndarray] = None
        if it["gt_features"] is not None:
            gt_feats = torch.from_numpy(it["gt_features"]).to(device).unsqueeze(0)
            gt_out = motion_rep.inverse(gt_feats, is_normalized=False, return_numpy=False)
            gt_joints = gt_out["posed_joints"][0].cpu().numpy()

        base = step_dir / f"{i:02d}_{it['id']}"
        np.savez(base.with_suffix(".npz"), **gen_dict, prompt=it["caption"])
        if not save_mp4:
            continue
        try:
            frames = render_sidebyside(
                gt_joints, gen_joints, joint_parents, caption=it["caption"],
            )
        except Exception as e:
            log.warning("Render failed for %s: %s", it["id"], e)
            continue
        try:
            iio.imwrite(
                str(base.with_suffix(".mp4")), frames,
                fps=float(fps), codec="h264", plugin="pyav",
            )
        except Exception as e:
            log.warning("MP4 write failed for %s: %s", it["id"], e)
        video_stack.append(frames)

    if tb_writer is not None and video_stack:
        Tmax = max(v.shape[0] for v in video_stack)
        H, W = video_stack[0].shape[1:3]
        vids = np.zeros((len(video_stack), Tmax, 3, H, W), dtype=np.float32)
        for n, v in enumerate(video_stack):
            vt = np.transpose(v, (0, 3, 1, 2)).astype(np.float32) / 255.0
            vids[n, : vt.shape[0]] = vt
            if vt.shape[0] < Tmax:
                vids[n, vt.shape[0]:] = vt[-1]
        try:
            tb_writer.add_video("viz", torch.from_numpy(vids), global_step=step, fps=fps)
        except Exception as e:
            log.warning("TensorBoard add_video failed (%s); skipping.", e)

    log.info("Wrote %d viz samples to %s", len(items), step_dir)


@torch.no_grad()
def _load_test_examples_soma(
    test_split_path: str | Path,
    data_root: str | Path,
    natural_csv_path: str | Path,
    motion_rep,
    skeleton,
    n_samples: int = 4,
    max_frames: int = 200,
    min_frames: int = 10,
    include_mirrored: bool = False,
) -> List[dict]:
    """Build held-out (id, caption, gt_features, length) examples for side-by-
    side GT-vs-gen viz on SOMA / BONES-SEED.

    Each entry pairs the first ``max_frames`` of a test motion with its first
    natural description from ``seed_metadata_v004.csv``. Features are
    canonicalised by ``motion_rep`` (with ``to_normalize=False`` — gen output
    is decoded with ``is_normalized=True`` so the codepaths stay symmetric).
    """
    import csv as _csv

    data_root = Path(data_root)
    natural_csv_path = Path(natural_csv_path)

    desc_by_name: Dict[str, str] = {}
    with open(natural_csv_path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            d = row.get("content_natural_desc_1", "")
            if isinstance(d, str):
                d = d.strip()
            if d:
                desc_by_name[fname] = d

    out: List[dict] = []
    with open(test_split_path, "r") as f:
        for line in f:
            if len(out) >= int(n_samples):
                break
            rel = line.strip()
            if not rel:
                continue
            fname = os.path.basename(rel)
            if not include_mirrored and fname.endswith("_M"):
                continue
            caption = desc_by_name.get(fname)
            if not caption:
                continue
            npz_path = data_root / (rel + ".npz")
            if not npz_path.is_file():
                continue
            try:
                with np.load(npz_path, mmap_mode="r") as data:
                    n_total = int(data["local_rot_mats"].shape[0])
                    T = min(n_total, int(max_frames))
                    if T < int(min_frames):
                        continue
                    local_rot_77 = np.asarray(data["local_rot_mats"][:T])
                    root_positions = np.asarray(data["root_positions"][:T])
            except (OSError, KeyError, ValueError):
                continue
            local_rot_30 = skeleton.from_SOMASkeleton77(
                torch.from_numpy(local_rot_77).float()
            )
            root_pos_t = torch.from_numpy(root_positions).float()
            feats = motion_rep(
                local_rot_30.unsqueeze(0),
                root_pos_t.unsqueeze(0),
                to_normalize=False,
                to_canonicalize=True,
                lengths=torch.tensor([T]),
            )[0]  # (T, D) unnormalized
            heading_cs = feats[0, motion_rep.slice_dict["global_root_heading"]]
            first_heading = float(torch.atan2(heading_cs[1], heading_cs[0]).item())
            out.append({
                "id": fname,
                "caption": caption,
                "gt_features": feats.cpu().numpy(),
                "first_heading": first_heading,
                "length": int(T),
            })
    return out


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
    pad_to = int(cfg.data.get("pad_to_max_frames", cfg.data.max_frames)) if cfg.data.get("pad_to_max_frames", True) else None
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

    # ----- held-out test examples for GT-vs-gen viz (rank 0, SOMA only) -----
    test_examples: List[dict] = []
    if env.is_main and bool(cfg.viz.get("test_vs_gt", False)):
        test_split_path = cfg.viz.get("test_split_file") or cfg.data.get("train_split_path")
        natural_csv_path = cfg.data.get("natural_csv_path")
        if test_split_path and natural_csv_path:
            try:
                test_examples = _load_test_examples_soma(
                    test_split_path=test_split_path,
                    data_root=cfg.data.data_root,
                    natural_csv_path=natural_csv_path,
                    motion_rep=dataset_motion_rep,
                    skeleton=dataset_motion_rep.skeleton,
                    n_samples=int(cfg.viz.get("num_test_samples", 4)),
                    max_frames=int(cfg.viz.get("test_max_frames", cfg.data.max_frames)),
                    min_frames=int(cfg.data.get("min_frames", 10)),
                    include_mirrored=False,
                )
                log.info(
                    "Loaded %d held-out test examples for side-by-side viz.",
                    len(test_examples),
                )
            except Exception as e:
                log.warning("Could not load test examples for viz: %s", e)

    def _run_viz(at_step: int) -> None:
        try:
            viz_step(
                denoiser, diffusion, text_encoder, device, cfg, at_step,
                output_dir / "viz",
                test_examples=test_examples,
                tb_writer=tb_writer,
            )
        except Exception as e:
            log.warning("Viz failed at step %d: %s", at_step, e)

    log.info("Starting training from step %d for %d steps", start_step, num_steps)
    denoiser.train()

    grad_accum = int(cfg.trainer.grad_accum)
    log_every = int(cfg.trainer.log_every)
    ckpt_every = int(cfg.trainer.ckpt_every)
    viz_every = int(cfg.trainer.viz_every)
    grad_clip = float(cfg.trainer.grad_clip)
    ema_every = int(cfg.trainer.ema_every)
    text_drop_prob = float(cfg.trainer.text_drop_prob)

    # Initial viz before any training so we can see the untrained baseline.
    if env.is_main and viz_every:
        _run_viz(start_step)

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
            params_for_clip = (
                denoiser.parameters() if not hasattr(denoiser, "module") else denoiser.module.parameters()
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
                comp = " ".join(f"{k.replace('l_',''):s}={v:.3f}" for k, v in agg_loss.items() if k.startswith("l_"))
                if comp:
                    line += " | " + comp
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
                _run_viz(step)

            if step >= num_steps:
                break
        epoch += 1

    if env.is_main:
        save_checkpoint(
            output_dir / "ckpt_final.pt",
            denoiser, optimizer, scheduler, scaler, ema, step, cfg,
        )
    cleanup_distributed(env)


def build_soma_dataset(cfg: DictConfig, dataset_motion_rep, seed: int):
    """Build the SOMA-family training dataset specified by ``cfg.data.kind``.

    Shared by the training loop and by tools that need to enumerate the same
    dataset (e.g. ``precompute_text_embeddings``). Dispatches on
    ``cfg.data.kind`` (default ``"segments"``): ``"segments"`` →
    :class:`SOMATextMotionDataset`, ``"bones_seed"`` → :class:`SOMABonesSeedDataset`.
    """
    dataset_kind = str(cfg.data.get("kind", "segments")).lower()
    if dataset_kind == "bones_seed":
        return SOMABonesSeedDataset(
            data_root=cfg.data.data_root,
            natural_csv_path=cfg.data.natural_csv_path,
            temporal_labels_path=cfg.data.temporal_labels_path,
            multi_timeline_path=cfg.data.multi_timeline_path,
            train_split_path=cfg.data.get("train_split_path"),
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
            packed_motions_path=cfg.data.get("packed_motions_path"),
            packed_features_path=cfg.data.get("packed_features_path"),
        )
    if dataset_kind != "segments":
        raise ValueError(
            f"Unknown data.kind={dataset_kind!r}; expected 'segments' or 'bones_seed'."
        )
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
