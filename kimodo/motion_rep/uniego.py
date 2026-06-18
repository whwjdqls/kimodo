# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UniEgoMotion-style head-centric (211-D) motion representation.

Motion-rep wrapper used by the MDM-style one-stage training script when feeding
the head-centric features produced by ``benchmark/kimodo_to_uniego.py`` (stored
as ``.npz`` with key ``features`` in ``HumanML3D/uniego_rep/``). Sibling to
``kimodo.motion_rep.humanml3d_native.HumanML3DNativeMotionRep`` — same minimal
interface (``motion_rep_dim``, ``nbjoints``, ``fps``, ``slice_dict``,
``skeleton``, ``normalize`` / ``unnormalize``), only the layout + decode differ.

211-D layout (J = 22 SMPL joints, head = joint 15):

    idx          name           width  description
    ---          ----           -----  -----------
    [0,   198)   local_pose      198   per-joint local SE(3) in the head-centric
                                       canonical frame: joint j = 6D rot (cols 0,1)
                                       ++ 3D translation, interleaved (9 per joint).
    [198, 207)   canon_delta       9   residual canonical-frame transform
                                       (6D rot ++ 3D trans). Frame 0 of a clip is
                                       the absolute cM[0]; frames 1+ are
                                       cM[t-1]^{-1} @ cM[t].
    [207, 211)   foot_contacts     4   binary contact flags (carried from kimodo).

Decode to world joints: cumulative-compose ``canon_delta`` to recover the
per-frame canonical frame ``cM``, then ``M_j = cM @ local_T_j`` and read the
translation. See :func:`uniego_world_joints_from_features`. No FK and no
skeleton offsets are needed — the per-joint translation already IS the joint
position in the canonical frame.

Frame-0 canonicalization (training): the stored rep keeps frame 0's
``canon_delta`` as the *absolute* cM[0] so the round-trip back to world / HML3D
is lossless. For windowed training we instead set each window's frame-0
``canon_delta`` to the IDENTITY transform (origin, +Z facing) via
:func:`canonicalize_frame0`, so every training window starts canonically — the
exact analog of HumanML3D's frame-0 canonicalization in the native rep. The
normalization stats are computed the same way.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

FEAT_DIM = 211
N_JOINTS = 22
HEAD_IDX = 15
_LOCAL = N_JOINTS * 9  # 198
_DELTA = _LOCAL + 9    # 207

# Encoding of the identity SE(3) in the (6D rot ++ 3 trans) convention:
# matrix_to_cont6d(I) = [I[:,0], I[:,1]] = [1,0,0, 0,1,0]; translation = 0.
IDENTITY_DELTA9: List[float] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def _feature_layout() -> "OrderedDict[str, slice]":
    return OrderedDict(
        [
            ("local_pose", slice(0, _LOCAL)),        # [0, 198)
            ("canon_delta", slice(_LOCAL, _DELTA)),  # [198, 207)
            ("foot_contacts", slice(_DELTA, FEAT_DIM)),  # [207, 211)
        ]
    )


# ----------------------------------------------------------------------------
# 6D <-> matrix (column convention; identical to humanml3d_to_kimodo's helpers
# and humanml3d_native._cont6d_to_matrix, so encode/decode are exact inverses).
# ----------------------------------------------------------------------------
def _cont6d_to_matrix(cont6d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / (torch.norm(x_raw, dim=-1, keepdim=True) + eps)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / (torch.norm(z, dim=-1, keepdim=True) + eps)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)  # (..., 3, 3), columns = x,y,z


def _build_se3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(R.shape[:-2] + (4, 4), device=R.device, dtype=R.dtype)
    out[..., :3, :3] = R
    out[..., :3, 3] = t
    out[..., 3, 3] = 1.0
    return out


# ----------------------------------------------------------------------------
# Frame-0 canonicalization (used by the dataset and the stats computation).
# ----------------------------------------------------------------------------
def canonicalize_frame0(features: torch.Tensor, n_joints: int = N_JOINTS) -> torch.Tensor:
    """Set the ``canon_delta`` block of frame 0 to the identity transform.

    Accepts ``(T, D)`` or ``(B, T, D)``; returns a copy (does not mutate the
    input). This drops the absolute world placement of the first frame so that
    every (windowed) clip starts at the origin facing +Z — matching HumanML3D's
    native frame-0 canonicalization. Frames 1+ (relative residuals) are
    untouched, so the trajectory *shape* is preserved exactly.

    Skeleton-agnostic: the ``canon_delta`` block lives at ``[n_joints*9 :
    n_joints*9+9]`` (J=22 for HML3D, 30 for SOMA-30).
    """
    lo = n_joints * 9
    out = features.clone()
    ident = out.new_tensor(IDENTITY_DELTA9)
    if out.dim() == 2:
        out[0, lo:lo + 9] = ident
    elif out.dim() == 3:
        out[:, 0, lo:lo + 9] = ident
    else:
        raise ValueError(f"expected (T,D) or (B,T,D); got {tuple(features.shape)}")
    return out


# ----------------------------------------------------------------------------
# Decode: 211-D (unnormalized) -> world joints (B, T, J, 3).
# ----------------------------------------------------------------------------
def uniego_world_joints_from_features(
    features_unnormalized: torch.Tensor,  # (B, T, 211) or (T, 211)
    n_joints: int = N_JOINTS,
) -> torch.Tensor:
    """Recover world joint positions from the head-centric rep.

    cumulative-compose the ``canon_delta`` residuals into per-frame canonical
    frames ``cM``, then ``M_j = cM @ local_T_j`` and read the translation.
    Returns ``(B, T, J, 3)`` (or ``(T, J, 3)`` if a 2-D input was given).
    """
    squeeze = False
    if features_unnormalized.dim() == 2:
        features_unnormalized = features_unnormalized.unsqueeze(0)
        squeeze = True
    B, T, D = features_unnormalized.shape
    lo = n_joints * 9
    if D < lo + 9:
        raise ValueError(f"feature dim {D} too small for n_joints={n_joints}")
    device, dtype = features_unnormalized.device, features_unnormalized.dtype

    # 1) per-joint local SE(3)
    fj = features_unnormalized[..., :lo].reshape(B, T, n_joints, 9)
    R_local = _cont6d_to_matrix(fj[..., :6])      # (B, T, J, 3, 3)
    t_local = fj[..., 6:9]                         # (B, T, J, 3)
    local_T = _build_se3(R_local, t_local)         # (B, T, J, 4, 4)

    # 2) residual canonical frame -> cM (cumulative compose over time)
    fd = features_unnormalized[..., lo:lo + 9]      # (B, T, 9)
    R_delta = _cont6d_to_matrix(fd[..., :6])        # (B, T, 3, 3)
    delta = _build_se3(R_delta, fd[..., 6:9])       # (B, T, 4, 4)
    cM = torch.empty_like(delta)
    cM[:, 0] = delta[:, 0]
    for t in range(1, T):
        cM[:, t] = cM[:, t - 1] @ delta[:, t]

    # 3) M = cM @ local_T ; joint position = translation block
    M = torch.matmul(cM[:, :, None], local_T)       # (B, T, J, 4, 4)
    world = M[..., :3, 3]                            # (B, T, J, 3)
    return world.squeeze(0) if squeeze else world


# ----------------------------------------------------------------------------
# The motion-rep wrapper (mirrors HumanML3DNativeMotionRep's interface).
# ----------------------------------------------------------------------------
class UniegoMotionRep(nn.Module):
    """Minimal motion-rep for the UniEgoMotion-style head-centric representation.

    Skeleton-agnostic: ``J = skeleton.nbjoints`` and the feature width is
    ``J*9 + 9 + n_foot`` (211 for HumanML3D SMPL-22; 283 for SOMA-30). Exposes
    exactly what the one-stage denoiser + MDM L2 loss need: ``motion_rep_dim``,
    ``nbjoints``, ``fps``, ``slice_dict``, ``skeleton``, ``normalize`` /
    ``unnormalize``. Mean/std are the precomputed ``Mean_uniego.npy`` /
    ``Std_uniego.npy`` (frame-0-canonicalized, matching the dataset).
    """

    motion_rep_dim: int = FEAT_DIM

    def __init__(
        self,
        mean_path: str | Path,
        std_path: str | Path,
        skeleton,
        fps: int = 20,
        eps: float = 1e-6,
        feat_bias: float = 1.0,
        n_foot: int = 4,
        # ``stats_path`` is injected by build_denoiser_from_model_config; unused
        # here (mean/std come from absolute paths). Accept & ignore.
        stats_path: Optional[str | Path] = None,
    ):
        super().__init__()
        self.skeleton = skeleton
        self.nbjoints = int(skeleton.nbjoints)
        self.n_foot = int(n_foot)
        lo = self.nbjoints * 9
        self.motion_rep_dim = lo + 9 + self.n_foot
        self.fps = int(fps)
        self.slice_dict: Dict[str, slice] = OrderedDict(
            [
                ("local_pose", slice(0, lo)),
                ("canon_delta", slice(lo, lo + 9)),
                ("foot_contacts", slice(lo + 9, self.motion_rep_dim)),
            ]
        )
        self.feature_names: List[str] = list(self.slice_dict.keys())
        self.feat_bias = float(feat_bias)

        mean = torch.from_numpy(np.load(str(mean_path))).float()
        std = torch.from_numpy(np.load(str(std_path))).float()
        if mean.shape[-1] != self.motion_rep_dim or std.shape[-1] != self.motion_rep_dim:
            raise ValueError(
                f"mean/std must be shape ({self.motion_rep_dim},) for a "
                f"{self.nbjoints}-joint skeleton; got {tuple(mean.shape)} / {tuple(std.shape)}"
            )

        # feat_bias rescaling (MDM/MoMask trick) — up-weights the canonical-frame
        # trajectory + foot-contact blocks (analog of native's root/contact
        # blocks). Default 1.0 = off, matching the native data path which
        # normalizes with raw mean/std.
        if self.feat_bias != 1.0:
            std = std.clone()
            for name in ("canon_delta", "foot_contacts"):
                std[self.slice_dict[name]] = std[self.slice_dict[name]] / self.feat_bias

        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self._eps = float(eps)

    def _safe_std(self) -> torch.Tensor:
        return torch.sqrt(self.std * self.std + self._eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(device=x.device, dtype=x.dtype)
        s = self._safe_std().to(device=x.device, dtype=x.dtype)
        return (x - m) / s

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(device=x.device, dtype=x.dtype)
        s = self._safe_std().to(device=x.device, dtype=x.dtype)
        return x * s + m
