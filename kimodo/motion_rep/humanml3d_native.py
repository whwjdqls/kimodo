# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HumanML3D-native (263-D) motion representation.

This is the minimal motion-rep wrapper used by the MDM-style one-stage
training script when feeding HumanML3D's native ``.npy`` files (as opposed
to the kimodo 273-D dicts in ``kimodo_rep/``). The 263-D layout is:

    idx          name                width  description
    ---          ----                -----  -----------
    [0,  1)      rot_velocity          1    per-frame yaw delta Δα
    [1,  3)      lin_velocity          2    root planar velocity in ego frame
    [3,  4)      root_height           1    root world y
    [4,  67)     ric_data             63    joints 1..21 positions in ego frame
                                            (root-relative XZ, world Y)
    [67, 193)    rot_data            126    joints 1..21 local 6D rotations
    [193, 259)   local_velocity       66    joints 0..21 velocities, ego frame
    [259, 263)   foot_contacts         4    binary contact flags

Two distinguishing properties vs the kimodo 273-D rep:

  * The root rotation lives in ``rot_velocity`` only (Δα). Joint-0 has no
    slot in ``rot_data`` (so ``rot_data`` is 21 joints, not 22).
  * Joint positions are stored ego-relative (rotated into the current root
    frame) plus a separate root height — there is no ``smooth_root_pos``.

For the FK consistency loss we re-run HumanML3D's standard parent-relative
``forward_kinematics_cont6d`` on the predicted 6D rotations against the
canonical SMPL-22 bone offsets (from motion ``012314`` frame 0, the same
template every HumanML3D motion is uniform-skeleton'd to during
preprocessing). Compared to the kimodo ``chainreset_hml3d`` FK, this one is
PARENT-RELATIVE and uses HML3D's standard chain. They are NOT
interchangeable.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


# HumanML3D's 22-joint SMPL kinematic chain. Identical to
# ``kimodo.motion_rep.fk_hml3d.HML3D_KINEMATIC_CHAIN`` (intentional — same
# topology, only the rotation convention differs between FK kinds).
HML3D_KINEMATIC_CHAIN: List[List[int]] = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

# HumanML3D's canonical per-joint unit-direction offsets (from
# ``paramUtil.t2m_raw_offsets``). The actual bone offset used in FK is
# ``raw_offset[j] * bone_length[j]`` where bone_length is derived from the
# T-pose joint positions of motion ``012314``. Same array as in
# ``kimodo.motion_rep.fk_hml3d.HML3D_RAW_OFFSETS``.
HML3D_RAW_OFFSETS: torch.Tensor = torch.tensor(
    [
        [ 0,  0,  0],   # 0  pelvis
        [ 1,  0,  0],   # 1  left_hip
        [-1,  0,  0],   # 2  right_hip
        [ 0,  1,  0],   # 3  spine1
        [ 0, -1,  0],   # 4  left_knee
        [ 0, -1,  0],   # 5  right_knee
        [ 0,  1,  0],   # 6  spine2
        [ 0, -1,  0],   # 7  left_ankle
        [ 0, -1,  0],   # 8  right_ankle
        [ 0,  1,  0],   # 9  spine3
        [ 0,  0,  1],   # 10 left_foot
        [ 0,  0,  1],   # 11 right_foot
        [ 0,  1,  0],   # 12 neck
        [ 1,  0,  0],   # 13 left_collar
        [-1,  0,  0],   # 14 right_collar
        [ 0,  0,  1],   # 15 head
        [ 0, -1,  0],   # 16 left_shoulder
        [ 0, -1,  0],   # 17 right_shoulder
        [ 0, -1,  0],   # 18 left_elbow
        [ 0, -1,  0],   # 19 right_elbow
        [ 0, -1,  0],   # 20 left_wrist
        [ 0, -1,  0],   # 21 right_wrist
    ],
    dtype=torch.float32,
)

FEAT_DIM = 263
N_JOINTS = 22


def _feature_layout() -> Dict[str, slice]:
    return OrderedDict([
        ("rot_velocity",    slice(0,   1)),
        ("lin_velocity",    slice(1,   3)),
        ("root_height",     slice(3,   4)),
        ("ric_data",        slice(4,   4 + (N_JOINTS - 1) * 3)),       # [4, 67)
        ("rot_data",        slice(4 + (N_JOINTS - 1) * 3,
                                  4 + (N_JOINTS - 1) * 3 + (N_JOINTS - 1) * 6)),  # [67, 193)
        ("local_velocity",  slice(4 + (N_JOINTS - 1) * 3 + (N_JOINTS - 1) * 6,
                                  4 + (N_JOINTS - 1) * 3 + (N_JOINTS - 1) * 6 + N_JOINTS * 3)),  # [193, 259)
        ("foot_contacts",   slice(FEAT_DIM - 4, FEAT_DIM)),
    ])


# ----------------------------------------------------------------------------
# Canonical bone offsets (loaded from any HumanML3D world-joint .npy).
# ----------------------------------------------------------------------------
def derive_canonical_offsets(
    canonical_joints_path: str | Path,
    chains: Sequence[Sequence[int]] = HML3D_KINEMATIC_CHAIN,
    raw_offsets: torch.Tensor = HML3D_RAW_OFFSETS,
) -> torch.Tensor:
    """Return canonical bone offsets ``(J, 3)`` matching HumanML3D's
    ``Skeleton.get_offsets_joints``: per-joint **unit raw_offset direction**
    times per-joint **bone length** measured from the template motion's
    T-pose. The canonical template is ``new_joints/012314.npy`` (every
    HumanML3D motion is uniform-skeleton'd to this actor during preprocessing).

    The offsets are NOT the literal (child-minus-parent) T-pose vectors —
    those would be ``parents.norm(dim=-1) * raw_offset_actual_direction``,
    which only equals ``raw_offset * bone_length`` if the actual T-pose
    direction matches the canonical raw_offset direction. For HumanML3D
    motion 012314 the two agree to numerical precision because uniform
    preprocessing already aligns the skeleton to ``t2m_raw_offsets``.
    """
    joints = np.load(str(canonical_joints_path))  # (T, J, 3)
    if joints.ndim != 3:
        raise ValueError(
            f"expected (T, J, 3) joints; got {joints.shape} from {canonical_joints_path}"
        )
    j0 = torch.from_numpy(joints[0]).float()  # (J, 3)
    J = j0.shape[0]
    raw = raw_offsets.float()
    offsets = torch.zeros(J, 3, dtype=torch.float32)
    for chain in chains:
        for k in range(1, len(chain)):
            child, parent = int(chain[k]), int(chain[k - 1])
            bone_len = (j0[child] - j0[parent]).norm()
            offsets[child] = raw[child] * bone_len
    return offsets


# ----------------------------------------------------------------------------
# Quaternion helpers — mirror HumanML3D's ``common/quaternion.py`` exactly
# so we reproduce ``recover_from_ric`` and ``recover_root_rot_pos`` to ≤1e-5 m.
# Convention: q = (w, x, y, z), unit-length; v = (x, y, z); v' = q * v * q^-1.
# HumanML3D's r_rot_quat uses (cos α, 0, sin α, 0) — α IS the FULL rotation
# angle (not half), because both r_rot_quat is unit norm and qrot's algebra
# yields rotation by α via that encoding (verified against new_joints/012314).
# ----------------------------------------------------------------------------
def _qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` (..., 3) by unit quaternion ``q`` (..., 4)."""
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    """Inverse (conjugate) of a unit quaternion."""
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


# ----------------------------------------------------------------------------
# 6D -> matrix (matches kimodo.geometry.cont6d_to_matrix; duplicated here so
# the loss can call it without import cycles).
# ----------------------------------------------------------------------------
def _cont6d_to_matrix(cont6d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / (torch.norm(x_raw, dim=-1, keepdim=True) + eps)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / (torch.norm(z, dim=-1, keepdim=True) + eps)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)  # (..., 3, 3)


# ----------------------------------------------------------------------------
# Standard parent-relative FK (HumanML3D convention, batched).
# ----------------------------------------------------------------------------
def hml3d_native_fk_world_joints(
    rot_data_6d_no_root: torch.Tensor,   # (B, T, J-1, 6) local 6D rotations (joints 1..21)
    root_rot_velocity: torch.Tensor,     # (B, T)  per-frame yaw Δα (rot_velocity block)
    lin_velocity: torch.Tensor,          # (B, T, 2) ego-frame planar velocity (lin_velocity block)
    root_height: torch.Tensor,           # (B, T)  root world y (root_height block)
    canonical_offsets: torch.Tensor,     # (J, 3)  template bone offsets from 012314
    chains: Sequence[Sequence[int]] = HML3D_KINEMATIC_CHAIN,
) -> torch.Tensor:
    """Run HumanML3D's parent-relative FK on the 263-D rotation block.

    Recovers the joint-0 (root) rotation from integrated ``rot_velocity``,
    the root world position from integrated ``lin_velocity`` + ``root_height``,
    then walks each kinematic chain accumulating
    ``parent_world = ... ; child_world = parent_world + R_world(child) @ offset(child)``.

    Returns ``(B, T, J, 3)`` world joint positions.
    """
    B, T, _, _ = rot_data_6d_no_root.shape
    J = canonical_offsets.shape[0]
    device, dtype = rot_data_6d_no_root.device, rot_data_6d_no_root.dtype

    # 1) Recover per-frame root quaternion (cos α, 0, sin α, 0), then root
    #    world position — same algorithm as recover_root_rot_pos.
    alpha = torch.zeros(B, T, device=device, dtype=dtype)
    if T > 1:
        alpha[:, 1:] = torch.cumsum(root_rot_velocity[:, :-1], dim=1)
    r_rot_quat = torch.zeros(B, T, 4, device=device, dtype=dtype)
    r_rot_quat[..., 0] = torch.cos(alpha)
    r_rot_quat[..., 2] = torch.sin(alpha)

    r_pos = torch.zeros(B, T, 3, device=device, dtype=dtype)
    if T > 1:
        r_pos[:, 1:, 0] = lin_velocity[:, :-1, 0]
        r_pos[:, 1:, 2] = lin_velocity[:, :-1, 1]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=1)
    r_pos[..., 1] = root_height
    root_xyz = r_pos

    # 2) Root rotation as a matrix. Joint-0 local rotation = R_root.
    #
    # HumanML3D encodes the root quaternion as q = (cos α, 0, sin α, 0) where
    # α is the cumulative ``rot_velocity``. Crucially, this is NOT a standard
    # half-angle quaternion encoding: ``qrot(q, v)`` produces a rotation by
    # **2α** around Y (verified by expanding the cross-product formula). So
    # the joint-0 rotation matrix matches a rotation by 2α (not α).
    two_a = 2.0 * alpha
    cos_2a = torch.cos(two_a)
    sin_2a = torch.sin(two_a)
    R_root = torch.zeros(B, T, 3, 3, device=device, dtype=dtype)
    R_root[..., 0, 0] = cos_2a
    R_root[..., 0, 2] = sin_2a
    R_root[..., 1, 1] = 1.0
    R_root[..., 2, 0] = -sin_2a
    R_root[..., 2, 2] = cos_2a

    # 3) Stack: joint-0 local rotation = R_root, joints 1..21 from rot_data.
    local_mats = torch.zeros(B, T, J, 3, 3, device=device, dtype=dtype)
    local_mats[:, :, 0] = R_root
    local_mats[:, :, 1:] = _cont6d_to_matrix(rot_data_6d_no_root)

    # 4) Walk the kinematic chains. HumanML3D's convention (verified against
    #    motion 012314): start each chain with matR = R_root, then for each
    #    joint along the chain do matR = matR @ R_local[chain[i]] AND apply
    #    matR (the now-updated accumulated rotation) to the child offset.
    #    Note this differs from standard parent-relative FK: we use the
    #    child's own accumulated rotation (not the parent's) on the offset.
    offsets = canonical_offsets.to(device=device, dtype=dtype)  # (J, 3)
    world_pos = torch.zeros(B, T, J, 3, device=device, dtype=dtype)
    world_pos[:, :, 0] = root_xyz

    for chain in chains:
        matR = local_mats[:, :, 0].clone()  # (B, T, 3, 3) — start with R_root each chain
        for k in range(1, len(chain)):
            parent_idx = int(chain[k - 1])
            child_idx = int(chain[k])
            matR = torch.einsum("btij,btjk->btik", matR, local_mats[:, :, child_idx])
            offset = offsets[child_idx]  # (3,)
            rotated_offset = torch.einsum("btij,j->bti", matR, offset)
            world_pos[:, :, child_idx] = world_pos[:, :, parent_idx] + rotated_offset

    return world_pos


# ----------------------------------------------------------------------------
# Inverse of the position block: 263-D -> world joints (no rotations needed).
# Mirrors HumanML3D's ``recover_from_ric``.
# ----------------------------------------------------------------------------
def hml3d_native_world_joints_from_features(
    features_unnormalized: torch.Tensor,   # (B, T, 263) or (T, 263)
    n_joints: int = N_JOINTS,
) -> torch.Tensor:
    """De-egoize HumanML3D's ric_data block + integrate the root state to get
    world joint positions ``(B, T, J, 3)``. Pure position-side recovery; the
    rotation block is not used.

    This matches HumanML3D's ``recover_from_ric`` exactly (algorithmically).
    """
    squeeze = False
    if features_unnormalized.dim() == 2:
        features_unnormalized = features_unnormalized.unsqueeze(0)
        squeeze = True
    B, T, D = features_unnormalized.shape
    if D != FEAT_DIM:
        raise ValueError(f"expected last dim {FEAT_DIM}, got {D}")
    device, dtype = features_unnormalized.device, features_unnormalized.dtype
    sd = _feature_layout()

    rot_vel = features_unnormalized[..., sd["rot_velocity"]].squeeze(-1)   # (B, T)
    lin_vel = features_unnormalized[..., sd["lin_velocity"]]               # (B, T, 2)
    root_y = features_unnormalized[..., sd["root_height"]].squeeze(-1)     # (B, T)
    ric = features_unnormalized[..., sd["ric_data"]]                       # (B, T, (J-1)*3)
    ric = ric.reshape(B, T, n_joints - 1, 3)

    # 1) Recover the per-frame root yaw quaternion (HumanML3D convention:
    #    q = (cos α, 0, sin α, 0), α = cumulative rot_velocity shifted by 1).
    alpha = torch.zeros(B, T, device=device, dtype=dtype)
    if T > 1:
        alpha[:, 1:] = torch.cumsum(rot_vel[:, :-1], dim=1)
    r_rot_quat = torch.zeros(B, T, 4, device=device, dtype=dtype)
    r_rot_quat[..., 0] = torch.cos(alpha)
    r_rot_quat[..., 2] = torch.sin(alpha)

    # 2) Integrate root world position from ego-frame lin_velocity. HumanML3D's
    #    flow: stuff lin_vel into r_pos[..., 1:, [0, 2]], apply qrot(qinv(q), .),
    #    cumsum, then override y with root_height.
    r_pos = torch.zeros(B, T, 3, device=device, dtype=dtype)
    if T > 1:
        r_pos[:, 1:, 0] = lin_vel[:, :-1, 0]
        r_pos[:, 1:, 2] = lin_vel[:, :-1, 1]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=1)
    r_pos[..., 1] = root_y

    # 3) De-egoize ric_data: same qrot(qinv(q), .), then add root XZ.
    ric_world = _qrot(_qinv(r_rot_quat)[..., None, :].expand(B, T, n_joints - 1, 4), ric)
    ric_world[..., 0] = ric_world[..., 0] + r_pos[..., 0:1]
    ric_world[..., 2] = ric_world[..., 2] + r_pos[..., 2:3]

    # 4) Concat root + non-root joints.
    world_joints = torch.cat([r_pos.unsqueeze(2), ric_world], dim=2)  # (B, T, J, 3)

    return world_joints.squeeze(0) if squeeze else world_joints


# ----------------------------------------------------------------------------
# The motion-rep wrapper.
# ----------------------------------------------------------------------------
class HumanML3DNativeMotionRep(nn.Module):
    """Minimal motion-rep for HumanML3D's 263-D representation.

    Exposes the attributes the one-stage denoiser + KimodoLoss need:
    ``motion_rep_dim``, ``nbjoints``, ``fps``, ``slice_dict``, ``skeleton``,
    ``normalize``, ``unnormalize``. Mean/std are HumanML3D's standard
    ``Mean.npy`` / ``Std.npy`` (the same ones MDM/T2M/MoMask train against).
    """

    motion_rep_dim: int = FEAT_DIM

    def __init__(
        self,
        mean_path: str | Path,
        std_path: str | Path,
        skeleton,
        canonical_joints_path: Optional[str | Path] = None,
        fps: int = 20,
        eps: float = 1e-6,
        feat_bias: float = 5.0,
        # ``stats_path`` is unused here (we read mean/std from absolute paths
        # rather than a split-stats subfolder layout), but the upstream helper
        # ``build_denoiser_from_model_config`` always injects this key. We
        # accept and silently ignore it so the Hydra wiring stays simple.
        stats_path: Optional[str | Path] = None,
    ):
        super().__init__()
        self.skeleton = skeleton
        self.nbjoints = int(skeleton.nbjoints)
        if self.nbjoints != N_JOINTS:
            raise ValueError(
                f"HumanML3DNativeMotionRep requires a 22-joint skeleton; got {self.nbjoints}"
            )
        self.fps = int(fps)
        self.slice_dict: Dict[str, slice] = _feature_layout()
        self.feature_names: List[str] = list(self.slice_dict.keys())
        self.feat_bias = float(feat_bias)

        mean = torch.from_numpy(np.load(str(mean_path))).float()
        std = torch.from_numpy(np.load(str(std_path))).float()
        if mean.shape[-1] != FEAT_DIM or std.shape[-1] != FEAT_DIM:
            raise ValueError(
                f"mean/std must be shape ({FEAT_DIM},); got {tuple(mean.shape)} / {tuple(std.shape)}"
            )

        # MDM/MoMask-style feat_bias rescaling on std (mean untouched). Smaller
        # std on the root/contact blocks up-weights them in the normalized
        # output, so the diffusion loss attends to them more.
        # Source: motion-diffusion-model/data_loaders/humanml/data/dataset.py:95-113
        # and momask-codes/data/t2m_dataset.py:42-60 — both divide
        # rot_velocity / lin_velocity / root_y / foot_contacts by feat_bias=5
        # and leave ric_data / rot_data / local_velocity at /1.0.
        if self.feat_bias != 1.0:
            std = std.clone()
            for name in ("rot_velocity", "lin_velocity", "root_height", "foot_contacts"):
                std[self.slice_dict[name]] = std[self.slice_dict[name]] / self.feat_bias

        # Persistent buffers — they ride along with .to(device) and ckpt save.
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self._eps = float(eps)

        # Canonical bone offsets for the FK loss. Optional at construction
        # time — the loss will derive them lazily if not pre-loaded.
        if canonical_joints_path is not None:
            off = derive_canonical_offsets(canonical_joints_path)
            self.register_buffer("canonical_offsets", off, persistent=False)
        else:
            self.canonical_offsets = None  # type: ignore[assignment]

    # --- normalization (z-score with sqrt(std^2 + eps), matching kimodo.Stats) ---
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
