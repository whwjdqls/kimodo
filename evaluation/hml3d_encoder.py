# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical HumanML3D encoder applied to world joints.

This is a faithful re-implementation of the ``process_file`` function in
``/home/jungbin_cho/HumanML3D/motion_representation.ipynb`` — the same
encoder HumanML3D used to produce its 263-D feature files from raw SMPL-X
joints. We expose it as a module-level function so the eval pipeline can
rebuild HumanML3D 263 from FK-decoded joint positions in *exactly* the same
way the GT dataset was built.

Flow inside ``encode_joints_to_hml3d_263``:

  1) uniform_skeleton (optional) — rescale source bones to the canonical
     target skeleton derived from motion 012314 frame 0.
  2) Put on floor: subtract min y so feet rest on y=0.
  3) XZ at origin: subtract frame-0 root xz.
  4) Face Z+: rotate so frame-0 character faces +z.
  5) Foot contact detection from per-frame squared joint position deltas.
  6) Inverse kinematics → cont6d_params, r_velocity (angular), velocity
     (linear in ego frame), r_rot (root quaternion).
  7) get_rifke: subtract root xz from each joint, then rotate by r_rot.
  8) Pack: [r_velocity, l_velocity, root_y, ric_data, rot_data, local_vel,
           feet_l, feet_r] → (T-1, 263).

For our use case the input joints already come from a canonicalized data
distribution (the model was trained on HumanML3D-canonicalized motion), so
the canonicalization steps (1-4) are approximately idempotent. We still
apply them to match what the GT preprocessing did exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Shim for HumanML3D's older numpy API (np.float / np.int) — same as eval_hml3d.py.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

# Make HumanML3D's helpers importable.
_HML3D_ROOT = Path("/home/jungbin_cho/HumanML3D")
if str(_HML3D_ROOT) not in sys.path:
    sys.path.insert(0, str(_HML3D_ROOT))

from common.skeleton import Skeleton  # noqa: E402
from common.quaternion import (  # noqa: E402
    qbetween_np,
    qfix,
    qinv_np,
    qmul_np,
    qrot_np,
    quaternion_to_cont6d_np,
)
from paramUtil import t2m_kinematic_chain, t2m_raw_offsets  # noqa: E402


# ----- Constants from HML3D's process_file (SMPL-22 layout) -----
FACE_JOINT_IDX = [2, 1, 17, 16]   # r_hip, l_hip, sdr_r, sdr_l
L_IDX1, L_IDX2 = 5, 8             # lower leg indices for the uniform_skeleton scale ratio
FID_R = [8, 11]                   # right foot joints (ankle, foot)
FID_L = [7, 10]                   # left foot joints
FEET_THRES = 0.002                # foot velocity² threshold
JOINTS_NUM = 22

# Lazy: target offsets are derived once from the canonical example motion.
_TGT_OFFSETS: Optional[np.ndarray] = None
_RAW_OFFSETS_T = torch.from_numpy(t2m_raw_offsets).float()


def _get_target_offsets() -> np.ndarray:
    """Bone offsets of the canonical target skeleton (motion 012314 frame 0
    after HumanML3D's preprocessing). These are the offsets every motion is
    rescaled to in step (1) of ``process_file``.
    """
    global _TGT_OFFSETS
    if _TGT_OFFSETS is not None:
        return _TGT_OFFSETS
    # Use the canonical PROCESSED motion 012314 as the target skeleton source.
    # Its frame 0 (after uniform_skeleton was applied) defines the canonical
    # bone lengths used across HumanML3D.
    canonical = np.load(
        "/home/jungbin_cho/HumanML3D/HumanML3D/new_joints/012314.npy"
    )  # (T, 22, 3)
    skel = Skeleton(_RAW_OFFSETS_T, t2m_kinematic_chain, "cpu")
    _TGT_OFFSETS = skel.get_offsets_joints(torch.from_numpy(canonical[0]).float()).numpy()
    return _TGT_OFFSETS


def _uniform_skeleton(positions: np.ndarray, target_offset: np.ndarray) -> np.ndarray:
    """Rescale ``positions`` to match the canonical bone lengths via
    leg-length ratio + IK + FK on the target offsets. Mirrors
    ``uniform_skeleton`` in the HML3D notebook.
    """
    skel = Skeleton(_RAW_OFFSETS_T, t2m_kinematic_chain, "cpu")
    src_offset = skel.get_offsets_joints(torch.from_numpy(positions[0]).float()).numpy()
    src_leg_len = np.abs(src_offset[L_IDX1]).max() + np.abs(src_offset[L_IDX2]).max()
    tgt_leg_len = np.abs(target_offset[L_IDX1]).max() + np.abs(target_offset[L_IDX2]).max()
    scale_rt = tgt_leg_len / src_leg_len
    tgt_root_pos = positions[:, 0] * scale_rt
    quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_IDX)
    skel.set_offset(torch.from_numpy(target_offset).float())
    new_joints = skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def _foot_detect(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binary foot contacts from per-frame squared position deltas (HML3D)."""
    velfactor = np.array([FEET_THRES, FEET_THRES])
    feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
    feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
    feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)
    feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
    feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
    feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
    return feet_l, feet_r


def encode_joints_to_hml3d_263(
    joints: np.ndarray | torch.Tensor,
    apply_uniform_skeleton: bool = True,
    apply_canonicalize: bool = True,
) -> np.ndarray:
    """Re-encode world joint positions to HumanML3D 263 — the exact path
    ``process_file`` in motion_representation.ipynb takes from raw SMPL-X
    joints to a 263-D feature vector.

    Args:
        joints: ``(T, 22, 3)`` world joint positions. Either np.ndarray or
            torch.Tensor (will be moved to CPU and converted).
        apply_uniform_skeleton: If True, rescale bones to the canonical target
            skeleton (HML3D's step 1). On FK-decoded joints from a kimodo
            model this is approximately idempotent; we still apply by default
            for byte-equivalence with the GT encoder.
        apply_canonicalize: If True, apply HML3D's frame-0 canonicalization
            (floor at y=0, root xz at origin, face +z). True by default for
            the same reason.

    Returns:
        ``(T-1, 263)`` float32 numpy array — the canonical HumanML3D feature
        layout. Note this is T-1 frames (the encoder consumes one trailing
        frame to compute deltas); the eval pipeline pads the last frame to
        keep length T.
    """
    if isinstance(joints, torch.Tensor):
        positions = joints.detach().cpu().numpy().astype(np.float32)
    else:
        positions = np.asarray(joints, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1] != JOINTS_NUM or positions.shape[2] != 3:
        raise ValueError(
            f"expected joints shape (T, {JOINTS_NUM}, 3); got {positions.shape}"
        )
    positions = positions.copy()

    # 1) Uniform skeleton (optional).
    if apply_uniform_skeleton:
        tgt_offsets = _get_target_offsets()
        positions = _uniform_skeleton(positions, tgt_offsets)

    # 2-4) Floor + XZ origin + face +Z (frame-0 canonicalization).
    if apply_canonicalize:
        # Put on floor.
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height
        # XZ at origin (subtract frame-0 root xz).
        root_pos_init = positions[0].copy()
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1], dtype=np.float32)
        positions = positions - root_pose_init_xz
        # All initially face Z+.
        r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_IDX
        across = (root_pos_init[r_hip] - root_pos_init[l_hip]) + (
            root_pos_init[sdr_r] - root_pos_init[sdr_l]
        )
        across = across / (np.linalg.norm(across, axis=-1, keepdims=True) + 1e-12)
        # Forward (3,) by cross(up, across) — rotate around y to align with +z.
        forward_init = np.cross(np.array([[0, 1, 0]], dtype=np.float32), across[None], axis=-1)
        forward_init = forward_init / (np.linalg.norm(forward_init, axis=-1, keepdims=True) + 1e-12)
        target = np.array([[0, 0, 1]], dtype=np.float32)
        root_quat_init = qbetween_np(forward_init, target)  # (1, 4)
        root_quat_init_full = np.ones(positions.shape[:-1] + (4,), dtype=np.float32) * root_quat_init
        positions = qrot_np(root_quat_init_full, positions)

    # "global_positions" is the canonicalized world joints — used for local_vel.
    global_positions = positions.copy()

    # 5) Foot contacts.
    feet_l, feet_r = _foot_detect(positions)

    # 6) IK → cont6d_params, r_velocity, velocity, r_rot.
    skel = Skeleton(_RAW_OFFSETS_T, t2m_kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_IDX, smooth_forward=True)
    cont_6d_params = quaternion_to_cont6d_np(quat_params)        # (T, J, 6)
    r_rot = quat_params[:, 0].copy()                             # (T, 4)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()      # (T-1, 3) world
    velocity = qrot_np(r_rot[1:], velocity)                       # to ego of frame t+1
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))          # (T-1, 4)

    # 7) get_rifke: subtract root xz then rotate by r_rot per frame.
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)

    # 8) Pack.
    r_velocity = np.arcsin(r_velocity[:, 2:3])                     # (T-1, 1) y-axis rotation
    l_velocity = velocity[:, [0, 2]]                                # (T-1, 2)
    root_y = positions[:, 0, 1:2]                                   # (T, 1)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)  # (T-1, 4)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)            # (T, 126)
    ric_data = positions[:, 1:].reshape(len(positions), -1)                       # (T, 63)

    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )                                                                              # (T-1, J, 3)
    local_vel = local_vel.reshape(len(local_vel), -1)                              # (T-1, 66)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    return data.astype(np.float32)
