
"""
HumanML3D <-> Kimodo-style (global-root) motion representation, with exact
round-trip conversion.

============================================================
What this script does
============================================================
HumanML3D encodes root motion as DELTAS in the character's egocentric
(yaw-aligned) frame. We convert that to a Kimodo-style "global-root"
representation where root position and heading are expressed in WORLD
coordinates, plus joint rotations/positions/velocities also in world frame.
Other channels (joint rotations stored as 6D, joint positions, joint
velocities, foot contacts) carry the same information; we only switch their
reference frame.

Two functions are provided:
  humanml3d_to_kimodo(data) -> dict   (forward)
  kimodo_to_humanml3d(dict) -> data   (inverse)
A `--verify-roundtrip N` CLI flag re-runs forward then inverse on the first
N files and reports the max element-wise difference vs the original 263-D
vector. It should be ~1e-6 (single-precision float noise).

============================================================
HumanML3D 263-D layout (J=22 SMPL joints, per frame)
============================================================
  data[..., 0      ]  root_rot_velocity     Δ-half-heading angle per frame.
                                            HumanML3D builds the heading
                                            quaternion as the standard
                                            half-angle quat
                                            (cos α, 0, sin α, 0) where
                                            α = cumsum(root_rot_velocity)
                                            with one frame of zero padding
                                            at the start. So
                                            root_rot_velocity[t] = α[t+1] - α[t].
  data[..., 1:3    ]  root_linear_velocity  (vx, vz) in the EGOCENTRIC
                                            (yaw-aligned) frame AT THAT
                                            FRAME. data[..., -1, 1:3] is
                                            never used in recovery (it would
                                            describe a non-existent frame
                                            T+1) -- we set it to 0 on the
                                            inverse path.
  data[..., 3      ]  root_height           world-frame y of the root joint.
  data[..., 4:67   ]  ric_data              (21,3) joint positions for joints
                                            1..21, expressed in the EGOCENTRIC
                                            frame with root xz subtracted.
                                            (root y NOT subtracted from y --
                                            see recover_from_ric below.)
  data[..., 67:193 ]  rot_data              (21,6) HumanML3D "local" rotations
                                            for joints 1..21. NOTE: these are
                                            NOT standard parent-relative
                                            rotations. They are stored such
                                            that HumanML3D's chain-reset FK
                                            (which resets matR to the root
                                            rotation at the start of every
                                            chain in the kinematic tree, even
                                            sub-chains like the arms which
                                            start at joint 9) produces correct
                                            world positions. See
                                            _compose_global_rotations_chainreset
                                            for the exact semantics.
                                            Root rotation is NOT here -- it is
                                            recoverable from root_rot_velocity.
  data[..., 193:259]  local_velocity        (22,3) joint velocities in the
                                            EGOCENTRIC frame at frame t.
                                            data[..., -1, 193:259] is the same
                                            artifact as last-frame lin_vel --
                                            never used in recovery, set to 0
                                            on inverse path. (HumanML3D
                                            actually computes this from
                                            position differences shifted by
                                            one frame, so it's actually
                                            available for all T frames in the
                                            source data; the recovery doesn't
                                            consume the last one is the point.)
  data[..., 259:263]  foot_contact          4 binary contact flags.

============================================================
Kimodo-style output layout (J=22, "KimodoMotionRep" pack order)
============================================================
The forward function returns a dict of these tensors, each (T, ...).

  smooth_root_pos        (3)        WORLD smoothed root xyz (Kimodo applies a
                                    causal smoother to the raw root xz; y is
                                    treated separately by the smoother).
  global_root_heading    (2)        [cos θ, sin θ] where θ is the actual
                                    world heading angle = 2α. We extract α
                                    from the recovered r_rot_quat and double
                                    it -- this matches Kimodo's
                                    [cos heading, sin heading] convention.
                                    (Note: when we round-trip we re-derive α
                                    by extracting the Y-axis rotation angle
                                    from global_rot_mats[..., 0], so we
                                    don't actually need to store θ here for
                                    round-trip correctness -- but it's the
                                    field a Kimodo training pipeline wants.)
  local_joints_positions (J, 3)     world joint xyz minus (smooth_root_x, 0,
                                    smooth_root_z). Matches Kimodo encode.
  global_rot_data        (J, 6)     6D rotation per joint computed via the
                                    chain-reset FK described above; these are
                                    HumanML3D-compatible "global rotations"
                                    suitable for round-trip recovery.
  velocities             (J, 3)     world-frame joint velocities. We get them
                                    by rotating HumanML3D's local_velocity
                                    out of the egocentric frame using
                                    qinv(r_rot_quat) -- this is the EXACT
                                    inverse of HumanML3D's processing pass,
                                    so round trip is lossless.
  foot_contacts          (4)        same as HumanML3D.

Plus, for convenience / sanity:
  posed_joints           (J, 3)     world joint positions from recover_from_ric.
  root_positions         (3)        world joint-0 position (== posed_joints[:,0])
  features               (T, 273)   packed tensor in Kimodo order (above).

Packed feature dim total = 3 + 2 + 22*3 + 22*6 + 22*3 + 4 = 273.

============================================================
Round-trip path
============================================================
forward (humanml3d -> kimodo):
  recover_from_ric            -> world_joints (T, 22, 3)
  recover_root_rot_pos        -> r_rot_quat (T, 4), r_pos (T, 3)
  build all_local_6d by concatenating quaternion_to_cont6d(r_rot_quat) with
  rot_data (T, 21, 6); chain-reset compose to global rotations.
  velocities = qrot(qinv(r_rot_quat_expanded), local_velocity)  -- to world.
  ric_data is implicitly captured by world_joints.

inverse (kimodo -> humanml3d):
  global_rot_mats[..., 0]    -> root rotation matrix per frame
                              -> extract α (half-angle Y rotation)
                              -> root_rot_velocity = Δα with last-frame zero
                              -> rebuild r_rot_quat
  Find world joint positions (we keep `posed_joints` for this), subtract
  root xz, rotate by r_rot_quat -> ric_data (21,3 per frame).
  Decompose global_rot_mats via reverse chain-reset to recover HumanML3D
  local rot matrices for joints 1..21; convert to 6D -> rot_data.
  Egocentric root velocity: world root delta -> rotate by r_rot_quat[t]
  -> data[t, 1:3]. Last frame unused, set to 0.
  Egocentric joint velocity: world velocities -> qrot(r_rot_quat_expanded,
  ...) -> local_velocity.
  Foot contacts: passthrough.
"""

from __future__ import annotations

# ---- Shims that must run BEFORE importing HumanML3D's old code -------------
import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch
from tqdm import tqdm

# Make HumanML3D modules importable.
_HML3D_ROOT = Path("/home/jungbin_cho/HumanML3D")
if str(_HML3D_ROOT) not in sys.path:
    sys.path.insert(0, str(_HML3D_ROOT))

import paramUtil  # type: ignore  # noqa: E402  # SMPL chain + raw offsets
from common.quaternion import (  # type: ignore  # noqa: E402
    qinv,
    qmul,
    qrot,
    quaternion_to_cont6d,
    quaternion_to_matrix,
)

# Kimodo's smooth-root smoother (the only Kimodo helper this script needs).
from kimodo.motion_rep.smooth_root import get_smooth_root_pos  # noqa: E402

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

NUM_JOINTS = 22  # SMPL body joints used by HumanML3D
FEAT_DIM_HML3D = 263

# 22-joint SMPL kinematic chain from HumanML3D's paramUtil.
# IMPORTANT: chains 4 ([9, 14, 17, 19, 21]) and 5 ([9, 13, 16, 18, 20]) start
# at joint 9, not at the global root. HumanML3D's FK RESETS its running matR
# to the root rotation at the start of every chain, so the stored local
# rotations for joints 14 and 13 are "root-relative" rather than
# "parent-relative". `_compose_global_rotations_chainreset` mirrors this so
# round-tripping is exact.
KINEMATIC_CHAIN: list[list[int]] = paramUtil.t2m_kinematic_chain


# ----------------------------------------------------------------------------
# HumanML3D recovery (local copy of the relevant inverses, so we don't depend
# on opening the notebook each time).
# ----------------------------------------------------------------------------


def recover_root_rot_pos(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mirror of HumanML3D's `recover_root_rot_pos`.

    Returns
    -------
    r_rot_quat : (..., T, 4)
        Standard half-angle quaternion for the per-frame Y-axis heading
        rotation, where α = cumsum(root_rot_velocity) shifted by 1.
    r_pos      : (..., T, 3)
        World-frame root joint position; y comes from data[..., 3], xz is
        cumsum of egocentric velocities rotated back to world via
        qinv(r_rot_quat).
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data: torch.Tensor, joints_num: int = NUM_JOINTS) -> torch.Tensor:
    """Mirror of HumanML3D's positions-branch FK. Returns world (T, J, 3)."""
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions,
    )
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


# ----------------------------------------------------------------------------
# 6D <-> matrix utilities. We use the SAME formulas HumanML3D uses, both for
# `cont6d_to_matrix` and the inverse direction, so conversions are exact.
# ----------------------------------------------------------------------------


def cont6d_to_matrix(cont6d: torch.Tensor) -> torch.Tensor:
    """
    (..., 6) -> (..., 3, 3). Columns of the output are basis vectors
    [x, y, z], same convention as HumanML3D's quaternion_to_cont6d which
    stores rotation_mat[..., 0] and rotation_mat[..., 1].
    """
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / x_raw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def matrix_to_cont6d(mat: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6), same as HumanML3D's quaternion_to_cont6d format
    (first two columns concatenated)."""
    return torch.cat([mat[..., 0], mat[..., 1]], dim=-1)


# ----------------------------------------------------------------------------
# Global rotation composition / decomposition mirroring HumanML3D's FK
# ----------------------------------------------------------------------------


def _compose_global_rotations_chainreset(
    local_rot_mats_J: torch.Tensor,
    kinematic_chain: list[list[int]],
    num_joints: int = NUM_JOINTS,
) -> torch.Tensor:
    """
    Compose local rotations along the kinematic chain using HumanML3D's
    "chain-reset" semantics: at the start of EVERY chain (including sub-chains
    that start at non-root joints like the arms starting at joint 9), the
    running matR is reset to the ROOT rotation (local_rot_mats_J[:, 0]).

    Parameters
    ----------
    local_rot_mats_J : (T, J, 3, 3)
        Per-joint rotation matrices in HumanML3D's stored convention. Slot 0
        must contain the root rotation (e.g. from `quaternion_to_cont6d` of
        r_rot_quat -> matrix). Slots 1..J-1 are HumanML3D's `rot_data` 6D
        -> matrix.
    kinematic_chain : list[list[int]]
        e.g. paramUtil.t2m_kinematic_chain.

    Returns
    -------
    global_rot : (T, J, 3, 3)
        Per-joint "global" rotation -- the matR HumanML3D's
        `forward_kinematics_cont6d` holds when it would multiply by the
        offset of that joint to recover its world-frame bone direction.

    Round-trip guarantee
    --------------------
    `_decompose_global_rotations_chainreset` reverses this exactly.
    """
    T = local_rot_mats_J.shape[0]
    device = local_rot_mats_J.device
    dtype = local_rot_mats_J.dtype
    global_rot = (
        torch.eye(3, device=device, dtype=dtype)
        .view(1, 1, 3, 3)
        .expand(T, num_joints, 3, 3)
        .clone()
    )
    global_rot[:, 0] = local_rot_mats_J[:, 0]
    for chain in kinematic_chain:
        # HumanML3D's FK does NOT continue matR from the previous chain --
        # it always resets to root_rot at the start of each chain.
        matR = local_rot_mats_J[:, 0].clone()
        for k in range(1, len(chain)):
            j = chain[k]
            matR = matR @ local_rot_mats_J[:, j]
            global_rot[:, j] = matR
    return global_rot


def _decompose_global_rotations_chainreset(
    global_rot_mats: torch.Tensor,
    kinematic_chain: list[list[int]],
    num_joints: int = NUM_JOINTS,
) -> torch.Tensor:
    """
    Inverse of `_compose_global_rotations_chainreset`.

    Given global_rot_mats (T, J, 3, 3) -- using HumanML3D chain-reset semantics
    -- recover the local rotation matrices that HumanML3D would store. Slot 0
    is the root rotation (passthrough). Slots 1..J-1 are recovered by walking
    the chains and dividing out the previous accumulated rotation, where the
    "previous accumulated rotation" RESETS to the root at the start of each
    chain just like the forward pass.
    """
    T = global_rot_mats.shape[0]
    device = global_rot_mats.device
    dtype = global_rot_mats.dtype
    local_rot_mats_J = (
        torch.eye(3, device=device, dtype=dtype)
        .view(1, 1, 3, 3)
        .expand(T, num_joints, 3, 3)
        .clone()
    )
    local_rot_mats_J[:, 0] = global_rot_mats[:, 0]
    for chain in kinematic_chain:
        # Match forward: chain starts from root rotation.
        matR_pre = global_rot_mats[:, 0].clone()
        for k in range(1, len(chain)):
            j = chain[k]
            # global_rot[j] = matR_pre @ local_rot_mats_J[j]
            # => local_rot_mats_J[j] = matR_pre.T @ global_rot[j]
            local_rot_mats_J[:, j] = matR_pre.transpose(-1, -2) @ global_rot_mats[:, j]
            matR_pre = global_rot_mats[:, j]
    return local_rot_mats_J


# ----------------------------------------------------------------------------
# Forward: HumanML3D -> Kimodo-style
# ----------------------------------------------------------------------------


def humanml3d_to_kimodo(
    hml_vec: np.ndarray | torch.Tensor,
    device: str | torch.device = "cpu",
) -> dict:
    """
    Convert one HumanML3D 263-D motion (T, 263) into a Kimodo-style dict.

    See module docstring for layout. The returned dict is lossless: passing
    it back through `kimodo_to_humanml3d` recovers the original 263-D vector
    to within float32 precision, modulo the never-used entries
    data[-1, 0:3] and data[-1, 193:259] (which we set to 0).
    """
    if isinstance(hml_vec, np.ndarray):
        hml = torch.from_numpy(hml_vec).float().to(device)
    else:
        hml = hml_vec.float().to(device)
    assert hml.ndim == 2 and hml.shape[-1] == FEAT_DIM_HML3D, (
        f"Expected (T, {FEAT_DIM_HML3D}); got {tuple(hml.shape)}"
    )

    T = hml.shape[0]
    J = NUM_JOINTS

    # 1) World joint positions (positions-branch recovery) -------------------
    world_joints = recover_from_ric(hml, joints_num=J)  # (T, J, 3)

    # 2) Root rotation quat (half-angle Y), root world position --------------
    r_rot_quat, r_pos = recover_root_rot_pos(hml)

    # 3) Heading angle and Kimodo's 2D heading vector ------------------------
    # α = atan2(quat_y, quat_w) is HALF the actual heading rotation θ.
    alpha = torch.atan2(r_rot_quat[..., 2], r_rot_quat[..., 0])  # (T,)
    heading_angle = 2.0 * alpha
    global_root_heading = torch.stack(
        [torch.cos(heading_angle), torch.sin(heading_angle)], dim=-1
    )  # (T, 2)

    # 4) Smoothed root position (Kimodo's smoother) --------------------------
    smooth_root_pos = get_smooth_root_pos(r_pos.unsqueeze(0))[0]  # (T, 3)

    # 5) Kimodo local joint positions (subtract smooth root xz) --------------
    local_joints_positions = world_joints.clone()
    local_joints_positions[..., 0] -= smooth_root_pos[..., None, 0]
    local_joints_positions[..., 2] -= smooth_root_pos[..., None, 2]

    # 6) Global rotations via chain-reset FK ---------------------------------
    # Build the full (T, J, 6) local-rotation tensor: slot 0 = root rotation
    # in HumanML3D's 6D form, slots 1..21 = data's rot_data as stored.
    root_rot_6d = quaternion_to_cont6d(r_rot_quat)            # (T, 6)
    rot_data_6d = hml[..., 67:193].view(T, J - 1, 6)          # (T, 21, 6)
    all_local_6d = torch.cat([root_rot_6d.unsqueeze(-2), rot_data_6d], dim=-2)
    all_local_mats = cont6d_to_matrix(all_local_6d)           # (T, J, 3, 3)
    global_rot_mats = _compose_global_rotations_chainreset(
        all_local_mats, KINEMATIC_CHAIN, num_joints=J
    )                                                          # (T, J, 3, 3)
    global_rot_data = matrix_to_cont6d(global_rot_mats)        # (T, J, 6)

    # 7) World joint velocities = qinv(r_rot_quat) applied to local_velocity -
    local_vel_ego = hml[..., 193:259].view(T, J, 3)            # (T, J, 3) egocentric
    quat_expanded = qinv(r_rot_quat).unsqueeze(-2).expand(T, J, 4)
    velocities_world = qrot(quat_expanded, local_vel_ego)      # (T, J, 3) world

    # 8) Foot contacts passthrough -------------------------------------------
    foot_contacts = hml[..., 259:263].clone()

    # 9) Pack features in Kimodo order ---------------------------------------
    feat = torch.cat(
        [
            smooth_root_pos,                              # (T, 3)
            global_root_heading,                          # (T, 2)
            local_joints_positions.reshape(T, -1),        # (T, J*3)
            global_rot_data.reshape(T, -1),               # (T, J*6)
            velocities_world.reshape(T, -1),              # (T, J*3)
            foot_contacts,                                # (T, 4)
        ],
        dim=-1,
    )

    return {
        # Kimodo pack channels:
        "smooth_root_pos": smooth_root_pos,
        "global_root_heading": global_root_heading,
        "local_joints_positions": local_joints_positions,
        "global_rot_data": global_rot_data,
        "global_rot_mats": global_rot_mats,
        "velocities": velocities_world,
        "foot_contacts": foot_contacts,
        # Convenience / round-trip:
        "posed_joints": world_joints,
        "root_positions": r_pos,
        "r_rot_quat": r_rot_quat,  # convenient for round-trip; not strictly needed
        # Packed feature tensor:
        "features": feat,
    }


# ----------------------------------------------------------------------------
# Inverse: Kimodo-style dict -> HumanML3D 263
# ----------------------------------------------------------------------------


def _matrix_to_quaternion_y_aligned(R: torch.Tensor) -> torch.Tensor:
    """
    Given (T, 3, 3) rotation matrices that we KNOW are pure Y-axis rotations
    (the root rotation in HumanML3D), recover the half-angle quaternion in
    HumanML3D's storage form (w, 0, y, 0).

    We extract α = atan2(R[2,0], R[0,0]) -- this is the angle θ such that
    R = R_y(θ) -- then return (cos(θ/2), 0, sin(θ/2), 0).

    Why: HumanML3D stores r_rot_quat as (cos α, 0, sin α, 0) where α is
    *half* the heading angle (standard half-angle quat). The matrix form is
    the active rotation by 2α = θ. So given R, θ = 2α can be read off, and
    α = θ/2 is what HumanML3D stores.
    """
    # For q = (cos α, 0, sin α, 0), HumanML3D's quaternion_to_matrix gives:
    #   R[0, 0] = R[2, 2] = 1 - 2 sin² α = cos(2α)
    #   R[0, 2] =  2 cos α · sin α =  sin(2α)
    #   R[2, 0] = -2 cos α · sin α = -sin(2α)
    # So the Y-axis full rotation angle is:
    #   2α = atan2( sin(2α), cos(2α) ) = atan2( R[..., 0, 2], R[..., 0, 0] )
    two_alpha = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    alpha = 0.5 * two_alpha
    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)
    q[..., 0] = torch.cos(alpha)
    q[..., 2] = torch.sin(alpha)
    return q


def _delta_rot_vel_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """
    Recover root_rot_velocity (T,) from per-frame α (T,) following HumanML3D's
    convention:
        r_rot_ang[0] = 0
        r_rot_ang[1:] = cumsum(rot_vel[..., :-1])
    => rot_vel[t] = alpha[t+1] - alpha[t] for t in 0..T-2; rot_vel[T-1] = 0
       (unused in recovery).
    Also wraps deltas into (-π, π] to handle the atan2 branch cut.
    """
    rot_vel = torch.zeros_like(alpha)
    diff = alpha[1:] - alpha[:-1]
    # Wrap to (-π, π]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    rot_vel[:-1] = diff
    return rot_vel


def kimodo_to_humanml3d(kimodo_dict: dict, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Inverse of `humanml3d_to_kimodo`. Takes a dict of tensors (see forward
    docstring for keys) and returns the HumanML3D 263-D vector.

    The fields actually required for inverse:
      global_rot_mats    (T, J, 3, 3)
      posed_joints       (T, J, 3)        world joints
      root_positions     (T, 3)           world root xyz (== posed_joints[:,0])
      velocities         (T, J, 3)        world joint velocities
      foot_contacts      (T, 4)

    The other entries (smooth_root_pos, global_root_heading, local_joints_positions,
    global_rot_data) are recomputable from those above; they are NOT used here.

    Returns
    -------
    data : (T, 263) torch.Tensor on `device`, dtype float32.
    """
    grm = kimodo_dict["global_rot_mats"].to(device).float()
    posed = kimodo_dict["posed_joints"].to(device).float()
    r_pos = kimodo_dict["root_positions"].to(device).float()
    velocities = kimodo_dict["velocities"].to(device).float()
    foot_contacts = kimodo_dict["foot_contacts"].to(device).float()

    T, J, _, _ = grm.shape
    assert J == NUM_JOINTS

    # 1) Recover root quat (half-angle Y) and α per frame --------------------
    # If the dict was produced by `humanml3d_to_kimodo` it carries the original
    # r_rot_quat; use it for an exact round trip. Otherwise reconstruct from
    # the root rotation matrix. NOTE: reconstructing from the matrix is lossy
    # when α leaves (-π/2, π/2] -- it loses a ±π ambiguity (the matrix
    # encodes 2α mod 2π). Round trip will still be CORRECT (physically same
    # rotation), but rot_vel can differ from the source by ±π at the
    # wrap-around frames. Pass `r_rot_quat` in the dict to avoid this.
    if "r_rot_quat" in kimodo_dict and kimodo_dict["r_rot_quat"] is not None:
        r_rot_quat = kimodo_dict["r_rot_quat"].to(device).float()
    else:
        r_rot_quat = _matrix_to_quaternion_y_aligned(grm[:, 0])
    alpha = torch.atan2(r_rot_quat[..., 2], r_rot_quat[..., 0])  # (T,)

    # 2) root_rot_velocity = Δα with last frame zero -------------------------
    rot_vel = _delta_rot_vel_from_alpha(alpha)                  # (T,)

    # 3) Root linear velocity in egocentric frame ---------------------------
    # Forward: r_pos[1:, [0,2]] = data[:-1, 1:3] then qrot(qinv(r_rot_quat), .)
    # then cumsum. Because qrot is broadcast per-frame, the actual relation is
    #   r_pos[t+1] - r_pos[t] = qrot(qinv(r_rot_quat[t+1]), [data[t,1], 0, data[t,2]])
    # so the inverse uses r_rot_quat at index t+1, NOT t.
    # data[T-1, 1:3] is unused in recovery -> set to 0.
    lin_vel = torch.zeros(T, 2, device=device, dtype=torch.float32)
    if T > 1:
        world_xz_delta = torch.zeros(T - 1, 3, device=device, dtype=torch.float32)
        world_xz_delta[:, 0] = r_pos[1:, 0] - r_pos[:-1, 0]
        world_xz_delta[:, 2] = r_pos[1:, 2] - r_pos[:-1, 2]
        ego_xz = qrot(r_rot_quat[1:], world_xz_delta)  # NB: r_rot_quat[1:], not [:-1]
        lin_vel[:-1, 0] = ego_xz[:, 0]
        lin_vel[:-1, 1] = ego_xz[:, 2]

    # 4) Root height passthrough --------------------------------------------
    root_height = r_pos[:, 1]                                   # (T,)

    # 5) ric_data: world joints excluding root, minus root xz, rotated to ego
    pos_minus_root_xz = posed[:, 1:].clone()                    # (T, J-1, 3)
    pos_minus_root_xz[..., 0] -= r_pos[:, None, 0]
    pos_minus_root_xz[..., 2] -= r_pos[:, None, 2]
    quat_J_minus_1 = r_rot_quat[:, None, :].expand(T, J - 1, 4)
    ric = qrot(quat_J_minus_1, pos_minus_root_xz)               # (T, J-1, 3)

    # 6) rot_data: reverse chain-reset decomposition ------------------------
    local_mats_J = _decompose_global_rotations_chainreset(
        grm, KINEMATIC_CHAIN, num_joints=J
    )                                                            # (T, J, 3, 3)
    rot_data_6d_full = matrix_to_cont6d(local_mats_J)            # (T, J, 6)
    # Drop slot 0 (root rotation lives in root_rot_velocity).
    rot_data = rot_data_6d_full[:, 1:].reshape(T, (J - 1) * 6)   # (T, 126)

    # 7) local_velocity: world velocities rotated to ego --------------------
    quat_J = r_rot_quat[:, None, :].expand(T, J, 4)
    local_vel_ego = qrot(quat_J, velocities)                     # (T, J, 3)
    local_vel_flat = local_vel_ego.reshape(T, J * 3)

    # 8) Pack into 263-D ----------------------------------------------------
    out = torch.zeros(T, FEAT_DIM_HML3D, device=device, dtype=torch.float32)
    out[:, 0] = rot_vel
    out[:, 1:3] = lin_vel
    out[:, 3] = root_height
    out[:, 4:67] = ric.reshape(T, -1)
    out[:, 67:193] = rot_data
    out[:, 193:259] = local_vel_flat
    out[:, 259:263] = foot_contacts
    return out


# ----------------------------------------------------------------------------
# I/O + CLI
# ----------------------------------------------------------------------------


def _convert_one(
    src: Path,
    dst_dir: Path,
    overwrite: bool,
) -> tuple[Path, str]:
    out_path = dst_dir / (src.stem + ".npz")
    if out_path.is_file() and not overwrite:
        return out_path, "skipped"
    try:
        arr = np.load(src)
        out = humanml3d_to_kimodo(arr)
        np_out = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out.items()}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **np_out)
        return out_path, "ok"
    except Exception as e:
        return out_path, f"error: {type(e).__name__}: {e}"


def _verify_roundtrip(src: Path) -> tuple[Path, float, float]:
    """Forward + inverse on one file. Returns (path, max_abs_err, mean_abs_err)
    over the "used" channels of the 263-D vector. Excludes data[-1, 0:3] and
    data[-1, 193:259] which are unused in HumanML3D's recovery."""
    arr = np.load(src)
    fwd = humanml3d_to_kimodo(arr)
    back = kimodo_to_humanml3d(fwd).cpu().numpy()
    diff = np.abs(back - arr)
    # Last-frame velocity / rot_vel are unused -- zero those out before
    # measuring so we report a fair error.
    diff[-1, 0] = 0.0
    diff[-1, 1:3] = 0.0
    diff[-1, 193:259] = 0.0
    return src, float(diff.max()), float(diff.mean())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert HumanML3D 263-D vectors <-> Kimodo-style global-root NPZs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs"),
        help="Folder of HumanML3D *.npy files.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/weka/jungbin/humanml3d_kimodo_motions_20fps"),
        help="Folder for *.npz outputs.",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Only process N files (debug).")
    ap.add_argument(
        "--verify-roundtrip",
        type=int,
        default=0,
        metavar="N",
        help="Run humanml3d->kimodo->humanml3d on the first N files and report max abs diff.",
    )
    ap.add_argument(
        "--skip-write",
        action="store_true",
        help="Don't write any NPZs (only useful with --verify-roundtrip for fast iteration).",
    )
    args = ap.parse_args()

    src_files = sorted(args.input_dir.glob("*.npy"))
    if args.limit is not None:
        src_files = src_files[: args.limit]
    if not src_files:
        raise SystemExit(f"No *.npy under {args.input_dir}")
    print(f"Discovered {len(src_files)} HumanML3D files.")

    # --- Conversion pass ---
    if not args.skip_write:
        print(f"Writing to {args.output_dir}")
        fn = partial(_convert_one, dst_dir=args.output_dir, overwrite=args.overwrite)
        n_ok = n_skip = n_err = 0
        errors: list[tuple[Path, str]] = []
        if args.workers <= 1:
            it = (fn(p) for p in src_files)
            pool = None
        else:
            pool = Pool(args.workers)
            it = pool.imap_unordered(fn, src_files)
        for out_path, status in tqdm(it, total=len(src_files), desc="Converting"):
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_err += 1
                errors.append((out_path, status))
        if pool is not None:
            pool.close()
            pool.join()
        print(f"Done. ok={n_ok}, skipped={n_skip}, errors={n_err}")
        if errors:
            print("First errors:")
            for path, msg in errors[:10]:
                print(f"  {path}: {msg}")

    # --- Optional round-trip verification ---
    if args.verify_roundtrip > 0:
        check_files = src_files[: args.verify_roundtrip]
        print(f"\nRound-trip verification on {len(check_files)} file(s):")
        overall_max = 0.0
        for src in check_files:
            _, mx, mn = _verify_roundtrip(src)
            overall_max = max(overall_max, mx)
            print(f"  {src.name}: max |Δ|={mx:.3e}, mean |Δ|={mn:.3e}")
        print(f"Overall max |Δ| across {len(check_files)} files: {overall_max:.3e}")
        if overall_max < 1e-4:
            print("Round-trip is exact (within single-precision float noise).")
        else:
            print("WARNING: round-trip error is larger than expected -- inspect.")


if __name__ == "__main__":
    main()
