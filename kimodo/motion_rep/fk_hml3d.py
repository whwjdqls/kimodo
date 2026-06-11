"""Chain-reset forward kinematics for HumanML3D-in-kimodo features.

This module exactly mirrors HumanML3D's ``Skeleton.forward_kinematics_cont6d``
(see ``HumanML3D/common/skeleton.py``), specialised to start from already-
chain-reset *global* rotations (which is what our kimodo features store).

HumanML3D's FK does two things per chain link:

  1. matR is reset to the root rotation at the START of every chain — even
     non-root sub-chains like ``spine3 -> left_collar -> ...``. This is the
     "chain-reset" rule.
  2. matR accumulates by right-multiplying each joint's local rotation:
     ``matR = matR @ local_rot[chain[k]]``. The accumulated matR at chain[k]
     is what we already store in ``global_rot_data`` — see
     ``benchmark/humanml3d_to_kimodo._compose_global_rotations_chainreset``.

So, **given** the chain-reset ``global_rot_mats``, the per-joint FK update is
simply::

    world_pos[chain[k]] = world_pos[chain[k-1]] + global_rot_mats[chain[k]] @ offset_vec[chain[k]]

The **critical** subtlety is what ``offset_vec[chain[k]]`` is. HumanML3D stores
it as ``raw_offset_direction[chain[k]] * bone_length[chain[k]]`` where:

  * ``raw_offset_direction`` is the canonical axis-aligned UNIT direction
    HumanML3D normalises every bone to. For the 22-joint SMPL chain these are
    the ``paramUtil.t2m_raw_offsets`` values — e.g. left hip is ``[+1, 0, 0]``,
    left shoulder is ``[0, -1, 0]``, head is ``[0, 0, +1]``. **They are not
    the bone direction in the kimodo canonical T-pose.**
  * ``bone_length[j]`` is the SCALAR L2 distance between joint j and its
    chain parent in the actor's rest skeleton. In HumanML3D's processing it is
    recovered by ``Skeleton.get_offsets_joints_batch`` from frame-0 joint
    positions and is per-motion. For our training loss we derive it the same
    way from each GT sample.

A previous version of this file used ``kimodo_neutral_joints[child] -
kimodo_neutral_joints[chain_parent]`` as the offset. Those vectors are NOT
axis-aligned (the kimodo SMPLXSkeleton22 T-pose has diagonal arm bones) and
so the FK produced visibly mis-oriented limbs. Use ``raw_offsets`` here.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


# HumanML3D's 22-joint SMPL kinematic chain (copied from
# ``paramUtil.t2m_kinematic_chain`` so we don't pull in HumanML3D's package
# at training time).
HML3D_KINEMATIC_CHAIN: List[List[int]] = [
    [0, 2, 5, 8, 11],          # right leg:  pelvis -> right_hip -> right_knee -> right_ankle -> right_foot
    [0, 1, 4, 7, 10],          # left  leg:  pelvis -> left_hip  -> left_knee  -> left_ankle  -> left_foot
    [0, 3, 6, 9, 12, 15],      # spine:      pelvis -> spine1 -> spine2 -> spine3 -> neck -> head
    [9, 14, 17, 19, 21],       # right arm:  spine3 -> right_collar -> right_shoulder -> right_elbow -> right_wrist
    [9, 13, 16, 18, 20],       # left  arm:  spine3 -> left_collar  -> left_shoulder  -> left_elbow  -> left_wrist
]


# HumanML3D's canonical raw bone-offset directions for SMPL-22 (from
# ``paramUtil.t2m_raw_offsets``). These are AXIS-ALIGNED UNIT vectors that
# define HumanML3D's bone convention; the actual offset used in FK is
# ``raw_offset[j] * bone_length[j]``.
HML3D_RAW_OFFSETS: torch.Tensor = torch.tensor(
    [
        [ 0,  0,  0],   # 0  pelvis        (root, no incoming bone)
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


def derive_bone_lengths_from_world_joints(
    world_joints: torch.Tensor,
    chains: Sequence[Sequence[int]] = HML3D_KINEMATIC_CHAIN,
    n_joints: int = 22,
    reduce: str = "median",
) -> torch.Tensor:
    """Recover per-sample per-joint bone lengths from a sequence of world joint positions.

    Args:
        world_joints: ``(B, T, J, 3)`` or ``(T, J, 3)`` world joint positions.
        chains: kinematic chains. Default is HumanML3D's 22-joint chain.
        n_joints: ``J``.
        reduce: how to collapse over the time axis — ``"median"`` (robust),
            ``"mean"``, or ``"first"`` (the value at t=0). For rigid bodies all
            values are identical; the reduction matters only for numerical
            noise.

    Returns:
        ``(B, J)`` (or ``(J,)`` if input was unbatched) where index 0 stays 0
        (root has no incoming bone).
    """
    squeeze_batch = False
    if world_joints.dim() == 3:
        world_joints = world_joints.unsqueeze(0)
        squeeze_batch = True
    if world_joints.dim() != 4:
        raise ValueError(
            f"world_joints must be (B, T, J, 3) or (T, J, 3); got {world_joints.shape}"
        )
    B, T, J, _ = world_joints.shape
    assert J == n_joints, f"world_joints J={J} != n_joints={n_joints}"

    bone_lengths = world_joints.new_zeros(B, n_joints)
    for chain in chains:
        for k in range(1, len(chain)):
            j = int(chain[k])
            parent = int(chain[k - 1])
            diff = world_joints[:, :, j] - world_joints[:, :, parent]  # (B, T, 3)
            per_frame = torch.linalg.norm(diff, dim=-1)                 # (B, T)
            if reduce == "median":
                val = per_frame.median(dim=-1).values
            elif reduce == "mean":
                val = per_frame.mean(dim=-1)
            elif reduce == "first":
                val = per_frame[:, 0]
            else:
                raise ValueError(f"unknown reduce: {reduce}")
            bone_lengths[:, j] = val
    return bone_lengths.squeeze(0) if squeeze_batch else bone_lengths


def chainreset_fk_world_joints(
    global_rot_mats: torch.Tensor,
    root_pos: torch.Tensor,
    bone_lengths: torch.Tensor,
    raw_offsets: torch.Tensor = HML3D_RAW_OFFSETS,
    chains: Sequence[Sequence[int]] = HML3D_KINEMATIC_CHAIN,
) -> torch.Tensor:
    """Compute world joint positions from chain-reset global rotations.

    Mirrors HumanML3D's ``Skeleton.forward_kinematics_cont6d`` but starts from
    already-composed chain-reset global rotations (the value of ``matR`` HumanML3D
    holds at the point of computing joint ``chain[k]``).

    Args:
        global_rot_mats: ``(B, T, J, 3, 3)`` or ``(T, J, 3, 3)`` chain-reset
            global rotation matrices (must match HumanML3D's chain-reset
            convention — see ``_compose_global_rotations_chainreset``).
        root_pos: ``(B, T, 3)`` or ``(T, 3)`` world root position.
        bone_lengths: ``(B, J)`` or ``(J,)`` per-joint chain-relative bone
            length scalars. Use :func:`derive_bone_lengths_from_world_joints`
            to derive them from GT positions, or set to a canonical vector.
        raw_offsets: ``(J, 3)`` unit-direction bone offsets, defaulting to
            HumanML3D's ``t2m_raw_offsets``.
        chains: kinematic chains, default ``HML3D_KINEMATIC_CHAIN``.

    Returns:
        ``world_pos`` with shape matching ``global_rot_mats[..., :J, 0, 0]``
        plus trailing ``(3,)`` — i.e. ``(B, T, J, 3)`` or ``(T, J, 3)``.
    """
    squeeze_batch = False
    if global_rot_mats.dim() == 4:
        global_rot_mats = global_rot_mats.unsqueeze(0)
        root_pos = root_pos.unsqueeze(0)
        if bone_lengths.dim() == 1:
            bone_lengths = bone_lengths.unsqueeze(0)
        squeeze_batch = True
    if global_rot_mats.dim() != 5:
        raise ValueError(
            f"global_rot_mats must be (B, T, J, 3, 3) or (T, J, 3, 3); got {global_rot_mats.shape}"
        )
    B, T, J, _, _ = global_rot_mats.shape
    if raw_offsets.shape != (J, 3):
        raise ValueError(f"raw_offsets must be (J={J}, 3); got {tuple(raw_offsets.shape)}")
    if bone_lengths.shape not in ((B, J), (J,)):
        raise ValueError(
            f"bone_lengths must be (B, J) or (J,); got {tuple(bone_lengths.shape)}"
        )

    device = global_rot_mats.device
    dtype = global_rot_mats.dtype
    raw = raw_offsets.to(device=device, dtype=dtype)
    bl = bone_lengths.to(device=device, dtype=dtype)
    if bl.dim() == 1:
        bl = bl.unsqueeze(0).expand(B, -1)  # (B, J)

    # Per-joint offset vector in HumanML3D convention: raw_offset * bone_length.
    # Shape: (B, J, 3).
    offsets_world = raw[None, :, :] * bl[..., None]  # (B, J, 3)

    world_pos = torch.zeros(B, T, J, 3, device=device, dtype=dtype)
    world_pos[:, :, 0] = root_pos

    for chain in chains:
        if len(chain) < 2:
            continue
        for k in range(1, len(chain)):
            parent_idx = int(chain[k - 1])
            child_idx = int(chain[k])
            offset = offsets_world[:, child_idx]  # (B, 3)
            # global_rot_mats[:, :, child_idx]: (B, T, 3, 3); offset: (B, 3) -> broadcast over T
            rotated = torch.einsum(
                "btij,bj->bti", global_rot_mats[:, :, child_idx], offset
            )  # (B, T, 3)
            world_pos[:, :, child_idx] = world_pos[:, :, parent_idx] + rotated

    return world_pos.squeeze(0) if squeeze_batch else world_pos


def world_joints_from_kimodo_features(
    features_unnormalized: torch.Tensor,
    slice_dict: dict,
    n_joints: int = 22,
) -> torch.Tensor:
    """Recover world joint positions from kimodo feature blocks.

    The conversion script stores ``local_joints_positions = world_joints -
    (smooth_root_x, 0, smooth_root_z)``, so the inverse is:

        world_x = local_x + smooth_root_x
        world_y = local_y
        world_z = local_z + smooth_root_z

    Args:
        features_unnormalized: ``(B, T, D)`` or ``(T, D)`` UNNORMALIZED kimodo
            features (273-dim for HumanML3D / 22 joints).
        slice_dict: ``KimodoMotionRep.slice_dict``.
        n_joints: J.

    Returns:
        ``(B, T, J, 3)`` or ``(T, J, 3)`` world joint positions.
    """
    smooth_root = features_unnormalized[..., slice_dict["smooth_root_pos"]]      # (..., 3)
    local_jp = features_unnormalized[..., slice_dict["local_joints_positions"]]   # (..., J*3)
    local_jp = local_jp.reshape(*local_jp.shape[:-1], n_joints, 3)
    out = local_jp.clone()
    out[..., 0] = out[..., 0] + smooth_root[..., None, 0]
    out[..., 2] = out[..., 2] + smooth_root[..., None, 2]
    return out
