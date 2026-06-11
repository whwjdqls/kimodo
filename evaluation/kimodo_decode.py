# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode kimodo (T, 273) features into the dict ``kimodo_to_humanml3d`` expects.

Our HumanML3D-in-kimodo pipeline stores motion as a 273-D feature vector with
blocks accessible via ``motion_rep.slice_dict``. The HumanML3D-263 converter
in ``kimodo_open/benchmark/humanml3d_to_kimodo.py`` (function
``kimodo_to_humanml3d``) consumes a dict with raw geometric quantities; this
module bridges the two.

The five fields required by the converter are:

    global_rot_mats : (T, J, 3, 3)  -- chain-reset global rotations, from the
                                       6D ``global_rot_data`` block via
                                       ``cont6d_to_matrix`` (see geometry.py).
    posed_joints    : (T, J, 3)     -- world joint positions, from
                                       ``world_joints_from_kimodo_features``.
    root_positions  : (T, 3)        -- ``posed_joints[:, 0]`` (root joint).
    velocities      : (T, J, 3)     -- direct from the ``velocities`` block.
    foot_contacts   : (T, 4)        -- direct from the ``foot_contacts`` block.

All inputs are expected **unnormalized** (already de-z-scored with the kimodo
mean/std). Output dict tensors live on the same device as the input.
"""

from __future__ import annotations

from typing import Dict

import torch

from kimodo.geometry import cont6d_to_matrix
from kimodo.motion_rep.fk_hml3d import world_joints_from_kimodo_features


def kimodo_features_to_decode_dict(
    features_unnormalized: torch.Tensor,
    slice_dict: dict,
    n_joints: int = 22,
) -> Dict[str, torch.Tensor]:
    """Decode a single ``(T, 273)`` unnormalized kimodo feature vector.

    Args:
        features_unnormalized: shape ``(T, 273)``. **Already unnormalized**
            (i.e. raw kimodo features, not the z-scored ones).
        slice_dict: ``motion_rep.slice_dict``.
        n_joints: J (22 for HumanML3D-22).

    Returns:
        Dict with keys ``global_rot_mats``, ``posed_joints``,
        ``root_positions``, ``velocities``, ``foot_contacts`` — all on
        ``features_unnormalized.device``.
    """
    if features_unnormalized.dim() != 2:
        raise ValueError(
            f"Expected (T, 273); got {tuple(features_unnormalized.shape)}"
        )
    T = features_unnormalized.shape[0]

    # 6D global rotations -> rotation matrices (chain-reset convention).
    rot6d = features_unnormalized[:, slice_dict["global_rot_data"]]
    rot6d = rot6d.reshape(T, n_joints, 6)
    grm = cont6d_to_matrix(rot6d)  # (T, J, 3, 3)

    # World joints (T, J, 3) via the existing converter.
    posed = world_joints_from_kimodo_features(
        features_unnormalized, slice_dict, n_joints=n_joints,
    )  # (T, J, 3)
    root_pos = posed[:, 0]  # (T, 3)

    # Velocities + foot_contacts read straight off the feature vector.
    vel = features_unnormalized[:, slice_dict["velocities"]].reshape(T, n_joints, 3)
    contacts = features_unnormalized[:, slice_dict["foot_contacts"]]  # (T, 4)

    return {
        "global_rot_mats": grm,
        "posed_joints": posed,
        "root_positions": root_pos,
        "velocities": vel,
        "foot_contacts": contacts,
    }
