# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight matplotlib renderer for HumanML3D (SMPL-22) skeleton motions.

Used for in-training visualisation: render a generated motion side-by-side
with its ground-truth, as an RGB frame stack suitable for TensorBoard
``add_video`` and/or an MP4.

World convention is HumanML3D / SMPL: **Y up, +Z forward**. matplotlib's
mplot3d is Z-up, so we remap world ``(x, y, z) -> plot (x, z, y)`` before
drawing (same trick as ``kimodo/scripts/visualize_hml3d.py``).

We render directly from world **joint positions** ``(T, 22, 3)`` — no FK and
no bone lengths required. For kimodo features, recover joints via
``kimodo.motion_rep.fk_hml3d.world_joints_from_kimodo_features``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402 (registers 3d projection)

from kimodo.motion_rep.fk_hml3d import HML3D_KINEMATIC_CHAIN


def _shared_bounds(joints_list: List[np.ndarray], pad: float = 0.3) -> Tuple[np.ndarray, float]:
    """Center + half-extent that contains every joint set passed in."""
    allj = np.concatenate([j.reshape(-1, 3) for j in joints_list], axis=0)
    center = allj.mean(axis=0)
    half = float(np.abs(allj - center).max()) + pad
    half = max(half, 0.6)
    return center, half


def _draw(ax, joints_xyz: np.ndarray, color: str, center: np.ndarray, half: float, title: str):
    # world (x, y, z) -> plot (x, z, y)
    xs, ys, zs = joints_xyz[:, 0], joints_xyz[:, 1], joints_xyz[:, 2]
    ax.scatter(xs, zs, ys, c=color, s=14)
    for chain in HML3D_KINEMATIC_CHAIN:
        ax.plot(joints_xyz[chain, 0], joints_xyz[chain, 2], joints_xyz[chain, 1],
                color=color, linewidth=2.0)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[2] - half, center[2] + half)
    ax.set_zlim(max(0.0, center[1] - half), center[1] + half)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x", fontsize=6)
    ax.set_ylabel("z", fontsize=6)
    ax.set_zlabel("y", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=15, azim=-70)


def render_sidebyside(
    gt_joints: np.ndarray,
    gen_joints: np.ndarray,
    caption: str = "",
    width: int = 640,
    height: int = 320,
    dpi: int = 80,
    max_frames: Optional[int] = 120,
    frame_stride: int = 1,
) -> np.ndarray:
    """Render GT (left) vs generated (right) skeleton, frame by frame.

    Args:
        gt_joints: ``(T, 22, 3)`` GT world joints.
        gen_joints: ``(T', 22, 3)`` generated world joints. If T' != T, the
            shorter length is used.
        caption: text prompt, drawn as the figure suptitle.
        width/height/dpi: output frame size.
        max_frames: cap on number of rendered frames (keeps viz fast).
        frame_stride: render every ``frame_stride``-th frame.

    Returns:
        ``(T_render, H, W, 3)`` uint8 frame stack.
    """
    T = min(gt_joints.shape[0], gen_joints.shape[0])
    idx = list(range(0, T, max(1, frame_stride)))
    if max_frames is not None and len(idx) > max_frames:
        idx = list(np.linspace(0, T - 1, max_frames, dtype=int))

    center, half = _shared_bounds([gt_joints[:T], gen_joints[:T]])

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    frames: List[np.ndarray] = []
    try:
        for t in idx:
            fig.clf()
            ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
            ax_gen = fig.add_subplot(1, 2, 2, projection="3d")
            _draw(ax_gt, gt_joints[t], "tab:blue", center, half, f"GT  t={t}")
            _draw(ax_gen, gen_joints[t], "tab:red", center, half, f"generated  t={t}")
            if caption:
                fig.suptitle(caption[:80], fontsize=8)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02, wspace=0.05)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(img)
    finally:
        plt.close(fig)
    return np.stack(frames, axis=0)  # (T_render, H, W, 3)
