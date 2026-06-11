# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matplotlib renderer for SOMA-skeleton motions (training-time viz).

A lighter analog of :mod:`kimodo.scripts.render_hml3d` for the BONES-SEED /
SOMA family: same tracking camera, floor, and grid, but bones are drawn from
the skeleton's ``joint_parents`` array (one segment per non-root joint)
instead of a hard-coded chain list — so the same renderer works for
``SOMASkeleton30`` and ``SOMASkeleton77`` interchangeably.

World convention matches the SOMA data: **right-handed, Y up, +Z forward**.
Because the basis identity ``ŷ × ẑ = x̂`` forces +X to point to the
character's LEFT side (verified empirically: in the somaskel30 T-pose,
``LeftFoot.x = +0.10`` and ``RightFoot.x = -0.10``). matplotlib's plot +X
renders on screen RIGHT, so a literal world→plot mapping would mirror the
scene (a "right turn" appears as a "left turn"). To match viewer intuition
we negate world-X at the entry of every render function and let downstream
helpers work in the resulting display frame, where +X (display) = screen
right = character's right.

matplotlib's mplot3d is Z-up, so we still remap ``display (x, y, z) ->
plot (x, z, y)``.

Inputs are world joint positions ``(T, J, 3)`` — typically
``motion_rep.inverse(...)["posed_joints"]``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402 (registers 3d projection)

from kimodo.scripts.render_hml3d import (
    DEFAULT_GRID_SPACING,
    DEFAULT_VIEW_HALF,
    _draw_floor_and_grid,
    _draw_trajectory,
    _frame_indices,
    _setup_axes,
    _viewport_for_center,
    _world_extent,
)


def _to_display(joints: np.ndarray) -> np.ndarray:
    """World → display frame. Negates X so character-right ends up on screen right.

    See module docstring for the right-handed (Y up, +Z forward) → +X-is-LEFT
    derivation that makes this flip necessary.
    """
    out = np.asarray(joints).copy()
    out[..., 0] = -out[..., 0]
    return out


def _draw_skel(
    ax,
    joints_xyz: np.ndarray,
    joint_parents: Sequence[int],
    color: str,
    lw: float = 2.2,
    scatter_s: float = 18.0,
) -> None:
    """Stick figure: one line segment per (parent, child) pair from ``joint_parents``."""
    ax.scatter(
        joints_xyz[:, 0], joints_xyz[:, 2], joints_xyz[:, 1],
        c=color, s=scatter_s, zorder=5,
    )
    for child, parent in enumerate(joint_parents):
        p = int(parent)
        if p < 0:
            continue
        ax.plot(
            [joints_xyz[p, 0], joints_xyz[child, 0]],
            [joints_xyz[p, 2], joints_xyz[child, 2]],
            [joints_xyz[p, 1], joints_xyz[child, 1]],
            color=color, linewidth=lw, zorder=4,
        )


def render_sidebyside(
    gt_joints: Optional[np.ndarray],   # (T, J, 3) or None for "no GT" (blank left panel)
    gen_joints: np.ndarray,            # (T, J, 3)
    joint_parents: Sequence[int],
    caption: str = "",
    width: int = 1200,
    height: int = 600,
    dpi: int = 120,
    max_frames: Optional[int] = 200,
    frame_stride: int = 1,
    view_half: float = DEFAULT_VIEW_HALF,
    grid_spacing: float = DEFAULT_GRID_SPACING,
) -> np.ndarray:
    """GT (left, blue) vs generated (right, red). Returns ``(T, H, W, 3)`` uint8.

    When ``gt_joints is None`` the left panel still gets the same floor +
    viewport as the right (so the layout stays consistent), but no skeleton
    or trajectory is drawn, and the title reads ``(no GT)``.
    """
    # Convert to display frame (negate world X) so character-right ends up
    # on screen right. See module docstring.
    gen_joints = _to_display(gen_joints)
    if gt_joints is not None:
        gt_joints = _to_display(gt_joints)
    if gt_joints is None:
        T = gen_joints.shape[0]
    else:
        T = min(gt_joints.shape[0], gen_joints.shape[0])
    idx = _frame_indices(T, max_frames, frame_stride)
    gen_trail = gen_joints[:T, 0]
    if gt_joints is None:
        gt_trail = gen_trail
        extent = _world_extent([gen_joints[:T]], extra_pts=[gen_trail])
    else:
        gt_trail = gt_joints[:T, 0]
        extent = _world_extent(
            [gt_joints[:T], gen_joints[:T]],
            extra_pts=[gt_trail, gen_trail],
        )

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    frames: List[np.ndarray] = []
    try:
        for t in idx:
            fig.clf()
            gt_root = gt_trail[t]
            gen_root = gen_trail[t]

            ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
            _setup_axes(
                ax_gt,
                _viewport_for_center((gt_root[0], gt_root[2]), view_half, extent["y_top"]),
                (f"GT  t={t}" if gt_joints is not None else "(no GT)"),
            )
            _draw_floor_and_grid(ax_gt, extent, spacing=grid_spacing)
            if gt_joints is not None:
                _draw_trajectory(ax_gt, gt_trail, t, "tab:blue")
                _draw_skel(ax_gt, gt_joints[t], joint_parents, "tab:blue")

            ax_gen = fig.add_subplot(1, 2, 2, projection="3d")
            _setup_axes(
                ax_gen,
                _viewport_for_center((gen_root[0], gen_root[2]), view_half, extent["y_top"]),
                f"generated  t={t}",
            )
            _draw_floor_and_grid(ax_gen, extent, spacing=grid_spacing)
            _draw_trajectory(ax_gen, gen_trail, t, "tab:red")
            _draw_skel(ax_gen, gen_joints[t], joint_parents, "tab:red")

            if caption:
                fig.suptitle(caption[:100], fontsize=10)
            fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.04, wspace=0.08)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(img)
    finally:
        plt.close(fig)
    return np.stack(frames, axis=0)


def render_single(
    joints: np.ndarray,            # (T, J, 3)
    joint_parents: Sequence[int],
    caption: str = "",
    color: str = "tab:red",
    width: int = 700,
    height: int = 700,
    dpi: int = 120,
    max_frames: Optional[int] = 200,
    frame_stride: int = 1,
    view_half: float = DEFAULT_VIEW_HALF,
    grid_spacing: float = DEFAULT_GRID_SPACING,
) -> np.ndarray:
    """One skeleton + its trail with a tracking camera and world-grid floor."""
    joints = _to_display(joints)
    T = joints.shape[0]
    idx = _frame_indices(T, max_frames, frame_stride)
    trail = joints[:T, 0]
    extent = _world_extent([joints[:T]], extra_pts=[trail])

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    frames: List[np.ndarray] = []
    try:
        for t in idx:
            fig.clf()
            root = trail[t]
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            _setup_axes(
                ax,
                _viewport_for_center((root[0], root[2]), view_half, extent["y_top"]),
                f"t={t}",
            )
            _draw_floor_and_grid(ax, extent, spacing=grid_spacing)
            _draw_trajectory(ax, trail, t, color)
            _draw_skel(ax, joints[t], joint_parents, color)
            if caption:
                fig.suptitle(caption[:100], fontsize=10)
            fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.04)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(img)
    finally:
        plt.close(fig)
    return np.stack(frames, axis=0)
