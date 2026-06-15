# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matplotlib renderer for HumanML3D (SMPL-22) skeleton motions.

Camera model: a **fixed-size viewport** (default 4 m × 4 m on the XZ floor)
**tracks the character**, so the skeleton stays at consistent on-screen size
even for long-distance locomotion. A gray floor with a 1 m **grid in world
coordinates** spans the entire motion — as the viewport moves, the grid
lines visibly pass by, so translation reads cleanly without the "ground
follows the character" effect. The root XZ trajectory is drawn on the floor
with a marker at the current frame.

World convention is HumanML3D / SMPL: **right-handed, Y up, +Z forward**.
Because the basis identity ``ŷ × ẑ = x̂`` forces +X to point to the
character's LEFT side (verified in ``HML3D_RAW_OFFSETS``: ``left_hip =
[+1,0,0]``, ``right_hip = [-1,0,0]``). matplotlib's plot +X renders on
screen RIGHT, so a literal world→plot mapping would mirror the scene (a
"right turn" appears as a "left turn"). To match viewer intuition we negate
world-X at the entry of every render function and let downstream helpers
work in the resulting display frame, where +X (display) = screen right =
character's right.

matplotlib's mplot3d is Z-up, so we still remap ``display (x, y, z) ->
plot (x, z, y)``.

We render directly from world **joint positions** ``(T, 22, 3)`` — no FK and
no bone lengths required. For kimodo features, recover joints via
``kimodo.motion_rep.fk_hml3d.world_joints_from_kimodo_features``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

from kimodo.motion_rep.fk_hml3d import HML3D_KINEMATIC_CHAIN

FLOOR_COLOR = "#cfcfcf"
FLOOR_EDGE = "#8a8a8a"
GRID_COLOR = "#a0a0a0"

DEFAULT_VIEW_HALF = 2.0   # half-extent of the (square) XZ viewport, in meters
DEFAULT_GRID_SPACING = 1.0


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def _to_display(joints: np.ndarray) -> np.ndarray:
    """World → display frame. Negates X so character-right ends up on screen right.

    See module docstring for the right-handed (Y up, +Z forward) → +X-is-LEFT
    derivation that makes this flip necessary.
    """
    out = np.asarray(joints).copy()
    out[..., 0] = -out[..., 0]
    return out


def _world_extent(
    joints_list: List[np.ndarray],
    extra_pts: Optional[List[np.ndarray]] = None,
    xz_pad: float = 1.0,
    y_top_pad: float = 0.2,
    min_y_top: float = 2.0,
) -> dict:
    """Full motion bounding box (used to size the floor + grid)."""
    all_pts = [j.reshape(-1, 3) for j in joints_list if j is not None]
    if extra_pts:
        all_pts.extend(p.reshape(-1, 3) for p in extra_pts if p is not None)
    arr = np.concatenate(all_pts, axis=0)
    return {
        "x": (float(arr[:, 0].min()) - xz_pad, float(arr[:, 0].max()) + xz_pad),
        "z": (float(arr[:, 2].min()) - xz_pad, float(arr[:, 2].max()) + xz_pad),
        "y_top": max(min_y_top, float(arr[:, 1].max()) + y_top_pad),
    }


def _viewport_for_center(
    center_xz: Tuple[float, float],
    view_half: float,
    y_top: float,
) -> dict:
    """Square XZ viewport centered at ``center_xz`` with Y running 0 -> ``y_top``."""
    cx, cz = float(center_xz[0]), float(center_xz[1])
    return {
        "x": (cx - view_half, cx + view_half),
        "z": (cz - view_half, cz + view_half),
        "y_top": float(y_top),
    }


# -----------------------------------------------------------------------------
# Scene element drawers
# -----------------------------------------------------------------------------
def _draw_floor_and_grid(ax, extent: dict, spacing: float = DEFAULT_GRID_SPACING,
                         floor_alpha: float = 0.55):
    """Gray floor quad + world-coordinate grid lines spanning the full motion extent.

    Grid lines live in world coords; as the per-frame viewport scrolls, they
    visibly pass through the camera.
    """
    (xmin, xmax) = extent["x"]
    (zmin, zmax) = extent["z"]
    quad_world = np.array(
        [
            [xmin, 0.0, zmin],
            [xmax, 0.0, zmin],
            [xmax, 0.0, zmax],
            [xmin, 0.0, zmax],
        ],
        dtype=np.float32,
    )
    quad_plot = quad_world[:, [0, 2, 1]]  # world (x,y,z) -> plot (x,z,y)
    ax.add_collection3d(
        Poly3DCollection(
            [quad_plot], facecolors=FLOOR_COLOR, edgecolors="none",
            alpha=floor_alpha, zorder=0,
        )
    )
    # Grid lines (also at world y=0). Each line is a plot-coord segment.
    segs: List[List[List[float]]] = []
    z0 = np.floor(zmin / spacing) * spacing
    z1 = np.ceil(zmax / spacing) * spacing
    for z in np.arange(z0, z1 + 0.5 * spacing, spacing):
        if zmin <= z <= zmax:
            segs.append([[xmin, float(z), 0.0], [xmax, float(z), 0.0]])
    x0 = np.floor(xmin / spacing) * spacing
    x1 = np.ceil(xmax / spacing) * spacing
    for x in np.arange(x0, x1 + 0.5 * spacing, spacing):
        if xmin <= x <= xmax:
            segs.append([[float(x), zmin, 0.0], [float(x), zmax, 0.0]])
    if segs:
        ax.add_collection3d(
            Line3DCollection(
                segs, colors=GRID_COLOR, linewidths=0.7, alpha=0.85, zorder=1,
            )
        )


def _draw_trajectory(
    ax, root_world: np.ndarray, current_t: int, color: str,
    lw: float = 1.8, marker_size: float = 40.0,
):
    """Root XZ trajectory projected onto the floor (y=0), with a marker at frame t."""
    if root_world is None or len(root_world) == 0:
        return
    trail = root_world.copy()
    trail[:, 1] = 0.0
    # world (x, 0, z) -> plot (x, z, 0)
    ax.plot(trail[:, 0], trail[:, 2], trail[:, 1],
            color=color, linewidth=lw, alpha=0.9, zorder=2)
    ti = min(int(current_t), len(trail) - 1)
    ax.scatter(
        trail[ti, 0], trail[ti, 2], trail[ti, 1],
        c=color, s=marker_size, edgecolors="white", linewidths=1.0, zorder=6,
    )


def _draw_skel(ax, joints_xyz: np.ndarray, color: str, lw: float = 2.2, scatter_s: float = 18.0):
    # world (x, y, z) -> plot (x, z, y)
    ax.scatter(joints_xyz[:, 0], joints_xyz[:, 2], joints_xyz[:, 1],
               c=color, s=scatter_s, zorder=5)
    for chain in HML3D_KINEMATIC_CHAIN:
        ax.plot(
            joints_xyz[chain, 0], joints_xyz[chain, 2], joints_xyz[chain, 1],
            color=color, linewidth=lw, zorder=4,
        )


def _setup_axes(ax, viewport: dict, title: str, view_elev: float = 20.0, view_azim: float = -60.0):
    (xmin, xmax) = viewport["x"]
    (zmin, zmax) = viewport["z"]
    ymax = viewport["y_top"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(0.0, ymax)
    try:
        ax.set_box_aspect([xmax - xmin, zmax - zmin, ymax])
    except Exception:
        pass
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("z (forward)", fontsize=7)
    ax.set_zlabel("y (up)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=view_elev, azim=view_azim)


# -----------------------------------------------------------------------------
# Frame selection
# -----------------------------------------------------------------------------
def _frame_indices(T: int, max_frames: Optional[int], frame_stride: int) -> List[int]:
    idx = list(range(0, T, max(1, frame_stride)))
    if max_frames is not None and len(idx) > max_frames:
        idx = list(np.linspace(0, T - 1, max_frames, dtype=int))
    return idx


# -----------------------------------------------------------------------------
# Top-level: animated mp4-style frame stacks
# -----------------------------------------------------------------------------
def render_sidebyside(
    gt_joints: Optional[np.ndarray],  # (T, J, 3) or None for "no GT" (blank left panel)
    gen_joints: np.ndarray,
    caption: str = "",
    width: int = 1200,
    height: int = 600,
    dpi: int = 120,
    max_frames: Optional[int] = 200,
    frame_stride: int = 1,
    view_half: float = DEFAULT_VIEW_HALF,
    grid_spacing: float = DEFAULT_GRID_SPACING,
) -> np.ndarray:
    """GT (left, blue) vs generated (right, red), tracking-camera per skeleton.

    When ``gt_joints is None`` the left panel still gets the same floor +
    viewport as the right (so the layout / frame size stays consistent
    across items with and without GT), but no skeleton or trajectory is
    drawn and the title reads ``(no GT)``.

    Returns ``(T_render, H, W, 3)`` uint8 frames.
    """
    # Convert to display frame (negate world X) so character-right ends up
    # on screen right. See module docstring.
    gen_joints = _to_display(gen_joints)
    if gt_joints is not None:
        gt_joints = _to_display(gt_joints)
    T = gen_joints.shape[0] if gt_joints is None else min(gt_joints.shape[0], gen_joints.shape[0])
    idx = _frame_indices(T, max_frames, frame_stride)
    gen_trail = gen_joints[:T, 0]
    if gt_joints is None:
        gt_trail = gen_trail  # share viewport center so the empty-GT panel still tracks something sensible
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
                _draw_skel(ax_gt, gt_joints[t], "tab:blue")

            ax_gen = fig.add_subplot(1, 2, 2, projection="3d")
            _setup_axes(
                ax_gen,
                _viewport_for_center((gen_root[0], gen_root[2]), view_half, extent["y_top"]),
                f"generated  t={t}",
            )
            _draw_floor_and_grid(ax_gen, extent, spacing=grid_spacing)
            _draw_trajectory(ax_gen, gen_trail, t, "tab:red")
            _draw_skel(ax_gen, gen_joints[t], "tab:red")

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
    joints: np.ndarray,
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
    """One skeleton + its trail, tracking camera + world-grid floor."""
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
            _draw_skel(ax, joints[t], color)
            if caption:
                fig.suptitle(caption[:100], fontsize=10)
            fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.04)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(img)
    finally:
        plt.close(fig)
    return np.stack(frames, axis=0)


# -----------------------------------------------------------------------------
# Still-PNG summaries
# -----------------------------------------------------------------------------
def save_still_pair(
    gt_joints: np.ndarray,
    gen_joints: np.ndarray,
    out_path: Path,
    caption: str = "",
    n_frames: int = 6,
    dpi: int = 130,
    view_half: float = DEFAULT_VIEW_HALF,
    grid_spacing: float = DEFAULT_GRID_SPACING,
) -> None:
    """2-row, ``n_frames``-column still summary (GT top blue, gen bottom red).

    Each cell uses a per-frame tracking viewport; the world-coord grid is
    consistent across cells so locomotion reads clearly.
    """
    gt_joints = _to_display(gt_joints)
    gen_joints = _to_display(gen_joints)
    T = min(gt_joints.shape[0], gen_joints.shape[0])
    idx = np.linspace(0, T - 1, n_frames, dtype=int)
    gt_trail = gt_joints[:T, 0]
    gen_trail = gen_joints[:T, 0]
    extent = _world_extent(
        [gt_joints[:T], gen_joints[:T]],
        extra_pts=[gt_trail, gen_trail],
    )

    fig = plt.figure(figsize=(n_frames * 3.0, 7.0))
    for i, fi in enumerate(idx):
        gt_root = gt_trail[fi]
        gen_root = gen_trail[fi]

        ax_gt = fig.add_subplot(2, n_frames, i + 1, projection="3d")
        _setup_axes(
            ax_gt,
            _viewport_for_center((gt_root[0], gt_root[2]), view_half, extent["y_top"]),
            f"GT t={fi}",
        )
        _draw_floor_and_grid(ax_gt, extent, spacing=grid_spacing)
        _draw_trajectory(ax_gt, gt_trail, fi, "tab:blue")
        _draw_skel(ax_gt, gt_joints[fi], "tab:blue")

        ax_gen = fig.add_subplot(2, n_frames, n_frames + i + 1, projection="3d")
        _setup_axes(
            ax_gen,
            _viewport_for_center((gen_root[0], gen_root[2]), view_half, extent["y_top"]),
            f"gen t={fi}",
        )
        _draw_floor_and_grid(ax_gen, extent, spacing=grid_spacing)
        _draw_trajectory(ax_gen, gen_trail, fi, "tab:red")
        _draw_skel(ax_gen, gen_joints[fi], "tab:red")
    if caption:
        fig.suptitle(caption[:120], fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)


def save_still_single(
    joints: np.ndarray,
    out_path: Path,
    caption: str = "",
    color: str = "tab:red",
    n_frames: int = 6,
    dpi: int = 130,
    view_half: float = DEFAULT_VIEW_HALF,
    grid_spacing: float = DEFAULT_GRID_SPACING,
) -> None:
    joints = _to_display(joints)
    T = joints.shape[0]
    idx = np.linspace(0, T - 1, n_frames, dtype=int)
    trail = joints[:T, 0]
    extent = _world_extent([joints[:T]], extra_pts=[trail])

    fig = plt.figure(figsize=(n_frames * 3.0, 3.5))
    for i, fi in enumerate(idx):
        root = trail[fi]
        ax = fig.add_subplot(1, n_frames, i + 1, projection="3d")
        _setup_axes(
            ax,
            _viewport_for_center((root[0], root[2]), view_half, extent["y_top"]),
            f"t={fi}",
        )
        _draw_floor_and_grid(ax, extent, spacing=grid_spacing)
        _draw_trajectory(ax, trail, fi, color)
        _draw_skel(ax, joints[fi], color)
    if caption:
        fig.suptitle(caption[:120], fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)
