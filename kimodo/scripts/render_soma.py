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

# Constraint-marker styling (used by the text+constraints viz).
_SPHERE_COLOR = "#9b30ff"        # purple sphere — keyframe joint / waypoint targets
_SPHERE_ACTIVE_EDGE = "#ffd000"  # gold ring on the target active at the current frame
_ROOT_PATH_COLOR = "#19c819"     # green — dense 2D root-path target on the floor


def _to_display(joints: np.ndarray) -> np.ndarray:
    """World → display frame. Negates X so character-right ends up on screen right.

    See module docstring for the right-handed (Y up, +Z forward) → +X-is-LEFT
    derivation that makes this flip necessary.
    """
    out = np.asarray(joints).copy()
    out[..., 0] = -out[..., 0]
    return out


def _fixed_viewport(extent: dict, margin: float = 0.0) -> dict:
    """Single static square XZ viewport covering the whole motion extent.

    Unlike :func:`_viewport_for_center` (which recenters on the root every
    frame, so the floor/grid appear to slide under a stationary character),
    this returns one viewport for the entire clip: the floor + grid stay put
    and the character visibly translates across them. ``extent`` already
    carries an XZ pad, so ``margin`` defaults to 0.
    """
    (xmin, xmax) = extent["x"]
    (zmin, zmax) = extent["z"]
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)
    half = 0.5 * max(xmax - xmin, zmax - zmin) + float(margin)
    return {
        "x": (cx - half, cx + half),
        "z": (cz - half, cz + half),
        "y_top": float(extent["y_top"]),
    }


def _draw_skel(
    ax,
    joints_xyz: np.ndarray,
    joint_parents: Sequence[int],
    color: str,
    lw: float = 2.2,
    scatter_s: float = 18.0,
    skip_joints: Optional[Sequence[int]] = None,
) -> None:
    """Stick figure: one line segment per (parent, child) pair from ``joint_parents``.

    ``skip_joints`` (joint indices) are omitted entirely — neither scattered
    nor connected by a bone — so clutter joints like the SOMASkeleton30
    fingertip ends can be dropped from the training viz.
    """
    skip = {int(j) for j in skip_joints} if skip_joints else set()
    keep = [j for j in range(joints_xyz.shape[0]) if j not in skip]
    ax.scatter(
        joints_xyz[keep, 0], joints_xyz[keep, 2], joints_xyz[keep, 1],
        c=color, s=scatter_s, zorder=5,
    )
    for child, parent in enumerate(joint_parents):
        p = int(parent)
        if p < 0 or child in skip or p in skip:
            continue
        ax.plot(
            [joints_xyz[p, 0], joints_xyz[child, 0]],
            [joints_xyz[p, 2], joints_xyz[child, 2]],
            [joints_xyz[p, 1], joints_xyz[child, 1]],
            color=color, linewidth=lw, zorder=4,
        )


def _draw_constraint_markers(ax, cons_display: dict, active_frame: Optional[int] = None) -> None:
    """Overlay constraint targets (already in the display frame) on a panel.

    ``cons_display`` keys:
      - ``spheres`` (K, 3): purple-sphere targets (full-body / end-effector
        keyframe joints + sparse 2D root waypoints).
      - ``sphere_frames`` (K,): the (display-snapped) frame each sphere is
        conditioned on. When ``active_frame`` matches, that sphere is enlarged
        and gold-ringed so you can read whether the skeleton lands on it.
      - ``root_path`` (P, 3): a *dense* 2D root path, drawn as a green polyline
        on the floor (y=0).
    Targets are static in the world and persist across all frames; the moving
    skeleton is what should pass through them.
    """
    pts = cons_display.get("spheres")
    if pts is not None and len(pts):
        pts = np.asarray(pts)
        frames = cons_display.get("sphere_frames")
        if frames is not None and active_frame is not None and len(frames) == len(pts):
            frames = np.asarray(frames)
            act = frames == int(active_frame)
        else:
            act = np.zeros(len(pts), dtype=bool)
        if (~act).any():
            ax.scatter(
                pts[~act, 0], pts[~act, 2], pts[~act, 1],
                c=_SPHERE_COLOR, marker="o", s=70, alpha=0.8,
                edgecolors="none", zorder=7, depthshade=True,
            )
        if act.any():
            ax.scatter(
                pts[act, 0], pts[act, 2], pts[act, 1],
                c=_SPHERE_COLOR, marker="o", s=210, alpha=1.0,
                edgecolors=_SPHERE_ACTIVE_EDGE, linewidths=2.0,
                zorder=8, depthshade=False,
            )
    rp = cons_display.get("root_path")
    if rp is not None and len(rp):
        floor = np.asarray(rp).copy()
        floor[:, 1] = 0.0
        ax.plot(
            floor[:, 0], floor[:, 2], floor[:, 1],
            color=_ROOT_PATH_COLOR, linewidth=2.6, alpha=0.95, zorder=3,
        )
        ax.scatter(
            floor[:, 0], floor[:, 2], floor[:, 1],
            c=_ROOT_PATH_COLOR, s=14, zorder=4, depthshade=False,
        )


def _constraints_to_display(cons: Optional[dict]) -> Optional[dict]:
    """Negate world-X on a constraint dict's point arrays (see :func:`_to_display`)."""
    if cons is None:
        return None
    out = dict(cons)
    for k in ("spheres", "root_path"):
        v = cons.get(k)
        if v is not None and len(v):
            out[k] = _to_display(np.asarray(v))
    return out


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
    skip_joints: Optional[Sequence[int]] = None,
    camera: str = "follow",
    gt_constraints: Optional[dict] = None,
    gen_constraints: Optional[dict] = None,
    pause_repeat: int = 0,
) -> np.ndarray:
    """GT (left, blue) vs generated (right, red). Returns ``(T, H, W, 3)`` uint8.

    When ``gt_joints is None`` the left panel still gets the same floor +
    viewport as the right (so the layout stays consistent), but no skeleton
    or trajectory is drawn, and the title reads ``(no GT)``.

    ``camera``: ``"follow"`` (default) is the legacy per-frame root-tracking
    viewport (floor/grid scroll past a centered character); ``"fixed"`` frames
    the whole clip with one static viewport. ``skip_joints`` drops clutter
    joints (e.g. fingertip ends). ``gt_constraints`` / ``gen_constraints``
    overlay constraint targets (see :func:`_draw_constraint_markers`); the
    target at the current frame is highlighted. ``pause_repeat`` holds each
    conditioned keyframe for that many extra frames so a viewer can see whether
    the skeleton lands on the target.
    """
    # Convert to display frame (negate world X) so character-right ends up
    # on screen right. See module docstring.
    gen_joints = _to_display(gen_joints)
    if gt_joints is not None:
        gt_joints = _to_display(gt_joints)
    gt_cons = _constraints_to_display(gt_constraints)
    gen_cons = _constraints_to_display(gen_constraints)
    if gt_joints is None:
        T = gen_joints.shape[0]
    else:
        T = min(gt_joints.shape[0], gen_joints.shape[0])
    idx = _frame_indices(T, max_frames, frame_stride)
    gen_trail = gen_joints[:T, 0]
    # Fold constraint markers into the extent so the viewport contains them
    # even if a target sits just outside the skeleton's bbox.
    extra = []
    for cons in (gt_cons, gen_cons):
        if cons is None:
            continue
        for k in ("spheres", "root_path"):
            v = cons.get(k)
            if v is not None and len(v):
                extra.append(np.asarray(v))
    if gt_joints is None:
        gt_trail = gen_trail
        extent = _world_extent([gen_joints[:T]], extra_pts=[gen_trail, *extra])
    else:
        gt_trail = gt_joints[:T, 0]
        extent = _world_extent(
            [gt_joints[:T], gen_joints[:T]],
            extra_pts=[gt_trail, gen_trail, *extra],
        )
    fixed_vp = _fixed_viewport(extent) if camera == "fixed" else None

    def _vp(root_xz):
        return fixed_vp if fixed_vp is not None else _viewport_for_center(
            (root_xz[0], root_xz[2]), view_half, extent["y_top"],
        )

    # Snap each conditioned frame to the nearest *rendered* frame (idx may be
    # subsampled), so highlight + pause land on a frame we actually draw.
    idx_arr = np.asarray(idx)

    def _snap_frames(cons):
        if cons is None:
            return
        f = cons.get("sphere_frames")
        if f is not None and len(f):
            cons["sphere_frames"] = np.asarray(
                [int(idx_arr[np.argmin(np.abs(idx_arr - int(v)))]) for v in f]
            )

    _snap_frames(gt_cons)
    _snap_frames(gen_cons)
    pause_at = set()
    if pause_repeat > 0:
        for cons in (gt_cons, gen_cons):
            for v in ((cons or {}).get("pause_frames") or []):
                pause_at.add(int(idx_arr[np.argmin(np.abs(idx_arr - int(v)))]))

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    frames: List[np.ndarray] = []
    try:
        for t in idx:
            fig.clf()

            ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
            _setup_axes(
                ax_gt, _vp(gt_trail[t]),
                (f"GT  t={t}" if gt_joints is not None else "(no GT)"),
            )
            _draw_floor_and_grid(ax_gt, extent, spacing=grid_spacing)
            if gt_cons is not None:
                _draw_constraint_markers(ax_gt, gt_cons, active_frame=t)
            if gt_joints is not None:
                _draw_trajectory(ax_gt, gt_trail, t, "tab:blue")
                _draw_skel(ax_gt, gt_joints[t], joint_parents, "tab:blue", skip_joints=skip_joints)

            ax_gen = fig.add_subplot(1, 2, 2, projection="3d")
            _setup_axes(
                ax_gen, _vp(gen_trail[t]),
                f"generated  t={t}",
            )
            _draw_floor_and_grid(ax_gen, extent, spacing=grid_spacing)
            if gen_cons is not None:
                _draw_constraint_markers(ax_gen, gen_cons, active_frame=t)
            _draw_trajectory(ax_gen, gen_trail, t, "tab:red")
            _draw_skel(ax_gen, gen_joints[t], joint_parents, "tab:red", skip_joints=skip_joints)

            if caption:
                fig.suptitle(caption[:100], fontsize=10)
            fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.04, wspace=0.08)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(img)
            if t in pause_at:
                frames.extend(img for _ in range(int(pause_repeat)))
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
    skip_joints: Optional[Sequence[int]] = None,
    camera: str = "follow",
    constraints: Optional[dict] = None,
) -> np.ndarray:
    """One skeleton + its trail with a tracking (or static) camera and world-grid floor."""
    joints = _to_display(joints)
    cons = _constraints_to_display(constraints)
    T = joints.shape[0]
    idx = _frame_indices(T, max_frames, frame_stride)
    trail = joints[:T, 0]
    extra = [trail]
    if cons is not None:
        for k in ("spheres", "root_path"):
            v = cons.get(k)
            if v is not None and len(v):
                extra.append(np.asarray(v))
    extent = _world_extent([joints[:T]], extra_pts=extra)
    fixed_vp = _fixed_viewport(extent) if camera == "fixed" else None

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    frames: List[np.ndarray] = []
    try:
        for t in idx:
            fig.clf()
            root = trail[t]
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            vp = fixed_vp if fixed_vp is not None else _viewport_for_center(
                (root[0], root[2]), view_half, extent["y_top"],
            )
            _setup_axes(ax, vp, f"t={t}")
            _draw_floor_and_grid(ax, extent, spacing=grid_spacing)
            if cons is not None:
                _draw_constraint_markers(ax, cons)
            _draw_trajectory(ax, trail, t, color)
            _draw_skel(ax, joints[t], joint_parents, color, skip_joints=skip_joints)
            if caption:
                fig.suptitle(caption[:100], fontsize=10)
            fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.04)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(img)
    finally:
        plt.close(fig)
    return np.stack(frames, axis=0)
