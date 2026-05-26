# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize a HumanML3D motion together with its Kimodo-rep reconstruction.

World coordinate convention
---------------------------
HumanML3D / SMPL data is **Y-up, +Z forward**. Matplotlib's mplot3d defaults to
Z-up. We bridge the two by remapping world (x, y, z) to plot (x, z, y) before
plotting, so the world's vertical axis (Y) becomes matplotlib's vertical axis.
Axis labels reflect the world meaning (X / Z forward / Y up), and the view
presets are picked in matplotlib coordinates accordingly.

What's drawn each frame
-----------------------
* Gray ground quad at world y = 0, large enough to contain the whole motion.
* Full root XZ trajectory for HumanML3D (joint 0), projected onto the floor.
* Full smoothed root XZ trajectory for Kimodo (from `smooth_root_pos`), also
  projected onto the floor.
* A circle marker at the *current* frame position on each trajectory.
* The current-frame skeleton(s): HumanML3D in blue, Kimodo in orange.

Camera is **fixed** in world coordinates (axes limits computed once across the
whole clip), so the character physically translates across the frame — root
motion is no longer hidden by axis tracking.

Modes
-----
  --mode overlay  (default) both skeletons + both trajectories in one axes.
  --mode side     two subplots side by side (left = HumanML3D, right = Kimodo).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# HumanML3D's older code uses np.float / np.int — shim them before importing.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

import matplotlib

matplotlib.use("Agg")  # headless rendering — required for cluster nodes.
import imageio.v3 as iio
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from tqdm import tqdm

# Locate HumanML3D's paramUtil for the SMPL-22 kinematic chain.
_HML3D_ROOT = Path("/home/jungbin_cho/HumanML3D")
if str(_HML3D_ROOT) not in sys.path:
    sys.path.insert(0, str(_HML3D_ROOT))
import paramUtil  # type: ignore  # noqa: E402

# Reuse `recover_from_ric` from the converter so HumanML3D joints come out
# identical to how they were computed during the kimodo_rep conversion.
_BENCH_ROOT = Path("/home/jungbin_cho/kimodo/benchmark")
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
from humanml3d_to_kimodo import recover_from_ric  # noqa: E402

KINEMATIC_CHAIN: list[list[int]] = paramUtil.t2m_kinematic_chain

# View presets, defined in **matplotlib-plot coords (z-up)** after our
# (x_world, y_world, z_world) -> (x_plot, z_world, y_world) remap.
# 'front' looks at the character from world +Z (which is plot +Y).
VIEW_PRESETS = {
    "general": dict(elev=20, azim=-60),
    "top": dict(elev=89, azim=-90),
    "front": dict(elev=8, azim=90),
}

COLORS = {
    "hml3d": dict(bone="#1f77b4", joint="#0d3a5e", trail="#1f77b4", label="HumanML3D"),
    "kimodo": dict(bone="#ff7f0e", joint="#7a3a04", trail="#ff7f0e", label="Kimodo (smooth root)"),
}

FLOOR_COLOR = "#bcbcbc"
FLOOR_EDGE = "#9a9a9a"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------


def load_hml3d_joints(path: Path) -> np.ndarray:
    arr = np.load(path)
    data = torch.from_numpy(arr).float()
    joints = recover_from_ric(data, 22).numpy().astype(np.float32)
    return joints  # (T, 22, 3) world (X, Y-up, Z-forward)


def load_kimodo_data(path: Path) -> dict:
    """Return dict with `posed_joints` (T, 22, 3) and `smooth_root_pos` (T, 3) if present."""
    out: dict = {}
    with np.load(path) as z:
        out["posed_joints"] = z["posed_joints"].astype(np.float32)
        if "smooth_root_pos" in z.files:
            out["smooth_root_pos"] = z["smooth_root_pos"].astype(np.float32)
    return out


# ----------------------------------------------------------------------------
# Coordinate remap: world (X, Y-up, Z) -> plot (X, Z, Y) so matplotlib's z-up
# convention aligns with the data's Y-up convention.
# ----------------------------------------------------------------------------


def _world_to_plot(pts: np.ndarray) -> np.ndarray:
    """Swap the last two axes of (..., 3). World (X, Y, Z) -> plot (X, Z, Y)."""
    return pts[..., [0, 2, 1]]


# ----------------------------------------------------------------------------
# Scene element drawers
# ----------------------------------------------------------------------------


def _draw_floor(ax, world_bounds):
    """Draw a gray quad at world y=0 covering the XZ bounds."""
    (xmin, xmax), _, (zmin, zmax) = world_bounds
    # World quad at y=0
    quad_world = np.array(
        [
            [xmin, 0.0, zmin],
            [xmax, 0.0, zmin],
            [xmax, 0.0, zmax],
            [xmin, 0.0, zmax],
        ],
        dtype=np.float32,
    )
    quad_plot = _world_to_plot(quad_world)
    poly = Poly3DCollection(
        [quad_plot],
        facecolors=FLOOR_COLOR,
        edgecolors=FLOOR_EDGE,
        linewidths=0.8,
        alpha=0.45,
        zorder=0,
    )
    ax.add_collection3d(poly)


def _draw_trajectory(
    ax,
    trail_world: np.ndarray,
    current_t: int,
    color: str,
    lw: float = 1.8,
    marker_size: float = 35.0,
):
    """Draw a root XZ trajectory projected to the floor (y=0), with a marker at frame t."""
    if trail_world is None or len(trail_world) == 0:
        return
    trail = trail_world.copy()
    trail[:, 1] = 0.0  # project onto the floor
    trail_plot = _world_to_plot(trail)
    ax.plot(
        trail_plot[:, 0], trail_plot[:, 1], trail_plot[:, 2],
        color=color, linewidth=lw, alpha=0.85, zorder=2,
    )
    ti = min(current_t, len(trail) - 1)
    ax.scatter(
        trail_plot[ti, 0], trail_plot[ti, 1], trail_plot[ti, 2],
        c=color, s=marker_size, edgecolors="white", linewidths=1.0, zorder=6,
    )


def _draw_skeleton(
    ax,
    joints_t_world: np.ndarray,
    color_bone: str,
    color_joint: str,
    lw: float = 2.4,
    joint_size: float = 16.0,
    alpha: float = 1.0,
):
    j = _world_to_plot(joints_t_world)
    ax.scatter(
        j[:, 0], j[:, 1], j[:, 2],
        c=color_joint, s=joint_size, zorder=5, alpha=alpha, edgecolors="none",
    )
    segs = []
    for chain in KINEMATIC_CHAIN:
        for k in range(len(chain) - 1):
            segs.append([j[chain[k]], j[chain[k + 1]]])
    lc = Line3DCollection(segs, colors=color_bone, linewidths=lw, alpha=alpha, zorder=4)
    ax.add_collection3d(lc)


# ----------------------------------------------------------------------------
# Bounds + axes
# ----------------------------------------------------------------------------


def _compute_world_bounds(
    joints_list: list[np.ndarray],
    extra_pts: list[np.ndarray] | None = None,
    xz_pad: float = 0.6,
    y_top_pad: float = 0.2,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute (xrange, yrange, zrange) covering all joints + any extra (e.g. trajectories)."""
    all_pts = [j.reshape(-1, 3) for j in joints_list if j is not None]
    if extra_pts:
        all_pts.extend(p.reshape(-1, 3) for p in extra_pts if p is not None)
    arr = np.concatenate(all_pts, axis=0)
    xmin, xmax = float(arr[:, 0].min()) - xz_pad, float(arr[:, 0].max()) + xz_pad
    ymax = float(arr[:, 1].max()) + y_top_pad
    zmin, zmax = float(arr[:, 2].min()) - xz_pad, float(arr[:, 2].max()) + xz_pad
    # Ensure a minimum extent so a stationary motion doesn't collapse to a point.
    if (xmax - xmin) < 1.5:
        cx = 0.5 * (xmin + xmax)
        xmin, xmax = cx - 0.75, cx + 0.75
    if (zmax - zmin) < 1.5:
        cz = 0.5 * (zmin + zmax)
        zmin, zmax = cz - 0.75, cz + 0.75
    if ymax < 2.0:
        ymax = 2.0
    return (xmin, xmax), (0.0, ymax), (zmin, zmax)


def _setup_axes(ax, world_bounds, view: str, show_axes: bool):
    """Configure the axes in plot coordinates (after world->plot swap)."""
    (xmin, xmax), (ymin_world, ymax_world), (zmin, zmax) = world_bounds
    # Plot coords: x_plot = x_world, y_plot = z_world, z_plot = y_world.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(ymin_world, ymax_world)
    preset = VIEW_PRESETS[view]
    ax.view_init(elev=preset["elev"], azim=preset["azim"])
    try:
        ax.set_box_aspect([xmax - xmin, zmax - zmin, ymax_world - ymin_world])
    except Exception:
        pass
    if show_axes:
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Z (forward)", fontsize=8)
        ax.set_zlabel("Y (up)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
    else:
        ax.set_axis_off()


def _fig_to_rgb(fig, target_size_px: tuple[int, int]) -> np.ndarray:
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    tw, th = target_size_px
    h, w = img.shape[:2]
    if (w, h) != (tw, th):
        out = np.full((th, tw, 3), 255, dtype=img.dtype)
        cw = min(w, tw)
        ch = min(h, th)
        out[:ch, :cw] = img[:ch, :cw]
        img = out
    return img


# ----------------------------------------------------------------------------
# Per-frame renderers
# ----------------------------------------------------------------------------


def render_overlay_frame(
    joints_hml: np.ndarray,
    joints_kim: np.ndarray | None,
    t: int,
    trail_hml: np.ndarray | None,
    trail_kim: np.ndarray | None,
    world_bounds,
    view: str,
    fig_size_inches: tuple[float, float],
    dpi: int,
    show_axes: bool,
    target_size_px: tuple[int, int],
) -> np.ndarray:
    fig = plt.figure(figsize=fig_size_inches, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    _setup_axes(ax, world_bounds, view, show_axes)

    _draw_floor(ax, world_bounds)
    if trail_hml is not None:
        _draw_trajectory(ax, trail_hml, t, COLORS["hml3d"]["trail"], lw=1.4)
    if trail_kim is not None:
        _draw_trajectory(ax, trail_kim, t, COLORS["kimodo"]["trail"], lw=1.8)

    _draw_skeleton(
        ax, joints_hml[t],
        COLORS["hml3d"]["bone"], COLORS["hml3d"]["joint"], alpha=0.95, lw=2.4,
    )
    if joints_kim is not None:
        _draw_skeleton(
            ax, joints_kim[t],
            COLORS["kimodo"]["bone"], COLORS["kimodo"]["joint"], alpha=0.55, lw=3.5,
        )
    handles = [
        Line2D([0], [0], color=COLORS["hml3d"]["bone"], lw=3, label=COLORS["hml3d"]["label"]),
        Line2D([0], [0], color=COLORS["kimodo"]["bone"], lw=3, label=COLORS["kimodo"]["label"]),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.85)

    fig.tight_layout(pad=0.4)
    img = _fig_to_rgb(fig, target_size_px=target_size_px)
    plt.close(fig)
    return img


def render_side_frame(
    joints_hml: np.ndarray,
    joints_kim: np.ndarray,
    t: int,
    trail_hml: np.ndarray | None,
    trail_kim: np.ndarray | None,
    world_bounds,
    view: str,
    fig_size_inches: tuple[float, float],
    dpi: int,
    show_axes: bool,
    target_size_px: tuple[int, int],
) -> np.ndarray:
    fig = plt.figure(figsize=fig_size_inches, dpi=dpi)
    ax_l = fig.add_subplot(121, projection="3d")
    ax_r = fig.add_subplot(122, projection="3d")
    for ax in (ax_l, ax_r):
        _setup_axes(ax, world_bounds, view, show_axes)
        _draw_floor(ax, world_bounds)

    ax_l.set_title("HumanML3D", fontsize=11, color=COLORS["hml3d"]["bone"])
    ax_r.set_title("Kimodo", fontsize=11, color=COLORS["kimodo"]["bone"])

    if trail_hml is not None:
        _draw_trajectory(ax_l, trail_hml, t, COLORS["hml3d"]["trail"], lw=1.6)
    if trail_kim is not None:
        _draw_trajectory(ax_r, trail_kim, t, COLORS["kimodo"]["trail"], lw=1.8)

    _draw_skeleton(ax_l, joints_hml[t], COLORS["hml3d"]["bone"], COLORS["hml3d"]["joint"])
    _draw_skeleton(ax_r, joints_kim[t], COLORS["kimodo"]["bone"], COLORS["kimodo"]["joint"])

    fig.tight_layout(pad=0.4)
    img = _fig_to_rgb(fig, target_size_px=target_size_px)
    plt.close(fig)
    return img


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _resolve_hml_path(input_arg: Path, hml_dir: Path) -> Path:
    if input_arg.is_file():
        return input_arg
    candidate = hml_dir / input_arg.name
    if candidate.suffix != ".npy":
        candidate = candidate.with_suffix(".npy")
    return candidate


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "input",
        type=Path,
        help="Either a full path to a HumanML3D .npy or a bare stem like '000005'.",
    )
    ap.add_argument(
        "--hml3d-dir",
        type=Path,
        default=Path("/home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs"),
        help="Folder of HumanML3D .npy files (used if `input` isn't an existing path).",
    )
    ap.add_argument(
        "--kimodo-dir",
        type=Path,
        default=Path("/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep"),
        help="Folder with the converted Kimodo-rep .npz files.",
    )
    ap.add_argument(
        "--mode", choices=["overlay", "side"], default="overlay",
        help="overlay = both skeletons in one axes (default), side = two subplots.",
    )
    ap.add_argument(
        "--view",
        default="general,top,front",
        help="Comma list of view names. One MP4 per view is written.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir, or *.mp4 path for single-view runs. "
             "Default: /home/jungbin_cho/HumanML3D/HumanML3D/viz/",
    )
    ap.add_argument("--fps", type=int, default=20, help="Playback fps (default 20).")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--xz-pad", type=float, default=0.6,
                    help="Padding (m) around the XZ bounding box of joints/trajectories.")
    ap.add_argument("--no-axes", action="store_true", help="Hide axis ticks/labels (cleaner shots).")
    ap.add_argument("--no-trail", action="store_true", help="Disable root trajectory traces.")
    args = ap.parse_args()

    hml_path = _resolve_hml_path(args.input, args.hml3d_dir)
    if not hml_path.is_file():
        raise SystemExit(f"HumanML3D file not found: {hml_path}")
    name = hml_path.stem
    kim_path = args.kimodo_dir / f"{name}.npz"

    print(f"Loading HumanML3D: {hml_path}")
    joints_hml = load_hml3d_joints(hml_path)
    joints_kim: np.ndarray | None = None
    smooth_root: np.ndarray | None = None
    if kim_path.is_file():
        print(f"Loading Kimodo:    {kim_path}")
        kimodo = load_kimodo_data(kim_path)
        joints_kim = kimodo["posed_joints"]
        smooth_root = kimodo.get("smooth_root_pos")
        if joints_kim.shape != joints_hml.shape:
            raise SystemExit(
                f"Shape mismatch: hml3d {joints_hml.shape} vs kimodo {joints_kim.shape}"
            )
    else:
        print(f"  (no Kimodo file at {kim_path} — drawing HumanML3D only)")

    T = joints_hml.shape[0]
    print(f"  T={T} frames, fps={args.fps}, mode={args.mode}")

    views = [v.strip() for v in args.view.split(",") if v.strip()]
    for v in views:
        if v not in VIEW_PRESETS:
            raise SystemExit(f"Unknown view '{v}'; choose from {sorted(VIEW_PRESETS)}.")

    # --- Trajectories ---
    trail_hml = joints_hml[:, 0, :] if not args.no_trail else None  # (T, 3) root over time
    trail_kim = smooth_root if (smooth_root is not None and not args.no_trail) else None

    # --- World bounds (fixed across the whole clip) ---
    joints_list: list[np.ndarray] = [joints_hml]
    if joints_kim is not None:
        joints_list.append(joints_kim)
    extra_pts: list[np.ndarray] = []
    if trail_hml is not None:
        extra_pts.append(trail_hml)
    if trail_kim is not None:
        extra_pts.append(trail_kim)
    world_bounds = _compute_world_bounds(joints_list, extra_pts, xz_pad=args.xz_pad)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = world_bounds
    print(f"  world bounds: X[{xmin:.2f},{xmax:.2f}] Y[{ymin:.2f},{ymax:.2f}] Z[{zmin:.2f},{zmax:.2f}]")

    # --- Output destination ---
    output_arg: Path | None = args.output
    if output_arg is None:
        out_dir = Path("/home/jungbin_cho/HumanML3D/HumanML3D/viz")
        out_dir.mkdir(parents=True, exist_ok=True)
        single_path_out: Path | None = None
    elif output_arg.suffix.lower() == ".mp4" and len(views) == 1:
        out_dir = output_arg.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        single_path_out = output_arg
    else:
        out_dir = output_arg
        out_dir.mkdir(parents=True, exist_ok=True)
        single_path_out = None

    # --- Figure sizing (h264 wants even pixel dims) ---
    target_w = (args.width // 2) * 2
    target_h = (args.height // 2) * 2
    dpi = 80
    if args.mode == "side":
        fig_size = (target_w / dpi * 1.6, target_h / dpi)
        out_w = (int(round(fig_size[0] * dpi)) // 2) * 2
        out_h = target_h
    else:
        fig_size = (target_w / dpi, target_h / dpi)
        out_w, out_h = target_w, target_h
    target_size_px = (out_w, out_h)

    for view in views:
        print(f"  rendering view: {view}")
        frames: list[np.ndarray] = []
        for t in tqdm(range(T), desc=view, leave=False):
            if args.mode == "side" and joints_kim is not None:
                img = render_side_frame(
                    joints_hml, joints_kim, t,
                    trail_hml, trail_kim,
                    world_bounds, view,
                    fig_size, dpi, not args.no_axes, target_size_px,
                )
            else:
                img = render_overlay_frame(
                    joints_hml, joints_kim, t,
                    trail_hml, trail_kim,
                    world_bounds, view,
                    fig_size, dpi, not args.no_axes, target_size_px,
                )
            frames.append(img)

        if single_path_out is not None:
            out_path = single_path_out
        else:
            out_path = out_dir / f"{name}_{args.mode}_{view}.mp4"
        iio.imwrite(
            str(out_path), np.stack(frames, axis=0),
            fps=float(args.fps), codec="h264", plugin="pyav",
        )
        print(f"  wrote {out_path}  ({len(frames)} frames @ {args.fps} fps, {out_w}x{out_h})")


if __name__ == "__main__":
    main()
