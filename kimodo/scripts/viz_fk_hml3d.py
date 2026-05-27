# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visualize FK consistency for HumanML3D-in-kimodo motions.

Loads a HumanML3D-in-kimodo NPZ, then draws two skeletons side-by-side per
frame:
  1. ``GT positions``  — the stored ``posed_joints``
  2. ``FK from rot``   — world joints recomputed by running
                          ``chainreset_fk_world_joints`` on the stored
                          ``global_rot_mats`` (with the canonical
                          ``SMPLXSkeleton22`` T-pose offsets).

If the chain-reset FK is implemented correctly, the two skeletons match in
shape, joint count and topology; the only residual error comes from the fact
that HumanML3D uses per-actor bone lengths whereas we use the canonical
T-pose offsets, so wrist/foot positions can be a few centimeters apart.

Outputs an MP4 animation and (optionally) a 6-frame still PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from kimodo.motion_rep.fk_hml3d import (
    HML3D_KINEMATIC_CHAIN,
    chainreset_fk_world_joints,
    derive_bone_lengths_from_world_joints,
)
from kimodo.skeleton import SMPLXSkeleton22  # noqa: F401  (left for back-compat)


def _draw_skel(ax, joints_xyz: np.ndarray, color: str, label: str | None = None):
    ax.scatter(joints_xyz[:, 0], joints_xyz[:, 2], joints_xyz[:, 1], c=color, s=18)
    for chain in HML3D_KINEMATIC_CHAIN:
        xs = joints_xyz[chain, 0]
        zs = joints_xyz[chain, 2]
        ys = joints_xyz[chain, 1]
        ax.plot(xs, zs, ys, color=color, linewidth=2.0, label=label)
        label = None  # only first line gets a label


def _setup_axes(ax, center: np.ndarray, half_extent: float, title: str):
    ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
    ax.set_ylim(center[2] - half_extent, center[2] + half_extent)  # plot Z on Y-axis
    ax.set_zlim(0, 2 * half_extent)                                 # height
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y (up)")
    ax.set_title(title)
    ax.view_init(elev=15, azim=-70)


def visualize_pair(
    npz_path: str | Path,
    out_dir: str | Path,
    fps: int = 20,
    n_still_frames: int = 6,
) -> dict:
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    needed = {"global_rot_mats", "root_positions", "posed_joints"}
    missing = needed - set(data.files)
    if missing:
        raise KeyError(f"{npz_path} missing keys: {missing}")

    grm = torch.from_numpy(data["global_rot_mats"]).float()      # (T, 22, 3, 3)
    rpos = torch.from_numpy(data["root_positions"]).float()      # (T, 3)
    gt_posed_t = torch.from_numpy(data["posed_joints"]).float()  # (T, 22, 3)
    gt_posed = gt_posed_t.numpy().astype(np.float32)
    T = grm.shape[0]

    # HumanML3D-style FK needs per-sample bone lengths (raw_offset_dir * bone_len).
    bone_lengths = derive_bone_lengths_from_world_joints(gt_posed_t)             # (22,)
    fk_posed = chainreset_fk_world_joints(grm, rpos, bone_lengths).numpy().astype(np.float32)

    err = np.abs(fk_posed - gt_posed)
    err_stats = {
        "mean": float(err.mean()),
        "max": float(err.max()),
        "per_joint_max": err.max(axis=(0, 2)).tolist(),
    }
    print(
        f"FK vs GT: mean |err|={err_stats['mean']:.4f} m,  max |err|={err_stats['max']:.4f} m"
    )

    # Center the plot around the average root position so the camera follows the motion.
    root_mean = gt_posed[:, 0].mean(axis=0)
    half_extent = max(1.2, float(np.linalg.norm(gt_posed - root_mean, axis=-1).max()))

    # ---------- 6-frame still: GT, FK, overlay --------------------------------
    frame_idx = np.linspace(0, T - 1, n_still_frames, dtype=int)
    fig = plt.figure(figsize=(n_still_frames * 3.5, 10.5))
    for i, fi in enumerate(frame_idx):
        ax_gt = fig.add_subplot(3, n_still_frames, i + 1, projection="3d")
        _draw_skel(ax_gt, gt_posed[fi], color="tab:blue", label="GT")
        _setup_axes(ax_gt, root_mean, half_extent, f"GT t={fi}")
        ax_fk = fig.add_subplot(3, n_still_frames, n_still_frames + i + 1, projection="3d")
        _draw_skel(ax_fk, fk_posed[fi], color="tab:red", label="FK")
        _setup_axes(ax_fk, root_mean, half_extent, f"FK t={fi}")
        ax_ov = fig.add_subplot(3, n_still_frames, 2 * n_still_frames + i + 1, projection="3d")
        _draw_skel(ax_ov, gt_posed[fi], color="tab:blue", label="GT")
        _draw_skel(ax_ov, fk_posed[fi], color="tab:red", label="FK")
        if i == 0:
            ax_ov.legend(loc="upper left", fontsize=7)
        _setup_axes(ax_ov, root_mean, half_extent, f"overlay t={fi}")
    fig.suptitle(
        f"{npz_path.name}    FK vs GT — mean|err|={err_stats['mean']:.3f}m  max={err_stats['max']:.3f}m",
        fontsize=12,
    )
    fig.tight_layout()
    png_path = out_dir / f"{npz_path.stem}_fk_vs_gt.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"wrote {png_path}")

    # ---------- MP4 animation ---------------------------------------------------
    fig = plt.figure(figsize=(8, 4))
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_fk = fig.add_subplot(1, 2, 2, projection="3d")

    def _frame(t: int):
        ax_gt.cla()
        ax_fk.cla()
        _draw_skel(ax_gt, gt_posed[t], color="tab:blue")
        _setup_axes(ax_gt, root_mean, half_extent, f"GT positions  t={t}")
        _draw_skel(ax_fk, fk_posed[t], color="tab:red")
        _setup_axes(ax_fk, root_mean, half_extent,
                    f"FK from rotations  t={t}    err={np.abs(fk_posed[t]-gt_posed[t]).mean():.3f}m")
        return []

    anim = FuncAnimation(fig, _frame, frames=T, interval=1000 / fps, blit=False)
    mp4_path = out_dir / f"{npz_path.stem}_fk_vs_gt.mp4"
    try:
        anim.save(str(mp4_path), fps=fps, dpi=120, codec="h264")
        print(f"wrote {mp4_path}")
    except Exception as e:
        print(f"MP4 save failed ({e}); falling back to GIF")
        gif_path = mp4_path.with_suffix(".gif")
        anim.save(str(gif_path), fps=fps, dpi=100)
        print(f"wrote {gif_path}")
    plt.close(fig)

    return err_stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=str, help="Kimodo-format HumanML3D NPZ")
    p.add_argument("--out", type=str, default="/tmp/fk_viz", help="output directory")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--n-still-frames", type=int, default=6)
    args = p.parse_args()
    visualize_pair(args.npz, args.out, fps=args.fps, n_still_frames=args.n_still_frames)


if __name__ == "__main__":
    main()
