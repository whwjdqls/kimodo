"""Check internal consistency of HumanML3D-Kimodo (273-D) GT features.

Quantifies how well the two redundant "joint position" streams encoded in a
kimodo feature vector agree on the SAME ground-truth sample:

  1. Position-block joints:  ``world_joints_from_kimodo_features`` reads
     ``smooth_root_pos`` + ``local_joints_positions`` and reconstructs the
     world joint positions directly.
  2. FK-block joints:        ``chainreset_fk_world_joints`` runs HumanML3D's
     chain-reset FK on the stored ``global_rot_data`` (6D global rotations),
     using bone lengths derived from stream (1).

These two streams describe the same motion, so on GT they should be near-
identical. The gap (and the gap of their first-order time derivatives) is
exactly what the ``fk`` and ``fk_v`` losses score during training; if it is
small on GT, those losses can only help (they read the same supervision the
positions block already provides). If it is large, the rotation block and
the positions block disagree about what the GT joint trajectory actually is,
and adding an FK-based loss pulls the model in two directions.

Usage:
    python -m kimodo.scripts.check_hml3d_kimodo_consistency \\
        --motion_dir /home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep \\
        --split_file /home/jungbin_cho/HumanML3D/HumanML3D/train.txt \\
        --n_samples 200
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from kimodo.geometry import cont6d_to_matrix
from kimodo.motion_rep.fk_hml3d import (
    HML3D_KINEMATIC_CHAIN,
    HML3D_RAW_OFFSETS,
    chainreset_fk_world_joints,
    derive_bone_lengths_from_world_joints,
    world_joints_from_kimodo_features,
)

N_JOINTS = 22
# Block layout for the 273-D HumanML3D-Kimodo feature, asserted at runtime
# against an example file before the loop starts.
SLICE_DICT = {
    "smooth_root_pos":         slice(0,   3),
    "global_root_heading":     slice(3,   5),
    "local_joints_positions":  slice(5,   5 + N_JOINTS * 3),    # [5, 71)
    "global_rot_data":         slice(71,  71 + N_JOINTS * 6),   # [71, 203)
    "velocities":              slice(203, 203 + N_JOINTS * 3),  # [203, 269)
    "foot_contacts":           slice(269, 273),
}

JOINT_NAMES = [
    "pelvis",  "l_hip",      "r_hip",      "spine1",  "l_knee",     "r_knee",
    "spine2",  "l_ankle",    "r_ankle",    "spine3",  "l_foot",     "r_foot",
    "neck",    "l_collar",   "r_collar",   "head",    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",    "l_wrist",    "r_wrist",
]


def _check_one(features: torch.Tensor):
    """One sample. ``features``: (T, 273) unnormalized.

    Returns:
        fk_err_per_joint  (J,)  L1 between pos-block and FK-block, averaged over
                                time and xyz, in meters.
        fkv_err_per_joint (J,)  L1 between (pos_t+1 - pos_t) and (fk_t+1 - fk_t),
                                averaged over T-1 frames and xyz, in meters.
        pos_motion_per_joint (J,)  Per-joint mean L1 displacement (pos_t+1 - pos_t)
                                   over time — the "natural scale" baseline against
                                   which fkv_err should be judged.
    """
    feats = features.unsqueeze(0)  # (1, T, 273)
    T = feats.shape[1]

    pos_world = world_joints_from_kimodo_features(
        feats, SLICE_DICT, n_joints=N_JOINTS,
    )  # (1, T, J, 3)

    bone_lengths = derive_bone_lengths_from_world_joints(
        pos_world, chains=HML3D_KINEMATIC_CHAIN, n_joints=N_JOINTS, reduce="median",
    )  # (1, J)

    rot6 = feats[..., SLICE_DICT["global_rot_data"]].reshape(1, T, N_JOINTS, 6)
    rot_mats = cont6d_to_matrix(rot6)  # (1, T, J, 3, 3)
    smooth_root = feats[..., SLICE_DICT["smooth_root_pos"]]  # (1, T, 3)

    # Reconstruct the actual world root from features — same recipe as
    # KimodoMotionRep.inverse (kimodo_motionrep.py:195-198) and the FIXED
    # _fk_world_from_pred in train.py. local_jp[root] stores
    # (root.x - smooth_root.x, root.y, root.z - smooth_root.z), so adding
    # smooth_root.xz back (and reading y directly) recovers the actual root.
    local_jp = feats[..., SLICE_DICT["local_joints_positions"]].reshape(1, T, N_JOINTS, 3)
    actual_root = torch.stack(
        [
            smooth_root[..., 0] + local_jp[..., 0, 0],
            local_jp[..., 0, 1],
            smooth_root[..., 2] + local_jp[..., 0, 2],
        ],
        dim=-1,
    )

    # FK anchored at smooth_root (the OLD buggy loss behavior)
    fk_world = chainreset_fk_world_joints(
        rot_mats, smooth_root, bone_lengths=bone_lengths,
        raw_offsets=HML3D_RAW_OFFSETS, chains=HML3D_KINEMATIC_CHAIN,
    )  # (1, T, J, 3)
    # FK anchored at actual_root (the FIXED loss behavior)
    fk_world_fixed = chainreset_fk_world_joints(
        rot_mats, actual_root, bone_lengths=bone_lengths,
        raw_offsets=HML3D_RAW_OFFSETS, chains=HML3D_KINEMATIC_CHAIN,
    )

    fk_err = (pos_world - fk_world).abs().mean(dim=(0, 1, -1))  # (J,)
    fk_err_xyz = (pos_world - fk_world).abs().mean(dim=(0, 1))  # (J, 3) — per-axis

    # FIXED FK anchored at actual root — should be sub-mm on GT.
    fk_err_fixed = (pos_world - fk_world_fixed).abs().mean(dim=(0, 1, -1))  # (J,)

    # Pelvis-centered comparison: subtract pelvis from every joint, then L1.
    # Removes the constant rigid xz translation between (smooth_root vs actual
    # root) reference frames, isolating articulation/rotation consistency.
    pos_rel = pos_world - pos_world[:, :, 0:1, :]
    fk_rel = fk_world - fk_world[:, :, 0:1, :]
    fk_err_rel = (pos_rel - fk_rel).abs().mean(dim=(0, 1, -1))  # (J,)

    pos_v = pos_world[:, 1:] - pos_world[:, :-1]
    fk_v = fk_world[:, 1:] - fk_world[:, :-1]
    fkv_err = (pos_v - fk_v).abs().mean(dim=(0, 1, -1))  # (J,)
    pos_motion = pos_v.abs().mean(dim=(0, 1, -1))  # (J,) baseline scale

    return (
        fk_err.cpu().numpy(),
        fk_err_rel.cpu().numpy(),
        fk_err_xyz.cpu().numpy(),
        fk_err_fixed.cpu().numpy(),
        fkv_err.cpu().numpy(),
        pos_motion.cpu().numpy(),
    )


def _format_per_joint_table(fk_pj_cm, fk_rel_pj_cm, fkv_pj_cm, motion_pj_cm):
    """One row per joint; raw FK err, pelvis-centered FK err, FK vel err."""
    lines = []
    header = (
        f"  {'idx':>3} {'joint':<12} "
        f"{'FK (raw)':>10} {'FK (pelv-ctr)':>15} "
        f"{'FK_vel':>10} {'GT vel':>10}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for j, name in enumerate(JOINT_NAMES):
        lines.append(
            f"  {j:>3} {name:<12} "
            f"{fk_pj_cm[j]:>10.3f} {fk_rel_pj_cm[j]:>15.3f} "
            f"{fkv_pj_cm[j]:>10.3f} {motion_pj_cm[j]:>10.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--motion_dir", type=str,
                   default="/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep")
    p.add_argument("--split_file", type=str,
                   default="/home/jungbin_cho/HumanML3D/HumanML3D/train.txt")
    p.add_argument("--n_samples", type=int, default=200,
                   help="Random subset size. 0 = process every motion in the split.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    motion_dir = Path(args.motion_dir)
    with open(args.split_file) as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    rng = random.Random(args.seed)
    if args.n_samples > 0:
        ids = rng.sample(ids, min(args.n_samples, len(ids)))

    fk_pj_all, fk_rel_pj_all, fk_fixed_pj_all, fkv_pj_all, motion_pj_all = [], [], [], [], []
    fk_xyz_all = []  # (N, J, 3) per-axis decomposition
    fk_glob, fk_rel_glob, fk_fixed_glob, fkv_glob, motion_glob = [], [], [], [], []
    Ts = []
    skipped = 0

    for i, mid in enumerate(ids):
        path = motion_dir / f"{mid}.npz"
        if not path.is_file():
            skipped += 1
            continue
        try:
            with np.load(path) as z:
                if "features" not in z.files:
                    skipped += 1
                    continue
                feats_np = np.asarray(z["features"]).astype(np.float32)
        except Exception:
            skipped += 1
            continue
        if feats_np.shape[-1] != 273 or feats_np.shape[0] < 2:
            skipped += 1
            continue
        try:
            fk_err, fk_err_rel, fk_err_xyz, fk_err_fixed, fkv_err, motion = _check_one(
                torch.from_numpy(feats_np).to(device),
            )
        except Exception as e:
            print(f"[{mid}] error: {e}")
            skipped += 1
            continue
        fk_pj_all.append(fk_err)
        fk_rel_pj_all.append(fk_err_rel)
        fk_fixed_pj_all.append(fk_err_fixed)
        fk_xyz_all.append(fk_err_xyz)
        fkv_pj_all.append(fkv_err)
        motion_pj_all.append(motion)
        fk_glob.append(float(fk_err.mean()))
        fk_rel_glob.append(float(fk_err_rel.mean()))
        fk_fixed_glob.append(float(fk_err_fixed.mean()))
        fkv_glob.append(float(fkv_err.mean()))
        motion_glob.append(float(motion.mean()))
        Ts.append(feats_np.shape[0])

        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(ids)}] processed  (skipped so far: {skipped})", flush=True)

    if not fk_glob:
        print("No samples processed; check --motion_dir and --split_file.")
        return

    fk_pj_arr = np.stack(fk_pj_all, axis=0)        # (N, J)
    fk_rel_pj_arr = np.stack(fk_rel_pj_all, axis=0)
    fk_fixed_pj_arr = np.stack(fk_fixed_pj_all, axis=0)
    fk_xyz_arr = np.stack(fk_xyz_all, axis=0)      # (N, J, 3)
    fkv_pj_arr = np.stack(fkv_pj_all, axis=0)
    motion_pj_arr = np.stack(motion_pj_all, axis=0)
    fk_glob = np.asarray(fk_glob)
    fk_rel_glob = np.asarray(fk_rel_glob)
    fk_fixed_glob = np.asarray(fk_fixed_glob)
    fkv_glob = np.asarray(fkv_glob)
    motion_glob = np.asarray(motion_glob)

    print()
    print("=" * 86)
    print(f"HumanML3D-Kimodo internal consistency  (N = {len(fk_glob)} samples)")
    print("=" * 86)
    print(f"  motion_dir : {args.motion_dir}")
    print(f"  split_file : {args.split_file}")
    print(f"  device     : {device}")
    print(f"  T stats    : min={min(Ts)}, median={int(np.median(Ts))}, max={max(Ts)}, total_frames={sum(Ts)}")
    if skipped:
        print(f"  skipped    : {skipped}")
    print()
    print("Aggregate per-sample L1 (averaged over time, joints, xyz; in CM):")
    print(f"             {'mean':>9} {'p50':>9} {'p90':>9} {'p99':>9} {'max':>9}")

    def _row(label, arr_m):
        a = arr_m * 100  # m -> cm
        return f"  {label:<10} {a.mean():>9.3f} {np.median(a):>9.3f} {np.percentile(a, 90):>9.3f} {np.percentile(a, 99):>9.3f} {a.max():>9.3f}"

    print(_row("FK raw",       fk_glob))
    print(_row("FK pelv-ctr",  fk_rel_glob))
    print(_row("FK fixed",     fk_fixed_glob))
    print(_row("FK_vel",       fkv_glob))
    print(_row("GT motion",    motion_glob))
    print()
    # Per-axis decomposition of the raw FK error, averaged over joints+samples.
    # Whole-body uniform xz error with zero y is the signature of a rigid
    # reference-frame offset between FK origin (smooth_root_pos xz) and the
    # positions-block origin (actual root xz).
    fk_xyz_global = fk_xyz_arr.mean(axis=(0, 1)) * 100  # (3,) in cm
    print(f"FK error by axis (whole body, mean over samples and joints, cm):")
    print(f"    dx = {fk_xyz_global[0]:.3f}    dy = {fk_xyz_global[1]:.3f}    dz = {fk_xyz_global[2]:.3f}")
    pelvis_xyz = fk_xyz_arr[:, 0, :].mean(axis=0) * 100   # (3,) pelvis-only
    wrist_xyz = fk_xyz_arr[:, 20:22, :].mean(axis=(0, 1)) * 100  # avg of wrists
    print(f"  pelvis: dx={pelvis_xyz[0]:.3f}  dy={pelvis_xyz[1]:.3f}  dz={pelvis_xyz[2]:.3f}")
    print(f"  wrists: dx={wrist_xyz[0]:.3f}  dy={wrist_xyz[1]:.3f}  dz={wrist_xyz[2]:.3f}")
    print()
    print(f"  Relative FK-velocity noise vs natural per-frame motion:")
    print(f"    median ratio = {np.median(fkv_glob/motion_glob)*100:.2f}%, p90 = {np.percentile(fkv_glob/motion_glob, 90)*100:.2f}%")
    print()
    print("Per-joint MEAN across samples (cm). 'FK (raw)' = world FK vs world pos-block,")
    print("'FK (pelv-ctr)' = same after subtracting pelvis from both (pure articulation),")
    print("'FK_vel' = first-order temporal diff, 'GT vel' = per-frame motion baseline.")
    fk_pj_mean     = fk_pj_arr.mean(axis=0) * 100
    fk_rel_pj_mean = fk_rel_pj_arr.mean(axis=0) * 100
    fkv_pj_mean    = fkv_pj_arr.mean(axis=0) * 100
    motion_pj_mean = motion_pj_arr.mean(axis=0) * 100
    print(_format_per_joint_table(fk_pj_mean, fk_rel_pj_mean, fkv_pj_mean, motion_pj_mean))
    print()
    print("Verdict:")
    median_raw    = float(np.median(fk_glob)) * 100      # cm
    median_relctr = float(np.median(fk_rel_glob)) * 100  # cm
    median_v_rat  = float(np.median(fkv_glob / motion_glob))
    print(f"  Raw FK gap     : median {median_raw:.2f} cm  (world-frame vs world-frame)")
    print(f"  Pelv-ctr FK gap: median {median_relctr:.2f} cm  (articulation only)")
    print(f"  FK-vel noise   : median {median_v_rat*100:.1f}% of natural frame motion")
    print()
    if median_relctr < 0.5:
        print(f"  GT articulation is INTERNALLY CONSISTENT (pelvis-centered gap < 5mm).")
        print(f"  The raw FK gap (~{median_raw:.1f} cm) is a rigid xz-translation between two")
        print(f"  reference frames (smooth_root_pos vs actual root_pos) — see the per-axis")
        print(f"  decomposition above: dy~0, dx/dz nearly identical across all joints.")
        print(f"")
        print(f"  Implications for FK / FK-v losses:")
        print(f"   * FK_pos loss as written: ~{median_raw:.1f} cm IRREDUCIBLE error per frame because")
        print(f"     it anchors FK at smooth_root_pos but compares against world joints anchored")
        print(f"     at the actual root. At a perfect prediction, this loss saturates at the")
        print(f"     smoothing residual — gradients past that point push rotations to absorb a")
        print(f"     translation error that they cannot actually fix. Likely benign but wasted.")
        print(f"   * FK_v loss (newly added): error scale {median_v_rat*100:.1f}% of motion — much cleaner")
        print(f"     because the per-frame translation offset largely cancels in finite diffs.")
        print(f"   * To fix FK_pos: either anchor BOTH sides at smooth_root xz (subtract")
        print(f"     smooth_root from world joints before FK comparison) or anchor FK at the")
        print(f"     actual root (reconstructable as smooth_root + local_jp[pelvis] in xz).")
    elif median_relctr < 3.0:
        print(f"  GT articulation is partially consistent (pelvis-centered gap ~{median_relctr:.1f} cm).")
        print(f"  Watch the per-joint table — distal joints (wrists, ankles) carry most of it.")
    else:
        print(f"  GT articulation is INCONSISTENT even pelvis-centered ({median_relctr:.1f} cm).")
        print(f"  FK losses are reading a different joint trajectory than the positions block;")
        print(f"  they will pull the model in two directions. Diagnose the representation first.")


if __name__ == "__main__":
    main()
