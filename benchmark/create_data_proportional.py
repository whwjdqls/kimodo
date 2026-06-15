# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shape-aware variant of create_data.py.

Walks /weka/jungbin/seed/soma_proportional/bvh and writes one Kimodo motion NPZ
per BVH, mirroring the input subdirectory tree under the output root.

Diff vs the uniform create_data.py:
  - DEFAULT_DATASET and DEFAULT_OUTPUT point at the proportional tree.
  - parse_bvh_motion(bvh_path) is still used for the animation (rotations +
    root translation). We do NOT call parse_neutral_joints=True (its naive
    world-frame OFFSET walk gives wrong joint positions for bone-axis BVHs).
    Instead we read HIERARCHY OFFSETs with the Bvh class and convert them
    into kimodo's Y-up frame using kimodo's own global_rot_offsets, via
    build_actor_neutrals_in_kimodo_frame().
  - skeleton.neutral_joints is *overridden* per-file with the actor's
    rest-pose joint positions. All downstream FK / canonicalization runs
    against the actor's body.
  - Saved NPZ gains a `neutral_joints` (77, 3) tensor so consumers know
    the body they were rendered against.

Training note: the saved motion features (KimodoMotionRep output) do NOT
encode body shape themselves. For shape-aware generation, your dataloader
needs to return `neutral_joints` (or a derived bone-length vector / shape
embedding) alongside the motion features so the model can condition on it,
and any FK / contact / position losses should use the per-sample
neutral_joints rather than the canonical baked ones.
"""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from kimodo.skeleton.bvh import Bvh  # kimodo's in-repo Bvh class (kimodo/skeleton/bvh.py)

from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton77
from kimodo.skeleton.bvh import parse_bvh_motion
from kimodo.tools import to_numpy, to_torch


# A kimodo-known joint can be legitimately absent from the BVH HIERARCHY only
# when it is intentionally not modelled in the bones_seed source. To date,
# every soma_proportional/bvh/*.bvh sampled has all 77 kimodo joints present.
# If a future BVH ships without some joint, list it here so the converter
# doesn't silently zero its bone length.
ALLOWED_MISSING_JOINTS: set[str] = set()


def bvh_offsets_in_kimodo_order(bvh_path: Path, skel: SOMASkeleton77) -> np.ndarray:
    """Read each kimodo-named joint's BVH OFFSET (in cm, parent-local) into a
    (J, 3) array indexed by skel.bone_order_names. Hips is zeroed because
    kimodo treats Hips as the root at origin (the ~(0, 100, 0) Hips OFFSET in
    BVH belongs to the wrapper Root joint and is recovered via the BVH
    animation's per-frame root translation, not via neutral_joints).

    Raises ValueError if any kimodo joint name is missing from the BVH so
    we never silently leave a joint at offset (0,0,0), which would collapse
    it onto its parent yet still pass downstream FK round-trips.
    """
    with open(bvh_path, encoding="utf-8") as f:
        mocap = Bvh(f.read(), backend="np")
    present = set(mocap.get_joints_names())

    missing = [n for n in skel.bone_order_names
               if n not in present and n not in ALLOWED_MISSING_JOINTS]
    if missing:
        raise ValueError(
            f"Missing BVH joints in {bvh_path.name}: "
            f"{missing[:20]}{' ...' if len(missing) > 20 else ''} "
            f"(total={len(missing)})"
        )

    out = np.zeros((skel.nbjoints, 3), dtype=np.float64)
    for i, name in enumerate(skel.bone_order_names):
        if name in present:
            out[i] = np.asarray(mocap.joint_offset(name), dtype=np.float64)
    hips_idx = skel.bone_order_names.index("Hips")
    out[hips_idx] = 0.0
    return out  # cm


def build_actor_neutrals_in_kimodo_frame(
    bvh_offsets_cm: np.ndarray,
    skel: SOMASkeleton77,
) -> np.ndarray:
    """Construct per-actor neutral_joints in kimodo's Y-up frame from the
    BVH HIERARCHY offsets, using kimodo's global_rot_offsets to map each
    bone-local offset into world frame.

    Formula (per joint j with parent p):
        pos[j] = pos[p] + global_rot_offsets[p] @ bvh_offset_cm[j]

    Validates exactly against kimodo's baked joints.p when fed the standard
    T-pose BVH (after zeroing Hips), and produces correctly-scaled actor
    skeletons (height ~1.7m, feet near Y=-1m, head near Y=+0.75m) when fed
    actor BVHs that use the BVH bone-axis convention.
    """
    gro = skel.global_rot_offsets.detach().cpu().numpy().astype(np.float64)  # (J, 3, 3)
    parents = skel.joint_parents.cpu().numpy().tolist()
    pos = np.zeros_like(bvh_offsets_cm)  # cm
    for j in range(len(pos)):
        p = parents[j]
        if p < 0:
            continue  # root stays at origin
        pos[j] = pos[p] + gro[p] @ bvh_offsets_cm[j]
    actor = (pos * 0.01).astype(np.float64)  # cm -> m

    # ----- per-bone length parity guard -----
    # gro[p] is a rotation matrix so by construction
    #     ||gro[p] @ bvh_offset|| == ||bvh_offset||.
    # This guards against future code changes / numerical drift, not against
    # bugs in this file as currently written.
    hips_idx = skel.bone_order_names.index("Hips")
    for j, p in enumerate(parents):
        if p < 0 or j == hips_idx:
            continue
        bvh_len     = float(np.linalg.norm(bvh_offsets_cm[j])) * 0.01  # m
        neutral_len = float(np.linalg.norm(actor[j] - actor[p]))       # m
        if abs(bvh_len - neutral_len) > 1e-5:
            raise ValueError(
                f"Bone-length mismatch at {skel.bone_order_names[j]}: "
                f"bvh={bvh_len:.6f}m  neutral={neutral_len:.6f}m"
            )
    return actor.astype(np.float32)

FPS = 20
DEFAULT_DATASET = Path("/weka/jungbin/seed/soma_proportional")
DEFAULT_OUTPUT = Path("/weka/jungbin/seed/soma_proportional_motions_20fps")


def discover_bvh_files(dataset_folder: Path, include_mirrored: bool) -> list[Path]:
    bvh_root = dataset_folder / "bvh"
    if not bvh_root.is_dir():
        raise FileNotFoundError(f"BVH folder not found: {bvh_root}")
    files = sorted(bvh_root.rglob("*.bvh"))
    if not include_mirrored:
        files = [p for p in files if not p.stem.endswith("_M")]
    return files


def convert_one(
    bvh_path: Path,
    dataset_folder: Path,
    output_folder: Path,
    fps: int,
    canonicalize: bool,
    overwrite: bool,
) -> tuple[Path, str]:
    bvh_root = dataset_folder / "bvh"
    rel = bvh_path.relative_to(bvh_root).with_suffix(".npz")
    out_path = output_folder / rel

    if out_path.is_file() and not overwrite:
        return out_path, "skipped"

    try:
        # ---- 1. parse BVH animation (rotations + root translation, no neutrals) ----
        local_rot_mats, root_trans, bvh_fps = parse_bvh_motion(bvh_path)

        # ---- 2. resample to target fps ----
        # Assert that source fps is (effectively) an integer multiple of target
        # fps; otherwise the simple stride drops/skews timing. BONES-SEED ships
        # at "Frame Time: 0.008333" → 120.00480 Hz (finite-precision header),
        # which rounds cleanly to 6×20 Hz. We allow a relative tolerance to
        # cover that, but still flag any future BVH that's substantively off.
        ratio = bvh_fps / fps
        step = max(1, round(ratio))
        if step < 1 or abs(ratio - step) / step > 1e-3:
            raise ValueError(
                f"Non-integer fps ratio at {bvh_path.name}: bvh_fps={bvh_fps}"
                f" target_fps={fps} ratio={ratio}; use real resampling"
            )
        root_trans = root_trans[::step]
        local_rot_mats = local_rot_mats[::step]

        # ---- 3. build per-actor neutral_joints in kimodo Y-up frame by
        #         applying global_rot_offsets[parent] @ bvh_offset[j] along
        #         the kinematic chain ----
        skeleton = SOMASkeleton77()
        bvh_off_cm = bvh_offsets_in_kimodo_order(bvh_path, skeleton)
        actor_neutrals = build_actor_neutrals_in_kimodo_frame(bvh_off_cm, skeleton)
        skeleton.neutral_joints = torch.from_numpy(actor_neutrals).to(
            device=skeleton.neutral_joints.device,
            dtype=skeleton.neutral_joints.dtype,
        )

        # ---- 4. rest of the pipeline (identical to create_data.py) ----
        local_rot_mats, _ = skeleton.to_standard_tpose(local_rot_mats)
        motion_rep = KimodoMotionRep(skeleton, fps)
        feats = motion_rep(local_rot_mats, root_trans, to_normalize=False)
        if canonicalize:
            feats = motion_rep.canonicalize(feats)
        motion = motion_rep.inverse(feats, is_normalized=False)
        motion = to_numpy(to_torch(motion, dtype=torch.float32))

        # ---- 5. save with per-actor neutrals so downstream knows ----
        motion["neutral_joints"] = actor_neutrals.astype(np.float32)

        # ---- 5a. Hips/root convention check (paranoia for the Hips=0 zeroing) ----
        # kimodo's FK: posed[Hips] = (neutral[Hips] - pelvis_offset) + root_pos
        # When neutral[Hips] == 0 (our convention), posed[Hips] must equal
        # root_positions exactly. If this ever differs by ~1m on Y the
        # Hips-zeroing assumption is wrong for this BVH source.
        hips_idx = skeleton.bone_order_names.index("Hips")
        d_hips = float(np.max(np.abs(motion["posed_joints"][:, hips_idx]
                                     - motion["root_positions"])))
        if d_hips > 1e-4:
            raise ValueError(
                f"Hips/root parity violated in {bvh_path.name}: "
                f"max |posed[Hips] - root_positions| = {d_hips:.6f} m"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **motion)
        return out_path, "ok"
    except Exception as e:
        return out_path, f"error: {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert SHAPE-AWARE (proportional) BONES-SEED BVH into Kimodo NPZs.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--no-canonicalize", action="store_true")
    parser.add_argument("--skip-mirrored", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards})")

    bvh_files = discover_bvh_files(args.dataset.resolve(), include_mirrored=not args.skip_mirrored)
    total = len(bvh_files)
    if args.num_shards > 1:
        bvh_files = bvh_files[args.shard_id :: args.num_shards]
    if args.limit is not None:
        bvh_files = bvh_files[: args.limit]
    if not bvh_files:
        raise SystemExit(f"No BVH files found under {args.dataset}/bvh")
    print(f"Discovered {total} BVH files; this shard handles {len(bvh_files)}.")
    print(f"Output root: {args.output.resolve()}")

    fn = partial(
        convert_one,
        dataset_folder=args.dataset.resolve(),
        output_folder=args.output.resolve(),
        fps=args.fps,
        canonicalize=not args.no_canonicalize,
        overwrite=args.overwrite,
    )

    if args.workers <= 1:
        iterator = (fn(p) for p in bvh_files)
    else:
        pool = Pool(args.workers)
        iterator = pool.imap_unordered(fn, bvh_files)

    n_ok = n_skip = n_err = 0
    errors = []
    for out_path, status in tqdm(iterator, total=len(bvh_files), desc="Converting BVH"):
        if status == "ok":
            n_ok += 1
        elif status == "skipped":
            n_skip += 1
        else:
            n_err += 1
            errors.append((out_path, status))

    if args.workers > 1:
        pool.close()
        pool.join()

    print(f"Done. ok={n_ok}, skipped={n_skip}, errors={n_err}")
    if errors:
        print("First failures:")
        for path, msg in errors[:10]:
            print(f"  {path}: {msg}")


if __name__ == "__main__":
    main()
