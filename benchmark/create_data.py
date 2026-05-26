# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert BONES-SEED BVH motions into Kimodo training NPZ files.

Walks the SEED dataset folder (default: /weka/jungbin/seed/soma_uniform/bvh) and writes one
Kimodo motion NPZ per BVH, mirroring the input subdirectory tree under the output root.
Pipeline matches benchmark/create_benchmark.py: parse BVH -> subsample to 30 fps -> remap to
standard T-pose -> compute motion features -> canonicalize -> inverse to motion dict -> save.
"""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton77
from kimodo.skeleton.bvh import parse_bvh_motion
from kimodo.tools import to_numpy, to_torch

FPS = 20
DEFAULT_DATASET = Path("/weka/jungbin/seed/soma_uniform")
DEFAULT_OUTPUT = Path("/weka/jungbin/seed/soma_uniform_motions_20fps")


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
        local_rot_mats, root_trans, bvh_fps = parse_bvh_motion(bvh_path)
        step = max(1, round(bvh_fps / fps))
        root_trans = root_trans[::step]
        local_rot_mats = local_rot_mats[::step]

        skeleton = SOMASkeleton77()
        local_rot_mats, _ = skeleton.to_standard_tpose(local_rot_mats)

        motion_rep = KimodoMotionRep(skeleton, fps)
        feats = motion_rep(local_rot_mats, root_trans, to_normalize=False)

        if canonicalize:
            feats = motion_rep.canonicalize(feats)
        motion = motion_rep.inverse(feats, is_normalized=False)
        motion = to_numpy(to_torch(motion, dtype=torch.float32))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **motion)
        return out_path, "ok"
    except Exception as e:
        return out_path, f"error: {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert BONES-SEED BVH dataset into Kimodo motion NPZs for training.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"SEED dataset folder containing a 'bvh/' subdirectory (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output root for Kimodo motion NPZs (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help=f"Target FPS to subsample BVH to (default: {FPS}).",
    )
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Skip the canonicalization step (keep original root translation and heading).",
    )
    parser.add_argument(
        "--skip-mirrored",
        action="store_true",
        help="Skip BVH files whose stem ends with '_M' (mirrored motions).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess even if the output NPZ already exists.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1, sequential).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N BVH files (for smoke testing).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards; use with --shard-id to split work across nodes.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Index of this shard in [0, num-shards). Files are assigned by index%%num-shards.",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards})")

    dataset = args.dataset.resolve()
    output = args.output.resolve()

    bvh_files = discover_bvh_files(dataset, include_mirrored=not args.skip_mirrored)
    total_files = len(bvh_files)
    if args.num_shards > 1:
        bvh_files = bvh_files[args.shard_id :: args.num_shards]
    if args.limit is not None:
        bvh_files = bvh_files[: args.limit]
    if not bvh_files:
        raise SystemExit(f"No BVH files found under {dataset}/bvh")
    print(
        f"Discovered {total_files} BVH files; shard {args.shard_id}/{args.num_shards} "
        f"handles {len(bvh_files)}."
    )
    print(f"Output root: {output}")

    fn = partial(
        convert_one,
        dataset_folder=dataset,
        output_folder=output,
        fps=args.fps,
        canonicalize=not args.no_canonicalize,
        overwrite=args.overwrite,
    )

    n_ok = n_skip = n_err = 0
    errors: list[tuple[Path, str]] = []
    if args.workers <= 1:
        iterator = (fn(p) for p in bvh_files)
    else:
        pool = Pool(args.workers)
        iterator = pool.imap_unordered(fn, bvh_files)

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
