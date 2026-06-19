# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack precomputed 369-D KimodoMotionRep features for a BONES-SEED split.

Eliminates the per-step motion_rep CPU work in dataset workers (FK on 30
joints, 6D rotation conversion, velocity, foot-contact detection) by baking
the features once at preprocessing time. Mirrors what
``benchmark/humanml3d_to_kimodo.py`` already does for HumanML3D: the
features key in those NPZs is exactly what makes HumanML3D training fast.

Features are packed with ``to_canonicalize=False, to_normalize=False``;
canonicalization, random heading augmentation, and stats normalization
happen at runtime on the sliced segment (cheap rotates / mean-std ops on the
369-D features, no FK).

Output layout:

    {
        'names':     List[str]                  # filenames in pack order
        'offsets':   Tensor int64 [N+1]         # cumulative frame counts
        'features':  Tensor f32 [N_total_frames, 369]
    }

Run via sbatch / srun — DO NOT run on the login node (motion_rep + the
~5 GB concat will hammer login RAM/CPU).

``--procs N`` (N>1) parallelizes the per-motion FK across N processes; output
is byte-identical to the serial path (each motion is processed independently
and deterministically; pack order is preserved). Serial 3.6 h -> ~15 min.

Usage:
    python -m kimodo.scripts.pack_bones_seed_features \\
        --split  /home/.../train_split_paths_small.txt \\
        --data-root /home/.../soma_uniform_motions_20fps \\
        --stats-path /home/.../Kimodo-SOMA-SEED-v1.1/stats/motion/ \\
        --fps 20 --procs 22 \\
        --out    /home/jungbin_cho/kimodo_caches/bones_seed_small_feats.pt
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton30

# Per-worker singletons (one motion_rep per process, built in the initializer).
_W: dict = {}


def _init_worker(stats_path: str, fps: int):
    torch.set_num_threads(1)  # 1 intra-op thread/worker -> N workers fill N cores cleanly
    skeleton = SOMASkeleton30()
    _W["skeleton"] = skeleton
    _W["mr"] = KimodoMotionRep(skeleton=skeleton, fps=fps, stats_path=str(stats_path))
    _W["data_root"] = stats_path  # placeholder; real data_root passed per-call


def _features_for(name: str, data_root: str):
    """Compute (T, 369) features for one motion, or None if the NPZ is missing."""
    p = os.path.join(data_root, f"{name}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as d:
        local_rot_77 = np.ascontiguousarray(d["local_rot_mats"], dtype=np.float32)
        root_pos = np.ascontiguousarray(d["root_positions"], dtype=np.float32)
    skeleton = _W["skeleton"]
    mr = _W["mr"]
    local_rot_30 = skeleton.from_SOMASkeleton77(torch.from_numpy(local_rot_77))
    T = int(local_rot_30.shape[0])
    with torch.no_grad():
        feats = mr(
            local_rot_30.unsqueeze(0),
            torch.from_numpy(root_pos).unsqueeze(0),
            to_normalize=False,
            to_canonicalize=False,
            lengths=torch.tensor([T]),
        )[0].float().contiguous()  # (T, 369)
    return feats.numpy()


def _worker(arg):
    i, name, data_root = arg
    feats = _features_for(name, data_root)
    return (i, name, feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--stats-path", required=True, type=Path)
    ap.add_argument("--fps", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--procs", type=int, default=1, help="parallel worker processes (1 = serial)")
    args = ap.parse_args()

    with open(args.split) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    print(f"split has {len(names)} entries; procs={args.procs}", flush=True)

    skeleton = SOMASkeleton30()
    motion_rep = KimodoMotionRep(
        skeleton=skeleton, fps=args.fps, stats_path=str(args.stats_path),
    )
    feat_dim = sum(int(np.prod(sz)) for sz in motion_rep.size_dict.values())
    print(f"motion_rep feature dim: {feat_dim}", flush=True)

    feature_parts: list = []
    kept_names: list = []
    offsets = [0]
    skipped = 0
    t0 = time.time()

    def _accumulate(name, feats):
        nonlocal skipped
        if feats is None:
            skipped += 1
            return
        if feats.shape[1] != feat_dim:
            raise RuntimeError(
                f"{name}: motion_rep returned {feats.shape[1]} dims, expected {feat_dim}"
            )
        feature_parts.append(torch.from_numpy(feats))
        kept_names.append(name)
        offsets.append(offsets[-1] + int(feats.shape[0]))

    if args.procs > 1:
        from multiprocessing import Pool
        data_root = str(args.data_root)
        tasks = [(i, name, data_root) for i, name in enumerate(names)]
        with Pool(
            args.procs, initializer=_init_worker, initargs=(str(args.stats_path), args.fps),
        ) as pool:
            # imap preserves input order -> pack order matches the split, no sort needed.
            for done, (i, name, feats) in enumerate(pool.imap(_worker, tasks, chunksize=16), 1):
                _accumulate(name, feats)
                if done % 5000 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{len(names)}  kept={len(kept_names)} skipped={skipped} "
                          f"frames={offsets[-1]} rate={done/el:.1f}/s eta={(len(names)-done)/(done/el):.0f}s",
                          flush=True)
    else:
        _init_worker(str(args.stats_path), args.fps)
        for i, name in enumerate(names):
            _accumulate(name, _features_for(name, str(args.data_root)))
            if (i + 1) % 1000 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(names)}  kept={len(kept_names)} skipped={skipped} "
                      f"frames={offsets[-1]} rate={(i+1)/el:.1f}/s eta={(len(names)-i-1)/((i+1)/el):.0f}s",
                      flush=True)

    print(f"concatenating {len(kept_names)} motions ({offsets[-1]} frames, skipped={skipped})", flush=True)
    features_cat = torch.cat(feature_parts, dim=0).contiguous()
    offsets_t = torch.tensor(offsets, dtype=torch.int64)
    print(f"sizes: features {tuple(features_cat.shape)} "
          f"({features_cat.numel() * 4 / 1024 / 1024:.0f} MiB)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save({"names": kept_names, "offsets": offsets_t, "features": features_cat}, tmp)
    os.replace(tmp, args.out)
    sz = args.out.stat().st_size / 1024 / 1024
    print(f"wrote {args.out} ({sz:.0f} MiB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
