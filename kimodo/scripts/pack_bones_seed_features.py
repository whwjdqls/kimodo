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

Usage:
    python -m kimodo.scripts.pack_bones_seed_features \\
        --split  /home/.../train_split_paths_small.txt \\
        --data-root /home/.../soma_uniform_motions_20fps \\
        --stats-path /home/.../Kimodo-SOMA-SEED-v1.1/stats/motion/ \\
        --fps 20 \\
        --out    /home/jungbin_cho/kimodo_caches/bones_seed_small_feats.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--stats-path", required=True, type=Path)
    ap.add_argument("--fps", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.split) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    print(f"split has {len(names)} entries")

    skeleton = SOMASkeleton30()
    motion_rep = KimodoMotionRep(
        skeleton=skeleton, fps=args.fps, stats_path=str(args.stats_path),
    )
    feat_dim = sum(int(np.prod(sz)) for sz in motion_rep.size_dict.values())
    print(f"motion_rep feature dim: {feat_dim}")

    feature_parts = []
    kept_names = []
    offsets = [0]
    skipped = 0

    t0 = time.time()
    for i, name in enumerate(names):
        p = args.data_root / f"{name}.npz"
        if not p.exists():
            skipped += 1
            continue
        with np.load(p) as d:
            local_rot_77 = np.ascontiguousarray(d["local_rot_mats"], dtype=np.float32)
            root_pos = np.ascontiguousarray(d["root_positions"], dtype=np.float32)
        local_rot_77 = torch.from_numpy(local_rot_77)
        root_pos = torch.from_numpy(root_pos)
        local_rot_30 = skeleton.from_SOMASkeleton77(local_rot_77)

        T = int(local_rot_30.shape[0])
        with torch.no_grad():
            feats = motion_rep(
                local_rot_30.unsqueeze(0),
                root_pos.unsqueeze(0),
                to_normalize=False,
                to_canonicalize=False,
                lengths=torch.tensor([T]),
            )[0].float().contiguous()  # (T, 369)

        if feats.shape[1] != feat_dim:
            raise RuntimeError(
                f"{name}: motion_rep returned {feats.shape[1]} dims, expected {feat_dim}"
            )

        feature_parts.append(feats)
        kept_names.append(name)
        offsets.append(offsets[-1] + T)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(names) - i - 1) / rate
            print(
                f"  {i+1}/{len(names)}  kept={len(kept_names)}  skipped={skipped}  "
                f"frames={offsets[-1]}  rate={rate:.1f}/s  eta={eta:.0f}s"
            )

    print(f"concatenating {len(kept_names)} motions ({offsets[-1]} frames, skipped={skipped})")
    features_cat = torch.cat(feature_parts, dim=0).contiguous()
    offsets_t = torch.tensor(offsets, dtype=torch.int64)
    print(
        f"sizes: features {tuple(features_cat.shape)} "
        f"({features_cat.numel() * 4 / 1024 / 1024:.0f} MiB)"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "names": kept_names,
            "offsets": offsets_t,
            "features": features_cat,
        },
        args.out,
    )
    sz = args.out.stat().st_size / 1024 / 1024
    print(f"wrote {args.out} ({sz:.0f} MiB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
