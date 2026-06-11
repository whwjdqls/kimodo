# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack per-split BONES-SEED motion NPZs into a single mmap-able .pt file.

Eliminates the per-step cost of opening, decompressing, and decoding 13k
individual NPZs from NFS. Each motion contributes only the two tensors the
dataset actually reads (``local_rot_mats``, ``root_positions``) — everything
else in the NPZ (precomputed FK outputs, smooth roots, etc.) is recomputed
by the worker at __getitem__ time anyway.

Output layout:

    {
        'names':           List[str]              # filenames in pack order
        'offsets':         Tensor int64 [N+1]     # cumulative frame counts
        'local_rot_mats':  Tensor f32 [N_total_frames, 77, 3, 3]
        'root_positions':  Tensor f32 [N_total_frames, 3]
    }

Random access for motion i:  rows ``offsets[i] : offsets[i+1]`` of either
tensor. Slicing an mmap-loaded tensor returns a view; the OS pages in only
the touched rows.

Usage:
    python -m kimodo.scripts.pack_bones_seed_motions \\
        --split  /home/.../train_split_paths_small.txt \\
        --data-root /home/.../soma_uniform_motions_20fps \\
        --out    /home/jungbin_cho/kimodo_caches/bones_seed_small_raw.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.split) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    print(f"split has {len(names)} entries")

    local_rot_parts = []
    root_pos_parts = []
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
            local_rot = np.ascontiguousarray(d["local_rot_mats"], dtype=np.float32)
            root_pos = np.ascontiguousarray(d["root_positions"], dtype=np.float32)
        local_rot_parts.append(torch.from_numpy(local_rot))
        root_pos_parts.append(torch.from_numpy(root_pos))
        kept_names.append(name)
        offsets.append(offsets[-1] + local_rot.shape[0])
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(names) - i - 1) / rate
            print(
                f"  {i+1}/{len(names)}  kept={len(kept_names)}  skipped={skipped}  "
                f"frames_so_far={offsets[-1]}  rate={rate:.1f}/s  eta={eta:.0f}s"
            )

    print(
        f"concatenating {len(kept_names)} motions, "
        f"{offsets[-1]} total frames, skipped={skipped}"
    )
    local_rot_cat = torch.cat(local_rot_parts, dim=0).contiguous()
    root_pos_cat = torch.cat(root_pos_parts, dim=0).contiguous()
    offsets_t = torch.tensor(offsets, dtype=torch.int64)

    print(
        f"sizes: local_rot_mats {tuple(local_rot_cat.shape)} "
        f"({local_rot_cat.numel() * 4 / 1024 / 1024:.0f} MiB)  "
        f"root_positions {tuple(root_pos_cat.shape)} "
        f"({root_pos_cat.numel() * 4 / 1024 / 1024:.1f} MiB)"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "names": kept_names,
            "offsets": offsets_t,
            "local_rot_mats": local_rot_cat,
            "root_positions": root_pos_cat,
        },
        args.out,
    )
    sz = args.out.stat().st_size / 1024 / 1024
    print(f"wrote {args.out} ({sz:.0f} MiB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
