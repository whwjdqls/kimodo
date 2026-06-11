# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild HumanML3D's 263-D per-motion .npy files from kimodo_rep dicts.

The MoMask T2M evaluator reads HumanML3D's standard 263-D feature .npy files
from ``<dataset>/new_joint_vecs/<id>.npy``. On this machine those originals
have been deleted; only the kimodo_rep .npz files (richer geometric dicts)
remain. Each kimodo_rep npz contains the five fields the inverse converter
needs (``global_rot_mats``, ``posed_joints``, ``root_positions``,
``velocities``, ``foot_contacts``) plus the optional ``r_rot_quat`` for
**exact** roundtrip.

Usage::

    python -m evaluation.rebuild_hml3d_263 \\
        --src /home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep \\
        --dst /home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs

Run once. Idempotent (skips existing files unless ``--force``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Bring in the converter from benchmark/.
BENCH = Path("/home/jungbin_cho/kimodo_open/benchmark")
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
from humanml3d_to_kimodo import kimodo_to_humanml3d  # noqa: E402


def convert_one(src_npz: Path, dst_npy: Path, device: str = "cpu") -> int:
    """Return the number of frames written (or 0 if skipped)."""
    with np.load(src_npz) as z:
        kimodo_dict = {
            "global_rot_mats": torch.from_numpy(z["global_rot_mats"]).float(),
            "posed_joints":    torch.from_numpy(z["posed_joints"]).float(),
            "root_positions":  torch.from_numpy(z["root_positions"]).float(),
            "velocities":      torch.from_numpy(z["velocities"]).float(),
            "foot_contacts":   torch.from_numpy(z["foot_contacts"]).float(),
        }
        if "r_rot_quat" in z.files:
            kimodo_dict["r_rot_quat"] = torch.from_numpy(z["r_rot_quat"]).float()
    hml = kimodo_to_humanml3d(kimodo_dict, device=device)  # (T, 263)
    dst_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst_npy, hml.cpu().numpy().astype(np.float32))
    return int(hml.shape[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=str, help="kimodo_rep directory")
    p.add_argument("--dst", required=True, type=str, help="output new_joint_vecs directory")
    p.add_argument("--force", action="store_true", help="overwrite existing .npy files")
    p.add_argument("--limit", type=int, default=None, help="process only first N motions (smoke test)")
    args = p.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"src={src}  dst={dst}  n={len(files)}")

    n_written = 0
    n_skipped = 0
    n_failed = 0
    for f in tqdm(files):
        out = dst / f"{f.stem}.npy"
        if out.is_file() and not args.force:
            n_skipped += 1
            continue
        try:
            convert_one(f, out)
            n_written += 1
        except Exception as e:
            n_failed += 1
            tqdm.write(f"failed {f.name}: {e}")
    print(f"done. written={n_written} skipped={n_skipped} failed={n_failed}")


if __name__ == "__main__":
    main()
