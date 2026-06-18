"""
Compute mean/std of the 211-D UniEgo-style features over the HumanML3D train
split, for normalization during training.

Streams (sum, sumsq, count) per dimension in float64 across all frames of all
train-split clips (the split already lists both base and mirrored 'M' ids), then
writes flat (211,) Mean / Std vectors -- mirroring the Mean_kimodo.npy / Std_kimodo.npy
convention. Constant channels (std ~ 0) get std=1 so normalization is a no-op there.

Usage:
  python benchmark/compute_uniego_stats.py                 # defaults below
  python benchmark/compute_uniego_stats.py --workers 4
"""

from __future__ import annotations

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

# identity SE(3) in (6D rot ++ 3 trans) form for the canon_delta block.
IDENTITY_DELTA9 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64)


def _partial_stats(clip_id: str, rep_dir: Path, n_joints: int):
    """Return (count, sum(D,), sumsq(D,)) in float64 for one clip, or None.

    Frame 0's canon_delta (at [n_joints*9 : n_joints*9+9]) is overridden to the
    identity transform, matching the dataset's per-window frame-0
    canonicalization — so the stats reflect what training actually sees (no
    absolute-placement pollution of the delta-trans channels).
    """
    p = rep_dir / f"{clip_id}.npz"
    if not p.is_file():
        return None
    try:
        f = np.load(p)["features"].astype(np.float64)  # (T, D)
    except Exception:  # noqa: BLE001
        return None
    lo = n_joints * 9
    f[0, lo:lo + 9] = IDENTITY_DELTA9
    return f.shape[0], f.sum(axis=0), (f * f).sum(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split-file",
        type=Path,
        default=Path("/home/jungbin_cho/HumanML3D/HumanML3D/train.txt"),
    )
    ap.add_argument(
        "--rep-dir",
        type=Path,
        default=Path("/home/jungbin_cho/HumanML3D/HumanML3D/uniego_rep"),
    )
    ap.add_argument(
        "--out-mean",
        type=Path,
        default=Path("/home/jungbin_cho/HumanML3D/HumanML3D/Mean_uniego.npy"),
    )
    ap.add_argument(
        "--out-std",
        type=Path,
        default=Path("/home/jungbin_cho/HumanML3D/HumanML3D/Std_uniego.npy"),
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--std-floor", type=float, default=1e-6)
    ap.add_argument("--n-joints", type=int, default=22,
                    help="22 for HumanML3D SMPL, 30 for SOMA-30 (sets the canon_delta offset).")
    args = ap.parse_args()

    ids = [ln.strip() for ln in args.split_file.read_text().splitlines() if ln.strip()]
    print(f"{len(ids)} train ids from {args.split_file}")

    # Infer feature width from the first available clip.
    feat_dim = None
    for i in ids:
        p = args.rep_dir / f"{i}.npz"
        if p.is_file():
            feat_dim = int(np.load(p)["features"].shape[-1])
            break
    if feat_dim is None:
        raise SystemExit(f"No clips found under {args.rep_dir}")
    print(f"feature dim = {feat_dim} (n_joints={args.n_joints})")

    count = 0
    ssum = np.zeros(feat_dim, dtype=np.float64)
    ssq = np.zeros(feat_dim, dtype=np.float64)

    fn = partial(_partial_stats, rep_dir=args.rep_dir, n_joints=args.n_joints)
    n_missing = 0
    if args.workers <= 1:
        it = (fn(i) for i in ids)
        pool = None
    else:
        pool = Pool(args.workers)
        it = pool.imap_unordered(fn, ids, chunksize=64)
    for r in tqdm(it, total=len(ids), desc="Accumulating"):
        if r is None:
            n_missing += 1
            continue
        c, s, sq = r
        count += c
        ssum += s
        ssq += sq
    if pool is not None:
        pool.close()
        pool.join()

    if count == 0:
        raise SystemExit("No frames accumulated -- check --rep-dir / --split-file.")

    mean = ssum / count
    var = ssq / count - mean * mean
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var)
    n_const = int((std < args.std_floor).sum())
    std[std < args.std_floor] = 1.0

    args.out_mean.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_mean, mean.astype(np.float32))
    np.save(args.out_std, std.astype(np.float32))

    lo = args.n_joints * 9
    print(f"\nFrames: {count:,}   missing clips: {n_missing}   constant dims: {n_const}")
    print(f"mean[:4]={mean[:4]}  std[:4]={std[:4]}")
    # quick per-block summary (local SE(3) | 9 delta | foot)
    print(
        f"std ranges -- joints[0:{lo}]: {std[:lo].min():.3g}..{std[:lo].max():.3g} | "
        f"delta[{lo}:{lo+9}]: {std[lo:lo+9].min():.3g}..{std[lo:lo+9].max():.3g} | "
        f"foot[{lo+9}:{feat_dim}]: {std[lo+9:feat_dim].min():.3g}..{std[lo+9:feat_dim].max():.3g}"
    )
    print(f"wrote {args.out_mean}")
    print(f"wrote {args.out_std}")


if __name__ == "__main__":
    main()
