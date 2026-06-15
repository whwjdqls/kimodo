"""Scan every produced NPZ for NaN in the critical fields. Output a JSON
report listing which files have NaN in which arrays."""
from __future__ import annotations
import argparse, glob, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = "/weka/jungbin/seed/soma_proportional_motions_20fps"
KEYS_TO_CHECK = (
    "local_rot_mats",
    "global_rot_mats",
    "posed_joints",
    "root_positions",
    "neutral_joints",
    "smooth_root_pos",
    "global_root_heading",
    "foot_contacts",
)


def scan_one(path: str) -> tuple[str, dict]:
    """Returns (path, {key: nan_count}) — only keys with nan>0 are included."""
    try:
        z = np.load(path)
    except Exception as e:
        return path, {"_LOAD_ERROR": repr(e)[:120]}
    nan_per_key = {}
    for k in KEYS_TO_CHECK:
        if k not in z.files:
            continue
        arr = z[k]
        if arr.dtype.kind != "f":
            continue
        n = int(np.isnan(arr).sum())
        if n:
            nan_per_key[k] = n
    return path, nan_per_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="/home/jungbin_cho/_nan_audit_proportional.json")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{ROOT}/**/*.npz", recursive=True))
    print(f"scanning {len(files)} files with {args.workers} workers...")

    n_clean = n_nan = n_loaderr = 0
    nan_records = []      # files with NaN
    loaderr_records = []  # files that failed to load
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(scan_one, p) for p in files]
        for i, fut in enumerate(as_completed(futs), 1):
            path, status = fut.result()
            if "_LOAD_ERROR" in status:
                n_loaderr += 1
                loaderr_records.append({"path": path, "error": status["_LOAD_ERROR"]})
            elif status:
                n_nan += 1
                nan_records.append({"path": path, "nan_per_key": status})
            else:
                n_clean += 1
            if i % 5000 == 0:
                print(f"  scanned {i}/{len(files)}  nan={n_nan}  loaderr={n_loaderr}")

    print()
    print(f"=== summary ({len(files)} files total) ===")
    print(f"  clean         : {n_clean}")
    print(f"  with NaN      : {n_nan}")
    print(f"  load error    : {n_loaderr}")

    if nan_records:
        # Group NaN files by source directory (date prefix) to see if some
        # batches are systematically bad.
        from collections import Counter
        date_counter = Counter()
        for r in nan_records:
            d = r["path"].split("/")[-2] if "/" in r["path"] else "?"
            date_counter[d] += 1
        print()
        print("  top 10 date subdirs by NaN count:")
        for d, n in date_counter.most_common(10):
            print(f"    {d}: {n}")

    out = {
        "totals": {"clean": n_clean, "with_nan": n_nan, "load_error": n_loaderr,
                   "total": len(files)},
        "nan_files": nan_records,
        "load_error_files": loaderr_records,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
