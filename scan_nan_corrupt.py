"""Scan soma_uniform_motions_20fps NPZs for NaN/Inf, corruption, AND outliers.

Training (SOMABonesSeedDataset) only reads `local_rot_mats` and
`root_positions`; the rest are recomputed via FK. The bones_seed run diverged
to a *finite* loss plateau (~0.2, not NaN) around step 223-224k, so the likely
culprit is finite-but-garbage data, which a pure NaN/Inf check would miss.
So per file we check, on those two arrays:

  local_rot_mats (should be valid rotation matrices):
    - NaN / Inf
    - |element| > 1.05      -> impossible for an orthonormal rotation (blow-up)
    - |det - 1| (and det<0)  -> non-rotation (scale/shear/reflection)
  root_positions (meters):
    - NaN / Inf
    - |value| > 1e3          -> absurd translation (>1 km)
    - per-frame jump > 20 m   -> teleport (20 m in 1/20 s = 400 m/s)

Plus: any file that fails to load (corruption), is empty, or missing keys.

Per-file metrics are collected for ALL files so we can report the corpus
distribution + worst offenders even if nothing crosses a (somewhat arbitrary)
hard threshold. Run on a compute node (srun); CPU/IO only, no GPU.
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = "/home/jungbin_cho/seed/soma_uniform_motions_20fps"
OUT = "/home/jungbin_cho/kimodo_open/nan_scan_results.json"
CHECK_KEYS = ("local_rot_mats", "root_positions")

# Hard-flag thresholds.
ROT_ABS_MAX = 1.05      # rotation-matrix element magnitude ceiling (+float slack)
ROT_DET_DEV = 0.05      # |det - 1| tolerance
ROOT_ABS_MAX = 1.0e3    # meters
ROOT_JUMP_MAX = 20.0    # meters per frame (20 fps)


def check_one(path):
    rel = os.path.relpath(path, ROOT)
    try:
        with np.load(path) as d:
            files = set(d.files)
            missing = [k for k in CHECK_KEYS if k not in files]
            if missing:
                return {"file": rel, "missing_keys": missing}

            rot = d["local_rot_mats"]          # (T, J, 3, 3)
            root = d["root_positions"]         # (T, 3)
            T = int(rot.shape[0])
            if T == 0 or root.shape[0] == 0:
                return {"file": rel, "empty": True, "frames": T}

            rot = np.asarray(rot, dtype=np.float32)
            root = np.asarray(root, dtype=np.float32)

            rot_nan = bool(np.isnan(rot).any())
            rot_inf = bool(np.isinf(rot).any())
            root_nan = bool(np.isnan(root).any())
            root_inf = bool(np.isinf(root).any())

            # Use nan-safe maxes so a NaN doesn't poison the magnitude stats.
            rot_absmax = float(np.nanmax(np.abs(rot))) if not (rot_nan and np.isnan(rot).all()) else float("nan")
            root_absmax = float(np.nanmax(np.abs(root))) if root.size else 0.0

            # Determinant deviation (finite entries only).
            det_dev = 0.0
            det_min = 1.0
            finite = np.isfinite(rot).all(axis=(-1, -2))  # (T, J)
            if finite.any():
                dets = np.linalg.det(rot[finite].astype(np.float32))  # (M,)
                det_dev = float(np.max(np.abs(dets - 1.0)))
                det_min = float(np.min(dets))

            # Root per-frame displacement (teleport detector).
            root_jump = 0.0
            if T >= 2 and np.isfinite(root).all():
                root_jump = float(np.max(np.linalg.norm(np.diff(root, axis=0), axis=1)))

            flags = []
            if rot_nan or rot_inf:
                flags.append("rot_nonfinite")
            if np.isfinite(rot_absmax) and rot_absmax > ROT_ABS_MAX:
                flags.append("rot_absmax")
            if det_dev > ROT_DET_DEV or det_min < 0:
                flags.append("rot_not_rotation")
            if root_nan or root_inf:
                flags.append("root_nonfinite")
            if np.isfinite(root_absmax) and root_absmax > ROOT_ABS_MAX:
                flags.append("root_huge")
            if root_jump > ROOT_JUMP_MAX:
                flags.append("root_teleport")

            return {
                "file": rel, "frames": T,
                "rot_absmax": rot_absmax, "rot_det_dev": det_dev, "rot_det_min": det_min,
                "rot_nan": rot_nan, "rot_inf": rot_inf,
                "root_absmax": root_absmax, "root_jump": root_jump,
                "root_nan": root_nan, "root_inf": root_inf,
                "flags": flags,
            }
    except Exception as e:
        return {"file": rel, "load_error": f"{type(e).__name__}: {e}"}


def _topn(metrics, key, n=30):
    vals = [m for m in metrics if key in m and m[key] == m[key]]  # drop NaN
    vals.sort(key=lambda m: m[key], reverse=True)
    return [{"file": m["file"], key: round(m[key], 4), "flags": m.get("flags", [])} for m in vals[:n]]


def _pcts(metrics, key):
    arr = np.array([m[key] for m in metrics if key in m and np.isfinite(m[key])], dtype=np.float64)
    if not arr.size:
        return {}
    return {p: round(float(np.percentile(arr, p)), 4) for p in (50, 90, 99, 99.9, 100)}


def main():
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(ROOT, "**", "*.npz"), recursive=True))
    n = len(paths)
    print(f"scanning {n} npz files under {ROOT}", flush=True)
    nproc = int(os.environ.get("SCAN_PROCS", "16"))
    metrics, load_err, empty, missing, flagged = [], [], [], [], []
    done = 0
    with Pool(nproc) as pool:
        for res in pool.imap_unordered(check_one, paths, chunksize=64):
            done += 1
            if "load_error" in res:
                load_err.append(res)
            elif res.get("empty"):
                empty.append(res)
            elif "missing_keys" in res:
                missing.append(res)
            else:
                metrics.append(res)
                if res["flags"]:
                    flagged.append(res)
            if done % 10000 == 0:
                el = time.time() - t0
                print(f"  {done}/{n}  flagged={len(flagged)} corrupt={len(load_err)}  "
                      f"rate={done/el:.0f}/s  eta={(n-done)/(done/el):.0f}s", flush=True)

    by_flag = {}
    for m in flagged:
        for fl in m["flags"]:
            by_flag[fl] = by_flag.get(fl, 0) + 1

    summary = {
        "root": ROOT, "total_files": n, "scanned": done,
        "elapsed_sec": round(time.time() - t0, 1),
        "thresholds": {"rot_absmax": ROT_ABS_MAX, "rot_det_dev": ROT_DET_DEV,
                       "root_absmax": ROOT_ABS_MAX, "root_jump": ROOT_JUMP_MAX},
        "counts": {
            "clean": len(metrics) - len(flagged),
            "flagged": len(flagged), "load_error_corrupt": len(load_err),
            "empty": len(empty), "missing_keys": len(missing),
            "by_flag": by_flag,
        },
        "distribution": {
            "rot_absmax": _pcts(metrics, "rot_absmax"),
            "rot_det_dev": _pcts(metrics, "rot_det_dev"),
            "root_absmax": _pcts(metrics, "root_absmax"),
            "root_jump": _pcts(metrics, "root_jump"),
        },
        "top_offenders": {
            "rot_absmax": _topn(metrics, "rot_absmax"),
            "rot_det_dev": _topn(metrics, "rot_det_dev"),
            "root_absmax": _topn(metrics, "root_absmax"),
            "root_jump": _topn(metrics, "root_jump"),
        },
        "corrupt_files": load_err, "empty_files": empty, "missing_files": missing,
        "flagged_files": flagged,
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SCAN SUMMARY =====", flush=True)
    for k, v in summary["counts"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"  elapsed: {summary['elapsed_sec']}s   results -> {OUT}", flush=True)
    print("\n  distribution (p50 / p99 / p99.9 / max):", flush=True)
    for k, p in summary["distribution"].items():
        if p:
            print(f"    {k:14s} {p.get(50)} / {p.get(99)} / {p.get(99.9)} / {p.get(100)}", flush=True)
    for label, lst in (("CORRUPT", load_err), ("FLAGGED", flagged)):
        if lst:
            print(f"\n-- {label} (up to 10 of {len(lst)}) --", flush=True)
            for b in lst[:10]:
                print(f"   {json.dumps({k: b[k] for k in b if k not in ('rot_nan','rot_inf','root_nan','root_inf')})}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
