"""Scan soma_uniform_motions_20fps NPZs for NaN/Inf and corruption.

Training (SOMABonesSeedDataset) only reads `local_rot_mats` and
`root_positions`; the rest are recomputed via FK. So we check those two
thoroughly for non-finite values, and additionally flag any file that fails to
load at all (corruption) or is missing keys / has zero frames.

Run on a compute node (srun), never the login shell. CPU/IO only, no GPU.
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
# Also nan-check these recomputed arrays if present (cheap once file is open),
# reported separately — they don't gate training but indicate a bad source clip.
EXTRA_KEYS = ("posed_joints", "smooth_root_pos", "global_root_heading", "foot_contacts")


def check_one(path):
    """Return None if clean, else a dict describing the problem(s)."""
    rel = os.path.relpath(path, ROOT)
    try:
        with np.load(path) as d:
            files = set(d.files)
            problems = {}
            missing = [k for k in CHECK_KEYS if k not in files]
            if missing:
                problems["missing_keys"] = missing
            for k in CHECK_KEYS:
                if k not in files:
                    continue
                a = d[k]
                if a.shape[0] == 0:
                    problems.setdefault("empty", []).append(k)
                    continue
                if np.issubdtype(a.dtype, np.floating):
                    nan = bool(np.isnan(a).any())
                    inf = bool(np.isinf(a).any())
                    if nan or inf:
                        problems.setdefault("nonfinite", {})[k] = {
                            "nan": nan, "inf": inf,
                            "nan_count": int(np.isnan(a).sum()),
                            "inf_count": int(np.isinf(a).sum()),
                            "frames": int(a.shape[0]),
                        }
            extra_bad = {}
            for k in EXTRA_KEYS:
                if k in files:
                    a = d[k]
                    if a.size and np.issubdtype(a.dtype, np.floating):
                        if bool(np.isnan(a).any()) or bool(np.isinf(a).any()):
                            extra_bad[k] = {
                                "nan_count": int(np.isnan(a).sum()),
                                "inf_count": int(np.isinf(a).sum()),
                            }
            if extra_bad:
                problems["extra_nonfinite"] = extra_bad
        if problems:
            problems["file"] = rel
            return problems
        return None
    except Exception as e:  # corrupted / unreadable
        return {"file": rel, "load_error": f"{type(e).__name__}: {e}"}


def main():
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(ROOT, "**", "*.npz"), recursive=True))
    n = len(paths)
    print(f"scanning {n} npz files under {ROOT}", flush=True)
    nproc = int(os.environ.get("SCAN_PROCS", "48"))
    bad = []
    done = 0
    with Pool(nproc) as pool:
        for res in pool.imap_unordered(check_one, paths, chunksize=64):
            done += 1
            if res is not None:
                bad.append(res)
            if done % 10000 == 0:
                el = time.time() - t0
                print(f"  {done}/{n}  bad_so_far={len(bad)}  "
                      f"rate={done/el:.0f}/s  eta={(n-done)/(done/el):.0f}s", flush=True)

    # Categorize
    load_err = [b for b in bad if "load_error" in b]
    nonfinite = [b for b in bad if "nonfinite" in b]
    empty = [b for b in bad if "empty" in b]
    missing = [b for b in bad if "missing_keys" in b]
    extra_only = [b for b in bad if set(b) <= {"file", "extra_nonfinite"}]

    summary = {
        "root": ROOT,
        "total_files": n,
        "scanned": done,
        "elapsed_sec": round(time.time() - t0, 1),
        "counts": {
            "clean": n - len(bad),
            "bad_total": len(bad),
            "load_error_corrupt": len(load_err),
            "nonfinite_training_keys": len(nonfinite),
            "empty": len(empty),
            "missing_keys": len(missing),
            "extra_nonfinite_only": len(extra_only),
        },
        "bad_files": bad,
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SCAN SUMMARY =====", flush=True)
    for k, v in summary["counts"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"  elapsed: {summary['elapsed_sec']}s", flush=True)
    print(f"  full results -> {OUT}", flush=True)
    # Show first few offenders of each critical category
    for label, lst in (("CORRUPT/LOAD-ERROR", load_err),
                       ("NONFINITE in training keys", nonfinite),
                       ("EMPTY", empty), ("MISSING KEYS", missing)):
        if lst:
            print(f"\n-- {label} (showing up to 10 of {len(lst)}) --", flush=True)
            for b in lst[:10]:
                print(f"   {json.dumps(b)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
