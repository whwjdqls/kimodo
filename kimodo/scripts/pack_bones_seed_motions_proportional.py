# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack the per-actor (shape-aware) BONES-SEED kimodo motion NPZs into a
single mmap-able .pt file, exactly mirroring pack_bones_seed_motions.py's
layout, plus an actor-deduplicated shape table.

The same actor (e.g. ``A001``) appears in many motions and has the same
body, so neutral_joints are stored once per actor (522 unique actors over
141,541 clean motions in BONES-SEED).  Each motion's body is looked up via:

    actor_idx = pack['motion_actor_idx'][i]
    neutrals  = pack['actor_neutrals'][actor_idx]   # (77, 3)

Walks /weka/jungbin/seed/soma_proportional_motions_20fps recursively, skips
the NaN-tainted files identified by _nan_audit_proportional.json, and writes
one .pt with keys:

    {
        'names':            List[str]              # motion stems in pack order
        'offsets':          Tensor int64 [N+1]     # cumulative frame counts
        'local_rot_mats':   Tensor f32 [N_frames, 77, 3, 3]
        'root_positions':  Tensor f32 [N_frames, 3]
        'actor_names':      List[str]              # length N_actors, e.g. 'A001'
        'actor_neutrals':   Tensor f32 [N_actors, 77, 3]
        'motion_actor_idx': Tensor int32 [N_motions]
    }

Sanity-checked at pack time: for each actor, neutral_joints across all of
that actor's motions must be identical (same BVH HIERARCHY -> same body).
A bone-length discrepancy >1e-5 m raises a ValueError.
"""
from __future__ import annotations
import argparse, glob, json, os, re, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


DEFAULT_ROOT       = Path("/weka/jungbin/seed/soma_proportional_motions_20fps")
DEFAULT_NAN_REPORT = Path("/home/jungbin_cho/_nan_audit_proportional.json")
DEFAULT_OUT        = Path("/weka/jungbin/kimodo_caches/bones_seed_proportional_raw.pt")


_ACTOR_RE = re.compile(r"__(A\d+)")


def _read_one(path_str: str):
    """Worker: load one NPZ and return what the packer needs.
    Runs in a multiprocessing child; returns numpy arrays (small overhead to
    pickle vs torch tensors)."""
    path = Path(path_str)
    with np.load(path) as d:
        try:
            lr = np.ascontiguousarray(d["local_rot_mats"], dtype=np.float32)
            rp = np.ascontiguousarray(d["root_positions"], dtype=np.float32)
            nj = np.ascontiguousarray(d["neutral_joints"], dtype=np.float32)
        except KeyError:
            return None
    if lr.shape[1:] != (77, 3, 3) or rp.shape[1:] != (3,) or nj.shape != (77, 3):
        return None
    m = _ACTOR_RE.search(path.name)
    if m is None:
        return None
    return path.stem, m.group(1), lr, rp, nj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",  type=Path, default=DEFAULT_ROOT,
                    help="root of soma_proportional_motions_20fps NPZ tree")
    ap.add_argument("--nan-report", type=Path, default=DEFAULT_NAN_REPORT,
                    help="JSON from _nan_audit_proportional.py (files to skip)")
    ap.add_argument("--out",        type=Path, default=DEFAULT_OUT,
                    help="output .pt path")
    ap.add_argument("--split",      type=Path, default=None,
                    help="optional split file (list of motion stems); "
                         "default = all NPZs under --data-root except NaN-flagged")
    ap.add_argument("--limit",      type=int, default=None,
                    help="for smoke testing; process only first N files")
    ap.add_argument("--workers",    type=int, default=32,
                    help="parallel NPZ readers (default 32)")
    args = ap.parse_args()

    # ---- 1. enumerate files ----
    if args.split is not None:
        with open(args.split) as f:
            stems = [ln.strip() for ln in f if ln.strip()]
        files = []
        for s in stems:
            matches = list(args.data_root.rglob(f"{s}.npz"))
            if matches:
                files.append(matches[0])
        print(f"split file has {len(stems)} stems; resolved {len(files)} NPZs on disk")
    else:
        files = sorted(args.data_root.glob("**/*.npz"))
        print(f"discovered {len(files)} NPZs under {args.data_root}")

    # ---- 2. drop NaN-flagged files ----
    nan_set: set[str] = set()
    if args.nan_report is not None and args.nan_report.is_file():
        nan_report = json.load(open(args.nan_report))
        nan_set = ({r["path"] for r in nan_report.get("nan_files", [])}
                   | {r["path"] for r in nan_report.get("load_error_files", [])})
        before = len(files)
        files = [p for p in files if str(p) not in nan_set]
        print(f"skipped {before - len(files)} NaN-tainted files "
              f"(left after filter: {len(files)})")

    if args.limit:
        files = files[: args.limit]
        print(f"limit applied -> {len(files)} files")

    # ---- 3. parallel read ----
    # Multiprocessing reads NPZs concurrently then returns small results to the
    # parent which stitches the pack together. Preserves input file order via
    # an index-keyed dict so the final pack ordering is deterministic.
    print(f"reading {len(files)} NPZ with {args.workers} workers...")
    results: dict[int, tuple] = {}
    skipped_missing_key = 0
    t0 = time.time()

    paths_str = [str(p) for p in files]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_read_one, p): i for i, p in enumerate(paths_str)}
        for done_count, fut in enumerate(as_completed(futs), 1):
            idx = futs[fut]
            r = fut.result()
            if r is None:
                skipped_missing_key += 1
            else:
                results[idx] = r
            if done_count % 5000 == 0:
                dt = time.time() - t0
                rate = done_count / dt
                eta  = (len(files) - done_count) / rate
                print(
                    f"  read {done_count}/{len(files)}  ok={len(results)}  "
                    f"bad={skipped_missing_key}  rate={rate:.0f}/s  "
                    f"eta={eta/60:.0f} min"
                )

    # ---- 4. stitch in original file order ----
    kept_names: list[str] = []
    actor_names: list[str] = []         # de-duplicated, order-preserving
    actor_to_idx: dict[str, int] = {}
    motion_actor_idx: list[int] = []
    actor_neutrals_by_id: dict[int, torch.Tensor] = {}

    local_rot_parts: list[torch.Tensor] = []
    root_pos_parts:  list[torch.Tensor] = []
    offsets: list[int] = [0]

    for i in range(len(paths_str)):
        r = results.get(i)
        if r is None:
            continue
        name, actor_id, lr, rp, nj = r
        # actor dedup + consistency check
        a_idx = actor_to_idx.get(actor_id)
        if a_idx is None:
            a_idx = len(actor_names)
            actor_names.append(actor_id)
            actor_to_idx[actor_id] = a_idx
            actor_neutrals_by_id[a_idx] = torch.from_numpy(nj)
        else:
            prev = actor_neutrals_by_id[a_idx]
            d = float((prev - torch.from_numpy(nj)).abs().max())
            if d > 1e-5:
                raise ValueError(
                    f"Actor {actor_id} has inconsistent neutral_joints across "
                    f"motions: max abs diff {d:.3e} m  (file: {name})"
                )

        kept_names.append(name)
        motion_actor_idx.append(a_idx)
        local_rot_parts.append(torch.from_numpy(lr))
        root_pos_parts.append(torch.from_numpy(rp))
        offsets.append(offsets[-1] + lr.shape[0])

    print(
        f"\nconcatenating {len(kept_names)} motions, "
        f"{offsets[-1]} total frames, distinct actors={len(actor_names)}, "
        f"skipped_bad={skipped_missing_key}"
    )

    local_rot_cat = torch.cat(local_rot_parts, dim=0).contiguous()
    root_pos_cat  = torch.cat(root_pos_parts,  dim=0).contiguous()
    actor_neutrals = torch.stack([actor_neutrals_by_id[i] for i in range(len(actor_names))], dim=0).contiguous()
    motion_actor_idx_t = torch.tensor(motion_actor_idx, dtype=torch.int32)
    offsets_t = torch.tensor(offsets, dtype=torch.int64)

    print(
        f"sizes:\n"
        f"  local_rot_mats     {tuple(local_rot_cat.shape)} "
        f"({local_rot_cat.numel() * 4 / 1024 / 1024:.0f} MiB)\n"
        f"  root_positions    {tuple(root_pos_cat.shape)} "
        f"({root_pos_cat.numel() * 4 / 1024 / 1024:.1f} MiB)\n"
        f"  actor_neutrals     {tuple(actor_neutrals.shape)} "
        f"({actor_neutrals.numel() * 4 / 1024 :.1f} KiB)\n"
        f"  motion_actor_idx   {tuple(motion_actor_idx_t.shape)} "
        f"({motion_actor_idx_t.numel() * 4 / 1024:.0f} KiB)\n"
        f"  offsets            {tuple(offsets_t.shape)} ({offsets_t.numel()*8/1024:.0f} KiB)"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "names":            kept_names,
            "offsets":          offsets_t,
            "local_rot_mats":   local_rot_cat,
            "root_positions":   root_pos_cat,
            "actor_names":      actor_names,
            "actor_neutrals":   actor_neutrals,
            "motion_actor_idx": motion_actor_idx_t,
        },
        args.out,
    )
    sz_mb = args.out.stat().st_size / 1024 / 1024
    print(f"wrote {args.out} ({sz_mb:.0f} MiB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
