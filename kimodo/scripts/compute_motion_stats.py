
"""Compute KIMODO motion-feature normalization stats from a dataset of NPZ motions.

Reproduces the directory layout used by ``KimodoMotionRep`` at inference time:

    <output_dir>/
        global_root/{mean,std}.npy   shape (5,)
        local_root/{mean,std}.npy    shape (4,)
        body/{mean,std}.npy          shape (364,)

The 369-dim full KIMODO motion features are split into three blocks:
- ``global_root`` (5): smooth_root_pos(3) + global_root_heading(2)
- ``body`` (364): local_joints_positions + global_rot_data + velocities + foot_contacts
- ``local_root`` (4): local_root_rot_vel + local_root_xz_vel + global_root_y
  (derived from the global root features via
  ``motion_rep.global_root_to_local_root``; matches the inference normalization).

Stats are accumulated across all valid frames in float64 using Welford's
algorithm to stay numerically stable.

Usage:
    python -m kimodo.scripts.compute_motion_stats \\
        --data-root /weka/jungbin/seed/soma_uniform_motions_20fps \\
        --output-dir /weka/jungbin/stats_soma_20fps \\
        --fps 20 [--num-workers 16] [--include-mirrored]
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton30

log = logging.getLogger("kimodo.compute_motion_stats")


# -----------------------------------------------------------------------------
# Single-motion feature extractor (runs in a DataLoader worker)
# -----------------------------------------------------------------------------
class _MotionFeatureDataset(Dataset):
    """Iterates motion NPZ files; emits unnormalized KIMODO features per motion."""

    def __init__(
        self,
        paths: List[Path],
        fps: int,
        random_heading: bool = True,
        seed: int = 0,
    ):
        self.paths = paths
        self.fps = int(fps)
        self.skeleton = SOMASkeleton30()
        self.motion_rep = KimodoMotionRep(skeleton=self.skeleton, fps=self.fps, stats_path=None)
        self.random_heading = bool(random_heading)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        p = self.paths[index]
        try:
            with np.load(p, mmap_mode="r") as data:
                local_rot_77 = np.asarray(data["local_rot_mats"])  # (T, 77, 3, 3)
                root_pos = np.asarray(data["root_positions"])      # (T, 3)
        except Exception as e:
            log.warning("Skipping %s due to load error: %s", p, e)
            return None

        local_rot_77 = torch.from_numpy(local_rot_77).float()
        root_pos = torch.from_numpy(root_pos).float()
        local_rot_30 = self.skeleton.from_SOMASkeleton77(local_rot_77)
        T = local_rot_30.shape[0]
        if T < 2:
            return None

        if self.random_heading:
            # Deterministic per-file random Y rotation, so reruns are reproducible.
            rng = np.random.default_rng(self.seed + index)
            angle = float(rng.uniform(-np.pi, np.pi))
            c, s = float(np.cos(angle)), float(np.sin(angle))
            Ry = torch.tensor(
                [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                dtype=local_rot_30.dtype,
            )
            root_pos = root_pos @ Ry.T
            root_idx = int(self.skeleton.root_idx)
            local_rot_30 = local_rot_30.clone()
            local_rot_30[:, root_idx] = torch.einsum(
                "ij,tjk->tik", Ry, local_rot_30[:, root_idx]
            )

        lengths = torch.tensor([T])
        # Unnormalized features in KIMODO layout (B=1, T, 369).
        feats = self.motion_rep(
            local_rot_30.unsqueeze(0),
            root_pos.unsqueeze(0),
            to_normalize=False,
            to_canonicalize=False,
            lengths=lengths,
        )[0]  # (T, 369)
        # Local-root features derived from the global root block.
        root_block = feats[..., self.motion_rep.root_slice]  # (T, 5)
        local_root = self.motion_rep.global_root_to_local_root(
            root_block.unsqueeze(0),
            normalized=False,
            lengths=lengths,
        )[0]  # (T, 4)
        body_block = feats[..., self.motion_rep.body_slice]  # (T, 364)

        return {
            "global_root": root_block.numpy().astype(np.float64),
            "local_root": local_root.numpy().astype(np.float64),
            "body": body_block.numpy().astype(np.float64),
        }


def _collate_pass_through(batch):
    return [x for x in batch if x is not None]


# -----------------------------------------------------------------------------
# Welford running mean/var
# -----------------------------------------------------------------------------
class WelfordAccum:
    """Numerically-stable streaming mean / variance per feature.

    Implements the chunk merge formula (Chan et al.) so we can update with batches
    of frames at a time.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n: int = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        # x shape: (M, dim)
        if x.size == 0:
            return
        m = x.shape[0]
        new_mean = x.mean(axis=0)
        new_m2 = ((x - new_mean) ** 2).sum(axis=0)

        if self.n == 0:
            self.n = m
            self.mean = new_mean
            self.m2 = new_m2
            return

        delta = new_mean - self.mean
        total = self.n + m
        self.mean = self.mean + delta * (m / total)
        self.m2 = self.m2 + new_m2 + (delta ** 2) * (self.n * m / total)
        self.n = total

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.mean)
        return np.sqrt(self.m2 / (self.n - 1))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def find_motion_files(data_root: Path, include_mirrored: bool) -> List[Path]:
    files: List[Path] = []
    for p in data_root.rglob("*.npz"):
        if not include_mirrored and p.stem.endswith("_M"):
            continue
        files.append(p)
    files.sort()
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str,
                   help="Where to write {global_root,local_root,body}/{mean,std}.npy.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--include-mirrored", action="store_true",
                   help="Also include filenames ending in '_M'. Default off.")
    p.add_argument("--random-heading", action="store_true", default=True,
                   help="Apply random Y-axis heading rotation per motion before computing features (matches training-time aug). Default on.")
    p.add_argument("--no-random-heading", dest="random_heading", action="store_false",
                   help="Disable the random heading rotation.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the per-file deterministic random heading rotation.")
    p.add_argument("--max-files", type=int, default=0,
                   help="Optional cap on number of files (for debugging).")
    p.add_argument("--std-eps", type=float, default=1e-4,
                   help="Floor applied to std before saving, to avoid 0 std features.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Scanning %s ...", data_root)
    files = find_motion_files(data_root, include_mirrored=args.include_mirrored)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No NPZ files found under {data_root}")
    log.info("Found %d motion files (include_mirrored=%s).", len(files), args.include_mirrored)

    dataset = _MotionFeatureDataset(
        files, fps=args.fps, random_heading=args.random_heading, seed=args.seed,
    )
    log.info("random_heading=%s seed=%d", args.random_heading, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=_collate_pass_through,
        pin_memory=False,
    )

    motion_rep = dataset.motion_rep
    gr_acc = WelfordAccum(motion_rep.global_root_dim)
    lr_acc = WelfordAccum(motion_rep.local_root_dim)
    body_acc = WelfordAccum(motion_rep.body_dim)

    t0 = time.time()
    n_processed = 0
    n_failed = 0
    for batch in tqdm(loader, desc="stats", unit="motion"):
        for item in batch:
            if item is None:
                n_failed += 1
                continue
            gr_acc.update(item["global_root"])
            lr_acc.update(item["local_root"])
            body_acc.update(item["body"])
            n_processed += 1

    dt = time.time() - t0
    log.info(
        "Done in %.1fs: %d motions ok, %d failed. Frames: gr=%d lr=%d body=%d",
        dt, n_processed, n_failed, gr_acc.n, lr_acc.n, body_acc.n,
    )

    # Save in the layout expected by KimodoMotionRep stats loader.
    def _save(name: str, acc: WelfordAccum) -> None:
        out = output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        std = acc.std.astype(np.float32)
        std = np.maximum(std, float(args.std_eps))
        np.save(out / "mean.npy", acc.mean.astype(np.float32))
        np.save(out / "std.npy", std)
        log.info("  wrote %s: mean.shape=%s std.shape=%s (min std=%.4g)",
                 name, acc.mean.shape, std.shape, float(std.min()))

    _save("global_root", gr_acc)
    _save("local_root", lr_acc)
    _save("body", body_acc)

    # Sanity: total frames must match across blocks (they are derived from the
    # same motions). Local-root has T-1 valid frames if velocities are end-padded
    # but our motion_rep duplicates the last frame so all blocks share T per file.
    log.info("Stats written to %s", output_dir)


if __name__ == "__main__":
    main()
