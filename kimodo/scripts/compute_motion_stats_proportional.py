"""Compute KIMODO motion-feature normalization stats for the SHAPE-AWARE
(proportional) BONES-SEED dataset.

Mirrors ``compute_motion_stats.py`` exactly, with two changes:

  1. The features are computed via ``motion_rep(..., neutral_joints=...)`` using
     each motion's per-actor rest pose. Position / velocity / FK feature blocks
     therefore reflect the actor's actual bone lengths — NOT the canonical
     SOMA-30 T-pose — so the resulting stats differ from the shape-unaware
     ones (typically by ~5-15% on local_joints_positions / velocities).

  2. The motion+actor index is read from
     ``/weka/jungbin/kimodo_caches/bones_seed_proportional_raw.pt`` (the
     actor-deduplicated pack with ``actor_neutrals`` and ``motion_actor_idx``).
     The pack's ``names`` list is the authoritative motion list — we don't walk
     the NPZ tree, so NaN-tainted files dropped at pack time stay dropped.

Output layout (same as the shape-unaware variant — drop-in replacement at the
``stats_path`` config knob)::

    <output_dir>/
        global_root/{mean,std}.npy   shape (5,)
        local_root/{mean,std}.npy    shape (4,)
        body/{mean,std}.npy          shape (364,)

Usage:
    python -m kimodo.scripts.compute_motion_stats_proportional \\
        --packed-motions-path /weka/jungbin/kimodo_caches/bones_seed_proportional_raw.pt \\
        --output-dir /weka/jungbin/kimodo_caches/stats/proportional \\
        --fps 20 --num-workers 16
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton30
from kimodo.scripts.compute_motion_stats import WelfordAccum, _collate_pass_through

log = logging.getLogger("kimodo.compute_motion_stats_proportional")


class _ProportionalMotionFeatureDataset(Dataset):
    """Iterates the actor-dedup pack; emits unnormalized KIMODO features per motion.

    Each worker opens the pack mmap independently; the OS pages in only the
    touched rows so memory stays bounded even on a 54 GiB pack.
    """

    def __init__(
        self,
        packed_motions_path: Path,
        fps: int,
        random_heading: bool = True,
        seed: int = 0,
        max_motions: int = 0,
    ):
        self.packed_motions_path = Path(packed_motions_path)
        self.fps = int(fps)
        self.skeleton = SOMASkeleton30()
        self.motion_rep = KimodoMotionRep(skeleton=self.skeleton, fps=self.fps, stats_path=None)
        self.random_heading = bool(random_heading)
        self.seed = int(seed)

        log.info("Loading proportional motion pack from %s ...", self.packed_motions_path)
        try:
            blob = torch.load(
                str(self.packed_motions_path), map_location="cpu",
                weights_only=False, mmap=True,
            )
        except (RuntimeError, TypeError):
            blob = torch.load(
                str(self.packed_motions_path), map_location="cpu", weights_only=False,
            )
        for key in ("offsets", "local_rot_mats", "root_positions",
                    "actor_neutrals", "motion_actor_idx", "names"):
            if key not in blob:
                raise KeyError(
                    f"Pack {self.packed_motions_path} missing required key '{key}' — "
                    "rebuild with pack_bones_seed_motions_proportional.py.",
                )
        self._offsets = blob["offsets"]
        self._local_rot = blob["local_rot_mats"]
        self._root_pos = blob["root_positions"]
        self._motion_actor_idx = blob["motion_actor_idx"]
        idx30_in_77 = self.skeleton.get_skel_slice(self.skeleton.somaskel77)
        self._actor_neutrals_30 = blob["actor_neutrals"][:, idx30_in_77].contiguous()
        self._n_motions = int(self._motion_actor_idx.shape[0])
        if max_motions > 0:
            self._n_motions = min(self._n_motions, int(max_motions))

        log.info(
            "Loaded pack: %d motions, %d actors, %d total frames.",
            self._n_motions, self._actor_neutrals_30.shape[0],
            int(self._offsets[-1]),
        )

    def __len__(self) -> int:
        return self._n_motions

    def __getitem__(self, index: int):
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        T = end - start
        if T < 2:
            return None

        local_rot_77 = self._local_rot[start:end]
        root_pos = self._root_pos[start:end]
        local_rot_77 = local_rot_77.float() if local_rot_77.dtype != torch.float32 else local_rot_77
        root_pos = root_pos.float() if root_pos.dtype != torch.float32 else root_pos
        local_rot_30 = self.skeleton.from_SOMASkeleton77(local_rot_77)

        a_idx = int(self._motion_actor_idx[index])
        neutrals_30 = self._actor_neutrals_30[a_idx].clone()                # (30, 3)

        if self.random_heading:
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
                "ij,tjk->tik", Ry, local_rot_30[:, root_idx],
            )
            # Rotate the actor's neutrals into the new heading too — otherwise FK
            # is run with mismatched rotational frames between rotations and bones.
            neutrals_30 = neutrals_30 @ Ry.T

        lengths = torch.tensor([T])
        feats = self.motion_rep(
            local_rot_30.unsqueeze(0),
            root_pos.unsqueeze(0),
            to_normalize=False,
            to_canonicalize=False,
            lengths=lengths,
            neutral_joints=neutrals_30.unsqueeze(0),                        # shape-aware FK
        )[0]                                                                # (T, 369)

        root_block = feats[..., self.motion_rep.root_slice]
        local_root = self.motion_rep.global_root_to_local_root(
            root_block.unsqueeze(0), normalized=False, lengths=lengths,
        )[0]
        body_block = feats[..., self.motion_rep.body_slice]

        return {
            "global_root": root_block.numpy().astype(np.float64),
            "local_root": local_root.numpy().astype(np.float64),
            "body": body_block.numpy().astype(np.float64),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--packed-motions-path", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--random-heading", action="store_true", default=True)
    p.add_argument("--no-random-heading", dest="random_heading", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-motions", type=int, default=0,
                   help="Optional cap (for debugging). Default 0 = use all.")
    p.add_argument("--std-eps", type=float, default=1e-4)
    p.add_argument("--canonical-stats-dir", type=str, default=None,
                   help="If set, prints the per-feature delta vs the canonical "
                        "(shape-unaware) stats at that path. Read-only.")
    return p.parse_args()


def _print_delta_vs_canonical(out_dir: Path, canonical_dir: Path) -> None:
    """Print stats deltas vs canonical — sanity check that proportional is different."""
    for name in ("global_root", "local_root", "body"):
        prop_mean = np.load(out_dir / name / "mean.npy")
        prop_std = np.load(out_dir / name / "std.npy")
        canon_mean = np.load(canonical_dir / name / "mean.npy")
        canon_std = np.load(canonical_dir / name / "std.npy")
        d_mean = np.abs(prop_mean - canon_mean)
        d_std = np.abs(prop_std - canon_std)
        log.info(
            "  %s: |Δmean| max=%.3e median=%.3e   |Δstd| max=%.3e median=%.3e",
            name, d_mean.max(), float(np.median(d_mean)),
            d_std.max(), float(np.median(d_std)),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _ProportionalMotionFeatureDataset(
        Path(args.packed_motions_path),
        fps=args.fps,
        random_heading=args.random_heading,
        seed=args.seed,
        max_motions=args.max_motions,
    )
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

    log.info(
        "Done in %.1fs: %d motions ok, %d failed. Frames: gr=%d lr=%d body=%d",
        time.time() - t0, n_processed, n_failed, gr_acc.n, lr_acc.n, body_acc.n,
    )

    def _save(name: str, acc: WelfordAccum) -> None:
        out = output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        std = acc.std.astype(np.float32)
        std = np.maximum(std, float(args.std_eps))
        np.save(out / "mean.npy", acc.mean.astype(np.float32))
        np.save(out / "std.npy", std)
        log.info(
            "  wrote %s: mean.shape=%s std.shape=%s (min std=%.4g)",
            name, acc.mean.shape, std.shape, float(std.min()),
        )

    _save("global_root", gr_acc)
    _save("local_root", lr_acc)
    _save("body", body_acc)
    log.info("Stats written to %s", output_dir)

    if args.canonical_stats_dir is not None:
        canon = Path(args.canonical_stats_dir)
        if canon.is_dir():
            log.info("Delta vs canonical stats (%s):", canon)
            _print_delta_vs_canonical(output_dir, canon)


if __name__ == "__main__":
    main()
