# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOMA text-to-motion dataset loader (segment-based, matches benchmark style).

Each training sample is a single temporal segment of a SOMA Uniform motion,
paired with the segment's natural-language description from
``seed_metadata_v002_temporal_labels.jsonl``.  This mirrors the
``Kimodo-Motion-Gen-Benchmark`` testsuite construction in
``benchmark/create_benchmark.py``, where every test example crops a specific
``[crop_start_frame_index, crop_end_frame_index]`` window from a BVH and uses
that segment's description as the text prompt.

Per-fetch processing:

1. Load the pre-canonicalized 30-joint-equivalent motion from a Kimodo NPZ
   (these NPZs are produced by ``benchmark/create_data.py``; that script
   canonicalises the FULL clip so frame 0 of the full motion is at origin with
   heading 0).
2. Slice to the temporal segment ``[start_frame_20fps, end_frame_20fps]``.
3. Cap to ``max_frames`` (default 200 = 10 s @ 20 fps); pad zeros if shorter.
4. Re-canonicalise the sub-clip via ``motion_rep(... to_canonicalize=True)``
   (frame 0 of the *sub-clip* gets heading=0 and xz=(0,0)).
5. Optional random Y-rotation augmentation on the canonicalised features (paper
   does this; the model is conditioned on ``first_heading_angle``).
6. Normalise with the motion_rep stats.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton30

log = logging.getLogger(__name__)


@dataclass
class SegmentEntry:
    motion_path: str
    filename: str
    start_frame: int      # inclusive, at dataset fps
    end_frame: int        # exclusive
    text: str


def _walk_motion_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.npz"))


class SOMATextMotionDataset(Dataset):
    """Text-to-motion dataset of temporal segments.

    Args:
        data_root: Directory of Kimodo motion NPZs (e.g. ``soma_uniform_motions_20fps``).
        temporal_labels_path: Path to ``seed_metadata_v002_temporal_labels.jsonl``.
        fps: Sampling rate of the on-disk motions (default 20).
        max_frames: Hard cap on segment length in frames (default 200 = 10 s).
        min_frames: Minimum segment length in frames; shorter segments are dropped (default 10).
        include_mirrored: Include ``*_M`` mirrored motions if their NPZs exist on disk.
        skeleton / motion_rep / stats_path: Optional preloaded helpers; if not given, instantiated.
        normalize: Normalise final features (must be True for training).
        random_heading_aug: Apply per-sample random Y rotation after canonicalisation.
        cache_index: Path to cache the segment index as JSON (much faster startup).
        seed: RNG seed for window/heading sampling.
    """

    def __init__(
        self,
        data_root: str | Path,
        temporal_labels_path: str | Path,
        fps: int = 20,
        max_frames: int = 200,
        min_frames: int = 10,
        include_mirrored: bool = True,
        skeleton: Optional[SOMASkeleton30] = None,
        motion_rep: Optional[KimodoMotionRep] = None,
        stats_path: Optional[str | Path] = None,
        normalize: bool = True,
        random_heading_aug: bool = True,
        cache_index: Optional[str | Path] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.temporal_labels_path = Path(temporal_labels_path)
        self.fps = int(fps)
        self.max_frames = int(max_frames)
        self.min_frames = int(min_frames)
        self.include_mirrored = bool(include_mirrored)
        self.normalize = bool(normalize)
        self.random_heading_aug = bool(random_heading_aug)

        if skeleton is None:
            skeleton = SOMASkeleton30()
        self.skeleton = skeleton

        if motion_rep is None:
            motion_rep = KimodoMotionRep(
                skeleton=self.skeleton,
                fps=self.fps,
                stats_path=str(stats_path) if stats_path is not None else None,
            )
        self.motion_rep = motion_rep

        if self.normalize and not hasattr(self.motion_rep, "stats"):
            raise ValueError(
                "normalize=True but motion_rep has no stats; pass stats_path."
            )

        self._rng = random.Random(seed)
        self.entries: List[SegmentEntry] = self._build_index(cache_index)
        if not self.entries:
            raise RuntimeError(f"No segments built from {self.data_root}")
        log.info(
            "SOMATextMotionDataset: %d segments (fps=%d, max_frames=%d, min_frames=%d).",
            len(self.entries), self.fps, self.max_frames, self.min_frames,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _build_index(self, cache_index: Optional[str | Path]) -> List[SegmentEntry]:
        if cache_index is not None and Path(cache_index).is_file():
            log.info("Loading cached segment index from %s", cache_index)
            try:
                with open(cache_index, "r") as f:
                    raw = json.load(f)
                return [SegmentEntry(**e) for e in raw]
            except (json.JSONDecodeError, TypeError) as e:
                log.warning("Cached index %s corrupted (%s); rebuilding.", cache_index, e)

        log.info("Scanning motion files under %s ...", self.data_root)
        motion_paths = _walk_motion_files(self.data_root)
        path_by_name = {p.stem: p for p in motion_paths}
        log.info("Found %d motion files.", len(motion_paths))

        if not self.temporal_labels_path.is_file():
            raise FileNotFoundError(
                f"Temporal labels file not found: {self.temporal_labels_path}"
            )
        log.info("Reading temporal labels from %s", self.temporal_labels_path)

        entries: List[SegmentEntry] = []
        skipped_no_motion = 0
        skipped_short = 0
        skipped_mirror = 0
        with open(self.temporal_labels_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                filename = obj["filename"]
                if not self.include_mirrored and filename.endswith("_M"):
                    skipped_mirror += 1
                    continue
                p = path_by_name.get(filename)
                if p is None:
                    skipped_no_motion += 1
                    continue
                for ev in obj["events"]:
                    start = max(0, int(round(float(ev["start_time"]) * self.fps)))
                    end = int(round(float(ev["end_time"]) * self.fps))
                    if end - start < self.min_frames:
                        skipped_short += 1
                        continue
                    text = ev.get("description")
                    if not text or not isinstance(text, str):
                        continue
                    entries.append(
                        SegmentEntry(
                            motion_path=str(p),
                            filename=filename,
                            start_frame=start,
                            end_frame=end,
                            text=text.strip(),
                        )
                    )

        log.info(
            "Built %d segments (skipped: no_motion=%d short<%d=%d mirror=%d).",
            len(entries), skipped_no_motion, self.min_frames, skipped_short, skipped_mirror,
        )

        if cache_index is not None:
            cache_path = Path(cache_index)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
            with open(tmp_path, "w") as fout:
                json.dump([e.__dict__ for e in entries], fout)
            os.replace(tmp_path, cache_path)
            log.info("Wrote segment-index cache: %s", cache_path)

        return entries

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def _load_segment(
        self, entry: SegmentEntry,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (local_rot_30, root_positions, valid_length) for the segment.

        Crops the segment from the NPZ, caps to ``max_frames``. ``valid_length`` is
        the actual frame count read (no padding here).
        """
        start = entry.start_frame
        end = entry.end_frame
        # Cap to max_frames by taking the FIRST N frames of the segment.
        end = min(end, start + self.max_frames)

        with np.load(entry.motion_path, mmap_mode="r") as data:
            n_total = int(data["local_rot_mats"].shape[0])
            start_c = max(0, min(start, n_total))
            end_c = max(start_c, min(end, n_total))
            local_rot_77 = np.asarray(data["local_rot_mats"][start_c:end_c])
            root_positions = np.asarray(data["root_positions"][start_c:end_c])

        T = int(local_rot_77.shape[0])
        local_rot_77 = torch.from_numpy(local_rot_77).float()
        root_positions = torch.from_numpy(root_positions).float()
        local_rot_30 = self.skeleton.from_SOMASkeleton77(local_rot_77)
        return local_rot_30, root_positions, T

    def _features_with_canonicalize(
        self, local_rot_30: torch.Tensor, root_positions: torch.Tensor, T: int,
    ) -> torch.Tensor:
        lengths = torch.tensor([T])
        feats = self.motion_rep(
            local_rot_30.unsqueeze(0),
            root_positions.unsqueeze(0),
            to_normalize=False,
            to_canonicalize=True,
            lengths=lengths,
        )[0]
        return feats  # (T, 369)

    def _apply_random_heading(self, features: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Rotate canonicalised features to a random heading.  Returns (features, angle)."""
        if not self.random_heading_aug:
            return features, 0.0
        angle = self._rng.uniform(-np.pi, np.pi)
        angle_t = torch.tensor([angle], dtype=features.dtype)
        rotated = self.motion_rep.rotate(features.unsqueeze(0), angle_t)[0]
        return rotated, float(angle)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        local_rot, root_pos, T = self._load_segment(entry)
        if T < self.min_frames:
            # Fall back to another sample deterministically (rare edge case).
            return self[(index + 1) % len(self)]

        features = self._features_with_canonicalize(local_rot, root_pos, T)  # (T, 369)
        features, _angle = self._apply_random_heading(features)

        # First-frame heading angle of the (rotated) features — feed to the model.
        heading_cs = features[0, self.motion_rep.slice_dict["global_root_heading"]]
        first_heading = float(torch.atan2(heading_cs[1], heading_cs[0]).item())

        if self.normalize:
            features = self.motion_rep.normalize(features)

        return {
            "motion": features,                          # (T, D)  T <= max_frames
            "length": int(features.shape[0]),
            "text": entry.text,
            "first_heading_angle": first_heading,
            "filename": entry.filename,
            "start_frame": entry.start_frame,
        }


def build_collate_fn(pad_value: float = 0.0, pad_to: Optional[int] = None) -> Callable[[List[dict]], dict]:
    """Collate function with zero-padding and a boolean pad mask.

    If ``pad_to`` is given, every batch is padded to that fixed length (e.g. 200
    frames). Otherwise pads to the max length in the batch.
    """

    def collate(batch: List[dict]) -> dict:
        lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
        max_len = int(pad_to) if pad_to is not None else int(lengths.max().item())
        feat_dim = batch[0]["motion"].shape[-1]
        B = len(batch)

        motions = torch.full((B, max_len, feat_dim), pad_value, dtype=batch[0]["motion"].dtype)
        pad_mask = torch.zeros((B, max_len), dtype=torch.bool)
        for i, item in enumerate(batch):
            T = item["length"]
            motions[i, :T] = item["motion"]
            pad_mask[i, :T] = True

        return {
            "motion": motions,
            "lengths": lengths,
            "pad_mask": pad_mask,
            "text": [item["text"] for item in batch],
            "first_heading_angle": torch.tensor(
                [item["first_heading_angle"] for item in batch], dtype=torch.float32,
            ),
            "filename": [item["filename"] for item in batch],
            "start_frame": [item["start_frame"] for item in batch],
        }

    return collate
