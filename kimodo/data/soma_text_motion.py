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

import csv
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton30

log = logging.getLogger(__name__)

# this is debug
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

        out = {
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
        # Shape-aware path: SOMABonesSeedDatasetShapeAware emits per-sample
        # rest-pose joints. Stack them when present; shape-unaware datasets
        # never include this key so the legacy collate output is unchanged.
        if "neutral_joints" in batch[0]:
            out["neutral_joints"] = torch.stack(
                [item["neutral_joints"] for item in batch], dim=0,
            )
        return out

    return collate


# =====================================================================
# BONES-SEED text-to-motion dataset (mixture of three source kinds)
# =====================================================================
@dataclass
class _BSEntry:
    """One (text, motion-segment) entry in the BONES-SEED mixture.

    ``seg_start_sec`` / ``seg_end_sec`` are the source segment bounds in seconds
    (within the on-disk NPZ at ``fps``).  The actual fetched window is
    determined per-call by :py:meth:`SOMABonesSeedDataset._resolve_segment`.
    """

    source: str               # "natural" | "single" | "multi"
    motion_path: str
    filename: str
    seg_start_sec: float
    seg_end_sec: float
    text: str


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "1.0", "yes", "y", "t")
    return False


class SOMABonesSeedDataset(Dataset):
    """BONES-SEED text-to-motion dataset.

    Samples (text, motion segment) pairs from a 1:1:1 mixture of three sources:

      * ``natural`` — a full SOMA clip paired with one of its 4 natural-language
        descriptions in ``seed_metadata_v004.csv``.
      * ``single``  — a single labeled temporal event from
        ``seed_metadata_v002_temporal_labels.jsonl``.
      * ``multi``   — a merged 2-to-3 event segment from ``multi_timeline.jsonl``.

    Per-source segment-duration rule (T = segment length in seconds):

      * ``T > max_segment_sec``         → disregard at index build time.
      * ``max_clip_sec ≤ T ≤ max_segment_sec``
        → start at a uniform offset in ``[0, min(rand_offset_max_sec,
        T - max_clip_sec)]`` and take a ``max_clip_sec`` window.
      * ``T < max_clip_sec``            → use the whole segment as-is.

    The window is then loaded from the 20 fps NPZ, re-canonicalised, optionally
    rotated by a random heading, and normalised — same pipeline as
    :class:`SOMATextMotionDataset`.

    Args:
        data_root: Directory of Kimodo motion NPZs (``soma_uniform_motions_20fps``).
        natural_csv_path: Path to ``seed_metadata_v004.csv``.
        temporal_labels_path: Path to ``seed_metadata_v002_temporal_labels.jsonl``.
        multi_timeline_path: Path to ``multi_timeline.jsonl``.
        train_split_path: Path to a text file with one ``<rel_dir>/<filename>``
            (no extension) per line. Restricts the dataset to those filenames.
            ``None`` falls back to a recursive glob of ``data_root``.
        fps: Sampling rate of the on-disk motions (default 20).
        max_clip_sec: Output clip cap in seconds (default 10).
        max_segment_sec: Disregard source segments longer than this (default 15).
        rand_offset_max_sec: Max random start-offset when ``T ≥ max_clip_sec``
            (default 2).
        min_frames: Drop segments whose final length is below this (default 10).
        include_mirrored: Include ``*_M`` mirrored motions if present.
        source_weights: Reserved for future weighted sampling; the current
            implementation interleaves indices 1:1:1 across the three sources.
        cache_index: JSON path to cache the segment index (much faster startup).
        seed: RNG seed for per-fetch offset & heading sampling.
    """

    SOURCES: Tuple[str, str, str] = ("natural", "single", "multi")

    def __init__(
        self,
        data_root: str | Path,
        natural_csv_path: str | Path,
        temporal_labels_path: str | Path,
        multi_timeline_path: str | Path,
        train_split_path: Optional[str | Path] = None,
        fps: int = 20,
        max_clip_sec: float = 10.0,
        max_segment_sec: float = 15.0,
        rand_offset_max_sec: float = 2.0,
        min_frames: int = 10,
        include_mirrored: bool = True,
        source_weights: Optional[Tuple[float, float, float]] = None,
        skeleton: Optional[SOMASkeleton30] = None,
        motion_rep: Optional[KimodoMotionRep] = None,
        stats_path: Optional[str | Path] = None,
        normalize: bool = True,
        random_heading_aug: bool = True,
        cache_index: Optional[str | Path] = None,
        seed: Optional[int] = None,
        packed_motions_path: Optional[str | Path] = None,
        packed_features_path: Optional[str | Path] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.natural_csv_path = Path(natural_csv_path)
        self.temporal_labels_path = Path(temporal_labels_path)
        self.multi_timeline_path = Path(multi_timeline_path)
        self.train_split_path = Path(train_split_path) if train_split_path else None
        self.packed_motions_path = Path(packed_motions_path) if packed_motions_path else None
        self.packed_features_path = Path(packed_features_path) if packed_features_path else None
        self._packed = False             # raw-rotations pack (Tier 1)
        self._packed_features = False    # precomputed-features pack (Tier 2)

        self.fps = int(fps)
        self.max_clip_sec = float(max_clip_sec)
        self.max_segment_sec = float(max_segment_sec)
        self.rand_offset_max_sec = float(rand_offset_max_sec)
        self.max_frames = int(round(self.max_clip_sec * self.fps))
        self.min_frames = int(min_frames)
        self.include_mirrored = bool(include_mirrored)
        self.normalize = bool(normalize)
        self.random_heading_aug = bool(random_heading_aug)

        if source_weights is not None:
            sw = np.asarray(source_weights, dtype=np.float64)
            if sw.shape != (3,) or float(sw.sum()) <= 0:
                raise ValueError("source_weights must be a length-3 positive tuple.")
            self.source_weights = tuple(float(x) for x in (sw / sw.sum()))
        else:
            self.source_weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)

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
        self._pools: Dict[str, List[_BSEntry]] = self._build_index(cache_index)
        for src in self.SOURCES:
            if not self._pools[src]:
                raise RuntimeError(f"Empty pool for BONES-SEED source '{src}'")

        self._pool_lens: Dict[str, int] = {s: len(self._pools[s]) for s in self.SOURCES}
        self._cycle_len: int = max(self._pool_lens.values())
        log.info(
            "SOMABonesSeedDataset: pools natural=%d single=%d multi=%d "
            "(cycle_len=%d, virtual_len=%d, fps=%d, max_frames=%d).",
            self._pool_lens["natural"], self._pool_lens["single"],
            self._pool_lens["multi"], self._cycle_len, len(self),
            self.fps, self.max_frames,
        )

        if self.packed_features_path is not None:
            self._load_packed_features()
        if self.packed_motions_path is not None and not self._packed_features:
            self._load_packed_motions()

    def _load_packed_features(self) -> None:
        """Load mmap-able pack of precomputed 369-D KimodoMotionRep features.

        Built by ``pack_bones_seed_features.py``. When present, the dataset
        skips per-step ``motion_rep`` (FK + 6D + foot-contact detection) and
        slices into the packed features instead. Canonicalize + random heading
        + normalize still happen at runtime; they're cheap rotates/stats on
        the 369-D vectors. Mirrors how HumanML3D kimodo_rep is fast.
        """
        path = self.packed_features_path
        if not path.is_file():
            log.warning("packed_features_path %s not found; falling back", path)
            return
        try:
            blob = torch.load(
                str(path), map_location="cpu", weights_only=False, mmap=True,
            )
        except (RuntimeError, TypeError):
            blob = torch.load(str(path), map_location="cpu", weights_only=False)
        # Key by BASENAME: pack names carry the split's subdir prefix
        # (e.g. "220705/Idle_Left_001__A017") but entry.filename is the bare
        # basename from the metadata. Without this, every lookup misses and the
        # dataset silently falls back to the slow NFS NPZ path.
        names = blob["names"]
        self._packed_feat_idx = {os.path.basename(n): i for i, n in enumerate(names)}
        if len(self._packed_feat_idx) != len(names):
            log.warning(
                "packed features: %d basename collisions among %d names — "
                "some motions will resolve to the wrong features",
                len(names) - len(self._packed_feat_idx), len(names),
            )
        self._packed_feat_offsets = blob["offsets"]
        self._packed_features_tensor = blob["features"]
        self._packed_features = True
        log.info(
            "SOMABonesSeedDataset: loaded packed features from %s "
            "(%d motions, %d frames, %d dims, %.0f MiB on disk)",
            path, len(self._packed_feat_idx), int(self._packed_feat_offsets[-1]),
            self._packed_features_tensor.shape[1], path.stat().st_size / 1024 / 1024,
        )

    def _load_packed_motions(self) -> None:
        """Load the mmap-able pack produced by ``pack_bones_seed_motions.py``.

        Replaces per-step NPZ open + zlib decompress with a slice into an
        mmap-resident concat tensor. Motions whose filename is not in the
        pack fall back to the NPZ path automatically in :meth:`_load_segment`.
        """
        path = self.packed_motions_path
        if not path.is_file():
            log.warning("packed_motions_path %s not found; using NPZs", path)
            return
        try:
            blob = torch.load(
                str(path), map_location="cpu", weights_only=False, mmap=True,
            )
        except (RuntimeError, TypeError):
            # Older PyTorch / legacy serialization — load without mmap.
            blob = torch.load(str(path), map_location="cpu", weights_only=False)
        # Key by basename (pack names carry the split subdir prefix; entry.filename
        # is the bare basename) — otherwise every lookup misses -> NFS fallback.
        names = blob["names"]
        self._packed_names_idx = {os.path.basename(n): i for i, n in enumerate(names)}
        if len(self._packed_names_idx) != len(names):
            log.warning(
                "packed motions: %d basename collisions among %d names",
                len(names) - len(self._packed_names_idx), len(names),
            )
        self._packed_offsets = blob["offsets"]
        self._packed_local_rot = blob["local_rot_mats"]
        self._packed_root_pos = blob["root_positions"]
        self._packed = True
        log.info(
            "SOMABonesSeedDataset: loaded packed motions from %s "
            "(%d motions, %d frames, %.0f MiB on disk)",
            path, len(self._packed_names_idx),
            int(self._packed_offsets[-1]), path.stat().st_size / 1024 / 1024,
        )

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _read_split_filenames(self) -> Optional[set]:
        if self.train_split_path is None:
            return None
        names: set = set()
        with open(self.train_split_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.add(os.path.basename(line))
        return names

    def _build_path_index(self) -> Dict[str, str]:
        """Map filename stem -> absolute NPZ path."""
        path_by_name: Dict[str, str] = {}
        if self.train_split_path is not None:
            with open(self.train_split_path, "r") as f:
                for line in f:
                    rel = line.strip()
                    if not rel:
                        continue
                    name = os.path.basename(rel)
                    path_by_name[name] = str(self.data_root / (rel + ".npz"))
        else:
            for p in _walk_motion_files(self.data_root):
                path_by_name[p.stem] = str(p)
        return path_by_name

    def _load_index_cache(
        self, cache_index: Optional[str | Path],
    ) -> Optional[Dict[str, List[_BSEntry]]]:
        if cache_index is None or not Path(cache_index).is_file():
            return None
        log.info("Loading cached BONES-SEED index from %s", cache_index)
        try:
            with open(cache_index, "r") as f:
                raw = json.load(f)
            return {src: [_BSEntry(**e) for e in raw.get(src, [])] for src in self.SOURCES}
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("Cached index %s corrupted (%s); rebuilding.", cache_index, e)
            return None

    def _save_index_cache(
        self, cache_index: str | Path, pools: Dict[str, List[_BSEntry]],
    ) -> None:
        cache_path = Path(cache_index)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        with open(tmp_path, "w") as fout:
            json.dump(
                {src: [e.__dict__ for e in pools[src]] for src in self.SOURCES},
                fout,
            )
        os.replace(tmp_path, cache_path)
        log.info("Wrote BONES-SEED index cache: %s", cache_path)

    def _build_index(
        self, cache_index: Optional[str | Path],
    ) -> Dict[str, List[_BSEntry]]:
        cached = self._load_index_cache(cache_index)
        if cached is not None:
            return cached

        path_by_name = self._build_path_index()
        log.info("Path index: %d motions resolved.", len(path_by_name))

        pools: Dict[str, List[_BSEntry]] = {s: [] for s in self.SOURCES}

        self._build_single_timeline_pool(pools["single"], path_by_name)
        self._build_multi_timeline_pool(pools["multi"], path_by_name)
        self._build_natural_pool(pools["natural"], path_by_name)

        if cache_index is not None:
            self._save_index_cache(cache_index, pools)
        return pools

    def _build_single_timeline_pool(
        self, out: List[_BSEntry], path_by_name: Dict[str, str],
    ) -> None:
        if not self.temporal_labels_path.is_file():
            raise FileNotFoundError(
                f"Temporal labels file not found: {self.temporal_labels_path}"
            )
        no_motion = long_seg = short_seg = no_text = 0
        with open(self.temporal_labels_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                fname = obj["filename"]
                if not self.include_mirrored and fname.endswith("_M"):
                    continue
                p = path_by_name.get(fname)
                if p is None:
                    no_motion += 1
                    continue
                for ev in obj.get("events", []):
                    s = float(ev["start_time"])
                    e = float(ev["end_time"])
                    T = e - s
                    if T > self.max_segment_sec:
                        long_seg += 1
                        continue
                    if T * self.fps < self.min_frames:
                        short_seg += 1
                        continue
                    text = ev.get("description")
                    if not isinstance(text, str) or not text.strip():
                        no_text += 1
                        continue
                    out.append(_BSEntry(
                        source="single", motion_path=p, filename=fname,
                        seg_start_sec=s, seg_end_sec=e, text=text.strip(),
                    ))
        log.info(
            "Pool 'single': %d events (skipped: no_motion=%d long>%.1fs=%d short=%d no_text=%d)",
            len(out), no_motion, self.max_segment_sec, long_seg, short_seg, no_text,
        )

    def _build_multi_timeline_pool(
        self, out: List[_BSEntry], path_by_name: Dict[str, str],
    ) -> None:
        if not self.multi_timeline_path.is_file():
            raise FileNotFoundError(
                f"Multi-timeline file not found: {self.multi_timeline_path}"
            )
        no_motion = long_seg = short_seg = no_text = 0
        with open(self.multi_timeline_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                fname = obj["filename"]
                if not self.include_mirrored and fname.endswith("_M"):
                    continue
                p = path_by_name.get(fname)
                if p is None:
                    no_motion += 1
                    continue
                s = float(obj["start_time"])
                e = float(obj["end_time"])
                T = e - s
                if T > self.max_segment_sec:
                    long_seg += 1
                    continue
                if T * self.fps < self.min_frames:
                    short_seg += 1
                    continue
                text = obj.get("merged_description")
                if not isinstance(text, str) or not text.strip():
                    no_text += 1
                    continue
                out.append(_BSEntry(
                    source="multi", motion_path=p, filename=fname,
                    seg_start_sec=s, seg_end_sec=e, text=text.strip(),
                ))
        log.info(
            "Pool 'multi': %d segments (skipped: no_motion=%d long>%.1fs=%d short=%d no_text=%d)",
            len(out), no_motion, self.max_segment_sec, long_seg, short_seg, no_text,
        )

    def _build_natural_pool(
        self, out: List[_BSEntry], path_by_name: Dict[str, str],
    ) -> None:
        if not self.natural_csv_path.is_file():
            raise FileNotFoundError(
                f"Natural CSV file not found: {self.natural_csv_path}"
            )
        desc_cols = (
            "content_natural_desc_1", "content_natural_desc_2",
            "content_natural_desc_3", "content_natural_desc_4",
        )
        descs_by_file: Dict[str, List[str]] = {}
        no_motion = mirror = no_text = 0
        with open(self.natural_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                if not self.include_mirrored and (
                    _truthy(row.get("is_mirror")) or fname.endswith("_M")
                ):
                    mirror += 1
                    continue
                if fname not in path_by_name:
                    no_motion += 1
                    continue
                descs: List[str] = []
                seen: set = set()
                for k in desc_cols:
                    v = row.get(k, "")
                    if isinstance(v, str):
                        v = v.strip()
                    if v and v not in seen:
                        seen.add(v)
                        descs.append(v)
                if not descs:
                    no_text += 1
                    continue
                descs_by_file[fname] = descs

        log.info(
            "Natural CSV: %d candidate files (skipped: no_motion=%d mirror=%d no_text=%d). "
            "Reading NPZ headers ...",
            len(descs_by_file), no_motion, mirror, no_text,
        )

        long_seg = short_seg = bad_read = 0
        for fname, descs in descs_by_file.items():
            p = path_by_name[fname]
            try:
                with np.load(p, mmap_mode="r") as data:
                    n_total = int(data["local_rot_mats"].shape[0])
            except (OSError, KeyError, ValueError, EOFError) as e:
                log.warning("Failed to read frame count from %s: %s", p, e)
                bad_read += 1
                continue
            T_sec = n_total / float(self.fps)
            if T_sec > self.max_segment_sec:
                long_seg += 1
                continue
            if T_sec * self.fps < self.min_frames:
                short_seg += 1
                continue
            for text in descs:
                out.append(_BSEntry(
                    source="natural", motion_path=p, filename=fname,
                    seg_start_sec=0.0, seg_end_sec=T_sec, text=text,
                ))
        log.info(
            "Pool 'natural': %d (file, desc) entries (skipped: long>%.1fs=%d short=%d bad_read=%d)",
            len(out), self.max_segment_sec, long_seg, short_seg, bad_read,
        )

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        # Interleave the three sources 1:1:1: virtual length = 3 * max(pool).
        # Smaller pools are cycled with random per-fetch augmentation.
        return 3 * self._cycle_len

    def _resolve_segment(self, entry: _BSEntry) -> Tuple[int, int]:
        """Apply the duration rule to ``entry`` and return (start_frame, end_frame).

        Rule:
          * ``T_sec >= max_clip_sec``:
              offset ~ U[0, min(rand_offset_max_sec, T_sec - max_clip_sec)]
              window = [start + offset, start + offset + max_clip_sec]
          * ``T_sec < max_clip_sec``: use the whole segment.

        T > ``max_segment_sec`` entries are filtered at index build time.
        """
        T_sec = entry.seg_end_sec - entry.seg_start_sec
        if T_sec >= self.max_clip_sec:
            slack = max(0.0, min(self.rand_offset_max_sec, T_sec - self.max_clip_sec))
            offset = self._rng.uniform(0.0, slack) if slack > 0 else 0.0
            eff_start = entry.seg_start_sec + offset
            eff_end = eff_start + self.max_clip_sec
        else:
            eff_start = entry.seg_start_sec
            eff_end = entry.seg_end_sec
        return int(round(eff_start * self.fps)), int(round(eff_end * self.fps))

    def _load_segment(
        self, entry: _BSEntry,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sf, ef = self._resolve_segment(entry)
        if self._packed and entry.filename in self._packed_names_idx:
            # Pack path: slice into mmap-resident concat tensors. The OS pages
            # in only the touched rows; subsequent epochs hit the page cache.
            i = self._packed_names_idx[entry.filename]
            off = int(self._packed_offsets[i])
            n_total = int(self._packed_offsets[i + 1]) - off
            sf = max(0, min(sf, n_total))
            ef = max(sf, min(ef, n_total))
            ef = min(ef, sf + self.max_frames)
            local_rot_77 = self._packed_local_rot[off + sf : off + ef]
            root_positions = self._packed_root_pos[off + sf : off + ef]
        else:
            with np.load(entry.motion_path, mmap_mode="r") as data:
                n_total = int(data["local_rot_mats"].shape[0])
                sf = max(0, min(sf, n_total))
                ef = max(sf, min(ef, n_total))
                # Cap to max_frames in case rounding produced an extra frame.
                ef = min(ef, sf + self.max_frames)
                local_rot_77 = np.asarray(data["local_rot_mats"][sf:ef])
                root_positions = np.asarray(data["root_positions"][sf:ef])
            local_rot_77 = torch.from_numpy(local_rot_77)
            root_positions = torch.from_numpy(root_positions)

        T = int(local_rot_77.shape[0])
        if local_rot_77.dtype != torch.float32:
            local_rot_77 = local_rot_77.float()
        if root_positions.dtype != torch.float32:
            root_positions = root_positions.float()
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
        return feats

    def _apply_random_heading(
        self, features: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        if not self.random_heading_aug:
            return features, 0.0
        angle = self._rng.uniform(-np.pi, np.pi)
        angle_t = torch.tensor([angle], dtype=features.dtype)
        rotated = self.motion_rep.rotate(features.unsqueeze(0), angle_t)[0]
        return rotated, float(angle)

    def _load_features_segment(self, entry) -> Optional[torch.Tensor]:
        """Slice precomputed features for ``entry``. Returns None on miss.

        Equivalent (modulo a one-frame velocity boundary at the slice end) to
        ``self._features_with_canonicalize(*self._load_segment(entry))``. The
        last frame's velocity here is the forward-diff against the original
        next frame instead of the slice-end duplicate; this affects 0.13% of
        the values in the sample.
        """
        if entry.filename not in self._packed_feat_idx:
            return None
        sf, ef = self._resolve_segment(entry)
        i = self._packed_feat_idx[entry.filename]
        off = int(self._packed_feat_offsets[i])
        n_total = int(self._packed_feat_offsets[i + 1]) - off
        sf = max(0, min(sf, n_total))
        ef = max(sf, min(ef, n_total))
        ef = min(ef, sf + self.max_frames)
        feats = self._packed_features_tensor[off + sf : off + ef]
        if feats.dtype != torch.float32:
            feats = feats.float()
        # Canonicalize at runtime — rotate first-frame heading to zero and
        # translate root xz to origin. Cheap rotates on the 369-D features
        # only; no FK. Mathematically equivalent to passing
        # ``to_canonicalize=True`` to motion_rep at compute time.
        feats = self.motion_rep.canonicalize(feats.unsqueeze(0), normalized=False)[0]
        return feats

    def __getitem__(self, index: int) -> dict:
        # Round-robin source assignment guarantees 1:1:1 across the virtual
        # epoch. Smaller pools cycle; per-fetch random offset & heading provide
        # augmentation between cycles.
        source = self.SOURCES[index % 3]
        pool = self._pools[source]
        entry = pool[(index // 3) % len(pool)]

        features = None
        if self._packed_features:
            features = self._load_features_segment(entry)
        if features is None:
            local_rot, root_pos, T = self._load_segment(entry)
            if T < self.min_frames:
                # Edge case (e.g., NPZ shorter than the JSONL claims). Advance
                # deterministically to the next index to keep the dataset usable.
                return self[(index + 1) % len(self)]
            features = self._features_with_canonicalize(local_rot, root_pos, T)
        elif features.shape[0] < self.min_frames:
            return self[(index + 1) % len(self)]

        features, _angle = self._apply_random_heading(features)

        heading_cs = features[0, self.motion_rep.slice_dict["global_root_heading"]]
        first_heading = float(torch.atan2(heading_cs[1], heading_cs[0]).item())

        if self.normalize:
            features = self.motion_rep.normalize(features)

        return {
            "motion": features,                       # (T, D) T <= max_frames
            "length": int(features.shape[0]),
            "text": entry.text,
            "first_heading_angle": first_heading,
            "filename": entry.filename,
            "source": entry.source,
            "start_frame": int(round(entry.seg_start_sec * self.fps)),
        }


# =====================================================================
# Shape-aware (per-actor neutrals) variant of the BONES-SEED mixture
# =====================================================================
class SOMABonesSeedDatasetShapeAware(SOMABonesSeedDataset):
    """BONES-SEED dataset that emits per-sample rest-pose joints (30-joint subset).

    Differences from :class:`SOMABonesSeedDataset`:
      * ``data_root`` should point at the proportional NPZ tree (each motion's
        FK has actor-specific bone lengths).
      * ``packed_motions_path`` should point at the proportional motion pack
        (``pack_bones_seed_motions_proportional.py`` output), which carries
        per-actor ``actor_neutrals (N_actors, 77, 3)`` + ``motion_actor_idx``.
      * ``nan_audit_path`` filters NaN-tainted source files at index build time.
      * Each ``__getitem__`` adds ``"neutral_joints"`` ``(30, 3)`` for the
        sample's actor — already sliced to the 30-joint SOMA subset.
      * Features are computed with ``motion_rep(..., neutral_joints=...)`` so
        FK uses the actor's true rest pose, NOT the canonical T-pose.

    Everything else (mixture interleave, canonicalize, random heading,
    normalization, packed-features fallback) is inherited verbatim.
    """

    def __init__(
        self,
        *args,
        nan_audit_path: Optional[str | Path] = None,
        **kwargs,
    ):
        self._nan_audit_path = Path(nan_audit_path) if nan_audit_path else None
        self._nan_filenames: set[str] = self._load_nan_audit_filenames()
        super().__init__(*args, **kwargs)

        if not getattr(self, "_packed", False):
            raise RuntimeError(
                "SOMABonesSeedDatasetShapeAware requires a proportional motion pack "
                "with actor_neutrals / motion_actor_idx; pass packed_motions_path.",
            )
        if not hasattr(self, "_actor_neutrals_30"):
            raise RuntimeError(
                "Loaded pack at %s lacks actor_neutrals / motion_actor_idx; rebuild "
                "with pack_bones_seed_motions_proportional.py." % self.packed_motions_path,
            )

    def _load_nan_audit_filenames(self) -> set[str]:
        if self._nan_audit_path is None or not self._nan_audit_path.is_file():
            return set()
        report = json.load(open(self._nan_audit_path))
        out: set[str] = set()
        for r in report.get("nan_files", []) + report.get("load_error_files", []):
            p = Path(r["path"])
            out.add(p.stem)
        log.info(
            "SOMABonesSeedDatasetShapeAware: %d NaN-tainted filenames will be skipped "
            "at index build time (audit=%s).", len(out), self._nan_audit_path,
        )
        return out

    def _build_natural_pool(self, out, path_by_name):
        # Drop NaN-flagged entries from the path index before building the pool.
        if self._nan_filenames:
            filtered = {k: v for k, v in path_by_name.items() if k not in self._nan_filenames}
            dropped = len(path_by_name) - len(filtered)
            path_by_name = filtered
            log.info("Natural pool: dropped %d NaN-tainted filenames", dropped)
        super()._build_natural_pool(out, path_by_name)

    def _build_single_timeline_pool(self, out, path_by_name):
        if self._nan_filenames:
            path_by_name = {k: v for k, v in path_by_name.items() if k not in self._nan_filenames}
        super()._build_single_timeline_pool(out, path_by_name)

    def _build_multi_timeline_pool(self, out, path_by_name):
        if self._nan_filenames:
            path_by_name = {k: v for k, v in path_by_name.items() if k not in self._nan_filenames}
        super()._build_multi_timeline_pool(out, path_by_name)

    def _load_packed_motions(self) -> None:
        # Same mmap load as the parent, but additionally read the actor-dedup
        # fields and pre-slice neutrals to the 30-joint SOMA subset.
        super()._load_packed_motions()
        if not self._packed:
            return
        path = self.packed_motions_path
        try:
            blob = torch.load(
                str(path), map_location="cpu", weights_only=False, mmap=True,
            )
        except (RuntimeError, TypeError):
            blob = torch.load(str(path), map_location="cpu", weights_only=False)
        if "actor_neutrals" not in blob or "motion_actor_idx" not in blob:
            return  # legacy pack without actor dedup; caller will raise
        actor_neutrals_77 = blob["actor_neutrals"]                         # (N_actors, 77, 3)
        motion_actor_idx = blob["motion_actor_idx"]                        # (N_motions,) int32
        idx_30_in_77 = self.skeleton.get_skel_slice(self.skeleton.somaskel77)
        self._idx_30_in_77 = torch.tensor(idx_30_in_77, dtype=torch.long)
        self._actor_neutrals_30 = actor_neutrals_77[:, self._idx_30_in_77].contiguous()
        # Map filename -> actor index for O(1) per-item lookup.
        self._name_to_actor_idx = {
            n: int(motion_actor_idx[i]) for i, n in enumerate(blob["names"])
        }
        log.info(
            "SOMABonesSeedDatasetShapeAware: indexed %d actors, %d motions "
            "(neutral_joints sliced to %d joints).",
            self._actor_neutrals_30.shape[0], len(self._name_to_actor_idx),
            self._actor_neutrals_30.shape[1],
        )

    def _neutrals_for(self, filename: str) -> Optional[torch.Tensor]:
        a_idx = self._name_to_actor_idx.get(filename)
        if a_idx is None:
            return None
        return self._actor_neutrals_30[a_idx]                              # (30, 3)

    def _features_with_canonicalize(
        self, local_rot_30: torch.Tensor, root_positions: torch.Tensor, T: int,
        neutral_joints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        lengths = torch.tensor([T])
        nj = neutral_joints.unsqueeze(0) if neutral_joints is not None else None
        feats = self.motion_rep(
            local_rot_30.unsqueeze(0),
            root_positions.unsqueeze(0),
            to_normalize=False,
            to_canonicalize=True,
            lengths=lengths,
            neutral_joints=nj,
        )[0]
        return feats

    def __getitem__(self, index: int) -> dict:
        # Mirrors the parent but feeds neutrals into the FK pass. The packed-
        # features Tier-2 shortcut is intentionally NOT supported here — the
        # 369-D features depend on the actor's neutrals, so a one-shot feature
        # pack would freeze that choice and break shape conditioning.
        source = self.SOURCES[index % 3]
        pool = self._pools[source]
        entry = pool[(index // 3) % len(pool)]

        neutrals = self._neutrals_for(entry.filename)
        if neutrals is None:
            # Unknown actor (e.g. file not in the proportional pack) — drop.
            return self[(index + 1) % len(self)]

        local_rot, root_pos, T = self._load_segment(entry)
        if T < self.min_frames:
            return self[(index + 1) % len(self)]

        features = self._features_with_canonicalize(local_rot, root_pos, T, neutral_joints=neutrals)
        features, _angle = self._apply_random_heading(features)

        heading_cs = features[0, self.motion_rep.slice_dict["global_root_heading"]]
        first_heading = float(torch.atan2(heading_cs[1], heading_cs[0]).item())

        if self.normalize:
            features = self.motion_rep.normalize(features)

        return {
            "motion": features,
            "length": int(features.shape[0]),
            "text": entry.text,
            "first_heading_angle": first_heading,
            "filename": entry.filename,
            "source": entry.source,
            "start_frame": int(round(entry.seg_start_sec * self.fps)),
            "neutral_joints": neutrals,                                    # (30, 3) for this actor
        }
