# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HumanML3D dataset reading native 263-D ``.npy`` files (no kimodo conversion).

Mirrors :class:`HumanML3DTextMotionDataset`'s batch contract so the same
training loop and collate function work — only the underlying file format
and feature width differ. Key differences vs the kimodo-rep version:

  * Loads ``.npy`` (T, 263) per motion from ``motion_dir`` (e.g. the real
    HumanML3D ``new_joint_vecs/``), not ``.npz`` dicts.
  * Uses HumanML3D's standard ``Mean.npy`` / ``Std.npy`` for z-scoring.
  * No random heading augmentation — HumanML3D's preprocessing already
    canonicalizes every motion to face +Z at frame 0, and rotating the
    263-D non-trivially would require unpacking → rotating → re-encoding
    via HumanML3D's IK. Not worth it; native is normally trained without.
  * No ``first_heading_angle`` to extract — always 0 (canonical). The
    collate still emits the field (set to 0.0) for API compatibility with
    the existing training loop.

Item dict returned (matches HumanML3DTextMotionDataset):

    {
        "motion":              Tensor (max_motion_length, 263) — z-scored, right-padded with zeros
        "length":              int  — true motion length in frames
        "text":                str  — chosen caption
        "first_heading_angle": float — always 0.0
        "filename":            str  — motion id (file stem)
    }
"""

from __future__ import annotations

import codecs as cs
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# Reuse the existing collate (signature-compatible items).
from .humanml3d_text_motion import build_collate_fn  # noqa: F401 (re-exported below)


log = logging.getLogger(__name__)


@dataclass
class _Record:
    motion_id: str
    feats: np.ndarray  # (T, 263), unnormalized
    captions: List[str] = field(default_factory=list)                  # full-motion captions
    sub_clips: List[Tuple[np.ndarray, str]] = field(default_factory=list)  # (sub_feats, caption)


def _parse_text_file(path: str) -> List[Tuple[str, float, float]]:
    """Same format as :mod:`humanml3d_text_motion._parse_text_file`."""
    out: List[Tuple[str, float, float]] = []
    with cs.open(path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            caption = parts[0]
            try:
                f_tag, to_tag = float(parts[2]), float(parts[3])
            except ValueError:
                continue
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            out.append((caption, f_tag, to_tag))
    return out


class HumanML3DNativeTextMotionDataset(Dataset):
    """HumanML3D 263-D ``.npy`` dataset with text captions and sub-clip support.

    Args:
        motion_dir: Directory holding ``{id}.npy`` files of shape (T, 263).
        text_dir: Directory holding ``{id}.txt`` files in HumanML3D format.
        split_file: Plain-text file, one motion id per line.
        mean: ``(263,)`` numpy array (z-score mean — e.g. HumanML3D's ``Mean.npy``).
        std:  ``(263,)`` numpy array.
        fps: Dataset fps (20 for HumanML3D).
        window_size: Random window length used at training time (frames).
        max_motion_length: Pad outputs to this length.
        min_motion_len: Skip motions shorter than this.
        unit_length: Window length is rounded down to a multiple of this.
        clip_normalized: Optional clamp on |z-score| (e.g. 15.0). None = off.
        motion_rep / skeleton: Optional; carried for API parity with the kimodo
            dataset (the training loop's first_heading branch reads slice_dict).
            Not required by ``__getitem__`` itself.
        seed: RNG seed for the dataset-private RNG.
    """

    FEAT_DIM = 263

    def __init__(
        self,
        motion_dir: str | Path,
        text_dir: str | Path,
        split_file: str | Path,
        mean: np.ndarray,
        std: np.ndarray,
        fps: int = 20,
        window_size: int = 200,
        max_motion_length: int = 200,
        min_motion_len: int = 40,
        unit_length: int = 4,
        clip_normalized: Optional[float] = None,
        motion_rep=None,
        skeleton=None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.motion_dir = Path(motion_dir)
        self.text_dir = Path(text_dir)
        self.split_file = Path(split_file)
        self.fps = int(fps)
        self.window_size = int(window_size)
        self.max_motion_length = int(max_motion_length)
        self.min_motion_len = int(min_motion_len)
        self.unit_length = int(unit_length)
        self.clip_normalized = float(clip_normalized) if clip_normalized else None
        self.skeleton = skeleton           # passed through; not used here
        self.motion_rep = motion_rep       # ditto

        if mean.shape != (self.FEAT_DIM,) or std.shape != (self.FEAT_DIM,):
            raise ValueError(
                f"Expected mean/std of shape ({self.FEAT_DIM},), "
                f"got {mean.shape}/{std.shape}"
            )
        std = np.where(std < 1e-4, np.float32(1.0), std).astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        with open(self.split_file, "r") as f:
            ids = [line.strip() for line in f if line.strip()]

        self.records: List[_Record] = []
        n_short = n_missing = n_no_text = 0
        for mid in tqdm(ids, desc="Indexing HumanML3D-native",
                        disable=os.environ.get("KIMODO_NO_TQDM")):
            mp = self.motion_dir / f"{mid}.npy"
            tp = self.text_dir / f"{mid}.txt"
            if not mp.is_file():
                n_missing += 1
                continue
            if not tp.is_file():
                n_no_text += 1
                continue
            try:
                feats = np.load(mp, mmap_mode="r")
            except Exception:
                n_missing += 1
                continue
            if feats.ndim != 2 or feats.shape[1] != self.FEAT_DIM:
                n_missing += 1
                continue
            if feats.shape[0] < self.min_motion_len:
                n_short += 1
                continue

            text_lines = _parse_text_file(str(tp))
            if not text_lines:
                n_no_text += 1
                continue

            # Materialize to a normal in-memory array (no mmap_mode for the
            # stored copy — we slice from it frequently). 263 floats × ~200
            # frames × 4 bytes ≈ 200 KB per motion, ~6 GB for 30k motions.
            rec = _Record(motion_id=mid, feats=np.asarray(feats, dtype=np.float32))

            for caption, f_tag, to_tag in text_lines:
                if f_tag == 0.0 and to_tag == 0.0:
                    rec.captions.append(caption)
                else:
                    s = int(round(f_tag * self.fps))
                    e = int(round(to_tag * self.fps))
                    if e <= s or e > rec.feats.shape[0]:
                        continue
                    sub = rec.feats[s:e]
                    if sub.shape[0] < self.min_motion_len:
                        continue
                    rec.sub_clips.append((sub.astype(np.float32), caption))

            if not rec.captions and not rec.sub_clips:
                n_no_text += 1
                continue
            self.records.append(rec)

        if not self.records:
            raise RuntimeError(
                f"No HumanML3D-native samples assembled from {self.split_file}"
            )

        # Flatten: each sample = (record_idx, sub_clip_idx_or_-1, caption).
        self.samples: List[Tuple[int, int, str]] = []
        for i, rec in enumerate(self.records):
            for cap in rec.captions:
                self.samples.append((i, -1, cap))
            for j, (_sub, cap) in enumerate(rec.sub_clips):
                self.samples.append((i, j, cap))

        log.info(
            "HumanML3D-native: %d motions, %d (motion, caption) samples "
            "(skipped: missing=%d short<%d=%d no_text=%d).",
            len(self.records), len(self.samples),
            n_missing, self.min_motion_len, n_short, n_no_text,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def inv_transform(self, x):
        """Undo z-scoring; works on both numpy and torch tensors."""
        if isinstance(x, torch.Tensor):
            return x * torch.from_numpy(self.std).to(x.device, x.dtype) + \
                torch.from_numpy(self.mean).to(x.device, x.dtype)
        return x * self.std + self.mean

    def _pick_window(self, total_T: int) -> Tuple[int, int]:
        """Pick start + length matching the kimodo dataset's recipe."""
        if self.unit_length < 10:
            coin = self._rng.choice(["single", "single", "double"])
        else:
            coin = "single"
        cap = min(total_T, self.window_size)
        if coin == "double":
            m_len = max(self.unit_length, (cap // self.unit_length - 1) * self.unit_length)
        else:
            m_len = (cap // self.unit_length) * self.unit_length
        m_len = max(m_len, self.min_motion_len)
        m_len = min(m_len, total_T)
        max_start = max(0, total_T - m_len)
        start = self._rng.randint(0, max_start)
        return start, m_len

    def __getitem__(self, item: int) -> dict:
        rec_idx, sub_idx, caption = self.samples[item]
        rec = self.records[rec_idx]
        feats_full = rec.feats if sub_idx == -1 else rec.sub_clips[sub_idx][0]

        start, m_length = self._pick_window(feats_full.shape[0])
        feats = feats_full[start : start + m_length].copy()

        # Z-score with HumanML3D's Mean/Std (native).
        feats = (feats - self.mean) / self.std
        if self.clip_normalized is not None:
            feats = np.clip(feats, -self.clip_normalized, self.clip_normalized)

        # Right-pad to max_motion_length with zeros.
        motion = np.zeros((self.max_motion_length, feats.shape[1]), dtype=np.float32)
        motion[: feats.shape[0]] = feats

        return {
            "motion": torch.from_numpy(motion),                  # (max_T, 263)
            "length": int(feats.shape[0]),
            "text": caption,
            # Always 0 — HumanML3D's preprocessing canonicalizes frame 0 to face +Z.
            "first_heading_angle": 0.0,
            "filename": rec.motion_id,
        }


def build_humanml3d_native_collate_fn():
    """Re-export the same collate as the kimodo version (item dicts match)."""
    return build_collate_fn()
