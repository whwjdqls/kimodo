# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HumanML3D dataset reading UniEgoMotion-style 211-D ``.npz`` files.

Counterpart to :class:`HumanML3DNativeTextMotionDataset`; same batch contract
(so the same training loop + collate work), only the file format and feature
width differ:

  * Loads ``.npz`` with key ``features`` (T, 211) from ``motion_dir`` (e.g.
    ``HumanML3D/uniego_rep/``), produced by ``benchmark/kimodo_to_uniego.py``.
  * Uses ``Mean_uniego.npy`` / ``Std_uniego.npy`` for z-scoring.
  * **Frame-0 canonicalization**: after windowing, each window's ``canon_delta``
    block at frame 0 is set to the identity transform so every window starts at
    the origin facing +Z (the analog of HumanML3D-native's frame-0
    canonicalization). The trajectory shape is preserved; only the absolute
    world placement of the first frame is dropped.

Item dict matches :class:`HumanML3DNativeTextMotionDataset`.
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

from .humanml3d_text_motion import build_collate_fn  # noqa: F401 (re-exported)
from kimodo.motion_rep.uniego import FEAT_DIM as UNIEGO_FEAT_DIM
from kimodo.motion_rep.uniego import IDENTITY_DELTA9

log = logging.getLogger(__name__)

# canon_delta block [198:207]; identity SE(3) in (6D rot ++ 3 trans) form.
_DELTA0, _DELTA1 = 198, 207
_IDENTITY_DELTA = np.asarray(IDENTITY_DELTA9, dtype=np.float32)


@dataclass
class _Record:
    motion_id: str
    feats: np.ndarray  # (T, 211), unnormalized
    captions: List[str] = field(default_factory=list)
    sub_clips: List[Tuple[np.ndarray, str]] = field(default_factory=list)


def _parse_text_file(path: str) -> List[Tuple[str, float, float]]:
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


class HumanML3DUniegoTextMotionDataset(Dataset):
    """HumanML3D UniEgo-style 211-D ``.npz`` dataset with text + sub-clips."""

    FEAT_DIM = UNIEGO_FEAT_DIM  # 211

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
        self.skeleton = skeleton
        self.motion_rep = motion_rep

        if mean.shape != (self.FEAT_DIM,) or std.shape != (self.FEAT_DIM,):
            raise ValueError(
                f"Expected mean/std of shape ({self.FEAT_DIM},), got {mean.shape}/{std.shape}"
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
        for mid in tqdm(ids, desc="Indexing HumanML3D-uniego",
                        disable=os.environ.get("KIMODO_NO_TQDM")):
            mp = self.motion_dir / f"{mid}.npz"
            tp = self.text_dir / f"{mid}.txt"
            if not mp.is_file():
                n_missing += 1
                continue
            if not tp.is_file():
                n_no_text += 1
                continue
            try:
                feats = np.asarray(np.load(mp)["features"], dtype=np.float32)
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

            rec = _Record(motion_id=mid, feats=feats)
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
            raise RuntimeError(f"No HumanML3D-uniego samples assembled from {self.split_file}")

        self.samples: List[Tuple[int, int, str]] = []
        for i, rec in enumerate(self.records):
            for cap in rec.captions:
                self.samples.append((i, -1, cap))
            for j, (_sub, cap) in enumerate(rec.sub_clips):
                self.samples.append((i, j, cap))

        log.info(
            "HumanML3D-uniego: %d motions, %d (motion, caption) samples "
            "(skipped: missing=%d short<%d=%d no_text=%d).",
            len(self.records), len(self.samples),
            n_missing, self.min_motion_len, n_short, n_no_text,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def inv_transform(self, x):
        if isinstance(x, torch.Tensor):
            return x * torch.from_numpy(self.std).to(x.device, x.dtype) + \
                torch.from_numpy(self.mean).to(x.device, x.dtype)
        return x * self.std + self.mean

    def _pick_window(self, total_T: int) -> Tuple[int, int]:
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

        # Frame-0 canonicalization: drop the window's absolute world placement so
        # every window starts at origin/+Z (matches the native rep + the stats).
        feats[0, _DELTA0:_DELTA1] = _IDENTITY_DELTA

        # Z-score with Mean_uniego / Std_uniego.
        feats = (feats - self.mean) / self.std
        if self.clip_normalized is not None:
            feats = np.clip(feats, -self.clip_normalized, self.clip_normalized)

        motion = np.zeros((self.max_motion_length, feats.shape[1]), dtype=np.float32)
        motion[: feats.shape[0]] = feats

        return {
            "motion": torch.from_numpy(motion),
            "length": int(feats.shape[0]),
            "text": caption,
            "first_heading_angle": 0.0,  # canonical at frame 0
            "filename": rec.motion_id,
        }


def build_humanml3d_uniego_collate_fn():
    return build_collate_fn()
