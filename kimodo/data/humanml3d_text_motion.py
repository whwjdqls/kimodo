# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HumanML3D text-to-motion dataset for KIMODO-style training.

The HumanML3D motions live as Kimodo NPZs at e.g.
``/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/{id}.npz``. Each NPZ has a
pre-computed ``features`` array of shape ``(T, 273)`` already in KIMODO order
(``smooth_root_pos`` 3 + ``global_root_heading`` 2 + ``local_joints_positions``
22*3 + ``global_rot_data`` 22*6 + ``velocities`` 22*3 + ``foot_contacts`` 4 =
273), produced by ``benchmark/humanml3d_to_kimodo.py``.

IMPORTANT — skeleton caveat
---------------------------
``SMPLXSkeleton22`` is used here **only as a container** that gives
``nbjoints=22`` and the correct feature-block slicing/dimension for
``KimodoMotionRep``. The npz's ``global_rot_data`` and ``posed_joints`` were
produced by HumanML3D's **chain-reset FK**, not the standard parent-relative
FK that ``skeleton.fk()`` implements. Therefore we MUST NOT run
``motion_rep.inverse()`` or any γ₇-style FK consistency loss against this data:
they'd give incorrect positions. The operations we use are skeleton-agnostic:

* ``motion_rep.rotate(...)`` — Y-rotation of root xz, heading (cos, sin), joint
  positions, joint 6D rotations and velocities. Pure linear algebra on feature
  blocks.
* ``motion_rep.normalize/unnormalize`` — stats only.
* ``motion_rep.global_root_to_local_root`` — uses fps + heading, no FK.

If you need to render HumanML3D motions, decode back to HumanML3D 263-D via
``benchmark/humanml3d_to_kimodo.kimodo_to_humanml3d`` and use HumanML3D's
``recover_from_ric``. Do not rely on ``skeleton.fk()`` here.

Each motion ``{id}`` has a text file at ``texts/{id}.txt`` with multiple
captions, one per line, in HumanML3D format:

    caption#tagged_tokens#start_time#end_time

A line with ``start_time == 0.0 == end_time`` refers to the whole motion;
non-zero tags pick a sub-segment in seconds. We treat each line as one
candidate (caption, segment) pair and sample uniformly at fetch time. This
matches the style of momask's ``data/t2m_dataset.Text2MotionDataset``.

Per-fetch processing:

1. Read ``features`` slice for the selected segment (or full motion).
2. Sample a random sub-window of length ``window_size`` (rounded to a multiple
   of ``unit_length`` to mirror momask). If shorter than the dataset's hard
   ``max_motion_length``, pad with zeros and return a pad mask.
3. Apply random Y-rotation augmentation (paper's first-frame heading aug). We
   work on the 273-dim KIMODO features via ``motion_rep.rotate``.
4. Normalize with the supplied 273-dim Mean / Std.
"""

from __future__ import annotations

import codecs as cs
import logging
import os
import random
from dataclasses import dataclass, field
from os.path import join as pjoin
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SMPLXSkeleton22

log = logging.getLogger(__name__)


@dataclass
class _MotionRecord:
    motion_id: str
    feats: np.ndarray  # (T, 273) unnormalized
    captions: List[str] = field(default_factory=list)
    # Lazily-extracted sub-clips with their own captions.
    sub_clips: List[Tuple[np.ndarray, str]] = field(default_factory=list)


def _parse_text_file(path: str) -> List[Tuple[str, float, float]]:
    """Return list of (caption, f_tag, to_tag) from a HumanML3D text file."""
    out: List[Tuple[str, float, float]] = []
    with cs.open(path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            caption = parts[0]
            try:
                f_tag = float(parts[2])
                to_tag = float(parts[3])
            except Exception:
                continue
            f_tag = 0.0 if (np.isnan(f_tag) or f_tag < 0) else f_tag
            to_tag = 0.0 if (np.isnan(to_tag) or to_tag < 0) else to_tag
            out.append((caption, f_tag, to_tag))
    return out


class HumanML3DTextMotionDataset(Dataset):
    """KIMODO-feature HumanML3D dataset.

    Args:
        motion_dir: Directory holding ``{id}.npz`` files with a ``features`` key (T, 273).
        text_dir: Directory holding ``{id}.txt`` files in HumanML3D format.
        split_file: Plain-text file with one motion id per line.
        mean: (273,) numpy array.
        std: (273,) numpy array.
        fps: Dataset fps (20 for HumanML3D).
        window_size: Random window length used at training time (frames).
        max_motion_length: Pad outputs to this length.
        min_motion_len: Skip motions shorter than this.
        unit_length: Window length is rounded to a multiple of this.
        random_heading_aug: Apply random Y-rotation to the features each fetch.
        skeleton / motion_rep: Optional shared CPU-resident helpers.
        seed: RNG seed for the dataset-private RNG.
    """

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
        random_heading_aug: bool = True,
        clip_normalized: Optional[float] = None,
        skeleton: Optional[SMPLXSkeleton22] = None,
        motion_rep: Optional[KimodoMotionRep] = None,
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
        self.random_heading_aug = bool(random_heading_aug)
        # Optional: clamp normalized features to +/- this value. The HumanML3D
        # kimodo stats are per-block-constant and the velocities block has a
        # small std, so rare fast frames normalize to |x| ~ 65; clamping caps
        # their contribution to the (x0) loss and gradients. None = no clamp.
        self.clip_normalized = float(clip_normalized) if clip_normalized else None

        if skeleton is None:
            skeleton = SMPLXSkeleton22()
        self.skeleton = skeleton
        if motion_rep is None:
            motion_rep = KimodoMotionRep(skeleton=self.skeleton, fps=self.fps)
        self.motion_rep = motion_rep

        if mean.shape != (273,) or std.shape != (273,):
            raise ValueError(
                f"Expected mean/std of shape (273,), got {mean.shape}/{std.shape}"
            )
        # Floor std to avoid divide-by-zero
        std = np.where(std < 1e-4, np.float32(1.0), std).astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Load split.
        with open(self.split_file, "r") as f:
            ids = [line.strip() for line in f if line.strip()]

        self.records: List[_MotionRecord] = []
        n_short = n_missing = n_no_text = 0
        for mid in tqdm(ids, desc="Indexing HumanML3D", disable=os.environ.get("KIMODO_NO_TQDM")):
            mp = self.motion_dir / f"{mid}.npz"
            tp = self.text_dir / f"{mid}.txt"
            if not mp.is_file():
                n_missing += 1
                continue
            if not tp.is_file():
                n_no_text += 1
                continue
            try:
                with np.load(mp, mmap_mode="r") as data:
                    if "features" not in data.files:
                        n_missing += 1
                        continue
                    feats = np.asarray(data["features"])
            except Exception:
                n_missing += 1
                continue
            if feats.shape[1] != 273:
                n_missing += 1
                continue
            if feats.shape[0] < self.min_motion_len:
                n_short += 1
                continue

            text_lines = _parse_text_file(str(tp))
            if not text_lines:
                n_no_text += 1
                continue

            rec = _MotionRecord(motion_id=mid, feats=feats.astype(np.float32))

            for caption, f_tag, to_tag in text_lines:
                if f_tag == 0.0 and to_tag == 0.0:
                    rec.captions.append(caption)
                else:
                    s = int(round(f_tag * self.fps))
                    e = int(round(to_tag * self.fps))
                    if e <= s or e > feats.shape[0]:
                        continue
                    sub = feats[s:e]
                    if sub.shape[0] < self.min_motion_len:
                        continue
                    rec.sub_clips.append((sub.astype(np.float32), caption))

            # Flatten into a list of (rec_index, slice_into_record).
            # We store records separately so we don't duplicate the full-clip array.
            if not rec.captions and not rec.sub_clips:
                n_no_text += 1
                continue
            self.records.append(rec)

        if not self.records:
            raise RuntimeError(
                f"No HumanML3D samples assembled from {self.split_file}"
            )

        # Build a flat sample list: each entry is (record_idx, sub_clip_idx_or_-1, caption).
        # If sub_clip_idx == -1 the sample uses the record's full features with one of the
        # full-clip captions.
        self.samples: List[Tuple[int, int, str]] = []
        for i, rec in enumerate(self.records):
            for cap in rec.captions:
                self.samples.append((i, -1, cap))
            for j, (_sub, cap) in enumerate(rec.sub_clips):
                self.samples.append((i, j, cap))

        log.info(
            "HumanML3D: %d motions, %d (motion, caption) samples (skipped: missing=%d short<%d=%d no_text=%d).",
            len(self.records), len(self.samples),
            n_missing, self.min_motion_len, n_short, n_no_text,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def inv_transform(self, x: np.ndarray | torch.Tensor):
        if isinstance(x, torch.Tensor):
            return x * torch.as_tensor(self.std, dtype=x.dtype, device=x.device) + torch.as_tensor(
                self.mean, dtype=x.dtype, device=x.device
            )
        return x * self.std + self.mean

    def _pick_window(self, full_len: int) -> Tuple[int, int]:
        """Return (start, length) — momask-style window pick rounded to ``unit_length``.

        Length never exceeds ``max_motion_length``. If the motion is shorter than the
        window, return the whole motion.
        """
        m_length = min(full_len, self.window_size)
        if m_length < self.min_motion_len:
            return 0, m_length

        if self.unit_length < 10:
            coin = self._rng.choice(["single", "single", "double"])
        else:
            coin = "single"
        if coin == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        m_length = max(self.unit_length, m_length)
        m_length = min(m_length, full_len)

        if full_len > m_length:
            start = self._rng.randint(0, full_len - m_length)
        else:
            start = 0
        return start, m_length

    def _apply_random_heading(self, features: np.ndarray) -> np.ndarray:
        if not self.random_heading_aug:
            return features
        angle = float(self._np_rng.uniform(-np.pi, np.pi))
        ft = torch.from_numpy(features)[None]
        angle_t = torch.tensor([angle], dtype=ft.dtype)
        rotated = self.motion_rep.rotate(ft, angle_t)[0].numpy()
        return rotated

    def __getitem__(self, item: int) -> dict:
        rec_idx, sub_idx, caption = self.samples[item]
        rec = self.records[rec_idx]
        feats_full = rec.feats if sub_idx == -1 else rec.sub_clips[sub_idx][0]

        start, m_length = self._pick_window(feats_full.shape[0])
        feats = feats_full[start : start + m_length].copy()

        # Random heading aug (operates on unnormalized features).
        feats = self._apply_random_heading(feats)

        # First-frame heading angle (post-rotation), for the model's conditioning.
        heading_cs = feats[0, self.motion_rep.slice_dict["global_root_heading"]]
        first_heading = float(np.arctan2(heading_cs[1], heading_cs[0]))

        # Normalize
        feats = (feats - self.mean) / self.std
        if self.clip_normalized is not None:
            feats = np.clip(feats, -self.clip_normalized, self.clip_normalized)

        # Pad to max_motion_length on the right with zeros.
        motion = np.zeros((self.max_motion_length, feats.shape[1]), dtype=np.float32)
        motion[: feats.shape[0]] = feats

        return {
            "motion": torch.from_numpy(motion),                  # (max_T, 273)
            "length": int(feats.shape[0]),
            "text": caption,
            "first_heading_angle": first_heading,
            "filename": rec.motion_id,
        }


def build_collate_fn(pad_value: float = 0.0) -> Callable[[List[dict]], dict]:
    """Collate already-padded items; produce a boolean pad_mask from `length`."""

    def collate(batch: List[dict]) -> dict:
        lengths = torch.tensor([it["length"] for it in batch], dtype=torch.long)
        max_T = int(batch[0]["motion"].shape[0])
        B = len(batch)
        motions = torch.stack([it["motion"] for it in batch], dim=0)  # (B, max_T, D)
        pad_mask = torch.zeros((B, max_T), dtype=torch.bool)
        for i, L in enumerate(lengths.tolist()):
            pad_mask[i, :L] = True
        return {
            "motion": motions,
            "lengths": lengths,
            "pad_mask": pad_mask,
            "text": [it["text"] for it in batch],
            "first_heading_angle": torch.tensor(
                [it["first_heading_angle"] for it in batch], dtype=torch.float32,
            ),
            "filename": [it["filename"] for it in batch],
        }

    return collate
