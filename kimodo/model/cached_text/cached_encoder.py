# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached text encoder.

Serves precomputed pooled text embeddings from a ``.pt`` cache file, with
the same call interface as ``LLM2VecEncoder`` / ``CLIPTextEncoder``:

    enc = CachedTextEncoder(cache_path=..., device="cuda")
    feats, lengths = enc(list_of_strings)
    # feats: (B, 1, D) float32 on `device`
    # lengths: [1] * B

The cache file is produced by
``kimodo/scripts/precompute_text_embeddings.py`` and contains:

    {
        "captions": List[str],       # N unique captions, in row order
        "features": Tensor(N, D),    # float32 pooled embeddings
        "meta":     dict,            # {"encoder_type", "model", "dim", ...}
    }

Cache misses raise a ``KeyError`` with a clear "re-run the precompute
script" message — there is **no live encoder fallback** by design (the
whole point is to never load a slow encoder during training).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import torch


class CachedTextEncoder:
    """Frozen lookup-table 'encoder'. Same call signature as the live ones."""

    def __init__(self, cache_path: Union[str, Path], device: str = "cpu"):
        cache_path = Path(cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"text-embedding cache not found: {cache_path}. "
                f"Build it with kimodo.scripts.precompute_text_embeddings."
            )

        # weights_only=False because the file contains a {captions, features, meta}
        # dict, not a pure state_dict. Source is your own precompute script.
        # mmap=True so the (potentially multi-GB) features tensor isn't pulled
        # into RAM at load — only rows actually indexed get paged in. OS page
        # cache lets sibling training jobs share pages of the same cache file.
        try:
            blob = torch.load(
                str(cache_path), map_location="cpu", weights_only=False, mmap=True,
            )
        except (RuntimeError, TypeError):
            # Older PyTorch (no mmap kwarg) or legacy serialization format.
            blob = torch.load(str(cache_path), map_location="cpu", weights_only=False)

        captions: List[str] = list(blob["captions"])
        features: torch.Tensor = blob["features"]
        if features.dtype != torch.float32:
            # Cast forces a full materialization; only pay it when needed.
            features = features.to(torch.float32)
        if features.dim() != 2 or features.shape[0] != len(captions):
            raise ValueError(
                f"cache shape mismatch: features={tuple(features.shape)} vs "
                f"#captions={len(captions)} in {cache_path}"
            )

        self._captions = captions
        self._caption_to_idx: dict = {c: i for i, c in enumerate(captions)}
        self._features = features  # CPU resident; per-call gather is cheap
        self._device = torch.device(device)
        self._cache_path = cache_path

        self.text_dim = int(features.shape[1])
        self.llm_dim = self.text_dim  # alias for parity with the live encoders
        self.meta = dict(blob.get("meta", {}))

    # --- Make the wrapper look like a torch module to callers that .to() it ---
    def to(self, device):
        self._device = torch.device(device)
        return self

    def get_device(self):
        return self._device

    def __len__(self) -> int:
        return len(self._captions)

    def __contains__(self, caption: str) -> bool:
        return caption in self._caption_to_idx

    # --- The call interface ---
    @torch.no_grad()
    def __call__(
        self, text: Union[str, List[str]],
    ) -> Tuple[torch.Tensor, List[int]]:
        is_string = isinstance(text, str)
        if is_string:
            text = [text]

        missing: List[str] = []
        idxs: List[int] = []
        for cap in text:
            i = self._caption_to_idx.get(cap)
            if i is None:
                missing.append(cap)
            else:
                idxs.append(i)

        if missing:
            sample = missing[:3]
            raise KeyError(
                f"{len(missing)} caption(s) missing from text-embedding cache "
                f"({self._cache_path}). Examples: {sample}. "
                f"Re-run kimodo.scripts.precompute_text_embeddings with the "
                f"current training config (and include viz/test prompts)."
            )

        feats = self._features[torch.tensor(idxs, dtype=torch.long)]  # (B, D)
        feats = feats.unsqueeze(1).to(self._device, non_blocking=True)  # (B, 1, D)
        lengths = [1] * feats.shape[0]
        if is_string:
            feats = feats[0]
            lengths = lengths[0]
        return feats, lengths
