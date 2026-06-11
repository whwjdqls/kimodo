# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached text encoder: serves precomputed embeddings from disk."""

from .cached_encoder import CachedTextEncoder

__all__ = ["CachedTextEncoder"]
