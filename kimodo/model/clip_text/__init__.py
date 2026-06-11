# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLIP text encoder wrapper for Kimodo (drop-in alternative to LLM2Vec)."""

from .clip_encoder import CLIPTextEncoder

__all__ = ["CLIPTextEncoder"]
