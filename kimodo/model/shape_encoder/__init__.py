# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-actor body-shape encoder (rest-pose 30-joint positions -> 1 prefix token)."""

from .shape_encoder import ShapeEncoder

__all__ = ["ShapeEncoder"]
