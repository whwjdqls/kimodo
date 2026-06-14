# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viser-based 3D visualization for skeletons and motion.

The interactive viewer (``viser_utils``) needs the optional ``viser`` package.
We import it lazily/guarded so that viser-free consumers — e.g. the offline MP4
renderer in ``kimodo/scripts/visualize.py``, which only needs ``soma_skin`` —
can import ``kimodo.viz`` without installing the interactive web stack.
"""

try:
    from . import viser_utils
    from .viser_utils import (
        Character,
        CharacterMotion,
        ConstraintSet,
        EEJointsKeyframeSet,
        FullbodyKeyframeSet,
        GuiElements,
        RootKeyframe2DSet,
        SkeletonMesh,
        WaypointMesh,
        load_example_cases,
    )

    __all__ = [
        "Character",
        "CharacterMotion",
        "ConstraintSet",
        "EEJointsKeyframeSet",
        "FullbodyKeyframeSet",
        "GuiElements",
        "RootKeyframe2DSet",
        "SkeletonMesh",
        "WaypointMesh",
        "load_example_cases",
        "viser_utils",
    ]
except ImportError:
    # viser not installed -> interactive viewer unavailable, but offline
    # rendering (soma_skin) still imports fine.
    viser_utils = None
    __all__ = []
