# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-dir config snapshotting.

When training scripts only save the top-level training ``config.yaml``,
re-reading it later is fragile: ``model_config_path`` points at a file under
``configs/`` that may have been edited since the run started, so the saved
training config no longer faithfully describes what was actually trained.

``snapshot_configs`` fixes this by copying the referenced model config into
the run directory and rewriting ``cfg.model_config_path`` to point at the
copy before saving the top-level config.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import OmegaConf


def snapshot_configs(cfg, output_dir: Path | str) -> None:
    """Snapshot the model config into ``output_dir`` and write the top-level
    ``config.yaml`` so it references the snapshot instead of the source file.

    Idempotent for resume: if ``cfg.model_config_path`` already resolves to
    a file inside ``output_dir``, the copy step is skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mc_key = "model_config_path"
    src_str = OmegaConf.select(cfg, mc_key, default=None)
    if src_str:
        src = Path(src_str).expanduser()
        dst = output_dir / "model_config.yaml"
        try:
            same = src.resolve() == dst.resolve()
        except OSError:
            same = False
        if not same:
            shutil.copy2(src, dst)
        # Rewrite the top-level config to point at the snapshot.
        cfg.model_config_path = str(dst)

    OmegaConf.save(cfg, output_dir / "config.yaml")
