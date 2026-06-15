# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLP body-shape encoder.

Consumes the per-actor rest-pose joint positions ``(B, J, 3)`` (the
SOMABonesSeedDatasetShapeAware ``neutral_joints`` field, already sliced to the
30-joint SOMA subset) and emits a single prefix-token embedding ``(B, 1, D)``
shaped like the text-encoder output so it slots into the denoiser's prefix
plumbing alongside text/timestep/heading tokens.

Design notes
------------
* The input is shape ``(B, J, 3)`` of rest-pose joints in the kimodo Y-up
  frame, **already centered on the actor's pelvis** by the dataset (or by FK's
  ``root_positions_is_global=True`` pelvis-recentering elsewhere) — the encoder
  is shape-only, not pose-only.
* Flat MLP rather than per-bone features: lets the network discover what
  matters (limb lengths, torso proportions, head height) instead of baking in
  one prior. Three layers @ hidden=256, GELU + LayerNorm.
* Output is L2-normalized per sample so the model has a clean magnitude prior
  (matches what CLIP-style pooled features look like). The downstream
  ``embed_shape`` linear in the denoiser will rescale.
* ``__call__`` returns ``(emb, lengths)`` matching ``CLIPTextEncoder`` so the
  same prefix-token concat path in the backbone accepts either.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


class ShapeEncoder(nn.Module):
    """Flat MLP body-shape encoder (rest-pose joints -> pooled (B, 1, D))."""

    def __init__(
        self,
        n_joints: int = 30,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 3,
        normalize_output: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.n_joints = int(n_joints)
        self.output_dim = int(output_dim)
        self.normalize_output = bool(normalize_output)

        in_dim = self.n_joints * 3
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, neutral_joints: torch.Tensor) -> torch.Tensor:
        """Project (B, J, 3) -> (B, 1, output_dim)."""
        if neutral_joints.ndim != 3 or neutral_joints.shape[-1] != 3:
            raise ValueError(
                f"neutral_joints must be (B, J, 3); got {tuple(neutral_joints.shape)}",
            )
        if neutral_joints.shape[1] != self.n_joints:
            raise ValueError(
                f"ShapeEncoder configured for J={self.n_joints} but got "
                f"J={neutral_joints.shape[1]}",
            )
        B = neutral_joints.shape[0]
        x = neutral_joints.reshape(B, -1)
        emb = self.mlp(x)
        if self.normalize_output:
            emb = nn.functional.normalize(emb, dim=-1)
        return emb.unsqueeze(1)                                   # (B, 1, D)

    def encode(
        self, neutral_joints: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Text-encoder-style ``(feats, lengths)`` interface for the trainer."""
        emb = self(neutral_joints)                                # (B, 1, D)
        lengths = [1] * emb.shape[0]
        return emb, lengths

    def get_device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")
