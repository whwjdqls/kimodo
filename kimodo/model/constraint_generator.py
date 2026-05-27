# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 of the 3-stage text-to-motion model: (text, mask) -> constraint values.

Given a (continuous, sigmoid-valued) constraint mask predicted by Stage 1 and
text features, the ``ConstraintGenerator`` predicts ``(B, T, motion_rep_dim)``
normalised motion-feature values. ``observed_motion = mask * values`` is fed
to the Stage 3 motion denoiser.

There is **no GT mask** anywhere in this training pipeline; Stage 2 is
supervised in two ways:

1. **Implicit** via backprop through the motion-denoiser loss (Stage 3 wants
   ``observed_motion`` to look like GT motion at the gated positions).
2. **Optional auxiliary supervision** via ``smooth_l1(values, GT_motion)``
   on every valid frame. The mask is NOT used in this aux loss, so it does
   not leak any "GT mask" information — it just gives Stage 2 a dense per-
   frame regression signal so the model has something useful to gate.

Architecture mirrors :class:`kimodo.model.mask_generator.MaskGenerator` but
the per-frame input is the (soft) mask itself, projected to the latent
dimension. No timestep embedding — this is a deterministic regression head,
not a diffusion model.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from omegaconf import ListConfig
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .backbone import PositionalEncoding, pad_x_and_mask_to_fixed_size


class ConstraintGenerator(nn.Module):
    """Transformer (text, mask) -> (T, D) constraint values."""

    def __init__(
        self,
        motion_rep,
        llm_shape: Union[List[int], ListConfig],
        use_text_mask: bool = False,
        latent_dim: int = 384,
        ff_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 4,
        activation: str = "gelu",
        dropout: float = 0.1,
        pe_dropout: float = 0.1,
        norm_first: bool = False,
        num_text_tokens_override: Optional[int] = 50,
    ):
        super().__init__()
        self.motion_rep = motion_rep
        self.input_dim = motion_rep.motion_rep_dim   # mask dim
        self.output_dim = motion_rep.motion_rep_dim  # value dim
        self.latent_dim = int(latent_dim)
        self.use_text_mask = bool(use_text_mask)

        llm_dim = int(llm_shape[-1])
        self.num_text_tokens = (
            int(num_text_tokens_override) if num_text_tokens_override is not None else int(llm_shape[0])
        )
        self.embed_text = nn.Linear(llm_dim, self.latent_dim)

        self.input_linear = nn.Linear(self.input_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, pe_dropout)
        self.output_linear = nn.Linear(self.latent_dim, self.output_dim)

        trans_enc_layer = TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_size),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.seqTransEncoder = TransformerEncoder(
            trans_enc_layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        text_feat: Tensor,
        text_pad_mask: Tensor,
        mask: Tensor,
        pad_mask: Tensor,
    ) -> Tensor:
        """Predict constraint values.

        Args:
            text_feat: ``(B, L_text, llm_dim)``.
            text_pad_mask: ``(B, L_text)`` bool.
            mask: ``(B, T, motion_rep_dim)`` binary mask (float 0/1 is fine).
            pad_mask: ``(B, T)`` bool — True for valid frames.

        Returns:
            ``(B, T, motion_rep_dim)`` predicted values (in normalized motion-feature space).
        """
        B, T, _ = mask.shape
        device = text_feat.device

        text_feat, text_pad_mask = pad_x_and_mask_to_fixed_size(
            text_feat, text_pad_mask, self.num_text_tokens,
        )
        emb_text = self.embed_text(text_feat)

        x = self.input_linear(mask.to(emb_text.dtype))  # (B, T, latent)

        if not self.use_text_mask:
            text_pad_mask = torch.ones(
                (B, emb_text.shape[1]), dtype=torch.bool, device=device,
            )

        xseq = torch.cat((emb_text, x), dim=1)
        src_key_padding_mask = ~torch.cat((text_pad_mask, pad_mask), dim=1)

        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=src_key_padding_mask)
        output = output[:, self.num_text_tokens:]
        values = self.output_linear(output)
        return values
