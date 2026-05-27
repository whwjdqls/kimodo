# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 of the 3-stage text-to-motion model: text -> constraint mask.

Given text features and a target sequence length, the ``MaskGenerator``
predicts ``(B, T, motion_rep_dim)`` logits whose sigmoid is used as a
**continuous** (soft) mask in [0, 1]. There is no ground-truth mask anywhere
in this pipeline — the mask is a latent the model learns end-to-end:

* During training, ``mask_probs = sigmoid(logits)`` is multiplied into the
  Stage 2 output to form ``observed_motion = mask_probs * values``, which is
  passed to the Stage 3 motion denoiser. The motion-reconstruction loss
  backprops all the way through Stage 3, Stage 2, and finally Stage 1.
* A **sparsity regulariser** on ``mean(mask_probs)`` prevents the trivial
  collapse to ``mask = 1`` everywhere (which would let Stage 2 just leak
  ground-truth motion features to Stage 3).

The architecture mirrors ``TransformerEncoderBlock`` from
``kimodo/model/backbone.py``: text features are projected and prepended as a
prefix, per-frame learnable tokens carry positional information, a
transformer encoder mixes them, and a linear head projects to
``motion_rep_dim`` logits. There is **no timestep embedding** — Stage 1 is a
deterministic regression, not a diffusion model.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from omegaconf import ListConfig
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .backbone import PositionalEncoding, pad_x_and_mask_to_fixed_size


class MaskGenerator(nn.Module):
    """Transformer text -> (T, D) binary-mask logits."""

    def __init__(
        self,
        motion_rep,
        llm_shape: Union[List[int], ListConfig],
        use_text_mask: bool = False,
        latent_dim: int = 256,
        ff_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        activation: str = "gelu",
        dropout: float = 0.1,
        pe_dropout: float = 0.1,
        norm_first: bool = False,
        num_text_tokens_override: Optional[int] = 50,
    ):
        super().__init__()
        self.motion_rep = motion_rep
        self.output_dim = motion_rep.motion_rep_dim
        self.latent_dim = int(latent_dim)
        self.use_text_mask = bool(use_text_mask)

        llm_dim = int(llm_shape[-1])
        self.num_text_tokens = (
            int(num_text_tokens_override) if num_text_tokens_override is not None else int(llm_shape[0])
        )
        self.embed_text = nn.Linear(llm_dim, self.latent_dim)

        # Per-frame zero input -> latent via a bias-only linear (just provides a
        # learnable per-frame "blank" token; positional encoding does the rest).
        self.frame_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
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
        pad_mask: Tensor,
    ) -> Tensor:
        """Predict mask logits.

        Args:
            text_feat: ``(B, L_text, llm_dim)``.
            text_pad_mask: ``(B, L_text)`` bool — True where the text token is valid.
            pad_mask: ``(B, T)`` bool — True for valid frames.

        Returns:
            ``(B, T, motion_rep_dim)`` BCE logits.
        """
        B, T = pad_mask.shape
        device = text_feat.device

        text_feat, text_pad_mask = pad_x_and_mask_to_fixed_size(
            text_feat, text_pad_mask, self.num_text_tokens,
        )
        emb_text = self.embed_text(text_feat)  # (B, num_text_tokens, latent)

        # Per-frame learnable token broadcast across time and batch.
        x = self.frame_token.expand(B, T, -1)  # (B, T, latent)

        if not self.use_text_mask:
            text_pad_mask = torch.ones(
                (B, emb_text.shape[1]), dtype=torch.bool, device=device,
            )

        xseq = torch.cat((emb_text, x), dim=1)
        src_key_padding_mask = ~torch.cat((text_pad_mask, pad_mask), dim=1)

        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=src_key_padding_mask)
        output = output[:, self.num_text_tokens:]  # (B, T, latent)
        logits = self.output_linear(output)        # (B, T, D)
        return logits

    @torch.no_grad()
    def sample_mask(
        self,
        text_feat: Tensor,
        text_pad_mask: Tensor,
        pad_mask: Tensor,
        threshold: float = 0.5,
        sample_bernoulli: bool = False,
    ) -> Tensor:
        """Return a binary mask ``(B, T, motion_rep_dim)``."""
        logits = self(text_feat, text_pad_mask, pad_mask)
        probs = torch.sigmoid(logits)
        if sample_bernoulli:
            mask = torch.bernoulli(probs)
        else:
            mask = (probs > float(threshold)).to(probs.dtype)
        # Zero out positions in padded frames.
        mask = mask * pad_mask.to(mask.dtype).unsqueeze(-1)
        return mask
