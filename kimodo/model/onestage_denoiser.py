# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-stage transformer denoiser — MDM-style direct full-motion prediction.

Single :class:`TransformerEncoderBlock` (the same block KIMODO's two-stage
denoiser uses) predicts **all** ``motion_rep_dim`` features in one pass —
no root-first / body-second split. Forward signature matches
:class:`TwostageDenoiser` so the training loop, viz code, and offline
sampler are drop-in compatible.

Use this for an MDM baseline that shares the kimodo transformer block and
motion representation, isolating the architectural difference (one vs two
stages) from the encoder + dataset + loss + text-conditioning recipe.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .backbone import TransformerEncoderBlock
from .loading import load_checkpoint_state_dict

# Accepted values for ``motion_mask_mode``:
#   "concat" — Constraint plumbing on. The binary motion mask is concatenated
#              onto the channel dim (input width doubles to 2D) and the
#              observed values overwrite x at masked positions before the
#              transformer. This is the default; use it for any run that may
#              later be fine-tuned with constraints (Phase 2).
#   "none"   — Pure text-to-motion. The input is the raw motion (width D),
#              motion_mask / observed_motion arguments are ignored entirely
#              (forward signature still accepts them for API compatibility).
#              Pick this when you know the model will NEVER see constraints,
#              so you don't waste input-projection capacity on dead mask
#              channels (see notes in the forward).
_VALID_MOTION_MASK_MODES = ("concat", "none")


def _normalize_motion_mask_mode(mode: Optional[str]) -> str:
    """Coerce ``None``/``null`` → ``"concat"`` and validate the accepted set."""
    if mode is None:
        return "concat"
    if mode not in _VALID_MOTION_MASK_MODES:
        raise ValueError(
            f"motion_mask_mode={mode!r} is not supported; "
            f"expected one of {_VALID_MOTION_MASK_MODES} (or omit / null for the default 'concat')."
        )
    return mode


class OnestageDenoiser(nn.Module):
    """Single transformer that predicts the full motion x0 in one shot."""

    def __init__(
        self,
        motion_rep,
        motion_mask_mode: Optional[str] = "concat",
        ckpt_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.motion_rep = motion_rep
        self.motion_mask_mode = _normalize_motion_mask_mode(motion_mask_mode)
        self._concat_mask = self.motion_mask_mode == "concat"

        input_dim = motion_rep.motion_rep_dim

        # When mask is concatenated, the block input is doubled to (D, D) =
        # (raw motion, binary mask). When motion_mask_mode == "none", we feed
        # raw motion only — pure text-to-motion, no dead mask channels.
        block_input_dim = input_dim * 2 if self._concat_mask else input_dim
        block_output_dim = input_dim

        self.model = TransformerEncoderBlock(
            input_dim=block_input_dim,
            output_dim=block_output_dim,
            skeleton=self.motion_rep.skeleton,
            **kwargs,
        )

        if ckpt_path:
            self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path: str) -> None:
        state_dict = load_checkpoint_state_dict(ckpt_path)
        state_dict = {k.replace("denoiser.backbone.", ""): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)

    def forward(
        self,
        x: torch.Tensor,
        x_pad_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor] = None,
        motion_mask: Optional[torch.Tensor] = None,
        observed_motion: Optional[torch.Tensor] = None,
        shape_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``(B, T, motion_rep_dim)`` — predicted clean motion (x0).

        When ``motion_mask_mode='none'`` (pure text-to-motion), the
        ``motion_mask`` and ``observed_motion`` kwargs are accepted but
        silently ignored — the model has no input channels for them.
        """
        if self._concat_mask:
            if motion_mask is None or observed_motion is None:
                motion_mask = torch.zeros_like(x)
                observed_motion = torch.zeros_like(x)
            x = x * (1 - motion_mask) + observed_motion * motion_mask
            x_in = torch.cat([x, motion_mask], axis=-1)
        else:
            x_in = x

        return self.model(
            x_in,
            x_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
            shape_feat=shape_feat,
        )
