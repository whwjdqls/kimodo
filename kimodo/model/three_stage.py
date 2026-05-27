# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""3-stage text-to-motion model: mask generator -> constraint generator -> motion generator.

Forward flow (training and inference)::

    text_feat        = LLM2Vec(text)                      # frozen, shared
    mask_logits      = MaskGenerator(text_feat, pad_mask)
    mask_probs       = sigmoid(mask_logits)               # continuous in [0, 1]
    values           = ConstraintGenerator(text_feat, mask_probs, pad_mask)
    observed_motion  = mask_probs * values                # gated values
    # Stage 3 — existing kimodo diffusion denoiser (TwostageDenoiser).
    pred_clean       = Denoiser(motion_t, ..., motion_mask=mask_probs, observed_motion=observed_motion)

There is **no ground-truth mask** anywhere. Training relies on the motion-
reconstruction loss to push gradients all the way back to Stage 1, plus a
sparsity regulariser on the mask (prevents trivial mask=1 collapse) and an
optional auxiliary smooth-L1 on Stage 2's values vs GT motion (gives Stage 2
a dense supervisory signal but does **not** require a GT mask).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from kimodo.model.constraint_generator import ConstraintGenerator
from kimodo.model.mask_generator import MaskGenerator


class ThreeStageKimodo(nn.Module):
    """Wraps Stage 1 (mask gen) + Stage 2 (constraint gen) + Stage 3 (denoiser).

    Holds the three trainable sub-modules in a single ``nn.Module`` so that a
    single ``DistributedDataParallel`` wrap covers the entire trainable graph
    and one ``loss.backward()`` updates every parameter.

    Args:
        mask_generator: :class:`MaskGenerator`.
        constraint_generator: :class:`ConstraintGenerator`.
        denoiser: the Stage 3 denoiser
            (typically ``kimodo.model.twostage_denoiser.TwostageDenoiser``).
            Its ``motion_rep`` provides feature-block slices used by the loss.
    """

    def __init__(
        self,
        mask_generator: MaskGenerator,
        constraint_generator: ConstraintGenerator,
        denoiser: nn.Module,
    ):
        super().__init__()
        self.mask_generator = mask_generator
        self.constraint_generator = constraint_generator
        self.denoiser = denoiser

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def motion_rep(self):
        return self.denoiser.motion_rep

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def predict_mask_and_values(
        self,
        text_feat: Tensor,
        text_pad_mask: Tensor,
        pad_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """Run Stages 1 and 2; return continuous mask probs and Stage 2 values.

        Returns a dict with::

            {
              "mask_logits":  (B, T, D),
              "mask_probs":   (B, T, D)   sigmoid(logits)
              "values":       (B, T, D)
            }
        """
        mask_logits = self.mask_generator(text_feat, text_pad_mask, pad_mask)
        mask_probs = torch.sigmoid(mask_logits)
        # Zero out positions in padded frames so they don't contaminate downstream sums.
        mask_probs = mask_probs * pad_mask.to(mask_probs.dtype).unsqueeze(-1)
        values = self.constraint_generator(text_feat, text_pad_mask, mask_probs, pad_mask)
        return {"mask_logits": mask_logits, "mask_probs": mask_probs, "values": values}

    def forward_denoiser(
        self,
        motion_t: Tensor,
        pad_mask: Tensor,
        text_feat: Tensor,
        text_pad_mask: Tensor,
        timesteps: Tensor,
        first_heading_angle: Optional[Tensor],
        mask_probs: Tensor,
        observed_motion: Tensor,
    ) -> Tensor:
        """Call the Stage 3 denoiser with the supplied conditioning."""
        return self.denoiser(
            motion_t,
            pad_mask,
            text_feat,
            text_pad_mask,
            timesteps,
            first_heading_angle=first_heading_angle,
            motion_mask=mask_probs,
            observed_motion=observed_motion,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_constraint_pack(
        self,
        text_feat: Tensor,
        text_pad_mask: Tensor,
        pad_mask: Tensor,
        mask_threshold: Optional[float] = None,
        mask_override: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run Stages 1 and 2 in eval mode for sampling.

        Args:
            mask_threshold: If given, threshold ``sigmoid(logits) > threshold``
                to produce a hard binary mask (matches the "keyframe"
                interpretation). If ``None``, use the continuous sigmoid
                directly (matches training).
            mask_override: If given, use this mask instead of the predicted
                one. Useful for user-supplied constraints / scene constraints.

        Returns:
            ``{"mask": (B, T, D), "values": (B, T, D), "observed_motion": (B, T, D)}``
        """
        if mask_override is not None:
            mask = mask_override.to(text_feat.dtype)
        else:
            mask_logits = self.mask_generator(text_feat, text_pad_mask, pad_mask)
            mask = torch.sigmoid(mask_logits)
            if mask_threshold is not None:
                mask = (mask > float(mask_threshold)).to(mask.dtype)
            mask = mask * pad_mask.to(mask.dtype).unsqueeze(-1)
        values = self.constraint_generator(text_feat, text_pad_mask, mask, pad_mask)
        observed = mask * values
        return {"mask": mask, "values": values, "observed_motion": observed}
