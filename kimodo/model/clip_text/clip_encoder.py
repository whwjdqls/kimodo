"""CLIP text encoder for Kimodo.

A drop-in replacement for ``LLM2VecEncoder`` with the same call interface:

    enc = CLIPTextEncoder(...)
    feats, lengths = enc(list_of_strings)  # feats: (B, L_text, D), lengths: list[int]

Unlike the LLM2Vec wrapper, this one **batches** the forward pass — all
captions in the list go through CLIP in a single call — which is the main
reason text encoding is fast here.

Two output modes:

* ``pooled=True``  : returns the pooled ``[EOS]`` embedding shaped
                     ``(B, 1, D)``. Matches LLM2Vec's interface; the
                     denoiser then pads to ``num_text_tokens_override``.
* ``pooled=False`` : returns the per-token sequence ``(B, L_max, D)`` plus
                     true token lengths, so cross-attention can use the
                     real token-level features.

Common model choices:

* ``openai/clip-vit-base-patch32`` (default) — text dim **512**, fast.
  This is the **same model MDM uses** (``clip.load('ViT-B/32')``).
* ``openai/clip-vit-large-patch14``          — text dim **768**.

When you change models, update the model config's ``llm_shape`` to
``[1, <text_dim>]`` (or ``[L_max, <text_dim>]`` if ``pooled=False``).

Text length truncation:

Default ``max_length=22`` matches MDM's HumanML3D recipe — they cap to
``max_text_len=20`` content tokens, then add SOS + EOS (= 22 total).
HuggingFace's ``CLIPTokenizer`` handles padding/truncation natively via
``attention_mask``, so we skip MDM's "pad to 77 with zeros" hack
(that was specific to OpenAI-CLIP's hardcoded positional embeddings).
"""

from __future__ import annotations

import os
from typing import List, Tuple, Union

import torch

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


class CLIPTextEncoder:
    """Frozen CLIP text encoder. Same call signature as ``LLM2VecEncoder``."""

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        dtype: str = "float32",
        pooled: bool = True,
        max_length: int = 22,  # MDM HumanML3D default: SOS + 20 content + EOS
    ):
        from transformers import CLIPTextModel, CLIPTokenizer  # local import

        torch_dtype = _DTYPE_MAP.get(str(dtype).lower(), torch.float32)
        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")

        # Local-mirror override: same convention as build_text_encoder so DDP
        # runs can avoid HF snapshot races by pointing at a flat local dir.
        local_root = os.environ.get("KIMODO_TEXT_ENCODER_LOCAL_DIR")
        resolved = model_name_or_path
        if local_root:
            stem = model_name_or_path.split("/")[-1]
            candidate = os.path.join(local_root, stem)
            if os.path.isdir(candidate):
                resolved = candidate

        self.tokenizer = CLIPTokenizer.from_pretrained(resolved, cache_dir=cache_dir)
        # Force safetensors loading — recent transformers refuses .bin checkpoints
        # on torch<2.6 (CVE-2025-32434). All CLIP repos publish safetensors.
        self.model = CLIPTextModel.from_pretrained(
            resolved, torch_dtype=torch_dtype, cache_dir=cache_dir,
            use_safetensors=True,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._device = torch.device(device)
        self.model.to(self._device)

        # CLIP's text projection dim (== last_hidden_size for the text tower).
        self.text_dim = int(self.model.config.hidden_size)
        self.llm_dim = self.text_dim  # alias for parity with LLM2VecEncoder
        self.pooled = bool(pooled)
        self.max_length = int(max_length)

    def to(self, device):
        self._device = torch.device(device)
        self.model.to(self._device)
        return self

    def get_device(self):
        return self._device

    @torch.no_grad()
    def __call__(
        self, text: Union[str, List[str]],
    ) -> Tuple[torch.Tensor, List[int]]:
        is_string = isinstance(text, str)
        if is_string:
            text = [text]

        tokens = self.tokenizer(
            list(text),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        out = self.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )

        if self.pooled:
            feats = out.pooler_output.unsqueeze(1)  # (B, 1, D)
            lengths = [1] * feats.shape[0]
        else:
            feats = out.last_hidden_state  # (B, L_max, D)
            # True per-sample token counts (incl. SOS/EOS) before padding.
            lengths = tokens["attention_mask"].sum(dim=1).tolist()

        # Always return float32 features (matches LLM2VecEncoder).
        feats = feats.to(torch.float32)

        if is_string:
            feats = feats[0]
            lengths = lengths[0]
        return feats, lengths
