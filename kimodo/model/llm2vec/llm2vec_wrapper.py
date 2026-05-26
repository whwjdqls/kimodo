# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import os

import numpy as np
import torch

from .llm2vec import LLM2Vec


class LLM2VecEncoder:
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
        device: str = "auto",
    ) -> None:
        torch_dtype = getattr(torch, dtype)
        self.llm_dim = llm_dim

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")

        if "TEXT_ENCODERS_DIR" in os.environ:
            base_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], base_model_name_or_path)
            peft_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], peft_model_name_or_path)

        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )

        env_device = os.environ.get("TEXT_ENCODER_DEVICE")
        if env_device:
            device = env_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        if device is not None:
            self.model = self.model.to(device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self._device = str(device) if not isinstance(device, str) else device
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        # NOTE: We bypass LLM2Vec.encode() and call _encode() directly per
        # sentence. encode()'s "multi-GPU" branch (torch.cuda.device_count()>1)
        # spawns a multiprocessing.Pool with one worker per visible GPU. Under
        # DDP every rank sees all GPUs, so 8 ranks each spawn 8 workers = 64
        # CUDA worker procs that contend for memory and crash with OOM /
        # "invalid device context" during pool teardown.
        #
        # Using batch_size=1 (as before) for repeatability — different batch
        # sizes change the output embeddings due to a transformers quirk:
        # https://github.com/huggingface/transformers/issues/25420
        sentences = [[""] + [t] for t in text]
        concatenated = [self.model._convert_to_str(s[0], s[1]) for s in sentences]
        self.model.eval()
        with torch.no_grad():
            chunks = []
            for s in concatenated:
                chunks.append(
                    self.model._encode([s], device=self._device, convert_to_numpy=False)
                )
            encoded_text = torch.cat(chunks, dim=0).to(torch.float32)

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.tensor(encoded_text).to(self._device)
        return encoded_text, lengths
