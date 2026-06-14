# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precompute pooled LLM2Vec text embeddings for a benchmark testsuite tree.

Mirror of ``kimodo/scripts/precompute_text_embeddings.py`` (same cache file
format, same encoder + ``encode_texts`` machinery) — only the caption source
differs. Instead of enumerating a *training* dataset, we walk a benchmark
testsuite folder tree and embed every prompt found in its ``meta.json`` files.

Benchmark generation (``benchmark/generate_eval.py``) normally builds a live
LLM2Vec-Meta-Llama-3-8B encoder on every invocation (the sweep does this 16x).
Precomputing every prompt once and serving it via
``kimodo.model.cached_text.CachedTextEncoder`` removes the 8B model entirely
from generation — it becomes a dict lookup.

CORRECTNESS — keys must be the SANITIZED prompts:
    The Kimodo model sanitizes prompts (``sanitize_texts``) *before* encoding
    (``kimodo/model/kimodo_model.py`` lines 145/492 -> 588). So we apply the
    exact same ``sanitize_texts`` here and key the cache on the result. A miss
    raises ``KeyError`` at generation time (no live fallback), so we enumerate
    the WHOLE testsuite tree and add the empty string ``""`` as insurance.

Usage (run on a COMPUTE node — loads the 8B encoder on GPU; never on login)::

    python benchmark/precompute_benchmark_text.py \
        --benchmark /home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite \
        --out       /home/jungbin_cho/kimodo_caches/benchmark_llm2vec.pt \
        [--batch-size 16] [--device cuda] [--fp32] [--resume]

Then point generation at it::

    python benchmark/generate_eval.py ... --text-cache /home/jungbin_cho/kimodo_caches/benchmark_llm2vec.pt

Cache file format (``torch.save``)::

    {
        "captions": List[str],          # N unique SANITIZED prompts, row order
        "features": Tensor(N, D),       # float32 pooled embeddings
        "meta":     {encoder_type, model, dim, n, benchmark_root, sanitized, created_at},
    }
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import List, Optional, Set

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from kimodo.meta import parse_prompts_from_meta
from kimodo.sanitize import sanitize_texts
from kimodo.scripts.train import build_text_encoder, encode_texts
from kimodo.tools import load_json

log = logging.getLogger("kimodo.precompute_benchmark_text")


# -----------------------------------------------------------------------------
# Caption enumeration — walk the testsuite tree's meta.json files
# -----------------------------------------------------------------------------
def enumerate_benchmark_captions(root: Path) -> Set[str]:
    """Return every SANITIZED prompt referenced by ``meta.json`` under ``root``.

    Uses the same parse path generation uses: ``parse_prompts_from_meta`` then
    ``sanitize_texts``. Metas without a recognized text field (e.g.
    ``constraints_notext``) are skipped — their empty text is covered by the
    ``""`` insurance row added in ``main``.
    """
    captions: Set[str] = set()
    n_meta = 0
    n_skipped = 0
    meta_paths = sorted(root.rglob("meta.json"))
    log.info("found %d meta.json under %s", len(meta_paths), root)
    for mp in tqdm(meta_paths, desc="scan meta.json"):
        try:
            meta = load_json(str(mp))
            texts, _durations = parse_prompts_from_meta(meta)
        except Exception:
            n_skipped += 1
            continue
        n_meta += 1
        for t in sanitize_texts(list(texts)):
            if isinstance(t, str):
                captions.add(t)
    log.info(
        "scanned %d meta.json (%d unparseable/text-less), %d unique sanitized prompts",
        n_meta, n_skipped, len(captions),
    )
    return captions


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--benchmark", required=True, type=str,
                   help="Testsuite root folder (recursively scanned for meta.json).")
    p.add_argument("--out", required=True, type=str, help="Output .pt cache path.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Captions per encoder call. LLM2Vec loops per sentence "
                        "internally, so this mainly controls log granularity.")
    p.add_argument("--device", type=str, default=None, help="'cuda', 'cuda:N', or 'cpu'.")
    p.add_argument("--fp32", action="store_true",
                   help="Build the LLM2Vec encoder in fp32 (default: bf16, matching "
                        "generation's default).")
    p.add_argument("--resume", action="store_true",
                   help="If --out exists, only encode prompts not already cached.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    root = Path(args.benchmark).resolve()
    if not root.is_dir():
        raise SystemExit(f"benchmark root is not a directory: {root}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Enumerate sanitized captions from the testsuite tree.
    captions: Set[str] = enumerate_benchmark_captions(root)
    # Empty string (CFG / empty-text target) — always include as insurance.
    captions.add("")
    captions_sorted: List[str] = sorted(captions)
    log.info("total unique captions to embed: %d", len(captions_sorted))

    # 2) Resume support — skip captions already in the file.
    existing_caps: List[str] = []
    existing_feats: Optional[torch.Tensor] = None
    if args.resume and out_path.is_file():
        blob = torch.load(str(out_path), map_location="cpu", weights_only=False)
        existing_caps = list(blob["captions"])
        existing_feats = blob["features"].to(torch.float32)
        existing_set = set(existing_caps)
        captions_to_embed = [c for c in captions_sorted if c not in existing_set]
        log.info("resume: %d already cached, %d new", len(existing_caps), len(captions_to_embed))
    else:
        captions_to_embed = captions_sorted

    # 3) Build the live text encoder + embed in batches.
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    text_cfg = OmegaConf.create(
        {"type": "llm2vec", "device": "auto", "fp32": bool(args.fp32)}
    )
    log.info("building LLM2Vec encoder (fp32=%s, device=%s) ...", args.fp32, device)
    text_encoder = build_text_encoder(text_cfg, device=device)

    feats_list: List[torch.Tensor] = []
    bs = max(1, int(args.batch_size))
    log.info("encoding %d captions in batches of %d ...", len(captions_to_embed), bs)
    for i in tqdm(range(0, len(captions_to_embed), bs), desc="encode"):
        batch = captions_to_embed[i : i + bs]
        feats, _pad_mask = encode_texts(text_encoder, batch, device)
        # CachedTextEncoder requires pooled output (L=1).
        if feats.shape[1] != 1:
            raise RuntimeError(
                f"precompute requires a POOLED encoder (L=1); got feats shape "
                f"{tuple(feats.shape)}."
            )
        feats_list.append(feats.squeeze(1).to("cpu", torch.float32))
    new_features = torch.cat(feats_list, dim=0) if feats_list else torch.empty(
        (0, getattr(text_encoder, "llm_dim", 0)), dtype=torch.float32,
    )

    # 4) Merge with existing rows (if --resume), save atomically.
    if existing_feats is not None:
        all_captions = existing_caps + captions_to_embed
        all_features = torch.cat([existing_feats, new_features], dim=0)
    else:
        all_captions = captions_to_embed
        all_features = new_features

    blob = {
        "captions": all_captions,
        "features": all_features,
        "meta": {
            "encoder_type": "llm2vec",
            "model": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "dim": int(all_features.shape[1]) if all_features.numel() else 0,
            "n": int(all_features.shape[0]),
            "benchmark_root": str(root),
            "sanitized": True,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        },
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(blob, str(tmp))
    tmp.replace(out_path)
    log.info("wrote %s  (%d captions, dim=%d, %.1f MB)",
             out_path, blob["meta"]["n"], blob["meta"]["dim"],
             out_path.stat().st_size / 1e6)
    log.info("Use it: python benchmark/generate_eval.py ... --text-cache %s", out_path)


if __name__ == "__main__":
    main()
