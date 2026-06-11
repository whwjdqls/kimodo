# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precompute pooled text embeddings for a training dataset and save to disk.

LLM2Vec-Llama-3-8B is far too slow to run live in the training loop
(~50-100ms / sentence in the wrapper's per-sentence forward), so we
embed all unique captions once, save them, and serve at training time via
``kimodo.model.cached_text.CachedTextEncoder`` (a lookup table).

Usage::

    python -m kimodo.scripts.precompute_text_embeddings \\
        --config configs/training/bones_seed.yaml \\
        --out  /weka/jungbin/kimodo_caches/bones_seed_llm2vec.pt \\
        [--encoder llm2vec|clip] \\
        [--batch-size 16] \\
        [--device cuda] \\
        [--no-viz-prompts] [--no-test-prompts] \\
        [--extra-texts file.txt] [--resume]

Then in the training config, point at the cache::

    text_encoder:
      cache_path: /weka/jungbin/kimodo_caches/bones_seed_llm2vec.pt
      type: llm2vec        # informational only — the live encoder isn't built

The training loop never instantiates the live LLM2Vec model.

Cache file format (``torch.save``)::

    {
        "captions": List[str],          # N unique captions, in row order
        "features": Tensor(N, D),       # float32 pooled embeddings
        "meta":     {                   # for provenance / sanity checks
            "encoder_type": "llm2vec" | "clip",
            "model": <hf repo id>,
            "dim":   D,
            "n":     N,
            "config_path": <path>,
            "created_at": <iso-8601>,
        },
    }
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from kimodo.scripts.train import build_text_encoder, encode_texts

log = logging.getLogger("kimodo.precompute_text")


# -----------------------------------------------------------------------------
# Caption enumeration — duck-typed across the three dataset families
# -----------------------------------------------------------------------------
def iter_captions_from_dataset(dataset) -> Iterable[str]:
    """Yield every caption referenced by ``dataset``.

    Knows about: HumanML3DTextMotionDataset (.samples), SOMATextMotionDataset
    (.entries), SOMABonesSeedDataset (._pools dict). Falls back to
    ``dataset[i]['text']`` if none match (slow — does the full per-item work).
    """
    # HumanML3D: flat (rec_idx, sub_idx, caption) triples.
    samples = getattr(dataset, "samples", None)
    if samples is not None and len(samples) and len(samples[0]) >= 3:
        for *_, cap in samples:
            if isinstance(cap, str):
                yield cap
        return

    # SOMATextMotion: list of SegmentEntry with .text.
    entries = getattr(dataset, "entries", None)
    if entries is not None and len(entries) and hasattr(entries[0], "text"):
        for e in entries:
            yield e.text
        return

    # SOMABonesSeed: pool dict {source -> list[_BSEntry]}.
    pools = getattr(dataset, "_pools", None)
    if isinstance(pools, dict) and pools:
        for src_entries in pools.values():
            for e in src_entries:
                yield e.text
        return

    # Fallback: full per-item materialisation (slow).
    log.warning(
        "Dataset %s has no .samples/.entries/._pools — falling back to "
        "per-item __getitem__ iteration (slow).", type(dataset).__name__,
    )
    for i in range(len(dataset)):
        item = dataset[i]
        t = item.get("text") if isinstance(item, dict) else None
        if isinstance(t, str):
            yield t


def build_dataset_for_captions(cfg):
    """Instantiate just enough of the training dataset to enumerate captions.

    Dispatches on ``cfg.data`` to the same builder the training loop uses:

      * HumanML3D family — detected via the ``motion_dir/text_dir/split_file/
        mean_path`` keys or ``cfg.data.type == "humanml3d"``.
      * SOMA family — anything else. ``cfg.data.kind`` (``"segments"`` |
        ``"bones_seed"``, default ``"segments"``) selects between
        :class:`SOMATextMotionDataset` and :class:`SOMABonesSeedDataset`.

    Motion features are not loaded here — only the index lists (``.entries``
    / ``._pools``) are populated, which is what caption enumeration needs.
    """
    data = cfg.data
    dtype = str(data.get("type", "")).lower()

    # HumanML3D-in-Kimodo training datasets (hml3d.yaml family).
    has_hml3d_keys = all(
        k in data for k in ("motion_dir", "text_dir", "split_file", "mean_path")
    )
    if dtype == "humanml3d" or has_hml3d_keys:
        from kimodo.data import HumanML3DTextMotionDataset
        from kimodo.scripts.train import _build_cpu_motion_rep

        mr = _build_cpu_motion_rep(
            cfg.model_config_path, cfg.stats_path,
            fps_override=cfg.get("denoiser_fps_override"),
        )
        mean = np.load(data.mean_path).astype(np.float32)
        std = np.load(data.std_path).astype(np.float32)
        return HumanML3DTextMotionDataset(
            motion_dir=data.motion_dir,
            text_dir=data.text_dir,
            split_file=data.split_file,
            mean=mean, std=std,
            fps=int(data.fps),
            window_size=int(data.window_size),
            max_motion_length=int(data.max_motion_length),
            min_motion_len=int(data.min_motion_len),
            unit_length=int(data.unit_length),
            random_heading_aug=False,
            clip_normalized=data.get("clip_normalized"),
            skeleton=mr.skeleton,
            motion_rep=mr,
            seed=0,
        )

    # SOMA family: segments OR bones_seed. Defer to train.py's builder so the
    # caption-enumeration path stays in lockstep with the training path.
    from kimodo.scripts.train import _build_cpu_motion_rep, build_soma_dataset
    mr = _build_cpu_motion_rep(
        cfg.model_config_path, cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    )
    return build_soma_dataset(cfg, mr, seed=0)


# -----------------------------------------------------------------------------
# Encoder model-name resolution (for cache metadata)
# -----------------------------------------------------------------------------
def _resolve_model_name(text_cfg) -> str:
    enc_type = str(text_cfg.get("type", "llm2vec")).lower()
    if enc_type == "clip":
        return str(text_cfg.get("model_name", "openai/clip-vit-base-patch32"))
    if enc_type == "llm2vec":
        return "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    return enc_type


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=str,
                   help="Training YAML config (we read cfg.data, cfg.viz, cfg.text_encoder).")
    p.add_argument("--out", required=True, type=str, help="Output .pt path.")
    p.add_argument("--encoder", default=None,
                   help="Override cfg.text_encoder.type (e.g. 'llm2vec' or 'clip').")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Captions per encoder call. For LLM2Vec the wrapper loops per "
                        "sentence internally, so this mainly controls log granularity.")
    p.add_argument("--device", type=str, default=None, help="'cuda', 'cuda:N', or 'cpu'.")
    p.add_argument("--no-viz-prompts", action="store_true",
                   help="Skip embedding cfg.viz.prompts.")
    p.add_argument("--no-test-prompts", action="store_true",
                   help="Skip embedding the held-out viz_test_vs_gt captions.")
    p.add_argument("--extra-texts", type=str, default=None,
                   help="File with extra captions to embed (one per line).")
    p.add_argument("--resume", action="store_true",
                   help="If --out exists, only encode captions not already cached.")
    p.add_argument("overrides", nargs="*",
                   help="OmegaConf dotted overrides, e.g. "
                        "data.train_split_path=/path/to/train_small.txt "
                        "data.cache_index=/path/to/seg_index_small.json")
    return p.parse_args()


def _collect_test_split_captions(cfg, n_samples: Optional[int] = None) -> List[str]:
    """First full-motion (#0#0) caption per id in cfg.viz.test_split_file."""
    from kimodo.scripts.train_w_hml3d import _load_test_examples
    n = int(n_samples or cfg.viz.get("num_test_samples", 64))
    try:
        exs = _load_test_examples(
            motion_dir=cfg.data.motion_dir,
            text_dir=cfg.data.text_dir,
            split_file=cfg.viz.get("test_split_file", cfg.data.split_file),
            n_samples=n,
            max_frames=int(cfg.viz.get("test_max_frames", cfg.data.max_motion_length)),
            min_frames=int(cfg.data.get("min_motion_len", 40)),
        )
        return [e["caption"] for e in exs]
    except Exception as e:
        log.warning("could not collect test-split captions: %s", e)
        return []


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.resolve(cfg)
    if args.encoder:
        cfg.text_encoder = OmegaConf.merge(cfg.text_encoder, {"type": args.encoder})
    # Drop any cache_path on the source config so build_text_encoder builds the live one.
    if "cache_path" in cfg.text_encoder:
        cfg.text_encoder.pop("cache_path", None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Enumerate captions.
    log.info("building dataset from %s to enumerate captions ...", args.config)
    dataset = build_dataset_for_captions(cfg)
    captions: Set[str] = set()
    for cap in iter_captions_from_dataset(dataset):
        if cap:
            captions.add(cap)
    log.info("collected %d unique training captions", len(captions))

    if not args.no_viz_prompts:
        for p in cfg.viz.get("prompts", []) or []:
            if isinstance(p, str) and p:
                captions.add(p)
    if not args.no_test_prompts and bool(cfg.viz.get("test_vs_gt", False)):
        # SOMA / bones_seed path: pull the same N test natural captions that
        # ``_load_test_examples_soma`` will use at viz time.
        if "natural_csv_path" in cfg.data:
            try:
                from kimodo.scripts.train import _load_test_examples_soma
                test_split = cfg.viz.get("test_split_file") or cfg.data.get("train_split_path")
                if test_split:
                    test_exs = _load_test_examples_soma(
                        test_split_path=test_split,
                        data_root=cfg.data.data_root,
                        natural_csv_path=cfg.data.natural_csv_path,
                        motion_rep=dataset.motion_rep,
                        skeleton=dataset.skeleton,
                        n_samples=int(cfg.viz.get("num_test_samples", 4)),
                        max_frames=int(cfg.viz.get("test_max_frames", cfg.data.get("max_frames", 200))),
                        min_frames=int(cfg.data.get("min_frames", 10)),
                        include_mirrored=False,
                    )
                    added = 0
                    for ex in test_exs:
                        c = ex.get("caption")
                        if isinstance(c, str) and c:
                            captions.add(c)
                            added += 1
                    log.info("Added %d SOMA test-split natural captions.", added)
            except Exception as e:
                log.warning("Could not collect SOMA test-split captions: %s", e)
        else:
            for c in _collect_test_split_captions(cfg):
                captions.add(c)
    if args.extra_texts:
        with open(args.extra_texts, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    captions.add(line)
    # Empty string (CFG-drop target) — always include.
    captions.add("")

    # Deterministic row ordering.
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
        new_caps = [c for c in captions_sorted if c not in existing_set]
        log.info("resume: %d already cached, %d new", len(existing_caps), len(new_caps))
        captions_to_embed = new_caps
    else:
        captions_to_embed = captions_sorted

    # 3) Build the live text encoder + embed in batches.
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("building text encoder (type=%s, device=%s) ...",
             cfg.text_encoder.get("type", "llm2vec"), device)
    text_encoder = build_text_encoder(cfg.text_encoder, device=device)

    feats_list: List[torch.Tensor] = []
    bs = max(1, int(args.batch_size))
    log.info("encoding %d captions in batches of %d ...", len(captions_to_embed), bs)
    for i in tqdm(range(0, len(captions_to_embed), bs), desc="encode"):
        batch = captions_to_embed[i : i + bs]
        feats, pad_mask = encode_texts(text_encoder, batch, device)
        # We require pooled output (L=1) — error if encoder returns per-token.
        if feats.shape[1] != 1:
            raise RuntimeError(
                f"precompute requires a POOLED encoder (L=1); got feats shape "
                f"{tuple(feats.shape)}. Per-token encoders are not supported "
                f"by CachedTextEncoder."
            )
        feats_list.append(feats.squeeze(1).to("cpu", torch.float32))
    new_features = torch.cat(feats_list, dim=0) if feats_list else torch.empty(
        (0, getattr(text_encoder, "llm_dim", 0)), dtype=torch.float32,
    )

    # 4) Merge with existing rows (if --resume), save.
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
            "encoder_type": str(cfg.text_encoder.get("type", "llm2vec")),
            "model": _resolve_model_name(cfg.text_encoder),
            "dim": int(all_features.shape[1]) if all_features.numel() else 0,
            "n": int(all_features.shape[0]),
            "config_path": str(Path(args.config).resolve()),
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        },
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(blob, str(tmp))
    tmp.replace(out_path)
    log.info("wrote %s  (%d captions, dim=%d, %.1f MB)",
             out_path, blob["meta"]["n"], blob["meta"]["dim"],
             out_path.stat().st_size / 1e6)
    log.info("In your training config, set: text_encoder.cache_path: %s", out_path)


if __name__ == "__main__":
    main()
