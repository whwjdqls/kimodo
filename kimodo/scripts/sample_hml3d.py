# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline sampler for HumanML3D-in-kimodo checkpoints.

Loads a trained denoiser checkpoint, pulls captions + ground-truth motions
from the test split, samples with classifier-free guidance, and writes a
per-sample folder containing:

    <out_dir>/<NN>_<id>/
        caption.txt           # the conditioning caption
        gt.npz                # ground-truth kimodo features (T, 273) + caption
        gen.npz               # generated kimodo features  (T, 273) + caption + cfg_scale
        sidebyside.mp4        # GT (blue, left) vs generated (red, right) animation
        still.png             # 2 rows x N frames overview

Uses the same ``_cfg_sample`` helper as the in-training viz path
(``kimodo.scripts.train_w_hml3d``) and the upgraded renderer in
``kimodo.scripts.render_hml3d``.

Usage:
    python -m kimodo.scripts.sample_hml3d \\
        --ckpt /path/to/runs/<run>/ckpt_step0050000.pt \\
        --out-dir /path/to/runs/<run>/sampled/step0050000 \\
        --n-samples 8 --cfg-scale 2.5

If ``--config`` is omitted, ``<ckpt_dir>/config.yaml`` is used.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Reuse the training-time helpers so sampling stays in lockstep with the model.
from kimodo.model.diffusion import Diffusion
from kimodo.scripts.train import (
    build_denoiser_from_model_config,
    build_text_encoder,
    encode_texts,
)
from kimodo.scripts.train_w_hml3d import (
    _cfg_sample,
    _load_test_examples,
)
from kimodo.motion_rep.fk_hml3d import world_joints_from_kimodo_features
from kimodo.scripts.render_hml3d import render_sidebyside, save_still_pair

log = logging.getLogger("kimodo.sample_hml3d")


def _resolve_num_steps(model_cfg_path: str) -> int:
    try:
        raw = OmegaConf.load(model_cfg_path)
        return int(raw.get("num_base_steps", 1000))
    except Exception:
        return 1000


def _load_state_dict(ckpt: dict, use_ema: bool) -> dict:
    """Pull denoiser state dict (raw weights or EMA shadow) out of a checkpoint."""
    if "denoiser" not in ckpt:
        raise KeyError(
            f"Checkpoint is missing the 'denoiser' key (got: {sorted(ckpt.keys())})."
        )
    if use_ema:
        if not ckpt.get("ema"):
            raise KeyError("Checkpoint has no 'ema' shadow; rerun without --use-ema.")
        # EMA shadow only stores float params; merge over the live state dict
        # so non-float buffers (positions, masks) are preserved.
        merged = dict(ckpt["denoiser"])
        for k, v in ckpt["ema"].items():
            merged[k] = v
        return merged
    return ckpt["denoiser"]


def _safe_name(s: str, n: int = 40) -> str:
    return s.lower().replace(" ", "_").replace(".", "").replace("/", "_")[:n]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=str, help="Path to ckpt_step*.pt")
    p.add_argument("--config", type=str, default=None,
                   help="Run config.yaml (default: <ckpt_dir>/config.yaml).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output root (default: <ckpt_dir>/sampled/<ckpt_stem>).")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--cfg-scale", type=float, default=2.5,
                   help="Classifier-free guidance scale. 1.0 disables CFG.")
    p.add_argument("--num-denoising-steps", type=int, default=None,
                   help="Override cfg.viz.num_denoising_steps.")
    p.add_argument("--split-file", type=str, default=None,
                   help="Override the test split file (one motion id per line).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap per-sample length in frames (default: cfg.viz.test_max_frames).")
    p.add_argument("--min-frames", type=int, default=None,
                   help="Skip test ids shorter than this (default: cfg.data.min_motion_len).")
    p.add_argument("--use-ema", action="store_true",
                   help="Sample with the EMA-averaged weights stored in the ckpt.")
    p.add_argument("--device", type=str, default=None,
                   help="cuda | cuda:N | cpu (default: cuda if available).")
    p.add_argument("--n-still-frames", type=int, default=6,
                   help="Number of frames in the still PNG summary.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    cfg_path = Path(args.config) if args.config else (ckpt_path.parent / "config.yaml")
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {cfg_path} (pass --config explicitly)")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)

    out_dir = Path(args.out_dir) if args.out_dir else (
        ckpt_path.parent / "sampled" / ckpt_path.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("checkpoint: %s", ckpt_path)
    log.info("config:     %s", cfg_path)
    log.info("out dir:    %s", out_dir)

    # ---- device ----
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device:     %s", device)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    # ---- denoiser ----
    denoiser = build_denoiser_from_model_config(
        cfg.model_config_path,
        cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    ).to(device)

    log.info("loading checkpoint weights (use_ema=%s) ...", args.use_ema)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = _load_state_dict(ckpt, use_ema=args.use_ema)
    denoiser.load_state_dict(state, strict=False)
    denoiser.eval()
    motion_rep = denoiser.motion_rep
    assert motion_rep.motion_rep_dim == 273, (
        f"Expected motion_rep_dim=273, got {motion_rep.motion_rep_dim}."
    )

    # ---- diffusion ----
    n_steps = int(args.num_denoising_steps or cfg.viz.num_denoising_steps)
    diffusion = Diffusion(num_base_steps=_resolve_num_steps(cfg.model_config_path)).to(device)
    log.info("diffusion: %d base steps; sampling with %d steps; cfg_scale=%.2f",
             diffusion.num_base_steps, n_steps, args.cfg_scale)

    # ---- stats (only needed to unnormalize the model output) ----
    mean = np.load(cfg.data.mean_path).astype(np.float32)
    std = np.load(cfg.data.std_path).astype(np.float32)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    # ---- text encoder ----
    log.info("building text encoder ...")
    text_encoder = build_text_encoder(cfg.text_encoder, device=device)

    # ---- test examples ----
    split_file = args.split_file or cfg.viz.get("test_split_file", cfg.data.split_file)
    max_frames = int(args.max_frames or cfg.viz.get("test_max_frames", cfg.data.max_motion_length))
    min_frames = int(args.min_frames or cfg.data.get("min_motion_len", 40))
    log.info("loading test examples from %s (n=%d, max_frames=%d) ...",
             split_file, args.n_samples, max_frames)
    test_examples = _load_test_examples(
        motion_dir=cfg.data.motion_dir,
        text_dir=cfg.data.text_dir,
        split_file=split_file,
        n_samples=int(args.n_samples),
        max_frames=max_frames,
        min_frames=min_frames,
    )
    if not test_examples:
        raise SystemExit("no test examples loaded — check --split-file and motion_dir.")
    log.info("loaded %d test examples", len(test_examples))

    # ---- sample (one batched diffusion pass) ----
    lengths = [ex["length"] for ex in test_examples]
    captions = [ex["caption"] for ex in test_examples]
    max_T = int(max(lengths))
    B = len(test_examples)

    text_feat, text_pad_mask = encode_texts(text_encoder, captions, device)
    pad_mask = torch.zeros(B, max_T, dtype=torch.bool, device=device)
    for i, L in enumerate(lengths):
        pad_mask[i, :L] = True
    first_heading = torch.zeros(B, device=device)
    motion_mask = torch.zeros(B, max_T, motion_rep.motion_rep_dim, device=device)
    observed = torch.zeros(B, max_T, motion_rep.motion_rep_dim, device=device)

    log.info("running diffusion (B=%d, max_T=%d, n_steps=%d) ...", B, max_T, n_steps)
    cur = _cfg_sample(
        denoiser, diffusion,
        text_feat=text_feat, text_pad_mask=text_pad_mask,
        pad_mask=pad_mask, first_heading=first_heading,
        motion_mask=motion_mask, observed=observed,
        n_steps=n_steps, cfg_scale=float(args.cfg_scale), device=device,
    )
    gen_unnorm = cur.float() * std_t + mean_t  # (B, max_T, 273)
    gen_joints_all = world_joints_from_kimodo_features(
        gen_unnorm, motion_rep.slice_dict, n_joints=motion_rep.nbjoints,
    ).cpu().numpy()  # (B, max_T, J, 3)
    gen_unnorm_np = gen_unnorm.cpu().numpy()

    fps = int(cfg.data.get("fps", 20))

    # ---- write per-sample folder ----
    import imageio.v3 as iio  # local import to avoid hard dep when --help

    for i, ex in enumerate(test_examples):
        L = ex["length"]
        sample_dir = out_dir / f"{i:02d}_{ex['id']}_{_safe_name(ex['caption'])}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # caption + features
        (sample_dir / "caption.txt").write_text(ex["caption"] + "\n")
        np.savez(sample_dir / "gt.npz", features=ex["gt_features"], caption=ex["caption"])
        np.savez(
            sample_dir / "gen.npz",
            features=gen_unnorm_np[i, :L],
            caption=ex["caption"],
            cfg_scale=float(args.cfg_scale),
        )

        # joint trajectories
        gen_joints = gen_joints_all[i, :L]
        gt_joints = world_joints_from_kimodo_features(
            torch.from_numpy(ex["gt_features"]).to(device).unsqueeze(0),
            motion_rep.slice_dict, n_joints=motion_rep.nbjoints,
        )[0].cpu().numpy()

        # mp4 + still png
        try:
            frames = render_sidebyside(gt_joints, gen_joints, caption=ex["caption"])
            iio.imwrite(
                str(sample_dir / "sidebyside.mp4"), frames,
                fps=float(fps), codec="h264", plugin="pyav",
            )
        except Exception as e:
            log.warning("MP4 render failed for %s (%s); skipping.", ex["id"], e)
        try:
            save_still_pair(
                gt_joints, gen_joints, sample_dir / "still.png",
                caption=ex["caption"], n_frames=int(args.n_still_frames),
            )
        except Exception as e:
            log.warning("Still PNG failed for %s (%s); skipping.", ex["id"], e)

        log.info("[%d/%d] %s  L=%d  -> %s", i + 1, B, ex["id"], L, sample_dir.name)

    log.info("done. wrote %d samples to %s", B, out_dir)


if __name__ == "__main__":
    main()
