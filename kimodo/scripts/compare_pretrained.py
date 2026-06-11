# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Side-by-side inference comparison: pretrained Kimodo vs our trained .pt.

Used as a format/scale sanity check while reproducing the upstream training:
both models should produce the same output dict layout, joint shape, and
roughly the same world scale and orientation. The motion content itself
won't match (different fps + different training maturity) — this script is
about plumbing parity, not quality parity.

Usage::

    python -m kimodo.scripts.compare_pretrained \\
        --our-ckpt /home/jungbin_cho/kimodo_open/runs/bones_seed_small/ckpt_step0050000.pt \\
        --out /home/jungbin_cho/kimodo_open/runs/compare/

The pretrained model is loaded via ``kimodo.load_model`` (it reads from the
HuggingFace cache, which is already warm on this node). Ours is built with
the same ``model_config_path`` + ``stats_path`` but with ``fps_override=20``
to match our training, then loaded from the .pt checkpoint and wrapped in
the same ``Kimodo`` class for sampling.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def _print_summary(name: str, out: dict) -> None:
    j = out["posed_joints"]
    j0 = j[0] if j.ndim == 4 else j  # (T, J, 3)
    print(f"[{name}]  keys={list(out.keys())}")
    print(f"[{name}]  posed_joints shape={tuple(j0.shape)}  dtype={j0.dtype}")
    print(
        f"[{name}]  bbox  x=[{j0[..., 0].min():+.3f}, {j0[..., 0].max():+.3f}]"
        f"  y=[{j0[..., 1].min():+.3f}, {j0[..., 1].max():+.3f}]"
        f"  z=[{j0[..., 2].min():+.3f}, {j0[..., 2].max():+.3f}]"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--our-ckpt", required=True, help="Our .pt training checkpoint.")
    p.add_argument("--model-config", default="/home/jungbin_cho/Kimodo-SOMA-SEED-v1.1/config.yaml",
                   help="Path to the same config.yaml our training uses (defines the denoiser arch).")
    p.add_argument("--stats-path", default="/home/jungbin_cho/Kimodo-SOMA-SEED-v1.1/stats/motion/")
    p.add_argument("--pretrained-name", default="kimodo-soma-seed",
                   help="Short key for kimodo.load_model.")
    p.add_argument("--our-fps", type=int, default=20,
                   help="Override fps for ours (matches the denoiser_fps_override used at training).")
    p.add_argument("--prompt", default="a person is walking forward.",
                   help="One prompt for both models.")
    p.add_argument("--num-frames", type=int, default=100,
                   help="Generate the same number of frames for both models so the MP4 aligns.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-denoising-steps", type=int, default=100,
                   help="Matches the upstream generate.py default (--diffusion_steps 100).")
    p.add_argument("--cfg-weight", type=float, nargs=2, default=[2.0, 2.0])
    p.add_argument("--num-base-steps", type=int, default=1000)
    p.add_argument("--cfg-type", default="separated")
    p.add_argument("--out", required=True, help="Output directory for NPZs + MP4s.")
    args = p.parse_args()

    # Force local LLM2Vec (skip the API probe + fallback dance).
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    # Point kimodo's load_model at the local HF snapshot mirror so we never hit network.
    # (`/home/jungbin_cho/Kimodo-SOMA-SEED-v1.1` is the unpacked snapshot; set
    # CHECKPOINT_DIR to its parent so load_model finds it by display name.)
    os.environ.setdefault("CHECKPOINT_DIR", "/home/jungbin_cho/")
    os.environ.setdefault("LOCAL_CACHE", "True")

    from kimodo import load_model
    from kimodo.model import Kimodo
    from kimodo.scripts.render_soma import render_sidebyside
    from kimodo.scripts.train import build_denoiser_from_model_config

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Pretrained reference ----------
    print(f"Loading pretrained '{args.pretrained_name}' ...")
    pre = load_model(args.pretrained_name, device=device_str)
    print(f"  pretrained fps     = {pre.fps}")
    print(f"  pretrained skeleton= {type(pre.skeleton).__name__} ({pre.skeleton.nbjoints} joints)")
    print(f"  output skeleton    = {type(pre.output_skeleton).__name__} ({pre.output_skeleton.nbjoints} joints)")

    # ---------- 2) Ours ----------
    print(f"Building 'ours' from {args.model_config} (fps={args.our_fps}) ...")
    ours_denoiser = build_denoiser_from_model_config(
        args.model_config, args.stats_path, fps_override=args.our_fps,
    ).to(device)
    obj = torch.load(args.our_ckpt, map_location=device, weights_only=False)
    sd = obj.get("denoiser", obj)
    missing, unexpected = ours_denoiser.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  state_dict mismatch — missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing[:3]:
            print(f"    first missing: {missing[:3]}")
        if unexpected[:3]:
            print(f"    first unexpected: {unexpected[:3]}")
    else:
        print("  state_dict loaded cleanly.")
    print(f"  ours step          = {obj.get('step')}")
    print(f"  ours denoiser fps  = {ours_denoiser.motion_rep.fps}")

    # Reuse the pretrained's live text encoder so we don't load LLM2Vec twice.
    ours = Kimodo(
        denoiser=ours_denoiser,
        text_encoder=pre.text_encoder,
        num_base_steps=args.num_base_steps,
        device=device_str,
        cfg_type=args.cfg_type,
    )
    print(f"  ours wrapper       = Kimodo(fps={ours.fps})")

    # ---------- 3) Sample both with the same seed ----------
    print(f"\nGenerating {args.num_frames} frames for prompt:\n  {args.prompt!r}\n")

    def _sample(model, label):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Sampling {label} ...")
        return model(
            args.prompt, args.num_frames,
            num_denoising_steps=args.num_denoising_steps,
            cfg_weight=list(args.cfg_weight),
            num_samples=1, return_numpy=True,
        )

    pre_out = _sample(pre, "pretrained")
    ours_out = _sample(ours, "ours")

    print()
    _print_summary("pretrained", pre_out)
    _print_summary("ours      ", ours_out)

    # ---------- 4) Save NPZ ----------
    np.savez(
        out_dir / "pretrained.npz", prompt=args.prompt,
        **{k: v for k, v in pre_out.items() if hasattr(v, "shape")},
    )
    np.savez(
        out_dir / "ours.npz", prompt=args.prompt,
        **{k: v for k, v in ours_out.items() if hasattr(v, "shape")},
    )

    # ---------- 5) Side-by-side stick-figure MP4 ----------
    pre_joints = pre_out["posed_joints"][0] if pre_out["posed_joints"].ndim == 4 else pre_out["posed_joints"]
    ours_joints = ours_out["posed_joints"][0] if ours_out["posed_joints"].ndim == 4 else ours_out["posed_joints"]

    pre_parents = pre.output_skeleton.joint_parents.cpu().numpy().tolist()
    ours_parents = ours.output_skeleton.joint_parents.cpu().numpy().tolist()
    if pre_parents != ours_parents:
        print("WARNING: output-skeleton parent arrays differ; rendering with pretrained's parents.")

    print(f"\nRendering side-by-side ...")
    import imageio.v3 as iio
    frames = render_sidebyside(
        pre_joints, ours_joints, pre_parents,
        caption=f"pretrained  |  ours  —  {args.prompt}",
    )
    mp4_path = out_dir / "compare.mp4"
    iio.imwrite(str(mp4_path), frames, fps=float(pre.fps), codec="h264", plugin="pyav")
    print(f"Wrote {mp4_path}")
    print(f"Wrote {out_dir / 'pretrained.npz'}, {out_dir / 'ours.npz'}")


if __name__ == "__main__":
    main()
