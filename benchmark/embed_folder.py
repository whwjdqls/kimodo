# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Step (3) of evaluation pipeline.

This script recursively embeds generated motions, ground-truth motions, and text prompts from a test suite folder tree with the pre-trained TMR model.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kimodo.meta import parse_prompts_from_meta
from kimodo.model.load_model import load_model
from kimodo.tools import load_json


def discover_motion_folders(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    out: list[Path] = []
    for meta_path in root.rglob("meta.json"):
        src_dir = meta_path.parent
        if (src_dir / "motion.npz").is_file() or (src_dir / "gt_motion.npz").is_file():
            out.append(src_dir)
    return sorted(out)


def _resample_time(posed_joints: torch.Tensor, src_fps: float, tgt_fps: float) -> torch.Tensor:
    """Linearly resample world joint positions [T, J, 3] from src_fps to tgt_fps.

    The benchmark/our models run at 20 fps but the TMR evaluator was trained at
    30 fps, where its motion_rep computes velocities (scaled by fps) and
    velocity-thresholded foot contacts. Feeding 20 fps motion to a 30 fps TMR
    makes everything look 1.5x slow and corrupts the embedding. We upsample both
    generated and GT positions to 30 fps so TMR sees physically correct dynamics.
    Positions interpolate cleanly under linear interp (unlike rotations).
    """
    if src_fps == tgt_fps:
        return posed_joints
    T, J, C = posed_joints.shape
    new_T = max(1, round(T * tgt_fps / src_fps))
    # [T, J, 3] -> [1, J*3, T] for 1D linear interp along time
    x = posed_joints.reshape(T, J * C).permute(1, 0).unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=new_T, mode="linear", align_corners=True)
    return x.squeeze(0).permute(1, 0).reshape(new_T, J, C)


def _load_posed_joints(
    npz_path: Path, device: str, src_fps: float | None = None, tgt_fps: float | None = None,
) -> torch.Tensor:
    data = np.load(npz_path)
    if "posed_joints" not in data:
        raise SystemExit(f"NPZ must contain 'posed_joints': {npz_path}")
    posed_joints = data["posed_joints"]
    if posed_joints.ndim == 4:
        if posed_joints.shape[0] != 1:
            raise SystemExit(f"Expected batch size 1 for posed_joints, got {posed_joints.shape[0]} in {npz_path}")
        posed_joints = posed_joints[0]
    if posed_joints.ndim != 3:
        raise SystemExit(f"Expected posed_joints shape [T, J, 3], got {posed_joints.shape} in {npz_path}")
    pj = torch.from_numpy(posed_joints).float().to(device)
    if src_fps is not None and tgt_fps is not None:
        pj = _resample_time(pj, src_fps, tgt_fps)
    return pj


def main():
    parser = argparse.ArgumentParser(
        description="Recursively embed motion, gt_motion, and text; save motion_embedding.npy, gt_motion_embedding.npy, and text_embedding.npy when present.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Root folder to search recursively for meta.json and motion.npz and/or gt_motion.npz",
    )
    parser.add_argument(
        "--model",
        default="tmr-soma-rp",
        help="Model for encoding (e.g. TMR-SOMA-RP-v1, tmr-soma-rp). Default: tmr-soma-rp",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-embed even if embedding files already exist",
    )
    parser.add_argument(
        "--text_encoder_fp32",
        action="store_true",
        help="Uses fp32 for the text encoder rather than default bfloat16.",
    )
    parser.add_argument(
        "--src-fps",
        type=float,
        default=None,
        help="Source fps of BOTH motion and gt npzs. Sets --motion-fps and --gt-fps unless "
        "those are given. If set with --tgt-fps, posed_joints are time-interpolated to "
        "--tgt-fps before TMR encoding (e.g. 20 -> 30 so a 30fps-trained TMR sees correct "
        "velocities).",
    )
    parser.add_argument(
        "--motion-fps",
        type=float,
        default=None,
        help="Source fps of the generated motion.npz (overrides --src-fps). Use when the model "
        "generates at a different fps than the GT, e.g. a 30fps model: --motion-fps 30 leaves it "
        "unchanged while --gt-fps 20 upsamples GT to --tgt-fps.",
    )
    parser.add_argument(
        "--gt-fps",
        type=float,
        default=None,
        help="Source fps of gt_motion.npz (overrides --src-fps).",
    )
    parser.add_argument(
        "--tgt-fps",
        type=float,
        default=None,
        help="Target fps to resample posed_joints to (should match the TMR training fps, e.g. 30).",
    )
    args = parser.parse_args()
    motion_fps = args.motion_fps if args.motion_fps is not None else args.src_fps
    gt_fps = args.gt_fps if args.gt_fps is not None else args.src_fps
    if args.tgt_fps is not None and (motion_fps is None or gt_fps is None):
        raise SystemExit("--tgt-fps requires a source fps (--src-fps, or both --motion-fps/--gt-fps).")
    if args.tgt_fps is None and (motion_fps is not None or gt_fps is not None):
        raise SystemExit("source fps given without --tgt-fps; set --tgt-fps too (or none).")

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {folder}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(modelname=args.model, device=device, default_family="TMR", text_encoder_fp32=args.text_encoder_fp32)

    dirs = discover_motion_folders(folder)
    if not dirs:
        raise SystemExit(f"No directories with meta.json and (motion.npz or gt_motion.npz) found under {folder}")
    print(f"Discovered {len(dirs)} motion folders.")

    skipped_motion = 0
    skipped_gt = 0
    skipped_text = 0
    for sample_dir in tqdm(dirs, desc="Embedding"):
        meta_path = sample_dir / "meta.json"
        meta = load_json(meta_path)
        texts, _ = parse_prompts_from_meta(meta)
        if len(texts) != 1:
            raise SystemExit(f"Expected exactly one text per motion; got {len(texts)} in {meta_path}")
        text = texts[0]

        # Embed motion.npz -> motion_embedding.npy
        if (sample_dir / "motion.npz").is_file():
            if not args.overwrite and (sample_dir / "motion_embedding.npy").is_file():
                skipped_motion += 1
            else:
                npz_path = sample_dir / "motion.npz"
                posed_joints = _load_posed_joints(npz_path, device, motion_fps, args.tgt_fps)
                with torch.inference_mode():
                    motion_emb = model.encode_motion(posed_joints, unit_vector=True)
                np.save(sample_dir / "motion_embedding.npy", motion_emb.cpu().numpy())

        # Embed gt_motion.npz -> gt_motion_embedding.npy
        if (sample_dir / "gt_motion.npz").is_file():
            if not args.overwrite and (sample_dir / "gt_motion_embedding.npy").is_file():
                skipped_gt += 1
            else:
                npz_path = sample_dir / "gt_motion.npz"
                posed_joints = _load_posed_joints(npz_path, device, gt_fps, args.tgt_fps)
                with torch.inference_mode():
                    gt_motion_emb = model.encode_motion(posed_joints, unit_vector=True)
                np.save(sample_dir / "gt_motion_embedding.npy", gt_motion_emb.cpu().numpy())

        # Embed text -> text_embedding.npy
        if not args.overwrite and (sample_dir / "text_embedding.npy").is_file():
            skipped_text += 1
        else:
            with torch.inference_mode():
                text_emb = model.encode_raw_text([text], unit_vector=True)
            np.save(sample_dir / "text_embedding.npy", text_emb.cpu().numpy())

    total_skipped = skipped_motion + skipped_gt + skipped_text
    if total_skipped:
        print(f"Embedded {len(dirs)} folders; skipped some existing files (use --overwrite to re-embed).")
    else:
        print(f"Saved motion_embedding.npy, gt_motion_embedding.npy, and text_embedding.npy in {len(dirs)} folders.")


if __name__ == "__main__":
    main()
