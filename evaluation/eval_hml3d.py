# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a kimodo HumanML3D model with the standard MoMask / T2M eval pipeline.

Pipeline:

    test caption + length  --(our kimodo model w/ CFG)-->  kimodo (T, 273)
        --(kimodo_to_humanml3d)-->  hml3d (T, 263)
        --(z-score with T2M evaluator's mean/std)-->  evaluator input
        --(MoMask T2M EvaluatorModelWrapper)-->  text/motion embeddings
        --> R-precision, MM-Dist, FID, Diversity, MultiModality

Metrics are computed exactly as MoMask's
``evaluation_mask_transformer_test_plus_res`` in
``momask-codes/utils/eval_t2m.py`` (the same paper-faithful T2M evaluator,
just driven by our model instead of MoMask's transformer).

=============================================================================
KIMODO (273-D) DECODING CONTRACT — read before tweaking the eval
=============================================================================
The kimodo rep predicts TWO views of joint positions (the ``smooth_root_pos`` +
``local_joints_positions`` block, and the ``global_rot_data`` rotation chain
that FK's to joint positions) that may *disagree* on predictions, since the
model outputs both blocks as independent linear heads. We commit the eval to
the **rotation-side view** ("Option B"):

  * Joint positions = ``KimodoLoss._fk_world_from_pred`` — chain-reset FK on
    the predicted rotation chain, anchored at
    ``smooth_root_pos + local_jp[root_idx]``.
  * World joint velocities = forward-difference of those FK joints.
  * Rotation channels (``rot_velocity``, ``rot_data``) = passed through from
    the predicted rotation block.
  * Foot contacts = passed through from the predicted ``foot_contacts`` block.

Why this choice (vs the positions-block view, "Option A"):
  * Matches training-time viz and the FK consistency loss — what you see in
    the .mp4s in ``runs/<name>/viz/`` is what gets evaluated.
  * Measures whether the predicted rotation chain produces a consistent
    skeleton, not just whether the position-head happens to land on plausible
    joints.

The cost — known imperfections measured against GT round-trip:
  1. **Heading-wrap edge** (FIXED): for ~7% of GT motions the root-rotation
     extraction had a ±π jump at the atan2 branch cut. Fixed in
     ``humanml3d_to_kimodo._delta_rot_vel_from_alpha`` by wrapping the half-
     angle delta to ``(-π/2, π/2]`` (the period of the half-angle, matching
     HML3D's ``arcsin``-based encoded range) instead of ``(-π, π]``. Post-fix
     verification on 30 random GT motions: ``rot_velocity`` max err = 5e-7.

  2. **FK-vs-positions-block disagreement on ~9% of test motions**
     (KNOWN, UNFIXABLE — upstream HumanML3D data issue). Measured by
     running HumanML3D's OWN ``recover_from_rot`` vs ``recover_from_ric``
     on its OWN test split (500 random motions):

         median        1.7e-6  (float noise)
         p90           4.6e-6  (float noise)
         p95           7.0e-2  (bimodal jump)
         p99           1.7e-1
         max in 500    3.5e-1

         > 1mm    9.0%     > 10cm   3.4%
         > 1cm    8.6%     > 50cm   0%

     91% of test motions are perfectly self-consistent (float noise);
     ~9% have FK-vs-positions disagreement, mostly cm-scale, max ~35cm
     in the test sample. Cause: HumanML3D's
     ``inverse_kinematics_np(smooth_forward=True)`` during preprocessing
     produces ``rot_data`` that doesn't always FK back to the same joint
     positions stored in ``ric_data``.

     Verified by running HML3D's OWN ``forward_kinematics_cont6d`` on its
     OWN ``rot_data`` for outlier motions — gives the EXACT SAME error
     numbers we measure in kimodo's FK (1.168 / 0.775 / 0.230 / 0.230 m on
     four corpus-wide outliers). So our chain-reset FK is byte-for-byte
     what HML3D's FK would compute; the disagreement is HML3D-intrinsic.
     Also verified NOT to be: bone-length derivation (per-motion median
     vs canonical from motion 012314 give identical error); raw_offsets
     vs tgt_offsets (byte-identical for the canonical skeleton); heading
     wrap (rot_velocity matches HML3D to 5e-7); root position (matches
     to 4e-9); a convention difference in kimodo's chain-reset compose
     (reproduces HML3D's chain walk exactly).

     **Effect on eval metrics — measured ceiling baseline**:
     Running this pipeline with a "perfect rotation predictor" (use HML3D's
     own stored rot_data + root as the source, run our exact Option B
     re-encoding) and comparing to real GT gives:

         FID(FK-GT vs real-GT)  = 0.083 ± 0.001
         R@1 real / FK-GT       = 0.513 / 0.507   (−0.6 pts)
         R@3 real / FK-GT       = 0.799 / 0.794   (−0.5 pts)
         Diversity real / FK-GT = 9.34  / 9.13    (−2%)

     So Option B adds ~0.083 to every reported FID — same direction for
     every model, so rankings are preserved exactly. Compared to typical
     model-vs-model FID differences (~0.1+), this is negligible. **Nothing
     to fix in our pipeline** — kimodo's FK is exactly faithful to
     HumanML3D's rotation-side representation, and the resulting bias is
     well below the resolution of model comparisons.

     Reproduce: ``python -m evaluation.eval_hml3d_fk_baseline``

     **Effect on metrics**: this adds a small additive bias on a fixed ~10%
     subset of test motions, which manifests as FID noise on the order of
     ~0.05. Same bias hits every model evaluated through this pipeline so
     model-vs-model comparisons remain valid; absolute FID is slightly
     pessimistic vs an Option-A eval.

If you want to flip the eval to Option A (positions block, no FK), remove
the ``decode['posed_joints'] = fk_joints`` / ``decode['velocities'] = fk_vel``
overrides in ``sample_and_convert_batch`` and ``_decode_gen_joints_contacts``.
Same change in both places.

Diagnostic scripts that established this contract:
  * /home/jungbin_cho/kimodo_open/verify_heading_wrap_fix.py
  * /home/jungbin_cho/kimodo_open/diag_compare_samples.py
  * /home/jungbin_cho/kimodo_open/diag_ric_outlier.py
  * /home/jungbin_cho/kimodo_open/diag_fk_vs_positions_block.py
  * /home/jungbin_cho/kimodo_open/diag_canonical_bones.py
=============================================================================

Resources (must exist locally):

  * MoMask repo + downloaded evaluator + glove:
      /home/jungbin_cho/momask-codes/
          checkpoints/t2m/Comp_v6_KLD005/opt.txt
          checkpoints/t2m/Comp_v6_KLD005/meta/{mean,std}.npy
          checkpoints/t2m/text_mot_match/model/finest.tar
          glove/our_vab_{data.npy,idx.pkl,words.pkl}

  * HumanML3D dataset:
      /home/jungbin_cho/HumanML3D/HumanML3D/
          test.txt, new_joint_vecs/*.npy, texts/*.txt

  * Our kimodo checkpoint, e.g.:
      /home/jungbin_cho/kimodo_open/runs/<run>/ckpt_step*.pt
      /home/jungbin_cho/kimodo_open/runs/<run>/config.yaml

Usage::

    python -m kimodo.scripts.eval_hml3d \\
        --ckpt /home/jungbin_cho/kimodo_open/runs/kim_hml3d_fp32/ckpt_step0500000.pt \\
        --batch-size 32 --repeat-times 5 --cfg-scale 2.5 --use-ema

(``--repeat-times`` controls the number of independent eval runs; we report
mean ± std across runs, matching MoMask's reporting convention.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Shim for HumanML3D's older numpy API (np.float / np.int).
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

# Make MoMask's modules importable. MoMask uses repo-relative imports
# ("from utils.metrics import ..."), so we add its root to sys.path AND chdir
# to it just before building the loader (so its hardcoded "./glove" resolves).
MOMASK_ROOT = Path("/home/jungbin_cho/momask-codes").resolve()
sys.path.insert(0, str(MOMASK_ROOT))

# Now we can import MoMask pieces.
from utils.get_opt import get_opt as momask_get_opt  # noqa: E402
from utils.metrics import (  # noqa: E402
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
    euclidean_distance_matrix,
)
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader  # noqa: E402
from models.t2m_eval_wrapper import EvaluatorModelWrapper  # noqa: E402

# Kimodo pieces.
from kimodo.scripts.train_w_hml3d import _cfg_sample  # noqa: E402
from kimodo.scripts.train import (  # noqa: E402
    build_denoiser_from_model_config,
    build_text_encoder,
    encode_texts,
)
from kimodo.model.diffusion import Diffusion  # noqa: E402

# Eval glue.
from evaluation.kimodo_decode import kimodo_features_to_decode_dict  # noqa: E402

# Foot-skating + contact metrics (reused from the kimodo SOMA-side benchmark;
# the metric classes themselves are skeleton-agnostic and only need
# (posed_joints (B,T,J,3), foot_contacts (B,T,4), lengths (B,)) + a skeleton
# with foot_joint_idx — both reps' motion_rep.skeleton (SMPLXSkeleton22) qualifies).
from kimodo.metrics import (  # noqa: E402
    FootContactConsistency,
    FootSkateFromContacts,
    FootSkateFromHeight,
    FootSkateRatio,
    aggregate_metrics,
)
from kimodo.motion_rep.humanml3d_native import (  # noqa: E402
    hml3d_native_world_joints_from_features,
)
from kimodo.motion_rep.uniego import (  # noqa: E402
    uniego_world_joints_from_features,
)

# Kimodo<->HumanML3D 263-D converter (lives in benchmark/).
BENCHMARK_ROOT = Path("/home/jungbin_cho/kimodo_open/benchmark").resolve()
sys.path.insert(0, str(BENCHMARK_ROOT))
from humanml3d_to_kimodo import kimodo_to_humanml3d  # noqa: E402
# UniEgo (211-D) -> HumanML3D 263 (chains uniego_to_kimodo -> kimodo_to_humanml3d).
from kimodo_to_uniego import uniego_to_humanml3d  # noqa: E402

log = logging.getLogger("eval_hml3d")


# -----------------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------------
def build_eval_wrapper_and_loader(
    batch_size: int,
    device: torch.device,
    humanml3d_root: Path = Path("/home/jungbin_cho/HumanML3D/HumanML3D"),
):
    """Load momask's EvaluatorModelWrapper + HumanML3D test loader.

    Overrides the path fields ``get_opt`` hardcodes (which assume cwd =
    momask-codes repo and ``./dataset/HumanML3D``) to point at the real
    dataset on this machine.
    """
    opt_path = MOMASK_ROOT / "checkpoints" / "t2m" / "Comp_v6_KLD005" / "opt.txt"

    # MoMask's get_opt builds paths from opt.checkpoints_dir; passing the
    # absolute path makes them resolve correctly regardless of cwd.
    opt = momask_get_opt(str(opt_path), device)
    opt.checkpoints_dir = str(MOMASK_ROOT / "checkpoints")
    opt.save_root = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = os.path.join(opt.save_root, "model")
    opt.meta_dir = os.path.join(opt.save_root, "meta")
    opt.data_root = str(humanml3d_root) + "/"
    opt.motion_dir = str(humanml3d_root / "new_joint_vecs")
    opt.text_dir = str(humanml3d_root / "texts")

    eval_wrapper = EvaluatorModelWrapper(opt)

    # The momask dataset reads ./glove relative to cwd, so chdir into the repo
    # right before building the loader, then restore cwd.
    prev_cwd = os.getcwd()
    os.chdir(MOMASK_ROOT)
    try:
        # Rewrite the opt file path too so get_opt inside dataset_motion_loader
        # sees the same overrides. We monkey-patch get_opt to inject them.
        loader, dataset = _build_motion_loader_with_overrides(
            str(opt_path), batch_size, "test", device, humanml3d_root,
        )
    finally:
        os.chdir(prev_cwd)
    return eval_wrapper, loader, dataset, opt


def _build_motion_loader_with_overrides(
    opt_path, batch_size, fname, device, humanml3d_root,
):
    """Same as motion_loaders.dataset_motion_loader.get_dataset_motion_loader,
    but with our path overrides applied to the inner opt before reading data.
    """
    from utils.get_opt import get_opt as _get
    from utils.word_vectorizer import WordVectorizer
    from data.t2m_dataset import Text2MotionDatasetEval, collate_fn
    from torch.utils.data import DataLoader
    from os.path import join as pjoin

    opt = _get(opt_path, device)
    opt.checkpoints_dir = str(MOMASK_ROOT / "checkpoints")
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.meta_dir = pjoin(opt.save_root, "meta")
    opt.data_root = str(humanml3d_root) + "/"
    opt.motion_dir = str(humanml3d_root / "new_joint_vecs")
    opt.text_dir = str(humanml3d_root / "texts")

    mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(opt.meta_dir, "std.npy"))
    w_vectorizer = WordVectorizer(str(MOMASK_ROOT / "glove"), "our_vab")
    split_file = pjoin(opt.data_root, f"{fname}.txt")
    dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, drop_last=True,
        collate_fn=collate_fn, shuffle=True,
    )
    return loader, dataset


# -----------------------------------------------------------------------------
# Kimodo model loading
# -----------------------------------------------------------------------------
def load_kimodo_model(ckpt_path: Path, device: torch.device, use_ema: bool):
    """Build the kimodo denoiser + text encoder from the run config + ckpt."""
    from omegaconf import OmegaConf

    cfg_path = ckpt_path.parent / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)

    denoiser = build_denoiser_from_model_config(
        cfg.model_config_path,
        cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    ).to(device)

    log.info("loading checkpoint weights from %s (use_ema=%s) ...", ckpt_path, use_ema)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = dict(ckpt["denoiser"])
    if use_ema:
        if not ckpt.get("ema"):
            raise KeyError("Checkpoint has no EMA shadow; rerun without --use-ema.")
        for k, v in ckpt["ema"].items():
            state[k] = v
    denoiser.load_state_dict(state, strict=False)
    denoiser.eval()

    # Diffusion (1000-step schedule by default; eval samples a strided subset).
    n_base = _resolve_num_base_steps(cfg.model_config_path)
    diffusion = Diffusion(num_base_steps=n_base).to(device)

    # Stats for unnormalizing the model output.
    mean_np = np.load(cfg.data.mean_path).astype(np.float32)
    std_np = np.load(cfg.data.std_path).astype(np.float32)

    # Text encoder.
    text_encoder = build_text_encoder(cfg.text_encoder, device=device)

    return denoiser, diffusion, text_encoder, mean_np, std_np, cfg


def _resolve_num_base_steps(model_cfg_path: str) -> int:
    from omegaconf import OmegaConf as _O
    try:
        raw = _O.load(model_cfg_path)
        return int(raw.get("num_base_steps", 1000))
    except Exception:
        return 1000


# -----------------------------------------------------------------------------
# Joints + contacts decoding (shared by gen and GT pathways)
# -----------------------------------------------------------------------------
def _decode_gen_joints_contacts(
    gen_unnorm: torch.Tensor,      # (B, pad_T, D) — UN-normalized model output
    m_lengths: torch.Tensor,       # (B,) ints, frames of valid motion per sample
    motion_rep,
    max_motion_length: int = 196,
    fk_helper=None,                # KimodoLoss instance for FK-on-rotations (273-D only)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode generated samples into (world_joints, binary foot_contacts) — the
    inputs the kimodo foot-skating metrics expect.

    For the kimodo 273-D rep, joints are derived via chain-reset FK on the
    predicted rotation chain (``fk_helper._fk_world_from_pred``), NOT read
    from the predicted positions block. This commits the eval to the
    rotation-side view of the prediction — matching what viz / FK loss
    measure during training (the positions and rotations blocks are
    independent model outputs and may disagree).

    Returns:
        joints   : (B, max_motion_length, 22, 3) float32, zero-padded past length
        contacts : (B, max_motion_length, 4)     float32 in {0,1}, zero-padded
    """
    device = gen_unnorm.device
    B = gen_unnorm.shape[0]
    D = int(motion_rep.motion_rep_dim)

    joints = torch.zeros(B, max_motion_length, int(motion_rep.nbjoints), 3,
                         dtype=torch.float32, device=device)
    contacts = torch.zeros(B, max_motion_length, 4, dtype=torch.float32, device=device)

    if D == 263:
        # HumanML3D native: foot_contacts live at slice_dict["foot_contacts"]
        # (== last 4 dims), world joints recovered from the ric_data + root state
        # via HumanML3D's ``recover_from_ric`` analog.
        c_sl = motion_rep.slice_dict["foot_contacts"]
        # Batched joint recovery handles (B, T, 263) directly.
        joints_full = hml3d_native_world_joints_from_features(gen_unnorm)  # (B, pad_T, 22, 3)
        for i in range(B):
            L = min(int(m_lengths[i]), max_motion_length)
            joints[i, :L] = joints_full[i, :L]
            contacts[i, :L] = (gen_unnorm[i, :L, c_sl] > 0.5).float()
    elif D == 273:
        # Kimodo 273-D: foot_contacts read from the features block; joints
        # derived by running chain-reset FK on the predicted rotation chain
        # (anchored at smooth_root + local_jp[root] for the root xyz).
        # See the module-level "KIMODO (273-D) DECODING CONTRACT" docstring
        # for the rationale and known FK-accumulation outliers.
        if fk_helper is None:
            raise ValueError(
                "fk_helper is required for kimodo (273-D) joint decoding "
                "(Option B: trust the predicted rotation chain)."
            )
        c_sl = motion_rep.slice_dict["foot_contacts"]
        for i in range(B):
            L = min(int(m_lengths[i]), max_motion_length)
            fk_joints = fk_helper._fk_world_from_pred(
                gen_unnorm[i, :L].unsqueeze(0),
            )[0]  # (L, J, 3)
            joints[i, :L] = fk_joints
            contacts[i, :L] = (gen_unnorm[i, :L, c_sl] > 0.5).float()
    elif D == 211:
        # UniEgo head-centric 211-D: foot_contacts read from the features block;
        # world joints decoded by cumulative-composing the residual canonical
        # frame, then reading per-joint translations (no FK, no ambiguity — the
        # rep stores joint positions directly, so there's a single view).
        c_sl = motion_rep.slice_dict["foot_contacts"]
        joints_full = uniego_world_joints_from_features(gen_unnorm)  # (B, pad_T, 22, 3)
        for i in range(B):
            L = min(int(m_lengths[i]), max_motion_length)
            joints[i, :L] = joints_full[i, :L]
            contacts[i, :L] = (gen_unnorm[i, :L, c_sl] > 0.5).float()
    else:
        raise ValueError(
            f"unsupported motion_rep_dim={D}; expected 263 (native), 273 (kimodo) or 211 (uniego)"
        )
    return joints, contacts


def _decode_gt_joints_contacts(
    gt_pose: torch.Tensor,          # (B, T_max, 263) z-scored with evaluator stats
    m_lengths: torch.Tensor,        # (B,) ints
    eval_mean_t: torch.Tensor,      # (263,)
    eval_std_t: torch.Tensor,       # (263,)
    max_motion_length: int = 196,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror of :func:`_decode_gen_joints_contacts` for the GT loader output.

    The GT motion is always the HumanML3D 263-D layout regardless of which
    motion_rep the model uses — so we always go through the native recovery.
    """
    device = gt_pose.device
    B, T_in, _ = gt_pose.shape
    T_eff = min(T_in, max_motion_length)
    gt_unnorm = gt_pose[:, :T_eff].float() * eval_std_t + eval_mean_t  # (B, T_eff, 263)

    joints = torch.zeros(B, max_motion_length, 22, 3, dtype=torch.float32, device=device)
    contacts = torch.zeros(B, max_motion_length, 4, dtype=torch.float32, device=device)

    joints_full = hml3d_native_world_joints_from_features(gt_unnorm)  # (B, T_eff, 22, 3)
    for i in range(B):
        L = min(int(m_lengths[i]), max_motion_length, T_eff)
        joints[i, :L] = joints_full[i, :L]
        contacts[i, :L] = (gt_unnorm[i, :L, -4:] > 0.5).float()
    return joints, contacts


def _build_foot_metrics(skeleton, fps: float):
    """Instantiate the four foot-quality metrics, fresh per eval run."""
    return [
        FootSkateFromHeight(skeleton=skeleton, fps=fps),
        FootSkateFromContacts(skeleton=skeleton, fps=fps),
        FootSkateRatio(skeleton=skeleton, fps=fps),
        FootContactConsistency(skeleton=skeleton, fps=fps),
    ]


# -----------------------------------------------------------------------------
# Sampling + conversion for a batch
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_and_convert_batch(
    captions: List[str],
    m_lengths: torch.Tensor,    # (B,) ints
    denoiser, diffusion, text_encoder,
    device, mean_np, std_np,
    cfg, cfg_scale: float, n_denoise_steps: int,
    eval_mean: np.ndarray, eval_std: np.ndarray,
    max_motion_length: int = 196,
    sampler: str = "ddim",
    fk_helper=None,             # KimodoLoss for FK-on-rotations (required for 273-D)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample model features, convert to HumanML3D-263 z-scored eval features.

    Returns a 2-tuple:
        out_eval   : ``(B, max_motion_length, 263)`` float32 on CPU — z-scored
                     with the evaluator's stats; matches the MoMask loader.
        gen_unnorm : ``(B, pad_T, D)`` float32 on the sampling device — raw
                     unnormalized model output in the *training* motion-rep
                     space (263 for native HumanML3D, 273 for kimodo). Used
                     downstream to decode (joints, contacts) for the foot-
                     skating metrics without resampling the diffusion model.
    """
    motion_rep = denoiser.motion_rep
    B = len(captions)
    max_T = int(m_lengths.max())
    # Round up to the kimodo data's unit_length so the model sees a length it
    # was trained on (HumanML3D loader uses unit_length=4 too).
    unit = int(cfg.data.get("unit_length", 4))
    pad_T = ((max_T + unit - 1) // unit) * unit
    pad_T = max(pad_T, unit)

    text_feat, text_pad_mask = encode_texts(text_encoder, captions, device)
    pad_mask = torch.zeros(B, pad_T, dtype=torch.bool, device=device)
    for i, L in enumerate(m_lengths.tolist()):
        pad_mask[i, :int(L)] = True
    first_heading = torch.zeros(B, device=device)
    motion_mask = torch.zeros(B, pad_T, motion_rep.motion_rep_dim, device=device)
    observed = torch.zeros(B, pad_T, motion_rep.motion_rep_dim, device=device)

    cur = _cfg_sample(
        denoiser, diffusion,
        text_feat=text_feat, text_pad_mask=text_pad_mask,
        pad_mask=pad_mask, first_heading=first_heading,
        motion_mask=motion_mask, observed=observed,
        n_steps=n_denoise_steps, cfg_scale=cfg_scale, device=device,
        sampler=sampler,
    )  # (B, pad_T, D) normalized

    # Unnormalize with the model's own training-time stats (data.mean_path /
    # data.std_path — raw HumanML3D Mean/Std for native, kimodo stats for the
    # 273-D rep).
    mean_t = torch.from_numpy(mean_np).to(cur.device)
    std_t = torch.from_numpy(std_np).to(cur.device)
    gen_unnorm = cur.float() * std_t + mean_t  # (B, pad_T, D)

    # Convert per-sample to HumanML3D-263 and z-score with the EVALUATOR's stats.
    eval_mean_t = torch.from_numpy(eval_mean).to(device).float()
    eval_std_t = torch.from_numpy(eval_std).to(device).float()
    out = torch.zeros(B, max_motion_length, 263, dtype=torch.float32, device=device)

    # Dispatch on motion_rep dim: native is already 263-D so skip the
    # kimodo_to_humanml3d conversion. Kimodo (273-D) goes through the
    # decode-dict + converter to get HumanML3D-263.
    D = int(motion_rep.motion_rep_dim)
    if D == 263:
        for i in range(B):
            L = min(int(m_lengths[i]), max_motion_length)
            out[i, :L] = (gen_unnorm[i, :L] - eval_mean_t) / eval_std_t
    elif D == 273:
        # Predictions decompose into INDEPENDENT positions + rotations blocks
        # that may disagree. We commit to the rotation-side view: decode the
        # prediction to world joints via chain-reset FK on the predicted
        # rotation chain (anchored at smooth_root + local_jp[root]), then
        # rebuild the ENTIRE HumanML3D 263-D from those FK joints + the same
        # predicted rotations:
        #   - rot_velocity, rot_data  <-  predicted global rotations (decomposed)
        #   - lin_velocity, root_height, ric_data, local_velocity
        #                             <-  FK joint positions + their time diffs
        #   - foot_contacts           <-  predicted foot_contacts (passthrough)
        # The model's predicted positions block and predicted velocities block
        # are NOT consulted — the source of truth is the FK-decoded motion.
        #
        # See the module-level "KIMODO (273-D) DECODING CONTRACT" docstring
        # for: (a) why Option B vs Option A, (b) the heading-wrap fix, and
        # (c) the known ~10% chain-reset FK accumulation outliers and what
        # they mean for FID interpretation.
        if fk_helper is None:
            raise ValueError(
                "fk_helper is required for kimodo (273-D) eval conversion "
                "(rebuild HumanML3D 263 from FK-decoded joints + rotations)."
            )
        for i in range(B):
            L = min(int(m_lengths[i]), max_motion_length)
            decode = kimodo_features_to_decode_dict(
                gen_unnorm[i, :L], motion_rep.slice_dict, n_joints=motion_rep.nbjoints,
            )
            fk_joints = fk_helper._fk_world_from_pred(
                gen_unnorm[i, :L].unsqueeze(0),
            )[0]  # (L, J, 3)
            decode["posed_joints"] = fk_joints
            decode["root_positions"] = fk_joints[:, 0]
            # Derive world-frame joint velocities from the FK joint trajectory.
            # HumanML3D convention: velocity at frame t is the forward difference
            # joints[t+1] - joints[t]; the last frame is unused in recovery and
            # set to 0 (matches how the converter zeros data[-1, 193:259]).
            fk_vel_world = torch.zeros_like(fk_joints)
            if fk_joints.shape[0] > 1:
                fk_vel_world[:-1] = fk_joints[1:] - fk_joints[:-1]
            decode["velocities"] = fk_vel_world
            hml = kimodo_to_humanml3d(decode, device=device)  # (L, 263)
            # z-score with evaluator stats (matches what the loader does to GT).
            out[i, :L] = (hml - eval_mean_t) / eval_std_t
    elif D == 211:
        # UniEgo head-centric 211-D -> HumanML3D 263 via uniego_to_humanml3d
        # (= uniego_to_kimodo -> kimodo_to_humanml3d). The generated features
        # carry only the 211-D core (no aux velocities / r_rot_quat), so the
        # decode recomputes world velocities from the decoded joint trajectory
        # and the root yaw from the decoded root rotation — the correct decode
        # for generated motion (mirrors the kimodo path's FK-velocity rebuild).
        for i in range(B):
            L = min(int(m_lengths[i]), max_motion_length)
            ud = {"features": gen_unnorm[i, :L]}
            hml = uniego_to_humanml3d(ud, device=device)  # (L, 263)
            out[i, :L] = (hml - eval_mean_t) / eval_std_t
    else:
        raise ValueError(
            f"unsupported motion_rep_dim={D}; expected 263 (native), 273 (kimodo) or 211 (uniego)"
        )
    return out.cpu(), gen_unnorm


# -----------------------------------------------------------------------------
# One pass over the eval loader, MoMask-style
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_one_pass(
    loader, eval_wrapper, denoiser, diffusion, text_encoder,
    mean_np, std_np, cfg, device,
    cfg_scale: float, n_denoise_steps: int,
    eval_mean: np.ndarray, eval_std: np.ndarray,
    num_mm_batch: int = 3, mm_repeats: int = 30,
    cal_mm: bool = True,
    sampler: str = "ddim",
) -> Dict[str, float]:
    """Mirror of ``evaluation_mask_transformer_test_plus_res`` from momask."""
    motion_annotation_list: List[torch.Tensor] = []
    motion_pred_list: List[torch.Tensor] = []
    motion_mm_batches: List[torch.Tensor] = []

    R_precision_real = 0.0
    R_precision = 0.0
    matching_score_real = 0.0
    matching_score_pred = 0.0
    nb_sample = 0

    # Foot-skate + contact metrics (one fresh accumulator per eval run, for both
    # generated and GT). Uses the model's own skeleton (SMPLXSkeleton22 for any
    # HumanML3D run, native or 273-D kimodo) — its foot_joint_idx defines the
    # heel/toe ordering the metric classes expect.
    motion_rep = denoiser.motion_rep
    fps = float(getattr(motion_rep, "fps", cfg.data.get("fps", 20)))
    skeleton = motion_rep.skeleton
    gen_foot_metrics = _build_foot_metrics(skeleton, fps)
    gt_foot_metrics = _build_foot_metrics(skeleton, fps)
    eval_mean_t = torch.from_numpy(eval_mean).to(device).float()
    eval_std_t = torch.from_numpy(eval_std).to(device).float()

    # Build the FK helper for kimodo-rep predictions. We commit to the
    # rotation-side view of the model output (Option B): joint positions
    # come from chain-reset FK on the predicted rotations rather than the
    # predicted positions block. This makes eval consistent with viz / FK
    # loss, which both use the same FK path. Native (263-D) doesn't need a
    # helper — its features ARE joint positions and there's only one block.
    fk_helper = None
    if int(motion_rep.motion_rep_dim) == 273:
        from kimodo.scripts.train import KimodoLoss
        fk_helper = KimodoLoss(
            motion_rep=motion_rep,
            weights={},
            fk_kind="chainreset_hml3d",
            fk_target="gt",
        ).to(device)
        fk_helper.eval()

    for i, batch in enumerate(loader):
        word_embs, pos_ohot, captions, sent_len, gt_pose, m_length, _tokens = batch
        captions = list(captions) if not isinstance(captions, list) else captions
        m_length = m_length.to(device)

        # MM batches: generate `mm_repeats` independent samples per caption
        # to compute MultiModality. Only on the first `num_mm_batch` batches
        # (matches momask).
        if cal_mm and i < num_mm_batch:
            mm_embeds = []
            pred_unnorm = None
            for _ in range(mm_repeats):
                pred, pred_unnorm = sample_and_convert_batch(
                    captions, m_length, denoiser, diffusion, text_encoder,
                    device, mean_np, std_np, cfg, cfg_scale, n_denoise_steps,
                    eval_mean, eval_std, sampler=sampler,
                    fk_helper=fk_helper,
                )
                pred = pred.to(device)
                _, em_pred = eval_wrapper.get_co_embeddings(
                    word_embs, pos_ohot, sent_len, pred, m_length,
                )
                mm_embeds.append(em_pred.unsqueeze(1))
            motion_mm_batches.append(torch.cat(mm_embeds, dim=1))  # (B, mm_repeats, D)
            # The last MM sample also counts as the prediction for FID/R-prec.
            pred_for_metrics = pred
            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embs, pos_ohot, sent_len, pred_for_metrics, m_length,
            )
        else:
            pred, pred_unnorm = sample_and_convert_batch(
                captions, m_length, denoiser, diffusion, text_encoder,
                device, mean_np, std_np, cfg, cfg_scale, n_denoise_steps,
                eval_mean, eval_std, sampler=sampler,
                fk_helper=fk_helper,
            )
            pred = pred.to(device)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embs, pos_ohot, sent_len, pred, m_length,
            )

        # Foot-skating + contact metrics — gen side. Same FK-on-rotations
        # convention as the 263-D conversion above.
        gen_joints, gen_contacts = _decode_gen_joints_contacts(
            pred_unnorm, m_length, motion_rep, fk_helper=fk_helper,
        )
        for metric in gen_foot_metrics:
            metric(
                posed_joints=gen_joints,
                foot_contacts=gen_contacts,
                lengths=m_length,
            )

        # GT side.
        gt_pose = gt_pose.to(device).float()
        et, em = eval_wrapper.get_co_embeddings(
            word_embs, pos_ohot, sent_len, gt_pose, m_length,
        )

        # Foot-skating + contact metrics — GT side. GT is always native 263-D
        # in this loader regardless of motion_rep, so we decode through the
        # HumanML3D ``recover_from_ric``-style path.
        gt_joints, gt_contacts = _decode_gt_joints_contacts(
            gt_pose, m_length, eval_mean_t, eval_std_t,
        )
        for metric in gt_foot_metrics:
            metric(
                posed_joints=gt_joints,
                foot_contacts=gt_contacts,
                lengths=m_length,
            )
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += pred.shape[0]

    # Aggregate.
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    multimodality = 0.0
    if cal_mm and motion_mm_batches:
        mm_arr = torch.cat(motion_mm_batches, dim=0).cpu().numpy()  # (Nmm, mm_repeats, D)
        multimodality = float(calculate_multimodality(mm_arr, 10))

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    fid = float(calculate_frechet_distance(gt_mu, gt_cov, mu, cov))

    diversity_real = float(calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100))
    diversity_pred = float(calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100))

    # Aggregate foot-skate + contact metrics (mean over all samples).
    gen_foot = {k: float(v.mean().item()) for k, v in aggregate_metrics(gen_foot_metrics).items()}
    gt_foot = {k: float(v.mean().item()) for k, v in aggregate_metrics(gt_foot_metrics).items()}

    return {
        "fid": fid,
        "matching_score_real": float(matching_score_real / nb_sample),
        "matching_score_pred": float(matching_score_pred / nb_sample),
        "r_precision_real": (R_precision_real / nb_sample).tolist(),   # length 3 (top1/2/3)
        "r_precision_pred": (R_precision / nb_sample).tolist(),
        "diversity_real": diversity_real,
        "diversity_pred": diversity_pred,
        "multimodality": multimodality,
        "n_samples": int(nb_sample),
        # Foot-skating + contact consistency. ``gen_*`` is the model's output;
        # ``gt_*`` is the empirical lower bound on the same test split.
        "foot_skate_from_height_gen":      gen_foot.get("foot_skate_from_height", float("nan")),
        "foot_skate_from_height_gt":       gt_foot.get("foot_skate_from_height", float("nan")),
        "foot_skate_from_pred_contacts_gen": gen_foot.get("foot_skate_from_pred_contacts", float("nan")),
        "foot_skate_from_pred_contacts_gt":  gt_foot.get("foot_skate_from_pred_contacts", float("nan")),
        "foot_skate_max_vel_gen":          gen_foot.get("foot_skate_max_vel", float("nan")),
        "foot_skate_max_vel_gt":           gt_foot.get("foot_skate_max_vel", float("nan")),
        "foot_skate_ratio_gen":            gen_foot.get("foot_skate_ratio", float("nan")),
        "foot_skate_ratio_gt":             gt_foot.get("foot_skate_ratio", float("nan")),
        "foot_contact_consistency_gen":    gen_foot.get("foot_contact_consistency", float("nan")),
        "foot_contact_consistency_gt":     gt_foot.get("foot_contact_consistency", float("nan")),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--out", type=str, default=None,
                   help="Where to write the metrics JSON. Default: alongside the ckpt.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--repeat-times", type=int, default=5,
                   help="Number of independent eval runs (momask uses 20). "
                        "Each run re-samples both the loader and the model.")
    p.add_argument("--cfg-scale", type=float, default=2.5)
    p.add_argument("--num-denoising-steps", type=int, default=50)
    p.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim",
                   help="Reverse-diffusion rule. 'ddim' (default): deterministic, "
                        "works well with subsampled --num-denoising-steps (e.g. 50). "
                        "'ddpm': stochastic ancestral sampling (Ho et al. 2020 / paper-MDM); "
                        "use with --num-denoising-steps close to the training schedule (e.g. 1000).")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--device", type=str, default=None, help="cuda | cuda:N | cpu")
    p.add_argument("--no-mm", action="store_true",
                   help="Skip MultiModality (saves time; 30 extra forwards per batch on the first few batches).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--humanml3d-root", type=str, default="/home/jungbin_cho/HumanML3D/HumanML3D")
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
    out_path = Path(args.out) if args.out else (ckpt_path.parent / f"eval_{ckpt_path.stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    # Load kimodo model.
    denoiser, diffusion, text_encoder, mean_np, std_np, cfg = load_kimodo_model(
        ckpt_path, device, use_ema=args.use_ema,
    )

    # Eval wrapper + GT loader. The evaluator's mean/std (eval_mean, eval_std)
    # are loaded once and reused per sample.
    eval_wrapper, loader, dataset, opt = build_eval_wrapper_and_loader(
        args.batch_size, device, Path(args.humanml3d_root),
    )
    log.info("Eval dataset: %d items (%d batches of %d)",
             len(dataset), len(loader), args.batch_size)
    eval_mean = np.load(os.path.join(opt.meta_dir, "mean.npy")).astype(np.float32)
    eval_std = np.load(os.path.join(opt.meta_dir, "std.npy")).astype(np.float32)

    runs: List[Dict[str, float]] = []
    for run_id in range(args.repeat_times):
        # Different seed each run -> different model noise + different loader shuffle.
        seed = int(args.seed) + run_id * 1000
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        t0 = time.time()
        log.info("==== run %d/%d (seed=%d) ====", run_id + 1, args.repeat_times, seed)
        metrics = evaluate_one_pass(
            loader, eval_wrapper, denoiser, diffusion, text_encoder,
            mean_np, std_np, cfg, device,
            args.cfg_scale, args.num_denoising_steps,
            eval_mean, eval_std,
            cal_mm=(not args.no_mm),
            sampler=args.sampler,
        )
        metrics["wall_sec"] = round(time.time() - t0, 1)
        log.info(
            "  FID=%.4f  MM-Dist=%.4f  R@1=%.4f  R@2=%.4f  R@3=%.4f  Div=%.4f  MM=%.4f  (%.1fs)",
            metrics["fid"], metrics["matching_score_pred"],
            metrics["r_precision_pred"][0], metrics["r_precision_pred"][1], metrics["r_precision_pred"][2],
            metrics["diversity_pred"], metrics["multimodality"], metrics["wall_sec"],
        )
        log.info(
            "  FootSkate[height] gen=%.4f gt=%.4f m/s | [pred-contacts] gen=%.4f gt=%.4f m/s | "
            "max=%.4f m/s | ratio gen=%.4f gt=%.4f | contact-consistency gen=%.4f gt=%.4f",
            metrics["foot_skate_from_height_gen"], metrics["foot_skate_from_height_gt"],
            metrics["foot_skate_from_pred_contacts_gen"], metrics["foot_skate_from_pred_contacts_gt"],
            metrics["foot_skate_max_vel_gen"],
            metrics["foot_skate_ratio_gen"], metrics["foot_skate_ratio_gt"],
            metrics["foot_contact_consistency_gen"], metrics["foot_contact_consistency_gt"],
        )
        runs.append(metrics)

    # Aggregate mean / std across runs.
    def _mean_std(vals: List[float]):
        a = np.asarray(vals, dtype=np.float64)
        return float(a.mean()), float(a.std())

    fid_m, fid_s = _mean_std([r["fid"] for r in runs])
    mm_d_m, mm_d_s = _mean_std([r["matching_score_pred"] for r in runs])
    div_m, div_s = _mean_std([r["diversity_pred"] for r in runs])
    div_real_m, div_real_s = _mean_std([r["diversity_real"] for r in runs])
    mm_m, mm_s = _mean_std([r["multimodality"] for r in runs])
    r1_m, r1_s = _mean_std([r["r_precision_pred"][0] for r in runs])
    r2_m, r2_s = _mean_std([r["r_precision_pred"][1] for r in runs])
    r3_m, r3_s = _mean_std([r["r_precision_pred"][2] for r in runs])

    foot_keys = [
        "foot_skate_from_height_gen",      "foot_skate_from_height_gt",
        "foot_skate_from_pred_contacts_gen","foot_skate_from_pred_contacts_gt",
        "foot_skate_max_vel_gen",          "foot_skate_max_vel_gt",
        "foot_skate_ratio_gen",            "foot_skate_ratio_gt",
        "foot_contact_consistency_gen",    "foot_contact_consistency_gt",
    ]
    foot_summary = {k: dict(zip(("mean", "std"), _mean_std([r[k] for r in runs]))) for k in foot_keys}

    summary = {
        "ckpt": str(ckpt_path),
        "use_ema": bool(args.use_ema),
        "cfg_scale": float(args.cfg_scale),
        "num_denoising_steps": int(args.num_denoising_steps),
        "sampler": str(args.sampler),
        "batch_size": int(args.batch_size),
        "repeat_times": int(args.repeat_times),
        "metrics_mean_std": {
            "fid":             {"mean": fid_m,     "std": fid_s},
            "mm_dist":         {"mean": mm_d_m,    "std": mm_d_s},
            "diversity_pred":  {"mean": div_m,     "std": div_s},
            "diversity_real":  {"mean": div_real_m,"std": div_real_s},
            "multimodality":   {"mean": mm_m,      "std": mm_s},
            "r_precision_top1":{"mean": r1_m,      "std": r1_s},
            "r_precision_top2":{"mean": r2_m,      "std": r2_s},
            "r_precision_top3":{"mean": r3_m,      "std": r3_s},
            **foot_summary,
        },
        "per_run": runs,
    }

    log.info("==================== SUMMARY ====================")
    log.info("FID            %.3f +- %.3f", fid_m, fid_s)
    log.info("MM-Dist        %.3f +- %.3f", mm_d_m, mm_d_s)
    log.info("R-Precision@1  %.3f +- %.3f", r1_m, r1_s)
    log.info("R-Precision@2  %.3f +- %.3f", r2_m, r2_s)
    log.info("R-Precision@3  %.3f +- %.3f", r3_m, r3_s)
    log.info("Diversity      %.3f +- %.3f (real %.3f +- %.3f)",
             div_m, div_s, div_real_m, div_real_s)
    log.info("MultiModality  %.3f +- %.3f", mm_m, mm_s)
    fsh_m_g, fsh_s_g = _mean_std([r["foot_skate_from_height_gen"] for r in runs])
    fsh_m_t, fsh_s_t = _mean_std([r["foot_skate_from_height_gt"] for r in runs])
    fsc_m_g, fsc_s_g = _mean_std([r["foot_skate_from_pred_contacts_gen"] for r in runs])
    fsc_m_t, fsc_s_t = _mean_std([r["foot_skate_from_pred_contacts_gt"] for r in runs])
    fsr_m_g, fsr_s_g = _mean_std([r["foot_skate_ratio_gen"] for r in runs])
    fsr_m_t, fsr_s_t = _mean_std([r["foot_skate_ratio_gt"] for r in runs])
    fcc_m_g, fcc_s_g = _mean_std([r["foot_contact_consistency_gen"] for r in runs])
    fcc_m_t, fcc_s_t = _mean_std([r["foot_contact_consistency_gt"] for r in runs])
    log.info("FootSkate[height]      gen %.4f +- %.4f m/s (gt %.4f +- %.4f)", fsh_m_g, fsh_s_g, fsh_m_t, fsh_s_t)
    log.info("FootSkate[contacts]    gen %.4f +- %.4f m/s (gt %.4f +- %.4f)", fsc_m_g, fsc_s_g, fsc_m_t, fsc_s_t)
    log.info("FootSkateRatio         gen %.4f +- %.4f      (gt %.4f +- %.4f)", fsr_m_g, fsr_s_g, fsr_m_t, fsr_s_t)
    log.info("FootContactConsistency gen %.4f +- %.4f      (gt %.4f +- %.4f)", fcc_m_g, fcc_s_g, fcc_m_t, fcc_s_t)
    log.info("Wrote: %s", out_path)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
