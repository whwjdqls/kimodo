# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rotation-side ceiling baseline for the kimodo HumanML3D eval.

What this measures
==================
What is the FID / R-precision a hypothetical model would achieve if it
**perfectly predicted HumanML3D's stored rotation chain** and we put those
rotations through the same Option-B pipeline our model goes through?

This is the *intrinsic noise floor* of our eval pipeline. Any actual model
evaluated via ``eval_hml3d.py`` cannot do better than this — it tells us
how much the ``rot_data`` vs ``ric_data`` self-inconsistency in HumanML3D's
test split (the ~9% of motions where IK doesn't fully recover the joints)
costs us in the metric.

Procedure
=========
For each test motion:
  1. Load GT 263-D from ``new_joint_vecs/``.
  2. Compute FK joints from its rotation block using HML3D's OWN
     ``forward_kinematics_cont6d`` — for ~91% of motions this matches the
     stored ``ric_data`` joints to float noise; for ~9% it differs by
     cm-scale.
  3. Re-encode 263-D from those FK joints, keeping HML3D's stored rotation
     channels (rot_velocity, rot_data) and foot_contacts unchanged — only
     ric_data / lin_velocity / root_height / local_velocity are rebuilt
     from FK joints. This mirrors what ``eval_hml3d.sample_and_convert_batch``
     does to model predictions (Option B), just with GT rotations as the
     source instead of model output.
  4. Z-score with the evaluator's mean/std and feed BOTH the original GT 263
     and the FK-reconstructed 263 to MoMask's T2M evaluator.
  5. Compute FID / R-precision / MM-Dist between the two embedding sets.

If the resulting FID is near 0, the FK-side reconstruction is essentially
in-distribution for the evaluator — our Option-B pipeline isn't adding
material bias. If it's large, the ~9% inconsistent motions are inflating
all our model FIDs by this amount and we should add it as a "baseline floor"
to any numbers we report.

Usage::

    python -m evaluation.eval_hml3d_fk_baseline \\
        --batch-size 32 --repeat-times 5
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
from typing import Dict, List

import numpy as np
import torch

# Shim for HumanML3D's older numpy API.
if not hasattr(np, 'float'): np.float = float  # type: ignore[attr-defined]
if not hasattr(np, 'int'): np.int = int        # type: ignore[attr-defined]

# Add HML3D + momask paths.
MOMASK_ROOT = Path("/home/jungbin_cho/momask-codes").resolve()
sys.path.insert(0, str(MOMASK_ROOT))
HML3D_ROOT = Path("/home/jungbin_cho/HumanML3D").resolve()
sys.path.insert(0, str(HML3D_ROOT))

# MoMask metric primitives.
from utils.metrics import (  # noqa: E402
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    euclidean_distance_matrix,
)

# HML3D building blocks.
from common.skeleton import Skeleton  # noqa: E402
from common.quaternion import qrot, qinv, quaternion_to_cont6d  # noqa: E402
from paramUtil import t2m_raw_offsets, t2m_kinematic_chain  # noqa: E402

# Reuse eval_hml3d.py's loader builder so we use the exact same dataset
# wiring as the real eval.
from evaluation.eval_hml3d import build_eval_wrapper_and_loader  # noqa: E402

log = logging.getLogger("eval_hml3d_fk_baseline")

JOINTS_NUM = 22
FACE_JOINT_IDX = [2, 1, 17, 16]


def _build_skel():
    """HumanML3D's Skeleton initialized with the canonical 012314 offsets."""
    canon = np.load('/home/jungbin_cho/HumanML3D/HumanML3D/new_joints/012314.npy')
    skel = Skeleton(torch.from_numpy(t2m_raw_offsets).float(), t2m_kinematic_chain, 'cpu')
    skel.set_offset(skel.get_offsets_joints(torch.from_numpy(canon[0]).float()))
    return skel


def _recover_root_rot_pos(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """HML3D's recover_root_rot_pos, inlined to avoid an extra import dance."""
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


@torch.no_grad()
def rebuild_gt_from_rotations(gt_263: torch.Tensor, skel: Skeleton) -> torch.Tensor:
    """Recompute the position-side channels of GT 263 from FK on GT rotations.

    Keeps: rot_velocity (slot 0), rot_data (67:193), foot_contacts (259:263).
    Recomputes from FK joints: lin_velocity, root_height, ric_data, local_velocity.

    Args:
        gt_263: ``(T, 263)`` unnormalized GT motion (float).
        skel: HML3D Skeleton with canonical offsets set.

    Returns: ``(T, 263)`` float tensor with same shape, position channels
        replaced by FK-side reconstruction.
    """
    T = gt_263.shape[0]
    device = gt_263.device

    # 1) FK joints from GT rot_data via HML3D's own forward_kinematics_cont6d.
    # Skeleton is on CPU; HML3D's FK expects same-device offsets, so we run
    # FK on CPU and move the joints back. The rest of the rebuild stays on
    # the input's device.
    r_rot_quat, r_pos = _recover_root_rot_pos(gt_263)
    r_rot_6d = quaternion_to_cont6d(r_rot_quat)
    rot_data = gt_263[..., 1 + 2 + 1 + 21 * 3 : 1 + 2 + 1 + 21 * 3 + 21 * 6]
    cont6d_full = torch.cat([r_rot_6d, rot_data], dim=-1).view(T, JOINTS_NUM, 6)
    fk_joints = skel.forward_kinematics_cont6d(
        cont6d_full.detach().cpu(), r_pos.detach().cpu()
    ).to(device)  # (T, J, 3)

    # 2) Rebuild ric_data from FK joints[:, 1:] minus FK_root.xz, rotated to ego.
    ric_pos = fk_joints[:, 1:].clone()
    ric_pos[..., 0] -= fk_joints[:, 0:1, 0]
    ric_pos[..., 2] -= fk_joints[:, 0:1, 2]
    # Rotate to ego: qrot(r_rot_quat, .)  (matches HML3D's get_rifke)
    quat_J1 = r_rot_quat[:, None, :].expand(T, JOINTS_NUM - 1, 4)
    ric_data_new = qrot(quat_J1, ric_pos).reshape(T, -1)  # (T, 63)

    # 3) Rebuild root_height from FK root y.
    root_y_new = fk_joints[:, 0, 1:2]  # (T, 1)

    # 4) Rebuild lin_velocity from FK root xz delta, rotated to ego at frame t+1.
    lin_vel_new = torch.zeros(T, 2, device=device, dtype=gt_263.dtype)
    if T > 1:
        world_delta = torch.zeros(T - 1, 3, device=device, dtype=gt_263.dtype)
        world_delta[:, 0] = fk_joints[1:, 0, 0] - fk_joints[:-1, 0, 0]
        world_delta[:, 2] = fk_joints[1:, 0, 2] - fk_joints[:-1, 0, 2]
        ego_xz = qrot(r_rot_quat[1:], world_delta)
        lin_vel_new[:-1, 0] = ego_xz[:, 0]
        lin_vel_new[:-1, 1] = ego_xz[:, 2]

    # 5) Rebuild local_velocity from FK joint deltas (per-joint world delta
    #    rotated to ego of frame t). Matches HML3D's local_vel computation.
    local_vel_new = torch.zeros(T, JOINTS_NUM * 3, device=device, dtype=gt_263.dtype)
    if T > 1:
        world_vel = fk_joints[1:] - fk_joints[:-1]  # (T-1, J, 3)
        ego_vel = qrot(r_rot_quat[:-1, None, :].expand(T - 1, JOINTS_NUM, 4), world_vel)
        local_vel_new[:-1] = ego_vel.reshape(T - 1, JOINTS_NUM * 3)

    # 6) Assemble. Keep rot_velocity / rot_data / foot_contacts from original GT.
    out = gt_263.clone()
    out[:, 1:3] = lin_vel_new
    out[:, 3:4] = root_y_new
    out[:, 4:67] = ric_data_new
    out[:, 193:259] = local_vel_new
    return out


def run_baseline(
    loader, eval_wrapper, eval_mean, eval_std, device, repeat_times: int, seed: int,
) -> Dict[str, float]:
    skel = _build_skel()
    eval_mean_t = torch.from_numpy(eval_mean).to(device).float()
    eval_std_t = torch.from_numpy(eval_std).to(device).float()

    runs: List[Dict[str, float]] = []
    for run_id in range(repeat_times):
        s = seed + run_id * 1000
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

        em_real, em_fake = [], []
        et_real, et_fake = [], []
        R_real_acc, R_fake_acc = 0.0, 0.0
        match_real, match_fake = 0.0, 0.0
        nb = 0

        t0 = time.time()
        for batch in loader:
            word_embs, pos_ohot, captions, sent_len, gt_pose, m_length, _tokens = batch
            m_length = m_length.to(device)
            gt_pose = gt_pose.to(device).float()              # (B, T_max, 263) z-scored
            gt_un = gt_pose * eval_std_t + eval_mean_t        # unnormalize

            # Per-sample rebuild of position channels from FK on GT rotations.
            fk_un = torch.empty_like(gt_un)
            for i in range(gt_un.shape[0]):
                L = int(m_length[i])
                rebuilt = rebuild_gt_from_rotations(gt_un[i, :L], skel)
                fk_un[i, :L] = rebuilt
                # Past length: leave zero (matches loader padding behavior).
                if L < gt_un.shape[1]:
                    fk_un[i, L:] = 0.0
            fk_pose = (fk_un - eval_mean_t) / eval_std_t      # re-z-score

            et_r, em_r = eval_wrapper.get_co_embeddings(word_embs, pos_ohot, sent_len, gt_pose, m_length)
            et_f, em_f = eval_wrapper.get_co_embeddings(word_embs, pos_ohot, sent_len, fk_pose, m_length)

            em_real.append(em_r); em_fake.append(em_f)
            et_real.append(et_r); et_fake.append(et_f)
            R_real_acc += calculate_R_precision(et_r.cpu().numpy(), em_r.cpu().numpy(), top_k=3, sum_all=True)
            R_fake_acc += calculate_R_precision(et_f.cpu().numpy(), em_f.cpu().numpy(), top_k=3, sum_all=True)
            match_real += euclidean_distance_matrix(et_r.cpu().numpy(), em_r.cpu().numpy()).trace()
            match_fake += euclidean_distance_matrix(et_f.cpu().numpy(), em_f.cpu().numpy()).trace()
            nb += gt_pose.shape[0]

        em_real_np = torch.cat(em_real, dim=0).cpu().numpy()
        em_fake_np = torch.cat(em_fake, dim=0).cpu().numpy()
        gt_mu, gt_cov = calculate_activation_statistics(em_real_np)
        fk_mu, fk_cov = calculate_activation_statistics(em_fake_np)
        fid = float(calculate_frechet_distance(gt_mu, gt_cov, fk_mu, fk_cov))
        div_real = float(calculate_diversity(em_real_np, 300 if nb > 300 else 100))
        div_fake = float(calculate_diversity(em_fake_np, 300 if nb > 300 else 100))

        m = {
            "fid": fid,
            "r_precision_real_top1": float(R_real_acc[0] / nb),
            "r_precision_real_top2": float(R_real_acc[1] / nb),
            "r_precision_real_top3": float(R_real_acc[2] / nb),
            "r_precision_fk_top1":   float(R_fake_acc[0] / nb),
            "r_precision_fk_top2":   float(R_fake_acc[1] / nb),
            "r_precision_fk_top3":   float(R_fake_acc[2] / nb),
            "matching_score_real":   float(match_real / nb),
            "matching_score_fk":     float(match_fake / nb),
            "diversity_real":        div_real,
            "diversity_fk":          div_fake,
        }
        log.info(
            "  run %d/%d (seed=%d, %.1fs): FID=%.4f  R@1 real=%.4f fk=%.4f  R@3 real=%.4f fk=%.4f",
            run_id + 1, repeat_times, s, time.time() - t0,
            m["fid"], m["r_precision_real_top1"], m["r_precision_fk_top1"],
            m["r_precision_real_top3"], m["r_precision_fk_top3"],
        )
        runs.append(m)

    out = {"per_run": runs}
    keys = runs[0].keys()
    out["metrics_mean_std"] = {
        k: {"mean": float(np.mean([r[k] for r in runs])),
            "std":  float(np.std([r[k] for r in runs]))}
        for k in keys
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--repeat-times", type=int, default=5)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--humanml3d-root", type=str, default="/home/jungbin_cho/HumanML3D/HumanML3D")
    p.add_argument("--out", type=str, default="/home/jungbin_cho/kimodo_open/runs/_fk_baseline.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("device=%s", device)

    eval_wrapper, loader, dataset, opt = build_eval_wrapper_and_loader(
        args.batch_size, device, Path(args.humanml3d_root),
    )
    eval_mean = np.load(os.path.join(opt.meta_dir, "mean.npy")).astype(np.float32)
    eval_std = np.load(os.path.join(opt.meta_dir, "std.npy")).astype(np.float32)
    log.info("Eval dataset: %d items (%d batches of %d)",
             len(dataset), len(loader), args.batch_size)

    result = run_baseline(
        loader, eval_wrapper, eval_mean, eval_std, device,
        repeat_times=args.repeat_times, seed=args.seed,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    log.info("==================== SUMMARY ====================")
    ms = result["metrics_mean_std"]
    log.info("FID (FK-GT vs real-GT)    %.4f +- %.4f", ms["fid"]["mean"], ms["fid"]["std"])
    log.info("R@1 real                  %.4f +- %.4f",
             ms["r_precision_real_top1"]["mean"], ms["r_precision_real_top1"]["std"])
    log.info("R@1 FK-GT                 %.4f +- %.4f",
             ms["r_precision_fk_top1"]["mean"], ms["r_precision_fk_top1"]["std"])
    log.info("R@3 real                  %.4f +- %.4f",
             ms["r_precision_real_top3"]["mean"], ms["r_precision_real_top3"]["std"])
    log.info("R@3 FK-GT                 %.4f +- %.4f",
             ms["r_precision_fk_top3"]["mean"], ms["r_precision_fk_top3"]["std"])
    log.info("Diversity real / FK-GT    %.4f / %.4f",
             ms["diversity_real"]["mean"], ms["diversity_fk"]["mean"])
    log.info("Wrote: %s", args.out)


if __name__ == "__main__":
    main()
