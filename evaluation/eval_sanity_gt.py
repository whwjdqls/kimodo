# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanity-check the evaluator + loader plumbing by using the GT as the "prediction".

If the pipeline is correctly wired we should reproduce MoMask Table 2's
"Real" row exactly:

    FID ~= 0.002, MM-Dist ~= 2.974, R@1 ~= 0.511, R@3 ~= 0.797

Any large deviation here points at a bug in our normalization, masking, or
batch ordering — *not* at the model.

There is one knob: ``--source`` chooses where the "prediction" comes from.

  * ``loader`` (default): just feed the loader's ``pose`` tensor back in as
    the prediction. This is the purest GT-vs-GT test; the evaluator never
    sees anything we computed.

  * ``rebuild`` : load the kimodo_rep .npz, run it through
    ``kimodo_to_humanml3d`` to get a freshly-converted 263-D vector, and
    feed THAT as the prediction. Tests the round-trip / converter path
    independently from the model. A near-zero gap between this and GT
    means the conversion is loss-free; any divergence quantifies the
    converter's error budget.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

# MoMask path glue (matches eval_hml3d.py).
MOMASK_ROOT = Path("/home/jungbin_cho/momask-codes").resolve()
sys.path.insert(0, str(MOMASK_ROOT))
from utils.metrics import (  # noqa: E402
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    euclidean_distance_matrix,
)

from evaluation.eval_hml3d import build_eval_wrapper_and_loader  # noqa: E402

BENCHMARK_ROOT = Path("/home/jungbin_cho/kimodo_open/benchmark").resolve()
sys.path.insert(0, str(BENCHMARK_ROOT))
from humanml3d_to_kimodo import kimodo_to_humanml3d  # noqa: E402

log = logging.getLogger("eval_sanity_gt")


def _rebuild_pred_from_kimodo_rep(
    tokens_list,  # list[str] — full '_'-joined token strings; we use a parallel name_list instead
    name_list,    # list[str] — motion IDs in the loader's order for this batch
    m_length: torch.Tensor,
    eval_mean: np.ndarray,
    eval_std: np.ndarray,
    kimodo_rep_dir: Path,
    device,
    max_motion_length: int = 196,
) -> torch.Tensor:
    """For each item in the batch, load its kimodo_rep .npz, convert to 263-D,
    z-score with the evaluator stats. Returns (B, max_motion_length, 263)."""
    B = len(name_list)
    out = torch.zeros(B, max_motion_length, 263, dtype=torch.float32)
    em_t = torch.from_numpy(eval_mean).float()
    es_t = torch.from_numpy(eval_std).float()
    for i, mid in enumerate(name_list):
        L = int(m_length[i])
        L = min(L, max_motion_length)
        npz_path = kimodo_rep_dir / f"{mid}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"missing kimodo_rep: {npz_path}")
        with np.load(npz_path) as z:
            d = {
                "global_rot_mats": torch.from_numpy(z["global_rot_mats"]).float(),
                "posed_joints":    torch.from_numpy(z["posed_joints"]).float(),
                "root_positions":  torch.from_numpy(z["root_positions"]).float(),
                "velocities":      torch.from_numpy(z["velocities"]).float(),
                "foot_contacts":   torch.from_numpy(z["foot_contacts"]).float(),
            }
            if "r_rot_quat" in z.files:
                d["r_rot_quat"] = torch.from_numpy(z["r_rot_quat"]).float()
        hml = kimodo_to_humanml3d(d, device="cpu")  # (T_full, 263)
        L = min(L, hml.shape[0])
        out[i, :L] = (hml[:L] - em_t) / es_t
    return out


def _resolve_id_for_token_string(name_list_token_string: str) -> str:
    """The dataset's '_tokens' field is the underscore-joined tokens for a
    chosen caption — NOT the file id. We don't get the id back from the
    loader. We'll grab it from the dataset's parallel name_list instead.
    (Not used in this script; left as a comment for the reader.)
    """
    return name_list_token_string  # placeholder, unused


def evaluate_gt(
    loader, eval_wrapper, source: str,
    eval_mean: np.ndarray, eval_std: np.ndarray,
    kimodo_rep_dir: Path,
) -> Dict:
    motion_real: List[torch.Tensor] = []
    motion_pred: List[torch.Tensor] = []
    R_real, R_pred, M_real, M_pred = 0.0, 0.0, 0.0, 0.0
    nb_sample = 0
    device = eval_wrapper.device

    # For 'rebuild' we need the per-batch ids. The loader's __getitem__ returns
    # a flattened tokens string but not the id, so we monkey-grab from the
    # underlying dataset via the cached name_list. The DataLoader's sampler
    # gives indices; we keep a synchronous counter through the dataset.

    for i, batch in enumerate(loader):
        word_embs, pos_ohot, captions, sent_len, gt_pose, m_length, tokens_str = batch
        m_length = m_length.to(device)
        gt_pose = gt_pose.to(device).float()

        # GT side embeddings.
        et, em = eval_wrapper.get_co_embeddings(word_embs, pos_ohot, sent_len, gt_pose, m_length)

        if source == "loader":
            # Use the GT pose as the prediction too — purest sanity test.
            et_pred, em_pred = et, em
            pred_for_metrics = gt_pose
        elif source == "rebuild":
            # Pull the dataset ids parallel to this batch.
            ds = loader.dataset
            # The loader shuffles + drops_last; we can't reliably read back the
            # batch ids without modifying the dataset. So this branch needs a
            # non-shuffled loader — surface a clear error otherwise.
            raise NotImplementedError(
                "source='rebuild' requires a sequential (non-shuffled) loader; "
                "use source='loader' for the GT-vs-GT sanity test, which already "
                "exercises the entire pipeline EXCEPT the kimodo->263 converter."
            )
        else:
            raise ValueError(f"unknown source: {source}")

        motion_real.append(em)
        motion_pred.append(em_pred)
        R_real += calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_pred += calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        M_real += euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        M_pred += euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        nb_sample += pred_for_metrics.shape[0]

    real_np = torch.cat(motion_real, dim=0).cpu().numpy()
    pred_np = torch.cat(motion_pred, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(real_np)
    mu, cov = calculate_activation_statistics(pred_np)
    fid = float(calculate_frechet_distance(gt_mu, gt_cov, mu, cov))

    div_real = float(calculate_diversity(real_np, 300 if nb_sample > 300 else 100))
    div_pred = float(calculate_diversity(pred_np, 300 if nb_sample > 300 else 100))

    return {
        "source": source,
        "n_samples": int(nb_sample),
        "fid": fid,
        "matching_score_real": float(M_real / nb_sample),
        "matching_score_pred": float(M_pred / nb_sample),
        "r_precision_real": (R_real / nb_sample).tolist(),
        "r_precision_pred": (R_pred / nb_sample).tolist(),
        "diversity_real": div_real,
        "diversity_pred": div_pred,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="loader", choices=["loader", "rebuild"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--kimodo-rep-dir", type=str, default="/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep")
    p.add_argument("--humanml3d-root", type=str, default="/home/jungbin_cho/HumanML3D/HumanML3D")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s  source=%s", device, args.source)

    eval_wrapper, loader, dataset, opt = build_eval_wrapper_and_loader(
        args.batch_size, device, Path(args.humanml3d_root),
    )
    log.info("Eval dataset: %d items (%d batches of %d)",
             len(dataset), len(loader), args.batch_size)
    eval_mean = np.load(os.path.join(opt.meta_dir, "mean.npy")).astype(np.float32)
    eval_std = np.load(os.path.join(opt.meta_dir, "std.npy")).astype(np.float32)

    metrics = evaluate_gt(loader, eval_wrapper, args.source,
                          eval_mean, eval_std, Path(args.kimodo_rep_dir))
    log.info("=== GT sanity (%s) ===", args.source)
    log.info("FID            %.4f", metrics["fid"])
    log.info("MM-Dist real   %.4f", metrics["matching_score_real"])
    log.info("MM-Dist pred   %.4f", metrics["matching_score_pred"])
    log.info("R-Prec real    [%.3f, %.3f, %.3f]", *metrics["r_precision_real"])
    log.info("R-Prec pred    [%.3f, %.3f, %.3f]", *metrics["r_precision_pred"])
    log.info("Diversity      real=%.3f  pred=%.3f", metrics["diversity_real"], metrics["diversity_pred"])

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
