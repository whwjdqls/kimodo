"""Train a KIMODO-style text-to-motion model on HumanML3D (in Kimodo features).

This is the HumanML3D counterpart to ``kimodo.scripts.train``. Everything
heavy (denoiser build, diffusion, optimizer, EMA, AMP, DDP setup, text
encoder loading, train step, FK-free smooth-L1 loss, checkpointing) is
imported from ``kimodo.scripts.train`` — we only override the dataset
construction and visualisation.

Key differences from the SOMA pipeline:

* Data: ``HumanML3DTextMotionDataset`` reads pre-converted ``(T, 273)``
  features from per-id NPZs + multi-caption HumanML3D text files.
* Skeleton: ``SMPLXSkeleton22`` is used as a CONTAINER ONLY (gives
  ``nbjoints=22`` so the motion_rep produces 273-dim features). We never
  call ``skeleton.fk()`` because the rotations follow HumanML3D's
  chain-reset semantics, not standard parent-relative FK.
* γ₇ FK consistency loss is OFF (config default ``loss_weights.fk: 0.0``).
* Visualisation: predicted features are decoded via the HumanML3D
  conversion pipeline (``benchmark/humanml3d_to_kimodo.kimodo_to_humanml3d``
  + HumanML3D ``recover_from_ric``) instead of ``motion_rep.inverse``.

Usage (single GPU):
    python -m kimodo.scripts.train_w_hml3d --config configs/training/hml3d.yaml

Multi-GPU (single node):
    torchrun --standalone --nproc_per_node=8 -m kimodo.scripts.train_w_hml3d \\
        --config configs/training/hml3d.yaml
"""

from __future__ import annotations

# Force HF offline mode when DDP, BEFORE any kimodo imports. (Same reasoning
# as train.py — see that file's header comment.)
import os as _os  # noqa: E402
if int(_os.environ.get("WORLD_SIZE", "1")) > 1:
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.cuda.amp import GradScaler  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from kimodo.data import (  # noqa: E402
    HumanML3DTextMotionDataset,
    build_humanml3d_collate_fn,
)
from kimodo.model.diffusion import Diffusion  # noqa: E402
from kimodo.scripts.train import (  # noqa: E402
    KimodoLoss,
    ModelEMA,
    build_denoiser_from_model_config,
    build_text_encoder,
    encode_texts,
    load_checkpoint,
    load_config,
    maybe_barrier,
    save_checkpoint,
    setup_distributed,
    train_one_step,
    _build_cpu_motion_rep,
    _resolve_num_steps,
)

log = logging.getLogger("kimodo.train_hml3d")


# -----------------------------------------------------------------------------
# Viz — HumanML3D-aware (chain-reset FK from rotations, shared with the loss)
# -----------------------------------------------------------------------------
@torch.no_grad()
def _cfg_sample(
    raw: nn.Module,
    diffusion: Diffusion,
    *,
    text_feat: torch.Tensor,
    text_pad_mask: torch.Tensor,
    pad_mask: torch.Tensor,
    first_heading: torch.Tensor,
    motion_mask: torch.Tensor,
    observed: torch.Tensor,
    n_steps: int,
    cfg_scale: float,
    device: torch.device,
    sampler: str = "ddim",
) -> torch.Tensor:
    """Classifier-free-guided x0 sampler with selectable reverse-step rule.

    The model is trained with ``text_drop_prob`` (zeroed text features = the
    unconditional case); we recover the conditional and unconditional x0
    predictions in a single doubled-batch forward pass and combine them as
    ``x0_cfg = x0_uncond + scale * (x0_cond - x0_uncond)``.

    ``cfg_scale <= 1.0`` short-circuits to plain (conditional-only) sampling.

    ``sampler``:
      - ``"ddim"`` (default): deterministic DDIM (η=0). Matches the previous
        behavior; safe to use with subsampled ``n_steps`` (e.g. 50).
      - ``"ddpm"``: stochastic ancestral sampling. Computes the DDPM posterior
        mean from the predicted x̂₀ and adds ``sqrt(posterior_variance) * z``
        at every step except the last (t=0). Use with ``n_steps`` close to the
        training schedule (e.g. 1000) for paper-faithful MDM-style sampling.
    """
    if sampler not in ("ddim", "ddpm"):
        raise ValueError(f"sampler must be 'ddim' or 'ddpm'; got {sampler!r}")

    B, T = pad_mask.shape
    D = raw.motion_rep.motion_rep_dim

    use_timesteps, map_tensor = diffusion.space_timesteps(n_steps)
    diffusion.calc_diffusion_vars(use_timesteps)
    cur = torch.randn(B, T, D, device=device)

    do_cfg = cfg_scale is not None and float(cfg_scale) > 1.0
    if do_cfg:
        text_feat_2x = torch.cat([text_feat, torch.zeros_like(text_feat)], dim=0)
        text_pad_2x = torch.cat([text_pad_mask, text_pad_mask], dim=0)
        pad_2x = torch.cat([pad_mask, pad_mask], dim=0)
        fh_2x = torch.cat([first_heading, first_heading], dim=0)
        mm_2x = torch.cat([motion_mask, motion_mask], dim=0)
        obs_2x = torch.cat([observed, observed], dim=0)

    for i in list(range(n_steps))[::-1]:
        t = torch.full((B,), i, device=device, dtype=torch.long)
        t_map = map_tensor[t]
        if do_cfg:
            cur_2x = torch.cat([cur, cur], dim=0)
            t_2x = torch.cat([t_map, t_map], dim=0)
            pred_2x = raw(
                cur_2x, pad_2x, text_feat_2x, text_pad_2x, t_2x,
                first_heading_angle=fh_2x,
                motion_mask=mm_2x, observed_motion=obs_2x,
            )
            pred_cond, pred_uncond = pred_2x[:B], pred_2x[B:]
            pred_clean = pred_uncond + float(cfg_scale) * (pred_cond - pred_uncond)
        else:
            pred_clean = raw(
                cur, pad_mask, text_feat, text_pad_mask, t_map,
                first_heading_angle=first_heading,
                motion_mask=motion_mask, observed_motion=observed,
            )

        if sampler == "ddim":
            eps = (
                diffusion.sqrt_recip_alphas_cumprod[t, None, None] * cur - pred_clean
            ) / diffusion.sqrt_recipm1_alphas_cumprod[t, None, None]
            alpha_bar_prev = diffusion.alphas_cumprod_prev[t, None, None]
            cur = pred_clean * alpha_bar_prev.sqrt() + (1 - alpha_bar_prev).sqrt() * eps
        else:  # "ddpm"
            mean = (
                diffusion.posterior_mean_coef1[t, None, None] * pred_clean
                + diffusion.posterior_mean_coef2[t, None, None] * cur
            )
            if i > 0:
                noise = torch.randn_like(cur)
                log_var = diffusion.posterior_log_variance_clipped[t, None, None]
                cur = mean + torch.exp(0.5 * log_var) * noise
            else:
                cur = mean
    return cur


@torch.no_grad()
def viz_step(
    denoiser: nn.Module,
    diffusion: Diffusion,
    text_encoder,
    device: torch.device,
    cfg: DictConfig,
    step: int,
    out_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    loss_fn,  # KimodoLoss — reused for FK so viz matches training-time supervision
    prompts: Optional[list] = None,
    test_examples: Optional[List[dict]] = None,
    tb_writer=None,
) -> None:
    """Unified per-step viz mirroring ``train.py``'s ``viz_step``.

    Layout: ``<out_dir>/<step:07d>/<i>_<id>.{npz,mp4}``.

    Items come from two sources, batched into a SINGLE diffusion forward:
      1. ``extra_prompts`` (passed via the ``prompts`` arg; defaults to
         ``cfg.viz.prompts``) — no GT. Length = ``cfg.viz.num_frames``.
      2. ``test_examples`` (when supplied) — held-out ``{id, caption,
         gt_features, length}`` tuples from ``_load_test_examples``.

    NPZ contains the generated unnormalized features ``(T, 273)`` + prompt
    (and, for test items, the GT features for inspection). MP4 is the
    side-by-side stick figure (GT left, gen right; left panel is blank when
    no GT exists).

    Joints are recomputed by FK on the predicted (or GT) global rotations,
    matching the inference convention (the model's deployment-time output is
    rotations + root; positions are reconstructed downstream). HML3D needs
    the chain-reset variant — standard parent-relative FK as in
    ``motion_rep.inverse`` would interpret the stored chain-reset rotations
    incorrectly. The actual world root is reconstructed from features the
    same way ``KimodoMotionRep.inverse`` does
    (``local_jp[pelvis] + (smooth_root_x, 0, smooth_root_z)``); per-sample
    bone lengths come from each item's own positions block (HML3D uniform-
    skeletons every sample, so these are constant across the dataset and
    deriving per-item is purely a cosmetic choice over caching).
    """
    import imageio.v3 as iio

    from kimodo.scripts.render_hml3d import render_sidebyside

    step_dir = out_dir / f"{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    raw = (denoiser.module if hasattr(denoiser, "module") else denoiser).eval()
    motion_rep = raw.motion_rep
    n_steps = int(cfg.viz.num_denoising_steps)
    cfg_scale = float(cfg.viz.get("cfg_scale", 2.5))
    fps = int(cfg.data.get("fps", 20))
    num_frames_default = int(cfg.viz.num_frames)
    save_mp4 = bool(cfg.viz.get("save_videos", True))

    if prompts is None:
        prompts = [
            p for p in (cfg.viz.get("prompts", []) or [])
            if isinstance(p, str) and p
        ]
    else:
        prompts = list(prompts)
    test_examples = test_examples or []

    items: List[dict] = []
    for i, p in enumerate(prompts):
        safe = p.lower().replace(" ", "_").replace(".", "")[:40] or f"prompt_{i}"
        items.append({
            "id": f"prompt_{i:02d}_{safe}",
            "caption": p,
            "gt_features": None,
            "length": num_frames_default,
        })
    for ex in test_examples:
        items.append({
            "id": ex["id"],
            "caption": ex["caption"],
            "gt_features": ex["gt_features"],  # already-unnormalized (T, 273)
            "length": int(ex["length"]),
        })
    if not items:
        return

    captions = [it["caption"] for it in items]
    lengths = [it["length"] for it in items]
    max_T = int(max(lengths))
    B = len(items)

    text_feat, text_pad_mask = encode_texts(text_encoder, captions, device)
    pad_mask = torch.zeros(B, max_T, dtype=torch.bool, device=device)
    for i, L in enumerate(lengths):
        pad_mask[i, :L] = True
    first_heading = torch.zeros(B, device=device)
    motion_mask = torch.zeros(B, max_T, motion_rep.motion_rep_dim, device=device)
    observed = torch.zeros(B, max_T, motion_rep.motion_rep_dim, device=device)

    cur = _cfg_sample(
        raw, diffusion,
        text_feat=text_feat, text_pad_mask=text_pad_mask,
        pad_mask=pad_mask, first_heading=first_heading,
        motion_mask=motion_mask, observed=observed,
        n_steps=n_steps, cfg_scale=cfg_scale, device=device,
    )  # (B, max_T, D) normalized
    raw.train()

    mean_t = torch.from_numpy(mean).to(cur.device)
    std_t = torch.from_numpy(std).to(cur.device)
    gen_unnorm = cur.float() * std_t + mean_t  # (B, max_T, 273)

    # TensorBoard add_video is OFF by default: it requires a (N, Tmax, 3, H, W)
    # *float32* stack which at the default render res (1200x600) for B=6 items
    # peaks at ~10 GB of host RAM and has caused OOM-kills on slurm nodes. The
    # per-item mp4 files already land on disk, so TB videos are redundant.
    # Set ``cfg.viz.tb_videos: true`` to opt in (and ideally lower the render
    # resolution or num_test_samples first).
    tb_videos = bool(cfg.viz.get("tb_videos", False)) and (tb_writer is not None)
    video_stack: List[np.ndarray] = []
    for i, it in enumerate(items):
        L = it["length"]
        gen_feats = gen_unnorm[i, :L]  # (L, 273)
        gen_joints = loss_fn._fk_world_from_pred(
            gen_feats.unsqueeze(0),
        )[0].cpu().numpy()
        gen_feats_np = gen_feats.cpu().numpy()

        gt_joints: Optional[np.ndarray] = None
        npz_kwargs = {"features": gen_feats_np, "prompt": it["caption"]}
        if it["gt_features"] is not None:
            gt_feats = torch.from_numpy(it["gt_features"]).to(device).unsqueeze(0)
            gt_joints = loss_fn._fk_world_from_pred(gt_feats)[0].cpu().numpy()
            npz_kwargs["gt_features"] = it["gt_features"]

        base = step_dir / f"{i:02d}_{it['id']}"
        np.savez(base.with_suffix(".npz"), **npz_kwargs)
        if not save_mp4:
            continue
        try:
            frames = render_sidebyside(gt_joints, gen_joints, caption=it["caption"])
        except Exception as e:
            log.warning("Render failed for %s: %s", it["id"], e)
            continue
        try:
            iio.imwrite(
                str(base.with_suffix(".mp4")), frames,
                fps=float(fps), codec="h264", plugin="pyav",
            )
        except Exception as e:
            log.warning("MP4 write failed for %s: %s", it["id"], e)
        if tb_videos:
            video_stack.append(frames)
        else:
            # Drop the (T,H,W,3) buffer immediately so peak RSS stays at ~1 item's
            # worth of frames (~430 MB at 1200x600x200) instead of B items'.
            del frames

    if tb_videos and video_stack:
        Tmax = max(v.shape[0] for v in video_stack)
        H, W = video_stack[0].shape[1:3]
        vids = np.zeros((len(video_stack), Tmax, 3, H, W), dtype=np.float32)
        for n, v in enumerate(video_stack):
            vt = np.transpose(v, (0, 3, 1, 2)).astype(np.float32) / 255.0
            vids[n, : vt.shape[0]] = vt
            if vt.shape[0] < Tmax:
                vids[n, vt.shape[0]:] = vt[-1]
        try:
            tb_writer.add_video("viz", torch.from_numpy(vids), global_step=step, fps=fps)
        except Exception as e:
            log.warning("TensorBoard add_video failed (%s); skipping.", e)

    log.info("Wrote %d viz samples to %s", len(items), step_dir)


# -----------------------------------------------------------------------------
# Test-set side-by-side visualization (generated vs GT)
# -----------------------------------------------------------------------------
def _load_test_examples(
    motion_dir: str,
    text_dir: str,
    split_file: str,
    n_samples: int,
    max_frames: int,
    min_frames: int = 40,
):
    """Load a fixed set of test examples for side-by-side viz.

    Returns a list of dicts: ``{id, caption, gt_features (T,273) unnormalized, length}``.
    Picks the first ``n_samples`` ids present on disk, takes the first
    (full-motion) caption, and uses the first ``max_frames`` frames (no random
    window / no heading aug — we want a deterministic, comparable clip).
    """
    import codecs as cs
    from pathlib import Path as _Path

    motion_dir = _Path(motion_dir)
    text_dir = _Path(text_dir)
    with open(split_file, "r") as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    out = []
    for mid in ids:
        if len(out) >= n_samples:
            break
        mp = motion_dir / f"{mid}.npz"
        tp = text_dir / f"{mid}.txt"
        if not mp.is_file() or not tp.is_file():
            continue
        try:
            with np.load(mp) as z:
                if "features" not in z.files:
                    continue
                feats = np.asarray(z["features"]).astype(np.float32)
        except Exception:
            continue
        if feats.shape[1] != 273 or feats.shape[0] < min_frames:
            continue
        # First full-motion caption (f_tag == to_tag == 0).
        caption = None
        with cs.open(tp, "r") as tf:
            for line in tf.readlines():
                parts = line.strip().split("#")
                if len(parts) < 4:
                    continue
                try:
                    f_tag, to_tag = float(parts[2]), float(parts[3])
                except Exception:
                    continue
                if f_tag == 0.0 and to_tag == 0.0:
                    caption = parts[0]
                    break
        if caption is None:
            continue
        T = min(feats.shape[0], max_frames)
        out.append({"id": mid, "caption": caption, "gt_features": feats[:T], "length": int(T)})
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    OmegaConf.resolve(cfg)

    env = setup_distributed()
    seed = int(cfg.trainer.seed) + env.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda", env.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    output_dir = Path(cfg.output_dir)
    if env.is_main:
        from kimodo.scripts._config_snapshot import snapshot_configs
        snapshot_configs(cfg, output_dir)
    maybe_barrier()

    log_path = output_dir / f"train_rank{env.rank}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )

    def _excepthook(exc_type, exc, tb):
        log.error("Uncaught exception on rank %d", env.rank, exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _excepthook

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("Distributed: world_size=%d rank=%d local_rank=%d", env.world_size, env.rank, env.local_rank)

    # ----- denoiser & diffusion -----
    denoiser = build_denoiser_from_model_config(
        cfg.model_config_path,
        cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    ).to(device)
    if cfg.get("init_from_safetensors"):
        from kimodo.model.loading import load_checkpoint_state_dict
        sd = load_checkpoint_state_dict(cfg.init_from_safetensors)
        sd = {k.replace("denoiser.backbone.", ""): v for k, v in sd.items()}
        denoiser.load_state_dict(sd, strict=False)
        log.info("Initialized denoiser weights from %s", cfg.init_from_safetensors)

    diffusion = Diffusion(num_base_steps=int(_resolve_num_steps(cfg))).to(device)
    motion_rep = denoiser.motion_rep
    assert motion_rep.motion_rep_dim == 273, (
        f"Expected HumanML3D-Kimodo motion_rep_dim=273, got {motion_rep.motion_rep_dim}. "
        f"Check that the model config uses SMPLXSkeleton22."
    )

    # ----- dataset / dataloader -----
    dataset_motion_rep = _build_cpu_motion_rep(
        cfg.model_config_path,
        cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    )

    mean = np.load(cfg.data.mean_path).astype(np.float32)
    std = np.load(cfg.data.std_path).astype(np.float32)

    def _build_dataset():
        return HumanML3DTextMotionDataset(
            motion_dir=cfg.data.motion_dir,
            text_dir=cfg.data.text_dir,
            split_file=cfg.data.split_file,
            mean=mean,
            std=std,
            fps=int(cfg.data.fps),
            window_size=int(cfg.data.window_size),
            max_motion_length=int(cfg.data.max_motion_length),
            min_motion_len=int(cfg.data.min_motion_len),
            unit_length=int(cfg.data.unit_length),
            random_heading_aug=bool(cfg.data.random_heading_aug),
            clip_normalized=cfg.data.get("clip_normalized"),
            skeleton=dataset_motion_rep.skeleton,
            motion_rep=dataset_motion_rep,
            seed=seed,
        )

    if env.is_main:
        dataset = _build_dataset()
    maybe_barrier()
    if not env.is_main:
        dataset = _build_dataset()
    maybe_barrier()

    sampler = None
    if env.is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=int(cfg.trainer.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.trainer.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=True,
        collate_fn=build_humanml3d_collate_fn(),
        persistent_workers=bool(cfg.data.num_workers) and True,
    )

    # ----- optimizer / scheduler -----
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=float(cfg.trainer.lr),
        betas=tuple(cfg.trainer.betas),
        weight_decay=float(cfg.trainer.weight_decay),
    )
    warmup_steps = int(cfg.trainer.warmup_steps)
    num_steps = int(cfg.trainer.num_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- AMP / scaler -----
    mp = str(cfg.trainer.mixed_precision).lower()
    autocast_dtype: Optional[torch.dtype]
    if mp == "fp16":
        autocast_dtype = torch.float16
        scaler = GradScaler()
    elif mp == "bf16":
        autocast_dtype = torch.bfloat16
        scaler = None
    else:
        autocast_dtype = None
        scaler = None

    ema: Optional[ModelEMA] = None
    if float(cfg.trainer.ema_decay) > 0:
        ema = ModelEMA(denoiser, decay=float(cfg.trainer.ema_decay))

    # ----- text encoder (frozen, serialized across ranks) -----
    _enc_type = str(cfg.text_encoder.get("type", "llm2vec")).lower()
    _enc_cache = cfg.text_encoder.get("cache_path", None)
    if _enc_cache:
        log.info("Building text encoder: cached (cache_path=%s) — live encoder NOT loaded.", _enc_cache)
    elif _enc_type == "llm2vec":
        log.info(
            "Building text encoder: llm2vec (HF_HUB_OFFLINE=%s, KIMODO_TEXT_ENCODER_LOCAL_DIR=%s) ...",
            os.environ.get("HF_HUB_OFFLINE", "0"),
            os.environ.get("KIMODO_TEXT_ENCODER_LOCAL_DIR", "(unset)"),
        )
    else:
        log.info("Building text encoder: %s (model=%s) ...",
                 _enc_type, cfg.text_encoder.get("model_name", "(default)"))

    if env.is_distributed:
        for r in range(env.world_size):
            if env.rank == r:
                log.info("Rank %d building text encoder ...", r)
                text_encoder = build_text_encoder(cfg.text_encoder, device=device)
                log.info("Rank %d text encoder built (class=%s).", r, type(text_encoder).__name__)
            maybe_barrier()
    else:
        text_encoder = build_text_encoder(cfg.text_encoder, device=device)
        log.info("Text encoder built (class=%s).", type(text_encoder).__name__)

    # ----- DDP wrap -----
    if env.is_distributed:
        denoiser = DDP(denoiser, device_ids=[env.local_rank], find_unused_parameters=False)

    # ----- loss -----
    # For HumanML3D, γ₇ is computed with the **chain-reset** FK to match the
    # data's rotation convention. Running the standard parent-relative FK on
    # these rotations would give wrong joint positions, so we dispatch to the
    # chain-reset variant here. The skeleton's neutral_joints (SMPLXSkeleton22
    # T-pose) and HumanML3D's t2m_kinematic_chain provide the bone offsets.
    weights = dict(cfg.trainer.loss_weights)
    loss_fn = KimodoLoss(
        motion_rep,
        weights,
        smooth_l1_beta=float(cfg.trainer.get("smooth_l1_beta", 1.0)),
        fk_kind="chainreset_hml3d",
        # Paper-faithful: compare FK(pred_rot, GT_bone_lengths) against the GT
        # positions block. With the corrected raw_offsets-based FK, this is a
        # physically meaningful per-meter error in the actor's actual skeleton.
        fk_target="gt",
    ).to(device)

    # Phase 2 (constraints) is supported by ConstraintSampler but disabled by
    # default here. Reuse the SOMA pipeline's sampler if needed.
    constraint_sampler = None
    phase = str(cfg.trainer.get("phase", "text"))
    if phase == "constraints":
        from kimodo.scripts.train import ConstraintSampler
        cw = cfg.trainer.get("constraint_weights", None)
        constraint_sampler = ConstraintSampler(
            motion_rep,
            weights=OmegaConf.to_container(cw) if cw is not None else None,
            seed=int(cfg.trainer.seed) + 7 * env.rank,
        )
        log.info("Phase 2 (constraints) active with weights %s", constraint_sampler.weights)
    else:
        log.info("Phase 1: text-only training (no constraints).")

    # ----- resume -----
    start_step = 0
    if cfg.trainer.get("resume_from"):
        start_step = load_checkpoint(
            Path(cfg.trainer.resume_from),
            denoiser, optimizer, scheduler, scaler, ema, device,
        )

    # ----- logging sinks -----
    tb_writer = None
    if env.is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(output_dir / "tb"))
        except Exception as e:
            log.warning("TensorBoard not available: %s", e)

    # ----- held-out test examples for side-by-side viz (rank 0 only) -----
    test_examples = []
    if env.is_main and bool(cfg.viz.get("test_vs_gt", True)):
        try:
            test_examples = _load_test_examples(
                motion_dir=cfg.data.motion_dir,
                text_dir=cfg.data.text_dir,
                split_file=cfg.viz.get("test_split_file", cfg.data.split_file),
                n_samples=int(cfg.viz.get("num_test_samples", 4)),
                max_frames=int(cfg.viz.get("test_max_frames", cfg.data.max_motion_length)),
                min_frames=int(cfg.data.get("min_motion_len", 40)),
            )
            log.info("Loaded %d held-out test examples for side-by-side viz.", len(test_examples))
        except Exception as e:
            log.warning("Could not load test examples for viz: %s", e)

    log.info("Starting training from step %d for %d steps", start_step, num_steps)
    denoiser.train()

    grad_accum = int(cfg.trainer.grad_accum)
    log_every = int(cfg.trainer.log_every)
    print_every = int(cfg.trainer.get("print_every", 50))
    ckpt_every = int(cfg.trainer.ckpt_every)
    viz_every = int(cfg.trainer.viz_every)
    grad_clip = float(cfg.trainer.grad_clip)
    ema_every = int(cfg.trainer.ema_every)
    text_drop_prob = float(cfg.trainer.text_drop_prob)

    step = start_step
    epoch = 0
    while step < num_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            step_t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            agg_loss: Dict[str, float] = {}
            for _accum in range(grad_accum):
                losses = train_one_step(
                    denoiser, diffusion, loss_fn, batch, text_encoder, device,
                    text_drop_prob, autocast_dtype, constraint_sampler=constraint_sampler,
                )
                loss = losses["loss"] / grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                for k, v in losses.items():
                    agg_loss[k] = agg_loss.get(k, 0.0) + float(v.detach()) / grad_accum

            if scaler is not None:
                scaler.unscale_(optimizer)
            params_for_clip = (
                denoiser.parameters() if not hasattr(denoiser, "module") else denoiser.module.parameters()
            )
            # Always compute the total grad norm for logging. When grad_clip>0,
            # clip_grad_norm_ both clips and returns the pre-clip norm; otherwise
            # compute it with max_norm=inf (no clipping).
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params_for_clip, grad_clip if grad_clip > 0 else float("inf"),
            )
            grad_norm_val = float(grad_norm)
            agg_loss["grad_norm"] = grad_norm_val

            # Guard against catastrophic batches that would corrupt Adam's
            # moment estimates (m, v). A non-finite or huge pre-clip grad_norm
            # almost always comes from a single bad batch (degenerate FK input,
            # bf16 underflow, etc.); skipping the step keeps the optimizer
            # state clean instead of letting one update poison the next 100k.
            # The spike is still logged below (train/grad_norm + train/grad_skip)
            # so it shows up in TensorBoard.
            grad_guard_max = float(cfg.trainer.get("grad_guard_max", 100.0))
            skipped_step = (not torch.isfinite(grad_norm)) or grad_norm_val > grad_guard_max
            agg_loss["grad_skip"] = 1.0 if skipped_step else 0.0

            if skipped_step:
                log.warning(
                    "step %d: SKIPPING optimizer step (grad_norm=%.3e non-finite or > %.1f).",
                    step + 1, grad_norm_val, grad_guard_max,
                )
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    # Tell the scaler we skipped, so it doesn't decay the scale.
                    scaler.update()
            else:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()

            if ema is not None and step % ema_every == 0:
                ema.update(denoiser.module if hasattr(denoiser, "module") else denoiser)

            step += 1
            step_dt = time.time() - step_t0

            if env.is_main and print_every > 0 and step % print_every == 0:
                comp_print = " ".join(
                    f"{k.replace('l_', '')}={v:.4f}"
                    for k, v in agg_loss.items() if k.startswith("l_")
                )
                print(
                    f"[step {step:>7d}] loss={agg_loss['loss']:.4f}"
                    + (f" | {comp_print}" if comp_print else ""),
                    flush=True,
                )

            if env.is_main and step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                line = (
                    f"step={step} loss={agg_loss['loss']:.4f} "
                    f"grad_norm={agg_loss.get('grad_norm', 0.0):.2f} "
                    f"lr={lr_now:.2e} dt={step_dt:.2f}s"
                )
                comp = " ".join(
                    f"{k.replace('l_',''):s}={v:.3f}"
                    for k, v in agg_loss.items() if k.startswith("l_")
                )
                if comp:
                    line += " | " + comp
                log.info(line)
                if tb_writer is not None:
                    for k, v in agg_loss.items():
                        tb_writer.add_scalar(f"train/{k}", v, step)
                    tb_writer.add_scalar("train/lr", lr_now, step)
                    tb_writer.add_scalar("train/step_time", step_dt, step)
                    # GPU memory (peak since last reset), in GB.
                    if torch.cuda.is_available():
                        dev_idx = device.index if device.index is not None else 0
                        tb_writer.add_scalar(
                            "gpu/mem_allocated_GB",
                            torch.cuda.max_memory_allocated(dev_idx) / 1e9, step,
                        )
                        tb_writer.add_scalar(
                            "gpu/mem_reserved_GB",
                            torch.cuda.max_memory_reserved(dev_idx) / 1e9, step,
                        )
                        free_b, total_b = torch.cuda.mem_get_info(dev_idx)
                        tb_writer.add_scalar("gpu/mem_used_GB", (total_b - free_b) / 1e9, step)
                        torch.cuda.reset_peak_memory_stats(dev_idx)

            if env.is_main and ckpt_every and step % ckpt_every == 0:
                save_checkpoint(
                    output_dir / f"ckpt_step{step:07d}.pt",
                    denoiser, optimizer, scheduler, scaler, ema, step, cfg,
                )
                latest = output_dir / "latest.pt"
                try:
                    if latest.exists() or latest.is_symlink():
                        latest.unlink()
                    latest.symlink_to(f"ckpt_step{step:07d}.pt")
                except OSError:
                    shutil.copy2(output_dir / f"ckpt_step{step:07d}.pt", latest)

            if env.is_main and viz_every and step % viz_every == 0:
                try:
                    # Unified viz: one diffusion forward, one folder per step
                    # with npz + side-by-side mp4 per item. test_examples carry
                    # GT (rendered on the left); any extra prompts have no GT
                    # (blank left panel). When test_examples is non-empty we
                    # skip the fixed cfg.viz.prompts list to avoid duplicating
                    # the work.
                    fallback_prompts = (
                        None if test_examples
                        else [p for p in (cfg.viz.get("prompts", []) or []) if isinstance(p, str) and p]
                    )
                    viz_step(
                        denoiser, diffusion, text_encoder, device, cfg, step,
                        output_dir / "viz", mean=mean, std=std,
                        loss_fn=loss_fn,
                        prompts=fallback_prompts,
                        test_examples=test_examples,
                        tb_writer=tb_writer,
                    )
                except Exception as e:
                    log.warning("Viz failed at step %d: %s", step, e)

            if step >= num_steps:
                break
        epoch += 1

    if env.is_main:
        save_checkpoint(
            output_dir / "ckpt_final.pt",
            denoiser, optimizer, scheduler, scaler, ema, step, cfg,
        )
    from kimodo.scripts.train import cleanup_distributed
    cleanup_distributed(env)


if __name__ == "__main__":
    main()
