"""Train a model on the UniEgoMotion-style 211-D representation (HumanML3D).

Counterpart to ``kimodo.scripts.train_w_hml3d_native`` (263-D). Same one-stage
MDM-style denoiser, same flat masked-L2 loss on x0, same CLIP text encoder —
only the data/rep differ (head-centric 211-D from ``HumanML3D/uniego_rep/``).
This makes an apples-to-apples comparison of native-263 vs uniego-211 (and vs
the kimodo-273 MDM run), varying only the motion representation.

Differences from ``train_w_hml3d_native``:
  * Dataset: :class:`HumanML3DUniegoTextMotionDataset` reads ``.npz`` (key
    ``features``, T×211), with per-window frame-0 canonicalization.
  * Stats: ``Mean_uniego.npy`` / ``Std_uniego.npy``.
  * Viz decode: :func:`uniego_world_joints_from_features` (cumulative-compose
    the residual canonical frame, then read joint translations).

Usage::

    python -m kimodo.scripts.train_w_hml3d_uniego \\
        --config configs/training/hml3d_uniego_mdm_clip.yaml
"""

from __future__ import annotations

import os as _os
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
from typing import Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.cuda.amp import GradScaler  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

# Shared training pieces — reused unmodified.
from kimodo.scripts.train import (  # noqa: E402
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
    _resolve_num_steps,
)
from kimodo.scripts.train_w_hml3d import _cfg_sample  # noqa: E402
# The flat masked-L2 loss is rep-agnostic (generic over feature dim) — reuse it.
from kimodo.scripts.train_w_hml3d_native import MDMNativeL2Loss  # noqa: E402
from kimodo.model.diffusion import Diffusion  # noqa: E402

# Uniego-side.
from kimodo.data.humanml3d_uniego_text_motion import (  # noqa: E402
    HumanML3DUniegoTextMotionDataset,
    build_humanml3d_uniego_collate_fn,
)
from kimodo.motion_rep.uniego import (  # noqa: E402
    FEAT_DIM as UNIEGO_FEAT_DIM,
    IDENTITY_DELTA9,
    uniego_world_joints_from_features,
)

log = logging.getLogger("kimodo.train_hml3d_uniego")

_DELTA0, _DELTA1 = 198, 207
_IDENTITY_DELTA = np.asarray(IDENTITY_DELTA9, dtype=np.float32)


# -----------------------------------------------------------------------------
# Held-out test examples — uniego loader (reads .npz "features", 211-D).
# -----------------------------------------------------------------------------
def _load_test_examples_uniego(
    motion_dir: str,
    text_dir: str,
    split_file: str,
    n_samples: int,
    max_frames: int,
    min_frames: int = 40,
) -> List[dict]:
    """First ``n_samples`` ids on disk with a full-clip caption.

    Returns dicts ``{id, caption, gt_features (T, 211), length}`` with frame 0
    canonicalized (matching the training dataset).
    """
    import codecs as cs
    motion_dir = Path(motion_dir)
    text_dir = Path(text_dir)
    with open(split_file, "r") as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    out: List[dict] = []
    for mid in ids:
        if len(out) >= n_samples:
            break
        mp = motion_dir / f"{mid}.npz"
        tp = text_dir / f"{mid}.txt"
        if not mp.is_file() or not tp.is_file():
            continue
        try:
            feats = np.asarray(np.load(mp)["features"]).astype(np.float32)  # (T, 211)
        except Exception:
            continue
        if feats.ndim != 2 or feats.shape[1] != UNIEGO_FEAT_DIM or feats.shape[0] < min_frames:
            continue
        caption: Optional[str] = None
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
        clip = feats[:T].copy()
        clip[0, _DELTA0:_DELTA1] = _IDENTITY_DELTA  # canonicalize frame 0
        out.append({"id": mid, "caption": caption, "gt_features": clip, "length": int(T)})
    return out


# -----------------------------------------------------------------------------
# Viz — mirrors viz_step_native with the 211-D decode swapped in.
# -----------------------------------------------------------------------------
@torch.no_grad()
def viz_step_uniego(
    denoiser: nn.Module,
    diffusion: Diffusion,
    text_encoder,
    device: torch.device,
    cfg: DictConfig,
    step: int,
    out_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    prompts: Optional[list] = None,
    test_examples: Optional[List[dict]] = None,
    tb_writer=None,
) -> None:
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
        prompts = [p for p in (cfg.viz.get("prompts", []) or []) if isinstance(p, str) and p]
    else:
        prompts = list(prompts)
    test_examples = test_examples or []

    items: List[dict] = []
    for i, p in enumerate(prompts):
        safe = p.lower().replace(" ", "_").replace(".", "")[:40] or f"prompt_{i}"
        items.append({"id": f"prompt_{i:02d}_{safe}", "caption": p,
                      "gt_features": None, "length": num_frames_default})
    for ex in test_examples:
        items.append({"id": ex["id"], "caption": ex["caption"],
                      "gt_features": ex["gt_features"], "length": int(ex["length"])})
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
    )  # (B, max_T, 211) normalized
    raw.train()

    mean_t = torch.from_numpy(mean).to(cur.device)
    std_t = torch.from_numpy(std).to(cur.device)
    gen_unnorm = cur.float() * std_t + mean_t  # (B, max_T, 211) raw uniego

    tb_videos = bool(cfg.viz.get("tb_videos", False)) and (tb_writer is not None)
    video_stack: List[np.ndarray] = []
    for i, it in enumerate(items):
        L = it["length"]
        gen_feats = gen_unnorm[i, :L]
        gen_joints = uniego_world_joints_from_features(
            gen_feats.unsqueeze(0), n_joints=motion_rep.nbjoints,
        )[0].cpu().numpy()
        gen_feats_np = gen_feats.cpu().numpy()

        gt_joints = None
        npz_kwargs = {"features": gen_feats_np, "prompt": it["caption"]}
        if it["gt_features"] is not None:
            gt_feats = torch.from_numpy(it["gt_features"]).to(device).unsqueeze(0)
            gt_joints = uniego_world_joints_from_features(
                gt_feats, n_joints=motion_rep.nbjoints,
            )[0].cpu().numpy()
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
            iio.imwrite(str(base.with_suffix(".mp4")), frames,
                        fps=float(fps), codec="h264", plugin="pyav")
        except Exception as e:
            log.warning("MP4 write failed for %s: %s", it["id"], e)
        if tb_videos:
            video_stack.append(frames)
        else:
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
        cfg.get("stats_path", ""),
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
    assert motion_rep.motion_rep_dim == UNIEGO_FEAT_DIM, (
        f"Expected uniego motion_rep_dim={UNIEGO_FEAT_DIM}, got {motion_rep.motion_rep_dim}. "
        f"Check that the model config uses UniegoMotionRep."
    )

    # ----- dataset / dataloader -----
    mean = np.load(cfg.data.mean_path).astype(np.float32)
    std = np.load(cfg.data.std_path).astype(np.float32)

    def _build_dataset():
        return HumanML3DUniegoTextMotionDataset(
            motion_dir=cfg.data.motion_dir,
            text_dir=cfg.data.text_dir,
            split_file=cfg.data.split_file,
            mean=mean, std=std,
            fps=int(cfg.data.fps),
            window_size=int(cfg.data.window_size),
            max_motion_length=int(cfg.data.max_motion_length),
            min_motion_len=int(cfg.data.min_motion_len),
            unit_length=int(cfg.data.unit_length),
            clip_normalized=cfg.data.get("clip_normalized"),
            motion_rep=motion_rep, skeleton=motion_rep.skeleton,
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
        collate_fn=build_humanml3d_uniego_collate_fn(),
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

    # ----- text encoder -----
    _enc_type = str(cfg.text_encoder.get("type", "llm2vec")).lower()
    _enc_cache = cfg.text_encoder.get("cache_path", None)
    if _enc_cache:
        log.info("Building text encoder: cached (cache_path=%s).", _enc_cache)
    else:
        log.info("Building text encoder: %s (model=%s) ...",
                 _enc_type, cfg.text_encoder.get("model_name", "(default)"))

    if env.is_distributed:
        for r in range(env.world_size):
            if env.rank == r:
                text_encoder = build_text_encoder(cfg.text_encoder, device=device)
            maybe_barrier()
    else:
        text_encoder = build_text_encoder(cfg.text_encoder, device=device)
        log.info("Text encoder built (class=%s).", type(text_encoder).__name__)

    # ----- DDP wrap -----
    if env.is_distributed:
        denoiser = DDP(denoiser, device_ids=[env.local_rank], find_unused_parameters=False)

    # ----- loss (paper-faithful MDM flat masked-L2 on x0) -----
    loss_fn = MDMNativeL2Loss(motion_rep).to(device)

    if str(cfg.trainer.get("phase", "text")) != "text":
        raise ValueError("hml3d-uniego script only supports trainer.phase='text'.")

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
    test_examples: List[dict] = []
    if env.is_main and bool(cfg.viz.get("test_vs_gt", True)):
        try:
            test_examples = _load_test_examples_uniego(
                motion_dir=cfg.data.motion_dir,
                text_dir=cfg.data.text_dir,
                split_file=cfg.viz.get("test_split_file", cfg.data.split_file),
                n_samples=int(cfg.viz.get("num_test_samples", 4)),
                max_frames=int(cfg.viz.get("test_max_frames", cfg.data.max_motion_length)),
                min_frames=int(cfg.data.get("min_motion_len", 40)),
            )
            log.info("Loaded %d held-out test examples (uniego).", len(test_examples))
        except Exception as e:
            log.warning("Could not load test examples for viz: %s", e)

    log.info("Starting training from step %d for %d steps", start_step, num_steps)
    denoiser.train()

    grad_accum = int(cfg.trainer.grad_accum)
    log_every = int(cfg.trainer.log_every)
    print_every = int(cfg.trainer.get("print_every", 1))
    ckpt_every = int(cfg.trainer.ckpt_every)
    viz_every = int(cfg.trainer.viz_every)
    grad_clip = float(cfg.trainer.grad_clip)
    ema_every = int(cfg.trainer.ema_every)
    text_drop_prob = float(cfg.trainer.text_drop_prob)
    grad_guard_max = float(cfg.trainer.get("grad_guard_max", 100.0))

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
                    text_drop_prob, autocast_dtype, constraint_sampler=None,
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
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params_for_clip, grad_clip if grad_clip > 0 else float("inf"),
            )
            grad_norm_val = float(grad_norm)
            agg_loss["grad_norm"] = grad_norm_val
            skipped_step = (not torch.isfinite(grad_norm)) or grad_norm_val > grad_guard_max
            agg_loss["grad_skip"] = 1.0 if skipped_step else 0.0

            if skipped_step:
                log.warning(
                    "step %d: SKIPPING optimizer step (grad_norm=%.3e non-finite or > %.1f).",
                    step + 1, grad_norm_val, grad_guard_max,
                )
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
            else:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()

            if ema is not None and step % ema_every == 0 and not skipped_step:
                ema.update(denoiser.module if hasattr(denoiser, "module") else denoiser)

            step += 1
            step_dt = time.time() - step_t0

            if env.is_main and print_every > 0 and step % print_every == 0:
                comp_print = " ".join(
                    f"{k.replace('l_', '')}={v:.4f}"
                    for k, v in agg_loss.items() if k.startswith("l_")
                )
                try:
                    print(
                        f"[step {step:>7d}] loss={agg_loss['loss']:.4f}"
                        + (f" | {comp_print}" if comp_print else ""),
                        flush=True,
                    )
                except OSError as e:
                    log.warning("stdout print failed (%s); continuing.", e)

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
                    if torch.cuda.is_available():
                        dev_idx = device.index if device.index is not None else 0
                        tb_writer.add_scalar(
                            "gpu/mem_allocated_GB",
                            torch.cuda.max_memory_allocated(dev_idx) / 1e9, step,
                        )
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
                    fallback_prompts = (
                        None if test_examples
                        else [p for p in (cfg.viz.get("prompts", []) or []) if isinstance(p, str) and p]
                    )
                    viz_step_uniego(
                        denoiser, diffusion, text_encoder, device, cfg, step,
                        output_dir / "viz", mean=mean, std=std,
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
