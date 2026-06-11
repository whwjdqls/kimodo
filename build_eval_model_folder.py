"""Assemble a load_model()-compatible model folder from a training run dir.

The benchmark's generate_eval.py calls `load_model(name)`, which expects a
HF-style model folder containing:
    config.yaml          (inference config; run dir's model_config.yaml IS this)
    model.safetensors    (denoiser weights, EMA-averaged)
    stats/motion/        (normalization stats)

This script converts a training checkpoint (ckpt_step*.pt, which holds
{"denoiser": sd, "ema": sd, "step": int, ...}) into that layout.

Run on a COMPUTE node (tmux 0), not login.

    python build_eval_model_folder.py \
        --run-dir runs/bones_seed_small_tiny \
        --ckpt latest.pt \
        --stats-src /home/jungbin_cho/Kimodo-SOMA-SEED-v1.1/stats/motion \
        --out /home/jungbin_cho/kimodo_eval_models/BS-Tiny
"""
import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--ckpt", default="latest.pt",
                    help="checkpoint filename inside run-dir (default latest.pt)")
    ap.add_argument("--stats-src", type=Path,
                    default=Path("/home/jungbin_cho/Kimodo-SOMA-SEED-v1.1/stats/motion"),
                    help="source stats/motion dir to copy into the model folder")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--use-ema", action="store_true", default=True,
                    help="overlay EMA float weights (default on)")
    ap.add_argument("--no-ema", dest="use_ema", action="store_false",
                    help="use raw (non-EMA) denoiser weights")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override motion_rep.fps in the output config.yaml. Needed when the "
                    "snapshot's model_config.yaml has a stale fps (e.g. fps:30) but the run was "
                    "trained with denoiser_fps_override (e.g. 20). Set to the TRUE training fps.")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    ckpt_path = (run_dir / args.ckpt).resolve()
    model_cfg = run_dir / "model_config.yaml"
    assert ckpt_path.is_file(), f"checkpoint not found: {ckpt_path}"
    assert model_cfg.is_file(), f"model_config.yaml not found: {model_cfg}"
    assert args.stats_src.is_dir(), f"stats src not found: {args.stats_src}"

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. weights: denoiser sd, EMA float weights overlaid ---
    print(f"loading {ckpt_path} ...")
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert "denoiser" in obj, f"no 'denoiser' key in checkpoint; keys={list(obj)[:8]}"
    sd = {k: v.clone() for k, v in obj["denoiser"].items()}
    n_ema = 0
    if args.use_ema and obj.get("ema"):
        for k, v in obj["ema"].items():
            if k in sd:
                sd[k] = v.clone()
                n_ema += 1
        print(f"overlaid {n_ema} EMA float tensors onto {len(sd)} denoiser tensors")
    else:
        print(f"using raw denoiser weights ({len(sd)} tensors), step={obj.get('step')}")

    # safetensors requires contiguous tensors and no shared storage
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, str(out / "model.safetensors"),
              metadata={"step": str(obj.get("step", "?")),
                        "ema": str(bool(n_ema))})
    print(f"wrote {out / 'model.safetensors'}  ({len(sd)} tensors, step={obj.get('step')})")

    # --- 2. config.yaml (run dir's model_config.yaml is already inference fmt) ---
    if args.fps is not None:
        from omegaconf import OmegaConf
        conf = OmegaConf.load(model_cfg)
        old_fps = conf.denoiser.motion_rep.fps
        conf.denoiser.motion_rep.fps = args.fps
        OmegaConf.save(conf, out / "config.yaml")
        print(f"wrote {out / 'config.yaml'}  (patched motion_rep.fps {old_fps} -> {args.fps})")
    else:
        shutil.copy(model_cfg, out / "config.yaml")
        print(f"wrote {out / 'config.yaml'}")

    # --- 3. stats/motion/ ---
    stats_dst = out / "stats" / "motion"
    if stats_dst.exists():
        shutil.rmtree(stats_dst)
    shutil.copytree(args.stats_src, stats_dst)
    print(f"copied stats -> {stats_dst}")

    print(f"\nDONE. Model folder ready: {out}")
    print(f"Use with: CHECKPOINT_DIR={out.parent} ... --model {out.name}")


if __name__ == "__main__":
    main()
