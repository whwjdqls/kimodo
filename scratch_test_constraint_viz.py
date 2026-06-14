"""End-to-end smoke test for the text+constraints training viz.

Mirrors train.py's model build, loads the latest bones_seed_small_constraints_warm
checkpoint, and runs viz_step (with constraint_sampler) on the 4 held-out test
examples — exercising BOTH the text-only and the new text+constraints passes,
the fixed camera, the finger drop, and the constraint-marker overlay.

Run on a compute node (srun), never the login shell.
"""
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from kimodo.model.diffusion import Diffusion
from kimodo.scripts.train import (
    ConstraintSampler,
    _build_cpu_motion_rep,
    _load_test_examples_soma,
    _resolve_num_steps,
    build_denoiser_from_model_config,
    build_text_encoder,
    viz_step,
)

RUN = Path("/home/jungbin_cho/kimodo_open/runs/bones_seed_small_constraints_warm")
OUT = Path("/home/jungbin_cho/kimodo_open/viz_constraint_train_test")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    cfg = OmegaConf.load(RUN / "config.yaml")

    # --- denoiser + checkpoint ---
    denoiser = build_denoiser_from_model_config(
        cfg.model_config_path, cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    ).to(device)
    ckpt = RUN / "latest.pt"
    obj = torch.load(ckpt, map_location=device)
    denoiser.load_state_dict(obj["denoiser"])
    step = int(obj.get("step", 0))
    print(f"loaded checkpoint {ckpt} @ step {step}", flush=True)
    denoiser.eval()

    diffusion = Diffusion(num_base_steps=int(_resolve_num_steps(cfg))).to(device)
    text_encoder = build_text_encoder(cfg.text_encoder, device=device)

    # --- held-out GT examples (same as training viz) ---
    dataset_motion_rep = _build_cpu_motion_rep(
        cfg.model_config_path, cfg.stats_path,
        fps_override=cfg.get("denoiser_fps_override"),
    )
    test_examples = _load_test_examples_soma(
        test_split_path=cfg.viz.test_split_file,
        data_root=cfg.data.data_root,
        natural_csv_path=cfg.data.natural_csv_path,
        motion_rep=dataset_motion_rep,
        skeleton=dataset_motion_rep.skeleton,
        n_samples=int(cfg.viz.get("num_test_samples", 4)),
        max_frames=int(cfg.viz.get("test_max_frames", cfg.data.max_frames)),
        min_frames=int(cfg.data.get("min_frames", 10)),
        include_mirrored=False,
    )
    print(f"loaded {len(test_examples)} test examples", flush=True)

    # --- constraint sampler (phase-2) matching the training config ---
    cw = cfg.trainer.get("constraint_weights", None)
    constraint_sampler = ConstraintSampler(
        denoiser.motion_rep,
        weights=OmegaConf.to_container(cw) if cw is not None else None,
        none_prob=float(cfg.trainer.get("constraint_none_prob", 0.10)),
        mix_prob=float(cfg.trainer.get("constraint_mix_prob", 0.25)),
        max_keyframes=int(cfg.trainer.get("constraint_max_keyframes", 20)),
        seed=0,
    )

    viz_step(
        denoiser, diffusion, text_encoder, device, cfg, step,
        OUT, test_examples=test_examples, tb_writer=None,
        constraint_sampler=constraint_sampler,
    )
    out_step = OUT / f"{step:07d}"
    mp4s = sorted(out_step.glob("*.mp4"))
    print(f"\n=== wrote {len(mp4s)} mp4 to {out_step} ===", flush=True)
    for m in mp4s:
        print(f"  {m.name}  ({m.stat().st_size // 1024}K)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
