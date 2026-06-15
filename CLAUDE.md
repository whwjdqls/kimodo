# CLAUDE.md

Guidance for Claude Code working in this repo (`kimodo_open`).

## What this is

Kimodo — a **ki**nematic **mo**tion **d**iffusi**o**n model: text- and constraint-conditioned 3D human/robot motion generation, trained on the BONES-SEED optical mocap dataset. This is a research working copy (open-source Kimodo + local training/eval/benchmark additions). Upstream docs: README.md and https://research.nvidia.com/labs/sil/projects/kimodo/.

Python 3.10, PyTorch 2.4, conda env **`kimodo`** at `/home/jungbin_cho/miniconda3/envs/kimodo`. Use its interpreter directly when scripting: `/home/jungbin_cho/miniconda3/envs/kimodo/bin/python`.

## Compute rules (important)

- **Never run heavy work on the login node.** Anything multi-GB or multi-minute (training, generation, eval, packing, GPU smoke tests) goes through **`srun` / `sbatch`**, or `tmux attach -t 0`. The login shell has **no GPU** (`nvidia-smi` fails there).
- SLURM partition **`a2`** = GCP a2-megagpu-16g nodes: 16× A100 **40 GB**, 96 vCPU, 1.35 TB RAM. Multi-GPU training uses 8 GPUs/node.
- One-off GPU check pattern:
  `srun -p a2 --gres=gpu:1 --cpus-per-task=8 --mem=48G python <script>.py`
- `#SBATCH --output` is parsed before the shell runs, so it can't contain shell vars; `sbatch --wrap` runs under `dash` (no `source`) — use a real `#!/bin/bash` script instead.
- The `bash` tool prints a harmless `libtinfo.so.6` warning on every call — ignore it.

## Running things

**Training** (`kimodo/scripts/train.py`), Hydra-style `key=value` overrides after `--config`:
```
# multi-GPU (8) via torchrun, from an sbatch script:
torchrun --standalone --nproc_per_node=8 -m kimodo.scripts.train \
    --config configs/training/bones_seed_full.yaml \
    trainer.batch_size=128 output_dir=runs/<name> \
    trainer.resume_from=runs/<name>/ckpt_step0200000.pt   # optional resume
```
Submit scripts live at repo root as `submit_train_*.sh` / `submit_resume_*.sh`. Each run writes to `runs/<name>/`: `ckpt_step*.pt`, `latest.pt` (symlink), `config.yaml` + `model_config.yaml` snapshots, `tb/`, `viz/<step>/*.{npz,mp4}`, `train_rank*.log`.

Config layering: `configs/training/*.yaml` references a `model_config_path` → `configs/model/*.yaml`. `trainer.phase` is `text` (text→motion only) or `constraints` (phase-2, adds the constraint sampler).

**Evaluation** (the benchmark, `benchmark/`): pipeline is generate → embed → evaluate over the testsuite. Driver scripts: `run_eval_*.sh`, `submit_eval_*.sh`. Results land under `runs/<model>/eval/{models,gen,results,logs}/step_<N>/`. Use `--text-cache` to skip live LLM2Vec at eval (see caching below).

**Viz**: two *separate* renderers (don't confuse them) —
- `kimodo/scripts/render_soma.py` — matplotlib stick figure, used **in-training** by `viz_step()`. Constraint runs also overlay purple-sphere targets + pause on keyframes; `camera="follow"` (default) tracks the root, fingertip-end joints dropped.
- `kimodo/scripts/visualize.py` — offline **pyrender/EGL** skinned-mesh MP4 renderer (separate `--constraints`/`--camera` flags). pyrender was installed into the env `--no-deps`.
- `kimodo/demo/` + `kimodo/viz/viser_utils.py` — interactive viser web viewer (optional `viser` dep; `kimodo/viz/__init__.py` guards the import so offline rendering works without it).

## Codebase map

- `kimodo/model/` — `kimodo_model.py` (top-level `Kimodo`), `twostage_denoiser.py` / `onestage_denoiser.py` / `three_stage.py`, `diffusion.py`, `backbone.py`. Text encoders: `llm2vec/`, `clip_text/`, `cached_text/` (lookup-table), `text_encoder_api.py`. `load_model.py` / `loading.py` (checkpoint/safetensors loading; both accept a `text_encoder=` injection). `tmr.py`, `mask_generator.py`, `constraint_generator.py`.
- `kimodo/motion_rep/` — `KimodoMotionRep` (the ~369-D feature: `smooth_root_pos`, `global_root_heading`, `local_joints_positions`, `global_rot_data`, `velocities`, `foot_contacts` via `slice_dict`). `normalize/unnormalize`, `inverse` (→ `posed_joints`), `canonicalize`, `rotate`. `stats.py`, `feet.py`, `fk_hml3d.py`, `humanml3d_native.py`.
- `kimodo/skeleton/` — `definitions.py`: **SOMASkeleton30** (training/feature skeleton, 30 joints) and **SOMASkeleton77** (model I/O, full hands), plus `G1Skeleton34`. `from_SOMASkeleton77`/`to_SOMASkeleton77` convert. `bone_index` maps name→idx.
- `kimodo/data/` — `soma_text_motion.py`: `SOMABonesSeedDataset` (1:1:1 mixture of natural / single-event / multi-event sources; segment-window rule; optional mmap **packs** via `packed_motions_path` raw or `packed_features_path` 369-D). `humanml3d_*` for HML3D.
- `kimodo/metrics/` — `tmr.py` (R-precision/retrieval over the full split, no 32-pooling), `foot_skate.py`, `constraints.py`.
- `kimodo/` misc — `constraints.py` (constraint types + `load_constraints_lst`), `sanitize.py` (`sanitize_texts` — strips/capitalizes/punctuates; **cache keys must be sanitized**), `meta.py` (`parse_prompts_from_meta`), `geometry.py`, `postprocess.py`, `exports/bvh.py`.
- `kimodo/scripts/` — entry points: `train.py`, `generate.py`, `sample_hml3d.py`, `pack_bones_seed_{motions,features}.py`, `precompute_text_embeddings.py`, `compute_motion_stats.py`, renderers above. HML3D variants: `train_w_hml3d*.py`, `train_3stage*.py`.
- `benchmark/` — `generate_eval.py`, `embed_folder.py`, `evaluate_folder.py`, `precompute_benchmark_text.py`, `parse_folder.py`.

## Domain gotchas (learned the hard way)

- **Text caching**: training and eval normally use a precomputed `CachedTextEncoder` (`.pt` lookup table) instead of live LLM2Vec-8B. It's **strict exact-string** — a miss raises `KeyError`, no live fallback. Keys must be `sanitize_texts`-normalized. Eval caches: `kimodo_caches/benchmark_llm2vec.pt`; training: `kimodo_caches/bones_seed_llm2vec_small.pt`. Eval is **diffusion-bound** (~87% of wall time); caching text only saves ~5%.
- **Resume OOM**: `train.load_checkpoint` loads to CPU (`map_location="cpu"`) and copies in-place to avoid a transient GPU duplicate of the 4.5 GB checkpoint — batch 128 already fills ~93% of a 40 GB A100. Resume sbatch also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Eval R-precision** is computed over the entire split (full N×N), so generation `batch_size` affects only per-sample init noise, not the metric protocol.
- `add_video needs package moviepy` in train logs is cosmetic (TB video embed); the `.mp4` files are still written.

## Data & caches (on this machine)

- Motions (20 fps NPZs): `/home/jungbin_cho/seed/soma_uniform_motions_20fps/`; metadata CSV/JSONL under `/home/jungbin_cho/seed/`.
- Splits: `/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/` (`train_split_paths.txt` full = 128k; `*_small.txt` = ~13k).
- Benchmark testsuite: `/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/`.
- Stats: `/home/jungbin_cho/Kimodo-SOMA-SEED-v1.1/stats/motion/`.
- Caches (text embeds, index, packs): `/home/jungbin_cho/kimodo_caches/`.

## Repo conventions

- `runs/`, `eval_logs/`, `*_test/`, `datasets` are **gitignored** — checkpoints/outputs aren't tracked.
- Root-level `diag_*.py`, `verify_*.py`, `profile_*.py`, `scratch_*.py` are ad-hoc diagnostics — fine to add more, keep them out of `kimodo/`.
- Don't commit or push unless asked; this is a working copy with a freshly `git init`'d history.
