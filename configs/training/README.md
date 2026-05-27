# Training Kimodo from scratch

Two training pipelines live in this repo:

| Pipeline | Dataset | Skeleton | Feature dim | Entrypoint | Config |
|---|---|---|---|---|---|
| `train.py` | BONES-SEED (SOMA Uniform NPZs) | `SOMASkeleton30` | 369 | `kimodo.scripts.train` | `default.yaml` |
| `train_w_hml3d.py` | HumanML3D (in Kimodo format) | `SMPLXSkeleton22` (container only) | 273 | `kimodo.scripts.train_w_hml3d` | `hml3d.yaml` |

The rest of this file covers **HumanML3D**. For BONES-SEED, see `default.yaml` and `kimodo/scripts/train.py`.

---

## 1. Data layout

HumanML3D in Kimodo format. Produced by `benchmark/humanml3d_to_kimodo.py`, which converts the standard HumanML3D 263‑D per-frame vector to KIMODO's global-root 273‑D layout (`smooth_root_pos` 3 + `global_root_heading` 2 + `local_joints_positions` 22*3 + `global_rot_data` 22*6 + `velocities` 22*3 + `foot_contacts` 4 = 273).

```
/home/jungbin_cho/HumanML3D/HumanML3D/
├── kimodo_rep/{id}.npz           # converted features. Each NPZ has `features` (T,273)
│                                  # + decomposed channels (smooth_root_pos, global_rot_mats, ...).
├── texts/{id}.txt                # standard HumanML3D text file:
│                                  #   caption # POS tokens # f_tag(sec) # to_tag(sec)
├── new_joint_vecs/{id}.npy       # original 263-D HumanML3D vectors (we don't read these
│                                  # at train time, but they pair 1:1 with kimodo_rep).
├── Mean_kimodo.npy , Std_kimodo.npy        # 273-dim training stats.
├── stats_kimodo/                  # split-layout stats for KimodoMotionRep:
│   ├── global_root/{mean,std}.npy        # (5,)
│   ├── local_root/{mean,std}.npy         # (4,)  — computed from features once
│   └── body/{mean,std}.npy               # (268,)
└── {train,val,test,all}.txt        # split files (one id per line)
```

> **Note**: at the time of writing, **val.txt** entries are not present on disk (the upstream HumanML3D processing didn't write them). `train.txt` has 23,024/23,384 present; `test.txt` is 100% present. Train on `train.txt`; use `test.txt` for held-out eval, or carve a held-out subset out of `train.txt`.

If `stats_kimodo/` is missing, regenerate it from `Mean_kimodo.npy` / `Std_kimodo.npy`:

```bash
python -c "
import numpy as np
from pathlib import Path
out = Path('/home/jungbin_cho/HumanML3D/HumanML3D/stats_kimodo')
mean = np.load('/home/jungbin_cho/HumanML3D/HumanML3D/Mean_kimodo.npy').astype('float32')
std  = np.load('/home/jungbin_cho/HumanML3D/HumanML3D/Std_kimodo.npy').astype('float32')
for sub, m, s in [('global_root', mean[:5], std[:5]), ('body', mean[5:], std[5:])]:
    (out/sub).mkdir(parents=True, exist_ok=True)
    np.save(out/sub/'mean.npy', m); np.save(out/sub/'std.npy', np.maximum(s, 1e-4))
"
```

`local_root` is derived from features (4-dim). It's already in the directory; if you ever need to recompute it, the recipe is in `kimodo/scripts/compute_motion_stats.py` adapted for SMPL-22.

---

## 2. Model

`configs/model/kimodo_humanml3d_small.yaml` defines an MDM-base-sized transformer:

* `latent_dim=512`, `ff_size=1024`, `num_layers=8`, `num_heads=4`, `dropout=0.1`.
* `motion_rep = KimodoMotionRep(SMPLXSkeleton22, fps=20)` → 273-dim features.
* `SMPLXSkeleton22` is used **as a container only** — it gives `nbjoints=22`, and that's it. We do **not** call its FK because HumanML3D rotations follow a chain-reset convention. See §5.

Want bigger? Bump `latent_dim`, `num_layers`, `ff_size` in the model config. The dataloader and training script are unchanged.

---

## 3. Single-GPU run

```bash
conda activate kimodo
cd /home/jungbin_cho/kimodo

python -m kimodo.scripts.train_w_hml3d \
  --config configs/training/hml3d.yaml \
  output_dir=/weka/jungbin/kimodo_runs/hml3d_small_v1
```

This uses LLM2Vec online; on first run it downloads the encoder weights to your HF cache (~16 GB Llama-3-8B + adapters).

---

## 4. 8-GPU run (single node)

Pre-download the text encoder to a flat local-dir once (avoids the HF cache snapshot/symlink races we hit when 8 ranks load concurrently):

```bash
python -c "
from huggingface_hub import snapshot_download
for repo, dst in [
  ('meta-llama/Meta-Llama-3-8B-Instruct', '/weka/jungbin/hf_models/Meta-Llama-3-8B-Instruct'),
  ('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp', '/weka/jungbin/hf_models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp'),
  ('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised', '/weka/jungbin/hf_models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised'),
]:
    snapshot_download(repo_id=repo, local_dir=dst, local_dir_use_symlinks=False)
"
# After download, patch the MNTP adapter configs to point at the local Llama dir:
python -c "
import json, pathlib
local_llama = '/weka/jungbin/hf_models/Meta-Llama-3-8B-Instruct'
for d in ['LLM2Vec-Meta-Llama-3-8B-Instruct-mntp', 'LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised']:
    p = pathlib.Path('/weka/jungbin/hf_models')/d/'adapter_config.json'
    cfg = json.loads(p.read_text())
    cfg['base_model_name_or_path'] = local_llama
    p.write_text(json.dumps(cfg, indent=2))
"
```

Then launch:

```bash
KIMODO_TEXT_ENCODER_LOCAL_DIR=/weka/jungbin/hf_models \
torchrun --standalone --nproc_per_node=8 --redirects 3 --tee 3 \
  -m kimodo.scripts.train_w_hml3d \
  --config configs/training/hml3d.yaml \
  trainer.batch_size=64 \
  output_dir=/weka/jungbin/kimodo_runs/hml3d_small_v1
```

* `KIMODO_TEXT_ENCODER_LOCAL_DIR` makes the encoder load from the flat local mirror (no HF API calls).
* `--redirects 3 --tee 3` writes each rank's stdout/stderr to disk and tees them.
* When `WORLD_SIZE>1`, `train_w_hml3d.py` automatically sets `HF_HUB_OFFLINE=1`.
* Ranks build the text encoder serially (rank 0 → barrier → rank 1 → …) — eliminates the remaining cache races we hit with concurrent first-time loads. Total cost is ~8×30 s for SMPL-22 / Llama-3.

---

## 5. Loss

Defined in `KimodoLoss` (in `kimodo/scripts/train.py`). Smooth-L1 on six feature blocks plus a chain-reset FK consistency term:

| Component | Weight (γ) | Notes |
|---|---|---|
| `smooth_root_pos` | 10.0 | γ₁ |
| `global_root_heading` | 2.0 | γ₂ |
| `local_joints_positions` | 10.0 | γ₃ |
| `velocities` | 3.0 | γ₄ |
| `global_rot_data` | 10.0 | γ₅ |
| `foot_contacts` | 4.0 | γ₆ |
| `fk` (chain-reset) | 5.0 | γ₇ — see below |

### FK consistency (γ₇) for HumanML3D

HumanML3D rotations are stored under the **chain-reset** convention: each sub-chain (arms, legs) starts with the root rotation as `matR`, not with its geometric parent's rotation. Standard parent-relative FK does not work on these rotations.

`kimodo/motion_rep/fk_hml3d.py` implements an FK that mirrors `HumanML3D/common/skeleton.py:forward_kinematics_cont6d` exactly:

```
world_pos[chain[k]] = world_pos[chain[k-1]] + global_rot_mats[chain[k]] @ (raw_offsets[chain[k]] * bone_length[chain[k]])
```

* `raw_offsets` are the **axis-aligned unit directions** from `paramUtil.t2m_raw_offsets` (e.g. left shoulder = `[0,-1,0]`, head = `[0,0,+1]`). They are NOT the bone vectors of any canonical T-pose. Baked into `HML3D_RAW_OFFSETS` so we don't depend on the HumanML3D package at training time.
* `bone_length` is **per-sample**, derived from the GT joint positions (`derive_bone_lengths_from_world_joints`). This makes the FK exact: `FK(gt_rot, derived_bone_lengths) == gt_positions` to ~1e-6 m.

`train_w_hml3d.py` sets `fk_kind="chainreset_hml3d"`, `fk_target="gt"` (paper-faithful — `||FK(pred_rot, bl) − GT positions||₁`).

To visually verify on a sample:
```bash
python -m kimodo.scripts.viz_fk_hml3d \
  /home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/000000.npz \
  --out /tmp/fk_viz
```
This writes a PNG (GT row / FK row / overlay row × 6 frames) and an MP4. With the correct FK, the overlay row should show red exactly on top of blue.

---

## 6. Resuming and visualization

* **Resume**: `trainer.resume_from=/path/ckpt.pt`. Restores denoiser, optimizer, scheduler, EMA, AMP scaler, and step counter.
* **Periodic checkpoints**: `trainer.ckpt_every=5000` → `output_dir/ckpt_step{N}.pt` and a `latest.pt` symlink. Final ckpt is always written.
* **In-training viz**: `trainer.viz_every=5000` writes the **raw kimodo features** of generated samples to `output_dir/viz/step{N}_*.npz`. We do **not** call `motion_rep.inverse()` here because it uses standard FK; to render, decode via `benchmark/humanml3d_to_kimodo.kimodo_to_humanml3d` → HumanML3D 263-D → `recover_from_ric` → render joints.
* **TensorBoard**: `output_dir/tb/` (scalars for `loss`, all `l_*` components, `lr`, `step_time`).
* **Per-rank logs**: `output_dir/train_rank{N}.log` — uncaught exceptions are captured into the log via `sys.excepthook`.

---

## 7. Common knobs (CLI overrides)

Append `key=value` after `--config`:

| Override | Effect |
|---|---|
| `trainer.batch_size=64` | per-GPU batch size |
| `trainer.lr=1e-5` | learning rate |
| `trainer.num_steps=500000` | total training steps |
| `trainer.ckpt_every=2000` | save more often |
| `trainer.viz_every=2000` | dump samples more often |
| `trainer.mixed_precision=fp32` | turn off AMP (debug) |
| `trainer.grad_accum=2` | effective batch ×2 without extra memory |
| `data.window_size=160` | shorter random windows |
| `data.max_motion_length=160` | shorter pad cap (must be ≥ window_size) |
| `data.split_file=/path/to/train.txt` | alternate split |
| `data.num_workers=0` | debug with no workers |
| `trainer.resume_from=/path/latest.pt` | resume |

---

## 8. Smoke test (CPU, no GPU)

```bash
head -30 /home/jungbin_cho/HumanML3D/HumanML3D/train.txt > /tmp/_split.txt

python -m kimodo.scripts.train_w_hml3d \
  --config configs/training/hml3d.yaml \
  trainer.num_steps=3 \
  trainer.log_every=1 \
  trainer.batch_size=1 \
  trainer.ckpt_every=0 trainer.viz_every=0 \
  trainer.mixed_precision=fp32 \
  data.num_workers=0 \
  data.split_file=/tmp/_split.txt \
  output_dir=/tmp/_hml3d_smoke
```

Expect three `step=N loss=...` lines with all seven component losses (including `l_fk`).

---

## 9. What is NOT trained / kept disabled

* **Text encoder is frozen** (`LLM2Vec` is loaded with `requires_grad=False`, called under `torch.no_grad()`, never in the optimizer).
* **Constraint training (Phase 2)** is supported (`trainer.phase=constraints` + `trainer.constraint_weights`) but disabled by default. HumanML3D doesn't ship the kinematic-constraint annotations that the BONES-SEED benchmark uses; Phase 2 here would still work via `ConstraintSampler`, which samples constraint masks directly from the GT features (full-body keyframes, root paths, end-effector keyframes), but you'd want to weigh whether that matches your evaluation pipeline.
* **LLM paraphrasing / motion stitching augmentations** from the KIMODO paper are intentionally NOT implemented here.
* **The kimodo skeleton's `fk()`** is **not** used during HumanML3D training (it's the wrong FK convention for these rotations). Use `kimodo/motion_rep/fk_hml3d.py` whenever you need joints from rotations.

---

## 10. Pointers

* `kimodo/data/humanml3d_text_motion.py` — segment-aware dataset.
* `kimodo/scripts/train_w_hml3d.py` — entrypoint. Reuses every helper from `kimodo/scripts/train.py` (denoiser build, diffusion, EMA, DDP, train step, checkpoint, text encoder, AMP).
* `kimodo/motion_rep/fk_hml3d.py` — chain-reset FK + bone-length derivation.
* `kimodo/scripts/viz_fk_hml3d.py` — FK-vs-GT visualisation tool.
* `benchmark/humanml3d_to_kimodo.py` — bidirectional 263↔273 conversion (used at data-prep time and for rendering predicted features back to HumanML3D format).
