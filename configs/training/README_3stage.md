# 3-stage text-to-motion model

> Variant of `train_w_hml3d.py` that inserts two **learned** stages in front of the kimodo motion denoiser:
> 1. a **mask generator** that produces a constraint mask from text,
> 2. a **constraint generator** that produces 3D constraint values for those masked slots,
> 3. the existing kimodo motion denoiser, conditioned on (mask, observed values).
>
> Hypothesis: predicting "where to keyframe" and "what to keyframe to" before generating the full motion makes text-to-motion easier than text-only conditioning alone. The same scaffolding cleanly accepts a Scene input in Stage 1 later.

---

## 1. Pipeline

Forward path, used identically at training and inference:

```
text_feat        = LLM2Vec(text)                       # frozen, shared across stages
mask_logits      = MaskGenerator(text_feat, pad_mask)             # (B, T, D)
mask_probs       = sigmoid(mask_logits)                # CONTINUOUS in [0,1]
values           = ConstraintGenerator(text_feat, mask_probs, pad_mask)  # (B, T, D)
observed_motion  = mask_probs * values                 # gated values
pred_clean       = Denoiser(motion_t, ..., motion_mask=mask_probs,
                                          observed_motion=observed_motion)
```

`D = motion_rep_dim` (273 for HumanML3D / 22 joints).

**Important:** there is no ground-truth mask anywhere in this design. The mask is a latent the model learns end-to-end via gradient signal from the motion-reconstruction loss.

---

## 2. Losses

Only **one** loss uses ground-truth motion (the rest are regularisers / dense auxiliary signals).

| Loss | Description | Default weight |
|---|---|---|
| `L_motion` | Existing 7-component kimodo loss (smooth-L1 on each feature block + chain-reset γ₇ FK). Backprops through Stage 3 and via `observed_motion = mask × values` all the way to Stages 1 and 2. **This is the only term that requires GT motion.** | `w_motion = 1.0` |
| `L_sparsity` | `mean(mask_probs)` over valid frames. With `sparsity_target > 0`, becomes a hinge `max(0, mean − target)` so the model is free to use up to that fraction of slots. **Prevents trivial collapse to `mask = 1`** (otherwise `observed_motion = values = GT` makes Stage 3 a copy and bypasses Stage 1). | `w_sparsity = 5.0`, `sparsity_target = 0.05` |
| `L_stage2_aux` (optional) | `smooth_L1(values, GT_motion)` over all valid frames. Gives Stage 2 a dense per-frame regression signal independent of the mask. **The mask is not used in this loss**, so it does not leak any "GT mask" information. Set `w_stage2_aux = 0` to disable. | `w_stage2_aux = 1.0` |

Total: `loss = w_motion * L_motion + w_stage2_aux * L_stage2_aux + w_sparsity * L_sparsity`.

### Sparsity matters

If you set `w_sparsity = 0`, the model has a trivial free-lunch optimum:
> `mask_probs = 1` everywhere ⇒ `observed_motion = values` ⇒ Stage 3 input is fully constrained ⇒ Stage 3's job collapses to a copy from `observed_motion` ⇒ `L_motion → 0` while `L_stage2_aux` pushes `values → GT_motion`. The whole 3-stage architecture degenerates into "Stage 2 is a text-to-motion regressor, Stage 3 is identity".

`L_sparsity` is what forces the model to use **few** constrained slots, which is the only configuration where the architecture actually does something interesting.

### Mask CFG-style dropout

Per sample with probability `mask_drop_prob`, the entire `(mask, observed_motion)` is zeroed out so Stage 3 keeps a usable text-only mode. Defaults to `0.1`, mirroring the existing text-CFG drop.

---

## 3. Files

| Path | Role |
|---|---|
| `kimodo/model/mask_generator.py` | Stage 1: text → `(T, D)` mask logits. |
| `kimodo/model/constraint_generator.py` | Stage 2: (text, mask) → `(T, D)` values. |
| `kimodo/model/three_stage.py` | `ThreeStageKimodo(mask_gen, constraint_gen, denoiser)` — single `nn.Module` so one DDP wraps everything trainable. |
| `kimodo/scripts/train_3stage_hml3d.py` | Training entrypoint (HumanML3D-in-kimodo features). |
| `configs/model/kimodo_3stage_humanml3d_small.yaml` | Sub-model schema (Stage 1 / Stage 2 / Stage 3 + diffusion). |
| `configs/training/hml3d_3stage.yaml` | Training config (data + trainer + viz + loss weights). |

All other helpers (kimodo loss with chain-reset FK γ₇, EMA, DDP setup, text encoder loading, AMP, checkpoint utilities) are imported unchanged from `kimodo/scripts/train.py`.

---

## 4. Run

### Single GPU
```bash
conda activate kimodo
cd /home/jungbin_cho/kimodo
python -m kimodo.scripts.train_3stage_hml3d \
  --config configs/training/hml3d_3stage.yaml \
  output_dir=/weka/jungbin/kimodo_runs/hml3d_3stage_v1
```

### 8 GPUs

Pre-download the text encoder once (avoids the HF cache races we hit when 8 ranks load concurrently — see the 2-stage README for the recipe). Then:

```bash
KIMODO_TEXT_ENCODER_LOCAL_DIR=/weka/jungbin/hf_models \
torchrun --standalone --nproc_per_node=8 --redirects 3 --tee 3 \
  -m kimodo.scripts.train_3stage_hml3d \
  --config configs/training/hml3d_3stage.yaml \
  trainer.batch_size=32 \
  output_dir=/weka/jungbin/kimodo_runs/hml3d_3stage_v1
```

Smaller default batch size than the 2-stage path: there are now **three** transformers in memory (Stage 1 ≈ 256-dim/4-layer, Stage 2 ≈ 384-dim/6-layer, Stage 3 ≈ 512-dim/8-layer) plus the frozen LLM2Vec text encoder. Bump back to 64 if memory allows.

---

## 5. Warm-starting Stage 3

The 3-stage model trains faster (and is less likely to collapse) if Stage 3 starts from a denoiser that already knows how to use `(mask, observed_motion)` conditioning. The 2-stage `train_w_hml3d.py` pipeline trains exactly that. To warm-start:

```yaml
# in hml3d_3stage.yaml
init_motion_denoiser_safetensors: /path/to/2stage_run/safetensors_or_pt_with_denoiser_weights
```

The training script tries to strip a `motion_denoiser.` prefix when present and falls back to non-strict loading.

---

## 6. Inference

`ThreeStageKimodo.sample_constraint_pack(text_feat, ..., mask_threshold=None)`:

```python
text_feat = text_encoder([prompt])
pad_mask  = torch.ones(1, num_frames, dtype=torch.bool, device=device)

pack = three_stage.sample_constraint_pack(
    text_feat, text_pad_mask, pad_mask,
    mask_threshold=None,    # None = use continuous sigmoid (matches training)
                            # 0.5  = threshold to binary keyframe-style mask
    mask_override=None,     # or supply a user/scene mask here
)
mask      = pack["mask"]              # (B, T, D)
values    = pack["values"]            # (B, T, D)
observed  = pack["observed_motion"]   # mask * values
```

Then run the existing DDIM loop on the Stage 3 denoiser with `motion_mask=mask` and `observed_motion=observed`. The viz code in `train_3stage_hml3d.py:viz_3stage_samples` does this end-to-end every `trainer.viz_every` steps and saves both the generated features and the mask used.

To pass user/scene constraints directly:

```python
user_mask = ...               # (B, T, D) — set 1 at slots the user constrains
pack = three_stage.sample_constraint_pack(
    text_feat, text_pad_mask, pad_mask,
    mask_override=user_mask,
)
```

This bypasses Stage 1 entirely. Useful at deployment time when constraints are externally specified.

---

## 7. Visualisation

Every `trainer.viz_every` steps (rank 0 only) the script dumps `step{N}_{i}_{prompt}.npz` files under `output_dir/viz/`. Each NPZ has:

- `features`: `(T, 273)` raw kimodo features (already un-normalised).
- `mask`: `(T, 273)` the mask used during sampling (continuous if `viz.mask_threshold=null`, binary otherwise).
- `prompt`: the text prompt.

To render the motion, run the standard HumanML3D-side decoding (see the 2-stage README §6). To visualise the mask itself, the `(T, 273)` array tells you which features were considered "constraints" at each frame — useful for sanity-checking that Stage 1 picks out interpretable patterns (e.g., concentrating density on root xz for "walk forward" prompts).

---

## 8. Knobs

CLI dot-overrides after `--config`:

| Override | Effect |
|---|---|
| `trainer.batch_size=64` | per-GPU batch (default 32 because three transformers + frozen Llama-3) |
| `trainer.w_sparsity=10.0` | stronger sparsity push (fewer constrained slots) |
| `trainer.sparsity_target=0.02` | tighter mask budget (≤ 2% of slots) |
| `trainer.w_stage2_aux=0.0` | disable Stage 2 aux supervision |
| `trainer.mask_drop_prob=0.0` | disable mask CFG drop (Stage 3 always sees the constraints) |
| `init_motion_denoiser_safetensors=…` | warm-start Stage 3 from a 2-stage checkpoint |
| `viz.mask_threshold=0.5` | binarise the mask at inference time |

---

## 9. Why this might work (the hypothesis)

The current kimodo text-to-motion path conditions a diffusion denoiser on text alone. Hard texts ("a person dances with feet alternating in syncopation") give the denoiser a single high-level signal and no per-frame anchors. Inserting a `text → keyframe mask → keyframe values → motion` chain gives the denoiser two pieces of conditioning:

1. **Which positions / channels are anchored** (the mask).
2. **What those anchored values should be** (the values).

If Stages 1 and 2 learn to pick out informative anchors (e.g., foot contacts at specific frames, hand positions at a wave's peak, root path waypoints), the motion model gets a much sharper conditioning signal than text alone. The architecture is identical to how kimodo already handles user-supplied keyframe constraints at inference — Stages 1+2 are essentially learning to *generate* those constraints from text.

---

## 10. Extending to scenes (future)

The `MaskGenerator.forward` signature is `(text_feat, text_pad_mask, pad_mask) → mask_logits`. To add a Scene input later, the natural change is:

```python
class MaskGenerator(nn.Module):
    def forward(self, text_feat, text_pad_mask, pad_mask,
                scene_feat=None, scene_pad_mask=None):
        ...
```

Concatenate `scene_feat` to the prefix tokens (alongside `text_feat`); the rest of the pipeline is unchanged. `ConstraintGenerator` and `ThreeStageKimodo.predict_mask_and_values` would forward the scene through similarly. Nothing about the training objective changes — `L_motion` still backprops the whole chain.

---

## 11. CPU smoke test

```bash
head -20 /home/jungbin_cho/HumanML3D/HumanML3D/train.txt > /tmp/_split.txt

python -m kimodo.scripts.train_3stage_hml3d \
  --config configs/training/hml3d_3stage.yaml \
  trainer.num_steps=2 trainer.log_every=1 trainer.batch_size=1 \
  trainer.ckpt_every=0 trainer.viz_every=0 trainer.mixed_precision=fp32 \
  data.num_workers=0 \
  data.split_file=/tmp/_split.txt \
  output_dir=/tmp/_3stage_smoke
```

Expect log lines like:
```
step=1 loss=3.91 ... | motion=0.78 stage2_aux=0.89 sparsity=0.45 mask_density=0.50
```
At step 0 the mask logits are random, so `mask_density ≈ 0.5` and `sparsity = max(0, 0.5 − target) = 0.45`. As training progresses `mask_density` should drop toward `sparsity_target` while `motion` decreases.
