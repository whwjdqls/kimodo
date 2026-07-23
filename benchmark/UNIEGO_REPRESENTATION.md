# UniEgoMotion-style head-centric representation (HumanML3D + SOMA)

A round-trip converter between the **Kimodo** motion representation and a
**UniEgoMotion-style head-centric SE(3)** representation, built to test whether
the head-centric encoding is a good target for **text-to-motion**. Validated on
HumanML3D (22 SMPL joints; an MDM trained on it beat the native-263 and kimodo-273
baselines on FID — see §10) and **generalized to SOMA** (BONES-SEED, 30 joints;
**§12–14**), with Nymeria as a documented follow-up.

- Converter (HML3D): `benchmark/kimodo_to_uniego.py`  (`kimodo_to_uniego`, `uniego_to_kimodo` — skeleton-agnostic)
- Converter (SOMA):  `benchmark/soma_to_uniego.py`
- Visualization:     `benchmark/viz_uniego_roundtrip.py` (HML3D); `benchmark/uniego_viz_soma/` (SOMA)
- Builds on:         `benchmark/humanml3d_to_kimodo.py` (HumanML3D ⇄ Kimodo, already verified)
- Shape-awareness analysis (the Nymeria research question): **§14**

---

## 1. What UniEgoMotion's representation is

UniEgoMotion (Patel et al., 2025 — https://github.com/chaitanya100100/UniEgoMotion)
is an egocentric motion model with a **head-centric** representation. Instead of a
pelvis-centric, kinematic-chain encoding (parent-relative joint rotations), it:

1. Runs forward kinematics to get every joint's **global SE(3) transform** — the
   head `M^h ∈ R^{4×4}` and the other joints `M^j ∈ R^{21×4×4}`. This **removes the
   kinematic-chain dependency**: each joint is described directly in world space.
2. Derives a per-frame **canonical reference frame** `cM` by projecting the head
   transform onto the floor — keeping only **yaw**, dropping pitch/roll and height.
3. Expresses the motion as `(cM, cM⁻¹·M^h, cM⁻¹·M^j)`: the canonical frame plus
   each joint relative to it. For trajectory invariance, `cM` is stored as a
   **residual** w.r.t. the previous frame, `cM_{t-1}⁻¹·cM_t`.

Their on-disk feature (v4) is `233 = 198 + 9 + 12 + 12 + 1 + 1`:
22 joints × (6D rot + 3 trans) = 198, residual canonical frame = 9, left/right
**hand-PCA** = 12+12 (SMPL-X only), and 2 foot contacts.

## 2. How we adapt it to Kimodo / HumanML3D

UniEgoMotion targets egocentric SMPL-X capture; we adapt the *idea*, not the bytes:

| Aspect            | UniEgoMotion        | Ours (HumanML3D)                    |
|-------------------|---------------------|-------------------------------------|
| Up axis           | +Z up               | **+Y up** (HumanML3D / Kimodo)      |
| Head anchor       | left-eye joint (23) | **head joint 15** (no eyes in SMPL-22) |
| Hand pose         | 12+12 PCA           | **dropped** (no hand articulation)  |
| Shape (betas)     | 10                  | **dropped**                         |
| Foot contacts     | 2 (L/R feet)        | **4** (Kimodo's, for lossless round-trip) |
| Forward axis      | (their convention)  | head **z-column** = +Z forward      |

**Why this is cheap:** the Kimodo dict (and the on-disk
`/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/*.npz`) already stores per-joint
**global rotation matrices** (`global_rot_mats`, T×22×3×3) and **global joint
positions** (`posed_joints`, T×22×3). Those *are* the global SE(3) transforms
UniEgoMotion gets from FK — so the converter is pure matrix algebra: **no forward
kinematics, no model, no GPU.**

## 3. Our feature layout — `211 = 198 + 9 + 4`

```
[0:198]    per-joint local SE(3), j = 0..21:
             matrix_to_cont6d(R_local_j) (6)  ++  trans_local_j (3)
[198:207]  delta canonical frame (residual trajectory):
             matrix_to_cont6d(R_delta)   (6)  ++  trans_delta   (3)
[207:211]  foot_contacts (4, passthrough from Kimodo)
```

- 6D rotation uses the **HumanML3D convention** (matrix columns 0 and 1), via the
  `matrix_to_cont6d` / `cont6d_to_matrix` helpers reused from `humanml3d_to_kimodo.py`.
- The delta slot at **frame 0 stores the absolute** `cM[0]`; for `i ≥ 1` it stores
  `cM_{i-1}⁻¹·cM_i`.
- **Auxiliary (not in the 211-D core):** `velocities` (T×22×3) and `r_rot_quat`
  (T×4) are carried alongside so the end-to-end HumanML3D 263-D round-trip stays
  exact. The 211-D core is deliberately velocity-free, faithful to UniEgoMotion.
  (`kimodo_to_humanml3d` needs world velocities for HML3D dims [193:259], which
  HML3D's joint recovery never reads; carrying `r_rot_quat` avoids a ±π half-angle
  wrap on ~5% of clips.)

## 4. The math

**Forward (kimodo → uniego), per frame `t`:**
```
M[t,j]   = [[R[t,j], p[t,j]], [0,0,0,1]]            # global SE(3), from the kimodo dict
fwd      = R[t, 15][:, 2]                           # head z-column (+Z forward)
           → fallback to R[t, 0][:, 2] when ‖fwd_xz‖ < forward_eps (1e-2)
yaw      = atan2(fwd_x, fwd_z)
cM[t]    = [[R_y(yaw), (p_head_x, 0, p_head_z)], [0,0,0,1]]   # height zeroed (cM == invtsfm_T)
tsfm[t]  = cM[t]⁻¹                                  # analytic SE(3) inverse
local_T[t,j] = tsfm[t] · M[t,j]
delta[0] = cM[0];   delta[i] = tsfm[i-1] · cM[i]   ( = cM[i-1]⁻¹·cM[i] )
```

**Inverse (uniego → kimodo):**
```
cM[0] = delta[0];   cM[i] = cM[i-1] · delta[i]     # cumulative compose
M[t,j] = cM[t] · local_T[t,j]
global_rot_mats = M[:,:,:3,:3];  posed_joints = M[:,:,:3,3];  root_positions = posed_joints[:,0]
```

The exact choice of "forward" axis **cancels** in the round-trip (applied in `tsfm`,
removed via the stored `delta`), so decode **never re-derives yaw** — it replays
`cM` purely from the residual channel. The forward-axis choice (head z-column,
empirically tracking motion direction with cosine ≈ 0.99 on walking clips) only
makes the canonical frame *interpretable and cross-sequence consistent*.

## 5. Verification

Two independent checks (`--verify-roundtrip N`):

- **Check A** (isolates the new code): `kimodo → uniego → kimodo`, diff arrays.
- **Check B** (end-to-end): `263 → humanml3d_to_kimodo → kimodo_to_uniego →
  uniego_to_kimodo → kimodo_to_humanml3d → 263'` (HML3D's never-recovered
  last-frame entries `[-1,0]`, `[-1,1:3]`, `[-1,193:259]` masked out).

Results over a 275-file scan (random + the 5 longest at T=201 + 20 mirrored clips):

| Check | Quantity                         | Worst \|Δ\|   | Tolerance |
|-------|----------------------------------|--------------|-----------|
| A     | `posed_joints` (m)               | **1.7e-5**   | < 1e-5–1e-4 |
| A     | `global_rot_mats`                | **~1e-5**    | < 1e-4    |
| A     | `foot_contacts`, `velocities`    | **0** (exact)| 0         |
| B     | HumanML3D 263-D vector           | **2.1e-5**   | < 1e-3    |

i.e. the representation is **exactly invertible** to float32 noise. Reproduce with:

```bash
/home/jungbin_cho/miniconda3/envs/kimodo/bin/python benchmark/kimodo_to_uniego.py \
    --verify-roundtrip 12 --skip-write
```

## 6. Visualization

```bash
python benchmark/viz_uniego_roundtrip.py                 # default 6 clips
python benchmark/viz_uniego_roundtrip.py --ids 000004 M000123
python benchmark/viz_uniego_roundtrip.py --n 10
```

Writes `benchmark/uniego_viz/<id>_uniego_roundtrip.mp4`: **original** Kimodo joints
(left, blue) vs **round-tripped** joints (right, red), with the per-clip max
position error in the caption. The two skeletons are visually identical (recon
~1e-5 m). Rendering uses `kimodo/scripts/render_hml3d.render_sidebyside` (CPU
matplotlib, SMPL-22 chain).

## 7. Full-dataset conversion

CPU-only, ~29,668 files of light matrix ops — run off the login node. This is
I/O-bound more than CPU-bound, so a small request backfills near-instantly and
still finishes in ~10 min; don't over-request cores (16 just makes Slurm wait for
that many free):

```bash
srun -p a2 --gres=gpu:0 --cpus-per-task=4 --mem=8G \
  /home/jungbin_cho/miniconda3/envs/kimodo/bin/python benchmark/kimodo_to_uniego.py \
    --input-dir  /home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep \
    --output-dir /home/jungbin_cho/HumanML3D/HumanML3D/uniego_rep \
    --workers 4
```

The CLI skips already-written outputs (unless `--overwrite`), so a re-submit
resumes cleanly from where a previous run stopped.

Each output `uniego_rep/<id>.npz` stores `features` (T×211) plus aux
`velocities`, `foot_contacts`, `r_rot_quat`. Train/val/test ids reuse HumanML3D's
`train.txt` / `val.txt` / `test.txt`.

## 8. Direct UniEgo → HumanML3D

`uniego_to_humanml3d(uniego_dict)` returns the original HumanML3D 263-D vector. It
simply chains `uniego_to_kimodo` → `kimodo_to_humanml3d` (there is no shortcut that
bypasses Kimodo — the HML3D inverse consumes exactly the world-frame quantities the
uniego decode reconstructs). Verified against the **original** `new_joint_vecs`
(not just the kimodo intermediate): worst |Δ| **1.7e-5** across root / position /
rotation channels, on disk-loaded uniego files. So the uniego rep is a faithful,
fully-invertible re-parameterization of HumanML3D.

## 9. Normalization stats

Computed by `benchmark/compute_uniego_stats.py` over the train split
(`train.txt`, 23,384 ids; 3.26M frames) → flat `(211,)` vectors:

- `/home/jungbin_cho/HumanML3D/HumanML3D/Mean_uniego.npy`
- `/home/jungbin_cho/HumanML3D/HumanML3D/Std_uniego.npy`

(360 of the 23,384 split ids have no data — a **pre-existing HumanML3D gap**:
the identical 360 are missing from `kimodo_rep` and the original `new_joint_vecs`
too.)

**11 dimensions are structurally constant** (std floored to 1 → normalization is a
no-op there); they are exact invariants of the representation, a useful correctness
check:

| dims | channel | why constant |
|------|---------|--------------|
| 1, 3, 4, 5     | joint0 (root) 6D rot | HML3D root rotation is pure-yaw `R_y(θ)`; its 6D is `[c,0,-s,0,1,0]` |
| 141, 143       | joint15 (head) trans x, z | head xy IS the canonical anchor → head local xz ≡ 0 (height varies) |
| 199,201,202,203| delta 6D rot | canonical frame is pure-yaw → same `[_,0,_,0,1,0]` structure |
| 205            | delta trans y | canonical frame height is zeroed by construction |

Regenerate: `python benchmark/compute_uniego_stats.py --workers 4` (CPU, ~2 min).

## 10. Training MDM on the uniego rep

An MDM run on the 211-D rep, mirroring the native-263 / kimodo-273 MDM recipes
(same 8-layer/4-head/dim-512 backbone, CLIP ViT-B/32, lr 1e-4, batch 64, 500k
steps, EMA 0.9999, warmup 2000, **fp32**, flat masked-L2 on x0). Only the
data/rep differ → apples-to-apples vs `runs/mdm_hml3d_native_fp32` and
`runs/mdm_hml3d_kimrep_fp32`.

New files:
- `kimodo/motion_rep/uniego.py` — `UniegoMotionRep` (211-D normalize/slice_dict),
  `uniego_world_joints_from_features` (decode → joints), `canonicalize_frame0`.
- `kimodo/data/humanml3d_uniego_text_motion.py` — dataset (loads `uniego_rep/*.npz`
  `features`, per-window frame-0 canonicalization).
- `configs/model/mdm_humanml3d_uniego_clip.yaml`, `configs/training/hml3d_uniego_mdm_clip.yaml`.
- `kimodo/scripts/train_w_hml3d_uniego.py` (reuses the native `MDMNativeL2Loss`).
- `submit_train_mdm_hml3d_uniego_fp32.sh` → `runs/mdm_hml3d_uniego_fp32`.

**Frame-0 canonicalization** (important): the stored rep keeps frame 0's
`canon_delta` as the absolute `cM[0]` (for the lossless round-trip), but each
*training window* sets frame-0 `canon_delta` to identity so every window starts
at origin/+Z — the analog of HML3D-native's frame-0 canonicalization. The stats
(§9) are computed the same way, so normalization matches what training sees.

Run:
```bash
sbatch submit_train_mdm_hml3d_uniego_fp32.sh
# smoke: python -m kimodo.scripts.train_w_hml3d_uniego --config configs/training/hml3d_uniego_mdm_clip.yaml \
#   output_dir=runs/_smoke data.split_file=<tiny> trainer.num_steps=25 trainer.viz_every=20 trainer.ckpt_every=0
```

## 11. Remaining follow-ups

- Run the benchmark eval on the trained uniego checkpoints (a TMR/FID/R-precision
  path that decodes 211-D → joints via `uniego_world_joints_from_features`).
- Decide whether the foot-contact channel should be predicted or supervised.

---

## 12. Generalizing to SOMA (BONES-SEED / Nymeria)

The conversion is **skeleton-agnostic** — the head-centric SE(3) math doesn't care
about joint count or topology, only about a per-joint `global_rot_mats`/`posed_joints`
pair, a head index, and a foot-contact count. `kimodo_to_uniego` / `uniego_to_kimodo`
take `head_idx` and `n_foot` and infer `J` (and the feature width `J*9 + 9 + n_foot`)
from the input. `UniegoMotionRep` infers `J` from `skeleton.nbjoints`.

| param | HumanML3D (SMPL-22) | SOMA-30 (BONES-SEED) |
|-------|---------------------|----------------------|
| J (joints) | 22 | 30 |
| head_idx | 15 | **6** ("Head") |
| root_idx | 0 (pelvis) | 0 (Hips) |
| n_foot | 4 | 4 |
| feature dim | 211 = 22·9+9+4 | **283 = 30·9+9+4** |
| up axis | +Y | +Y (same) |
| 6D convention | columns 0,1 | same |

**SOMA converter:** `benchmark/soma_to_uniego.py`. Source = raw SOMA `.npz`
(77-joint; already carry `global_rot_mats`, `posed_joints`, `foot_contacts`),
sliced to the SOMA-30 subset (the same 77→30 name map the dataset uses), then
`kimodo_to_uniego(head_idx=6, n_foot=4)`. Verified round-trip on bones_seed clips
(incl. mirrored): worst `posed_joints`/`global_rot_mats` |Δ| **2.0e-5**, foot
contacts exact.

```bash
# verify: python benchmark/soma_to_uniego.py --verify-roundtrip 12 --skip-write
# convert all: srun -p a2 --gres=gpu:0 --cpus-per-task=8 --mem=16G \
#   python benchmark/soma_to_uniego.py --workers 8 \
#     --input-dir /home/jungbin_cho/seed/soma_uniform_motions_20fps \
#     --output-dir /home/jungbin_cho/seed/soma_uniego_rep_20fps
# stats (283-D): python benchmark/compute_uniego_stats.py --n-joints 30 \
#     --rep-dir <soma_uniego> --split-file <soma_train_split> --out-mean ... --out-std ...
```

**Forward-axis nuance (SOMA):** HML3D head z-column tracks travel direction at
cosine ≈0.99 (the body is uniform-skeleton'd and faces +Z); SOMA mocap heads turn
independently, so on a slow walk clip the cosine is ≈0.69. This is *expected and
fine*: (a) the forward axis **cancels in the round-trip** (correctness unaffected —
verified 2e-5), and (b) a head-driven canonical frame *is* the egocentric signal
UniEgoMotion models. If a more travel-aligned canonical frame is wanted for
cross-sequence consistency, swap the head yaw for the kimodo hip-vector heading
(`kimodo/motion_rep/feature_utils.py` heading) — a one-line change to the encode's
`fwd`, decode is unaffected (it replays `cM` from the stored delta).

## 13. Dataset differences (why each needs care)

| Aspect | HumanML3D | BONES-SEED (SOMA) | NymeriaPlus |
|--------|-----------|-------------------|-------------|
| Source body model | SMPL → joints | SOMA (Momentum Human Rig) | SOMA + egocentric capture |
| Skeleton in rep | SMPL-22 | SOMA-30 (from 77) | SOMA-30 (expected) |
| Per-actor **shape** | **none** (uniform-skeleton'd to one actor) | **yes** (per-actor bones; "proportional" tree) | **yes**, and load-bearing |
| Camera / video / HMD | none | none (pure mocap) | **yes** (egocentric video + camera + head device) |
| On-disk kimodo rep | `kimodo_rep/*.npz` (273-D + joints) | raw `.npz` (rot+joints) / 369-D packs | (not present here) |
| Canonicalization | frame-0 faces +Z at origin | world frame (heading aug) | (egocentric, head-anchored) |

Practical upshot: HumanML3D and **uniform** BONES-SEED both encode a single
canonical body, so the plain uniego conversion captures pose-only (great for the
text-to-motion test). **Per-actor shape** lives in BONES-SEED's proportional tree
(per-actor `neutral_joints`) and is central to Nymeria.

> **NymeriaPlus converter now exists — see §15.** (Data lives on `/weka`, not in this
> repo tree.) The egocentric modalities (video, camera, floor grounding) are built out
> in `/home/jungbin_cho/nymeria_kimodo_pipeline/` (video mp4s + camera sidecars +
> manifest); the uniego motion rep for it is `benchmark/nymeria_to_uniego.py`.

## 14. Is the head-centric rep good for **shape-aware** generation?

Short answer: **yes, and arguably better-suited than a rotation rep for the
camera/video setting — with one caveat to engineer around.**

**Why it fits shape-aware / egocentric generation:**
- uniego stores each joint's **global position directly** (the translation block of
  the per-joint SE(3)). So **per-actor bone lengths are encoded implicitly** in the
  data — a taller actor's joints are simply farther apart. If `posed_joints` come
  from the per-actor (proportional) pipeline, the uniego rep **preserves that
  actor's shape for free** (the converter is shape-transparent: it copies whatever
  positions it's given).
- Camera/video supervision cares about **where joints are in image/world space** —
  exactly what uniego outputs. A rotation rep must FK through a skeleton to get
  positions; uniego gives them directly. And the head-anchored canonical frame
  aligns naturally with an egocentric (head-mounted) device.

**The caveat — no rigid-bone constraint:**
- Because positions are free per frame, uniego does **not** guarantee constant bone
  lengths over time, nor that generated positions match a *conditioned* target
  shape. Bone length is an emergent statistic of the prediction, not a hard
  invariant (unlike the kimodo rep, which is rigid-by-construction because positions
  come from FK on fixed `neutral_joints`).

**How to make uniego shape-aware (recommended design):**
1. **Condition** on the target body via the existing `ShapeEncoder` — it maps
   `neutral_joints (B,30,3)` → a prefix token and is **rep-agnostic** (conditions the
   denoiser, not the features), so it plugs into a uniego model unchanged
   (`kimodo/model/shape_encoder/shape_encoder.py`, used by `train_skel_aware.py`).
2. **Regularize** generated positions toward the conditioned shape: a bone-length
   loss `Σ |‖joint_c − joint_p‖ − target_len|` (target from the conditioned
   `neutral_joints`; reuse `derive_bone_lengths_from_world_joints` in
   `kimodo/motion_rep/fk_hml3d.py`), optionally plus a temporal bone-length-variance
   penalty so limbs don't wobble.
3. **Or decode shape-exact via the rotation view.** The uniego rep *also* stores
   per-joint **rotations** (the 6D in the local block — unused for the position
   decode). So one can decode `FK(rotations, target neutral_joints)` to get **rigid,
   shape-exact** joints — a head-centric analog of the kimodo rep's
   positions-vs-rotations ("Option A/B") choice. This gives an exactness guarantee
   when shape fidelity matters (e.g. aligning to video).

**Bottom line for Nymeria:** the head-centric rep is a good target for shape-aware,
egocentric, camera-grounded generation — it natively represents shape and position
and aligns with the head/device — provided you (a) condition on shape and (b) add a
bone-length-consistency regularizer (or use the rotation-view FK decode for
shape-exactness). The uniform-SOMA conversion here is shape-free (canonical body);
the shape-aware variant needs the proportional tree mounted (currently on `/weka`).

---

## 15. NymeriaPlus converter (shape-aware, ran on all 732)

`benchmark/nymeria_to_uniego.py` — the SOMA-30 uniego rep (**283-D**, head_idx 6,
n_foot 4) for the shape-aware NymeriaPlus motions at
`/weka/jungbin/nymeriaplus_kimodo_proportional/{Sxx}/{seq}.npz`. Unlike the raw
BONES-SEED SOMA `.npz` (which already carry global SE(3)), the proportional NPZs store
**local** rotations + a per-actor rest skeleton, so the converter runs FK first:

1. **FK (77)** `SOMASkeleton77.fk(local_rot_mats, root, neutral_joints)` →
   `global_rot_mats`, `posed_joints`. Runs on **this actor's** `neutral_joints`, so the
   joint positions carry per-actor bone lengths → the uniego rep is **shape-aware for
   free** (§14). 2. **slice** 77→30. 3. **feet** — per-frame *local-floor* contacts
   (below). 4. `kimodo_to_uniego(head_idx=6, n_foot=4)` → `features (T,283)`.

**Grounding (important, differs from the HML3D/SOMA path).** Positions are kept **RAW**
(`grounded=False`, the proportional NPZ's own convention — raw SLAM-world height).
Downstream windowed training grounds each window by the floor pipeline's per-slice
`ground_offset_y` (subtract from the per-joint trans-Y block; rotations + canonical
`delta` unaffected) — **exactly how the aligned video + camera + motion are grounded**
in `nymeria_kimodo_pipeline/video/manifest_video.jsonl`. `--ground` bakes whole-seq
`floor_offset` (lowest-floor) instead, but that mis-grounds multi-floor clips.

**Foot contacts vs a local floor.** The rep's plain `foot_detect_from_pos_and_vel`
gates on absolute foot height above a single global floor — wrong for raw / multi-floor
Nymeria (it zeroed ~40% of contacts: median contact_frac 0.015, 41% of seqs <1%).
`_robust_foot_contacts` instead estimates a **per-frame local floor** (10th-percentile
of min foot height over ±20 frames) and gates on (foot within 0.10 m of it) AND (speed
< 0.15 m/s) — grounding- and stairs-independent. Post-fix over all 732: contact_frac
**mean 0.69 / median 0.69 / p10 0.50, 0 seqs <1%**.

**Round-trip precision.** Per-window (training use, frame-0 re-canonicalized per §10):
**float32-exact ~2–9e-5 (PASS <1e-4)**. Whole-sequence cumulative decode drifts to
~1–2 mm on these ~19k-frame (16-min) clips — inherent to the residual-canonical-frame
encoding over very long sequences (BONES-SEED clips ≤201 frames hit 2e-5), physically
negligible (< the SMPL fit's ~1.6 mm/frame foot jitter). `--verify-roundtrip` reports
both.

**Output** `uniego_rep/{Sxx}/{seq}.npz`: `features (T,283)`, `foot_contacts (T,4)`,
`neutral_joints (30,3)` [SOMA-30, per-actor — feed the `ShapeEncoder`], `identity_coeffs
(1,10)`, `floor_offset`, `grounded` (False), `fps`, **`timestamps_us (T,)`** (so the rep
stays frame-aligned 1:1 with the ego video / camera / motion windows). All 732 motion
NPZs converted (0 errors, ~15 GB).

```bash
# verify (per-window + whole-seq):
python benchmark/nymeria_to_uniego.py --verify-roundtrip 6 --skip-write
# convert all 732 (CPU; ssh into an a3ultra node, do NOT srun a GPU node for CPU work):
python benchmark/nymeria_to_uniego.py --workers 24 --overwrite
```

---

## 16. Proportional BONES-SEED converter (shape-aware, ran on all 142,220)

`benchmark/soma_proportional_to_uniego.py` — the SOMA-30 uniego rep (**283-D**) for the
**shape-aware** BONES-SEED motions at `/weka/jungbin/seed/soma_proportional_motions_20fps`
→ `/weka/jungbin/seed/soma_proportional_uniegomotion_20fps` (date-subdir structure
preserved). Distinct from §12's `soma_to_uniego.py` (uniform / shape-free set) in two ways:

- The proportional NPZs were FK'd on **each actor's own rest skeleton**, so the
  precomputed `posed_joints` carry per-actor bone lengths → the uniego rep is
  **shape-aware for free**. We also **carry the per-actor `neutral_joints` (30,3)** into
  each output for downstream `ShapeEncoder` conditioning (the uniform set has none).
- The NPZs already store precomputed `global_rot_mats` + `posed_joints` + `foot_contacts`,
  and the BVH source is **floor-referenced** (min foot Y ≈ +0.01 m). So — unlike the raw
  Nymeria path (§15) — no FK, no grounding, and no foot re-detection: contacts pass
  straight through (contact_frac mean 0.75).

**Output `{date}/{name}.npz`:** `features (T,283)`, `foot_contacts (T,4)`,
`neutral_joints (30,3)` [SOMA-30, per-actor].

**Verification** (`_verify.json`, 200 random files): feature dim 283, **0 NaN**, stored
features == recomputed (Δ=0, deterministic), round-trip decode→joints `posed_joints`
**4.9e-5** / `global_rot_mats` 3.7e-5 (PASS <1e-4), `foot_contacts` == source (Δ=0).
Clips short (T 44–789) so no long-sequence drift. Reproduce:
`python benchmark/soma_proportional_to_uniego.py --verify-roundtrip 12 --skip-write`.

**Stats** (`_stats.json`, `Mean_uniego.npy` / `Std_uniego.npy`, 283-D): computed over
the **141,541 clean** clips (20,667,071 frames) by `compute_uniego_stats.py --n-joints 30`.
**7 structurally-constant dims** `[60,62, 271,273,274,275,277]` (std floored to 1.0) =
the expected invariants — head joint local x/z (the canonical anchor) + canonical-delta
pure-yaw 6D + zeroed-height — a correctness check.

**Data-quality caveat — 679 corrupt SOURCE clips (0.48%).** `_nan_files.txt` lists 679
files whose features contain NaN. The NaN is **pre-existing in the source**
(`soma_proportional_motions_20fps`'s own `local_rot_mats`/`global_rot_mats`/`posed_joints`
are NaN for these clips — verified) — the converter faithfully propagated it, it is not a
conversion bug. These are **excluded from the normalization stats** (`_clean_ids.txt` =
141,541 vs `_all_ids.txt` = 142,220) and should be filtered downstream.

**Run:** all 142,220 converted, 0 conversion errors, ~23 GB. CPU — ssh into an a3ultra
node (`--workers 32`), do NOT srun a GPU node for CPU work.
