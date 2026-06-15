# SOMA Proportional (shape-aware) data pipeline

How `/weka/jungbin/seed/soma_proportional_motions_20fps` is produced, audited,
packed, and turned into normalization stats for **shape-aware** Kimodo
text-to-motion training.

This is the per-actor-body counterpart of the uniform pipeline. The uniform
pipeline renders every actor against one canonical ~1.74 m SOMA body; this one
preserves each actor's true bone lengths so the model can be conditioned on body
shape (see `kimodo/scripts/train_skel_aware.py`).

---

## At a glance

```
 source BVH (per-actor bodies)
   /weka/jungbin/seed/soma_proportional/bvh/<date>/<move>__A<actor>[_M].bvh
        │
        │  (1) benchmark/create_data_proportional.py      ← BVH → Kimodo NPZ
        ▼
 per-motion NPZ  (adds neutral_joints (77,3))
   /weka/jungbin/seed/soma_proportional_motions_20fps/<date>/<stem>.npz   (142,220 files)
        │
        │  (2) benchmark/nan_audit_proportional.py        ← flag NaN-tainted NPZs
        ▼
   /home/jungbin_cho/_nan_audit_proportional.json   (679 NaN / 0 load-error / 141,541 clean)
        │
        │  (3) kimodo/scripts/pack_bones_seed_motions_proportional.py   ← dedup + concat
        ▼
   /weka/jungbin/kimodo_caches/bones_seed_proportional_raw.pt   (~54 GB, 141,541 motions, 522 actors)
        │
        │  (4) kimodo/scripts/compute_motion_stats_proportional.py      ← per-actor-FK stats
        ▼
   /weka/jungbin/kimodo_caches/stats/proportional/{global_root,local_root,body}/{mean,std}.npy
```

The training config `configs/training/bones_seed_skel_aware.yaml` points at the
pack (3) and the stats (4).

---

## Path map (authoritative)

| Role | Path |
|---|---|
| Source per-actor BVH tree | `/weka/jungbin/seed/soma_proportional/bvh/` |
| **Per-motion NPZ output** | **`/weka/jungbin/seed/soma_proportional_motions_20fps/`** (142,220 NPZ) |
| NaN audit report | `/home/jungbin_cho/_nan_audit_proportional.json` |
| Actor-dedup motion pack | `/weka/jungbin/kimodo_caches/bones_seed_proportional_raw.pt` (~54 GB) |
| Shape-aware stats | `/weka/jungbin/kimodo_caches/stats/proportional/` |
| Canonical (uniform) stats, for comparison | `/weka/jungbin/Kimodo-SOMA-SEED-v1.1/stats/motion/` |

The uniform analogue of the NPZ tree is
`/weka/jungbin/seed/soma_uniform_motions_20fps/` (produced by the unchanged
`benchmark/create_data.py`).

---

## Step 1 — BVH → Kimodo NPZ (`create_data_proportional.py`)

Walks the source BVH tree and writes one Kimodo motion NPZ per BVH, mirroring the
`<date>/` subdirectory layout. Resamples to 20 fps.

**Defaults** (in the script):
- `--dataset /weka/jungbin/seed/soma_proportional`
- `--output  /weka/jungbin/seed/soma_proportional_motions_20fps`
- `--fps 20`

**What makes it shape-aware** (vs `create_data.py`): instead of rendering
against the canonical `SOMASkeleton77()` body, each BVH's HIERARCHY `OFFSET`s are
read with the `Bvh` parser and converted into Kimodo's Y-up frame with

```
pos[j] = pos[parent[j]] + global_rot_offsets[parent[j]] @ bvh_offset_cm[j]
```

(`build_actor_neutrals_in_kimodo_frame`). The resulting `(77, 3)` rest pose
**overrides** `skeleton.neutral_joints` for that file, so all downstream FK,
canonicalization, and contact detection run against the actor's real body. The
saved NPZ therefore gains a **`neutral_joints` (77, 3) float32** key that the
uniform NPZs do not have. All other keys (`local_rot_mats`, `root_positions`,
`posed_joints`, `global_rot_mats`, `foot_contacts`, `smooth_root_pos`,
`global_root_heading`) are identical in schema to the uniform NPZs.

**Hardening guards** (raise rather than silently corrupt): missing-joint check,
per-bone-length parity (`‖actor[j]−actor[parent]‖ ≈ ‖bvh_offset[j]‖`, 1e-5 m),
Hips/root parity (`posed[Hips] == root_positions`), and integer-FPS-ratio check
(1e-3 rel tol). See `_create_data_proportional_changes.md` for the full
reviewer-driven hardening table and validation numbers (FK round-trip ≤ 4e-7 m,
standard-BVH self-check 4.5e-08 m).

**Run** (sharded across machines; ~12 h full run, I/O-bound on weka):

```bash
PYTHONPATH=/home/jungbin_cho/kimodo_open \
  python benchmark/create_data_proportional.py \
    --workers 16 --num-shards 4 --shard-id 0   # repeat shard-id 1..3 on other nodes
```

> Known residual: `to_standard_tpose` reuses the canonical `global_rot_offsets`
> for every actor, leaving ~0.2 cm/s foot-skating vs the uniform set. Below
> visible-quality threshold; documented in `_create_data_proportional_changes.md` §5.

---

## Step 2 — NaN audit (`nan_audit_proportional.py`)

Some source BVHs contain NaN frames (e.g. the A497 batch went NaN at frame 833),
which propagate into `local_rot_mats` / `posed_joints`. This scan flags every
NPZ with a NaN in any critical field so the packer can skip them.

```bash
PYTHONPATH=/home/jungbin_cho/kimodo_open \
  python benchmark/nan_audit_proportional.py --workers 16 \
    --out /home/jungbin_cho/_nan_audit_proportional.json
```

**Result on the current tree:** `clean=141,541  with_nan=679  load_error=0
total=142,220`. The JSON has `totals`, `nan_files` (path + per-key NaN counts),
and `load_error_files`.

---

## Step 3 — Pack (`pack_bones_seed_motions_proportional.py`)

Concatenates all clean NPZ motions into one mmap-able `.pt`, mirroring the
uniform packer's layout plus an **actor-deduplicated** shape table. The same
actor (e.g. `A001`, parsed from the `__A\d+` filename token) shares one body, so
`neutral_joints` are stored once per actor (522 unique) instead of per motion.

**Defaults:** `--data-root` = the NPZ tree, `--nan-report` =
`_nan_audit_proportional.json`, `--out` =
`/weka/jungbin/kimodo_caches/bones_seed_proportional_raw.pt`.

```bash
PYTHONPATH=/home/jungbin_cho/kimodo_open \
  python -m kimodo.scripts.pack_bones_seed_motions_proportional --workers 32
```

**Pack keys:**

| key | shape / dtype | notes |
|---|---|---|
| `names` | list[str], len 141,541 | motion stems in pack order |
| `offsets` | int64 `[N+1]` | cumulative frame counts |
| `local_rot_mats` | f32 `[20,667,071, 77, 3, 3]` | ~54.6 GiB |
| `root_positions` | f32 `[20,667,071, 3]` | |
| `actor_names` | list[str], len 522 | e.g. `A001` |
| `actor_neutrals` | f32 `[522, 77, 3]` | one rest pose per actor |
| `motion_actor_idx` | int32 `[141,541]` | motion → actor-index lookup |

A per-actor consistency check raises if any actor's `neutral_joints` differ
across its motions by >1e-5 m. Output is ~54 GB; packing takes a few minutes
once the NPZs are warm in page cache.

Look up a motion's body with:
```python
neutrals_77 = pack["actor_neutrals"][pack["motion_actor_idx"][i]]   # (77, 3)
```

---

## Step 4 — Normalization stats (`compute_motion_stats_proportional.py`)

Streams every motion in the pack through `KimodoMotionRep` **with the actor's
per-sample neutrals** and accumulates mean/std (Welford) over all valid frames.
Because FK runs against each actor's real bones, the position/velocity feature
blocks shift relative to the canonical (uniform) stats — so these **must be
recomputed; do not reuse the canonical stats**.

```bash
PYTHONPATH=/home/jungbin_cho/kimodo_open \
  python -m kimodo.scripts.compute_motion_stats_proportional \
    --packed-motions-path /weka/jungbin/kimodo_caches/bones_seed_proportional_raw.pt \
    --output-dir /weka/jungbin/kimodo_caches/stats/proportional \
    --fps 20 --num-workers 16 --random-heading \
    --canonical-stats-dir /weka/jungbin/Kimodo-SOMA-SEED-v1.1/stats/motion
```

Output layout (drop-in for the `stats_path` config knob):
```
stats/proportional/global_root/{mean,std}.npy   (5,)
stats/proportional/local_root/{mean,std}.npy    (4,)
stats/proportional/body/{mean,std}.npy          (364,)
```

**Last run:** 141,541 motions / 20,667,071 frames in 482 s, 0 failures.
Delta vs canonical confirms shape genuinely matters:

| block | max \|Δmean\| | max \|Δstd\| |
|---|---|---|
| `global_root` | 4.4e-2 | 1.2e-1 |
| `local_root` | 4.4e-2 | 6.5e-2 |
| `body` | **4.7e-1** | 1.8e-1 |

(The 0.47 m `body` mean shift is on peripheral joint positions — hands/head —
exactly where bone-length differences accumulate.)

---

## Consuming this in training

`configs/training/bones_seed_skel_aware.yaml` wires it together:
- `data.packed_proportional_motions_path` → the Step 3 pack
- `data.nan_audit_path` → the Step 2 report (drops the 679 at index build)
- `stats_path` → the Step 4 stats dir
- dataset class `SOMABonesSeedDatasetShapeAware` emits `neutral_joints` (sliced
  to the 30-joint SOMA subset) per sample; `ShapeEncoder` turns it into a prefix
  conditioning token, and `KimodoLoss` runs FK against it.

See the repo's shape-aware training plan / `kimodo/scripts/train_skel_aware.py`
for the model side.
