"""Profile SOMABonesSeedDataset.__getitem__ to find the real per-item bottleneck.

Times each section (load/slice, canonicalize, random-heading, normalize) over
many fetches, for both the Tier-2 features pack and the plain NPZ path. Run on
a COMPUTE node (tmux 0), NOT login.

    conda activate kimodo
    python profile_bones_getitem.py
"""
import time
import torch

from kimodo.data import SOMABonesSeedDataset
from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton30

torch.set_num_threads(1)  # mimic a single dataloader worker

DATA_ROOT = "/home/jungbin_cho/seed/soma_uniform_motions_20fps"
SPLIT = "/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths_small.txt"
CACHE_INDEX = "/home/jungbin_cho/kimodo_caches/seg_index_small.json"
STATS = "/home/jungbin_cho/Kimodo-SOMA-SEED-v1.1/stats/motion/"
FEATS_PACK = "/home/jungbin_cho/kimodo_caches/bones_seed_small_feats.pt"
N = 300

skeleton = SOMASkeleton30()
motion_rep = KimodoMotionRep(skeleton=skeleton, fps=20, stats_path=STATS)

common = dict(
    data_root=DATA_ROOT,
    natural_csv_path="/home/jungbin_cho/seed/metadata/seed_metadata_v004.csv",
    temporal_labels_path="/home/jungbin_cho/seed/metadata/seed_metadata_v002_temporal_labels.jsonl",
    multi_timeline_path="/home/jungbin_cho/seed/multi_timeline.jsonl",
    train_split_path=SPLIT,
    fps=20, min_frames=10, include_mirrored=True, random_heading_aug=True,
    skeleton=skeleton, motion_rep=motion_rep, normalize=True,
    cache_index=CACHE_INDEX, seed=0,
)


def time_sections(ds, label):
    """Manually replay __getitem__ steps with per-section timers."""
    secs = {"load": 0.0, "canon": 0.0, "heading": 0.0, "norm": 0.0, "total": 0.0}
    n_ok = 0
    sl = motion_rep.slice_dict["global_root_heading"]
    for i in range(N):
        idx = i * 7 % len(ds)
        t_all = time.perf_counter()
        source = ds.SOURCES[idx % 3]
        entry = ds._pools[source][(idx // 3) % len(ds._pools[source])]

        t0 = time.perf_counter()
        feats = None
        if ds._packed_features:
            feats = ds._load_features_segment(entry)
        if feats is None:
            lr, rp, T = ds._load_segment(entry)
            if T < ds.min_frames:
                continue
            feats = ds._features_with_canonicalize(lr, rp, T)
            canon_in_load = True
        else:
            canon_in_load = False  # canonicalize is inside _load_features_segment
        secs["load"] += time.perf_counter() - t0

        # canonicalize is bundled into load for both paths; we measure the
        # extra runtime rotates separately below.
        t0 = time.perf_counter()
        feats, _ = ds._apply_random_heading(feats)
        secs["heading"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        feats = motion_rep.normalize(feats)
        secs["norm"] += time.perf_counter() - t0

        secs["total"] += time.perf_counter() - t_all
        n_ok += 1

    print(f"\n=== {label}  ({n_ok} fetches, T~{feats.shape[0]} frames) ===")
    for k in ["load", "heading", "norm", "total"]:
        print(f"  {k:10s}: {secs[k]/n_ok*1000:7.2f} ms/item  "
              f"({secs[k]/secs['total']*100:5.1f}%)")
    print(f"  → batch64/8workers est: {secs['total']/n_ok*1000*8:.0f} ms/batch")


# ---- Tier 2: features pack ----
ds2 = SOMABonesSeedDataset(packed_features_path=FEATS_PACK, **common)
time_sections(ds2, "Tier 2 (features pack)")

# ---- Plain NPZ path ----
ds0 = SOMABonesSeedDataset(**common)
time_sections(ds0, "Plain NPZ (motion_rep at runtime)")

# ---- Micro-bench the individual motion_rep ops on a fixed feature tensor ----
print("\n=== micro-bench motion_rep ops (T=200, single item) ===")
feats = torch.randn(200, 369)
for name, fn in [
    ("canonicalize", lambda x: motion_rep.canonicalize(x.unsqueeze(0))[0]),
    ("rotate(head)", lambda x: motion_rep.rotate(x.unsqueeze(0), torch.tensor([1.3]))[0]),
    ("normalize", lambda x: motion_rep.normalize(x)),
]:
    fn(feats)  # warmup
    t0 = time.perf_counter()
    for _ in range(200):
        fn(feats)
    print(f"  {name:14s}: {(time.perf_counter()-t0)/200*1000:7.3f} ms")
