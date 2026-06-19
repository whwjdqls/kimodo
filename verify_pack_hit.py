"""Verify the features pack is actually HIT (not NFS-fallback) after the
basename-key fix. CPU-only; run via srun."""
import time
import numpy as np
import torch

from kimodo.data.soma_text_motion import SOMABonesSeedDataset
from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton import SOMASkeleton30

STATS = "/home/jungbin_cho/Kimodo-SOMA-SEED-v1.1/stats/motion/"
skel = SOMASkeleton30()
mr = KimodoMotionRep(skeleton=skel, fps=20, stats_path=STATS)

ds = SOMABonesSeedDataset(
    data_root="/home/jungbin_cho/seed/soma_uniform_motions_20fps",
    natural_csv_path="/home/jungbin_cho/seed/metadata/seed_metadata_v004.csv",
    temporal_labels_path="/home/jungbin_cho/seed/metadata/seed_metadata_v002_temporal_labels.jsonl",
    multi_timeline_path="/home/jungbin_cho/seed/multi_timeline.jsonl",
    train_split_path="/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths.txt",
    fps=20, skeleton=skel, motion_rep=mr, normalize=True, random_heading_aug=True,
    cache_index="/home/jungbin_cho/kimodo_caches/seg_index.json",
    seed=0,
    packed_features_path="/home/jungbin_cho/kimodo_caches/bones_seed_feats.pt",
)

print(f"packed_features active: {ds._packed_features}")
idx = ds._packed_feat_idx
# Hit rate across all entries in all source pools.
total = hit = 0
for src in ds.SOURCES:
    for e in ds._pools[src]:
        total += 1
        if e.filename in idx:
            hit += 1
print(f"entry filename hit rate: {hit}/{total} = {100.0*hit/max(1,total):.2f}%")

# Direct check: does _load_features_segment return features (hit) not None (miss)?
miss = 0
for src in ds.SOURCES:
    e = ds._pools[src][0]
    f = ds._load_features_segment(e)
    print(f"  [{src}] {e.filename}: {'HIT '+str(tuple(f.shape)) if f is not None else 'MISS (NFS fallback)'}")
    if f is None:
        miss += 1

# Time 30 __getitem__ calls (should be fast if pack-hit, no NFS).
t0 = time.time()
N = 30
for i in range(N):
    _ = ds[i * 37 % len(ds)]
dt = (time.time() - t0) / N
print(f"\nmean __getitem__ over {N} samples: {dt*1000:.1f} ms/sample")
print("RESULT:", "PASS (pack hit, fast)" if (hit == total and miss == 0 and dt < 0.05)
      else "CHECK (still missing or slow)")
