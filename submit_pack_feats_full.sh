#!/bin/bash
#SBATCH --job-name=pack_feats_full
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_caches/pack_feats_full_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_caches/pack_feats_full_%j.err

set -euo pipefail
source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

DATA_ROOT=/home/jungbin_cho/seed/soma_uniform_motions_20fps
STATS=/home/jungbin_cho/Kimodo-SOMA-SEED-v1.1/stats/motion/
SPLIT_SMALL=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths_small.txt
SPLIT_FULL=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths.txt
REF_SMALL=/home/jungbin_cho/kimodo_caches/bones_seed_small_feats.pt
VAL_SMALL=/home/jungbin_cho/kimodo_caches/_val_small_feats.pt
OUT_FULL=/home/jungbin_cho/kimodo_caches/bones_seed_feats.pt
PROCS=22

echo "[$(date)] === STEP 1: parallel-build SMALL pack for validation ==="
python -m kimodo.scripts.pack_bones_seed_features \
    --split "$SPLIT_SMALL" --data-root "$DATA_ROOT" --stats-path "$STATS" \
    --fps 20 --procs "$PROCS" --out "$VAL_SMALL"

echo "[$(date)] === STEP 2: validate parallel build vs known-good $REF_SMALL ==="
python - "$VAL_SMALL" "$REF_SMALL" <<'PY'
import sys, torch
new = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ref = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
assert new["names"] == ref["names"], "names differ (order/content mismatch)"
assert torch.equal(new["offsets"], ref["offsets"]), "offsets differ"
fa, fb = new["features"], ref["features"]
assert fa.shape == fb.shape, f"shape {tuple(fa.shape)} vs {tuple(fb.shape)}"
maxdiff = (fa - fb).abs().max().item()
print(f"  names match ({len(new['names'])}), offsets match, shape {tuple(fa.shape)}")
print(f"  max feature diff = {maxdiff:.3e}")
# Serial(ref) used multi-thread torch; parallel uses 1 thread/worker -> tiny
# reduction-order diffs possible. A real bug (wrong mapping/flags) is >> 1e-2.
assert maxdiff < 1e-2, f"features diverge by {maxdiff} -> parallel builder BUG, aborting"
print("  VALIDATION PASS")
PY

echo "[$(date)] === STEP 3: parallel-build FULL pack ==="
python -m kimodo.scripts.pack_bones_seed_features \
    --split "$SPLIT_FULL" --data-root "$DATA_ROOT" --stats-path "$STATS" \
    --fps 20 --procs "$PROCS" --out "$OUT_FULL"

rm -f "$VAL_SMALL"
echo "[$(date)] === DONE: full features pack at $OUT_FULL ==="
ls -lh "$OUT_FULL"
