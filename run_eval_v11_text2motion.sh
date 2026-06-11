#!/bin/bash
#SBATCH --job-name=eval_v11_t2m
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/eval_logs/eval_v11_t2m_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/eval_logs/eval_v11_t2m_%j.err

set -euo pipefail
mkdir -p /home/jungbin_cho/kimodo_open/eval_logs
echo "Job started on $(hostname) at $(date)"
source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

# Kimodo-SOMA-SEED-v1.1: 30fps model, already a loadable folder (no build).
# ASYMMETRIC fps: generated motion is 30fps (native, no TMR resample, geometry
# at 30), GT is 20fps (upsample 20->30 for TMR, geometry at 20).
MODEL_DIR=/home/jungbin_cho/Kimodo-SOMA-SEED-v1.1
GEN_ROOT=/home/jungbin_cho/kimodo_eval_gen/SOMA-SEED-v1.1
TS=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite

echo; echo "=== sanity: load v1.1 + 1-motion generate (expect fps=30, J=77) ==="
python - <<PY
import numpy as np
from kimodo.model import load_model_from_dir
m = load_model_from_dir("${MODEL_DIR}", device="cuda")
print("loaded; fps=", m.fps, "skeleton=", type(m.skeleton).__name__)
out = m(["a person walks forward."], [90], constraint_lst=[[]],
        num_denoising_steps=20, multi_prompt=False, return_numpy=True)
print("posed_joints:", np.asarray(out["posed_joints"]).shape)
assert abs(m.fps - 30) < 1e-6, f"expected fps=30, got {m.fps}"
PY

echo; echo "=== verify TMR fps ==="
python - <<PY
from kimodo.model import load_model
m = load_model("tmr-soma-rp", default_family="TMR", device="cuda")
print("TMR motion_rep fps =", float(m.motion_rep.fps))
assert abs(float(m.motion_rep.fps) - 30) < 1e-6
PY

echo; echo "=== [3/5] generate motions (30fps native) ==="
for split in content repetition; do
    echo "--- generating ${split}/text2motion ---"
    python benchmark/generate_eval.py \
        --benchmark "${TS}/${split}/text2motion" \
        --output "${GEN_ROOT}/${split}/text2motion" \
        --model-dir "${MODEL_DIR}" \
        --batch_size 32 --num_workers 8 --diffusion_steps 100
done

echo; echo "=== [4/5] embed: motion 30fps (no resample), GT 20->30 ==="
for split in content repetition; do
    echo "--- embedding ${split}/text2motion ---"
    python benchmark/embed_folder.py "${GEN_ROOT}/${split}/text2motion" \
        --motion-fps 30 --gt-fps 20 --tgt-fps 30
done

echo; echo "=== [5/5] evaluate: gen geometry @30, GT geometry @20 ==="
for split in content repetition; do
    echo "--- evaluating ${split}/text2motion ---"
    python benchmark/evaluate_folder.py "${GEN_ROOT}/${split}/text2motion" \
        --motion-fps 30 --gt-fps 20
done

echo; echo "=== aggregate ==="
python benchmark/parse_folder.py "${GEN_ROOT}" --format md || echo "parse partial (text2motion only)"

echo; echo "Job finished at $(date)"
