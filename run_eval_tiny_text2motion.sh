#!/bin/bash
#SBATCH --job-name=eval_tiny_t2m
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/eval_logs/eval_tiny_t2m_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/eval_logs/eval_tiny_t2m_%j.err

set -euo pipefail
mkdir -p /home/jungbin_cho/kimodo_open/eval_logs

echo "Job started on $(hostname) at $(date)"
source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

# ---- paths ----
RUN_DIR=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny
CKPT=latest.pt                              # step 190000
MODEL_DIR=/home/jungbin_cho/kimodo_eval_models/BS-Tiny-190k
GEN_ROOT=/home/jungbin_cho/kimodo_eval_gen/BS-Tiny-190k
TS=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite
SRC_FPS=20
TGT_FPS=30

# ---- 1. build a load_model-compatible model folder from the training ckpt ----
echo; echo "=== [1/5] build model folder ==="
python build_eval_model_folder.py \
    --run-dir "${RUN_DIR}" --ckpt "${CKPT}" --out "${MODEL_DIR}"

# fail-fast sanity: load the custom model + generate ONE short motion
echo; echo "=== sanity: load custom model + 1-motion generate ==="
python - <<PY
import numpy as np, torch
from kimodo.model import load_model_from_dir
m = load_model_from_dir("${MODEL_DIR}", device="cuda")
print("loaded; fps=", m.fps, "skeleton=", type(m.skeleton).__name__)
out = m(["a person walks forward."], [60], constraint_lst=[[]],
        num_denoising_steps=20, multi_prompt=False, return_numpy=True)
pj = out["posed_joints"]
print("posed_joints:", np.asarray(pj).shape, "(expect [.,T,J,3] with J=77)")
PY

# ---- 2. verify TMR fps (also triggers TMR-SOMA-RP-v1 download) ----
echo; echo "=== [2/5] verify TMR evaluator fps (downloads TMR on first use) ==="
python - <<PY
from kimodo.model import load_model
m = load_model("tmr-soma-rp", default_family="TMR", device="cuda")
fps = float(m.motion_rep.fps)
print("TMR motion_rep fps =", fps)
assert abs(fps - ${TGT_FPS}) < 1e-6, f"TMR fps {fps} != --tgt-fps ${TGT_FPS}; adjust resampling target"
print("OK: TMR fps matches --tgt-fps=${TGT_FPS}")
PY

# ---- 3. generate motions for text2motion (content + repetition) ----
echo; echo "=== [3/5] generate motions ==="
for split in content repetition; do
    echo "--- generating ${split}/text2motion ---"
    python benchmark/generate_eval.py \
        --benchmark "${TS}/${split}/text2motion" \
        --output "${GEN_ROOT}/${split}/text2motion" \
        --model-dir "${MODEL_DIR}" \
        --batch_size 32 --num_workers 8 --diffusion_steps 100
done

# ---- 4. embed motion/gt/text with TMR, resampling 20->30 fps ----
echo; echo "=== [4/5] embed with TMR (resample ${SRC_FPS}->${TGT_FPS} fps) ==="
for split in content repetition; do
    echo "--- embedding ${split}/text2motion ---"
    python benchmark/embed_folder.py "${GEN_ROOT}/${split}/text2motion" \
        --src-fps ${SRC_FPS} --tgt-fps ${TGT_FPS}
done

# ---- 5. compute metrics ----
echo; echo "=== [5/5] evaluate metrics ==="
for split in content repetition; do
    echo "--- evaluating ${split}/text2motion ---"
    python benchmark/evaluate_folder.py "${GEN_ROOT}/${split}/text2motion"
done

echo; echo "=== aggregate (parse_folder; may warn about missing constraint categories) ==="
python benchmark/parse_folder.py "${GEN_ROOT}" --format md || \
    echo "parse_folder skipped/partial (expected: only text2motion present)"

echo; echo "Job finished at $(date)"
