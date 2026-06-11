#!/bin/bash
#SBATCH --job-name=eval_bseed_t2m
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/eval_logs/eval_bseed_t2m_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/eval_logs/eval_bseed_t2m_%j.err

set -euo pipefail
mkdir -p /home/jungbin_cho/kimodo_open/eval_logs
echo "Job started on $(hostname) at $(date)"
source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

# bones_seed: 20fps model (trained with denoiser_fps_override=20; snapshot fps:30
# is stale, so we force --fps 20 when building). Symmetric: gen 20fps, GT 20fps.
RUN_DIR=/home/jungbin_cho/kimodo_open/runs/bones_seed
CKPT=latest.pt
MODEL_DIR=/home/jungbin_cho/kimodo_eval_models/BS-Full
GEN_ROOT=/home/jungbin_cho/kimodo_eval_gen/BS-Full
TS=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite

echo; echo "=== [1/5] build model folder (force fps=20) ==="
python build_eval_model_folder.py \
    --run-dir "${RUN_DIR}" --ckpt "${CKPT}" --out "${MODEL_DIR}" --fps 20

echo; echo "=== sanity: load + 1-motion generate (expect fps=20, J=77) ==="
python - <<PY
import numpy as np
from kimodo.model import load_model_from_dir
m = load_model_from_dir("${MODEL_DIR}", device="cuda")
print("loaded; fps=", m.fps, "skeleton=", type(m.skeleton).__name__)
out = m(["a person walks forward."], [60], constraint_lst=[[]],
        num_denoising_steps=20, multi_prompt=False, return_numpy=True)
print("posed_joints:", np.asarray(out["posed_joints"]).shape)
assert abs(m.fps - 20) < 1e-6, f"expected fps=20, got {m.fps}"
PY

echo; echo "=== [2/5] verify TMR fps ==="
python - <<PY
from kimodo.model import load_model
m = load_model("tmr-soma-rp", default_family="TMR", device="cuda")
print("TMR motion_rep fps =", float(m.motion_rep.fps))
assert abs(float(m.motion_rep.fps) - 30) < 1e-6
PY

echo; echo "=== [3/5] generate motions (20fps) ==="
for split in content repetition; do
    echo "--- generating ${split}/text2motion ---"
    python benchmark/generate_eval.py \
        --benchmark "${TS}/${split}/text2motion" \
        --output "${GEN_ROOT}/${split}/text2motion" \
        --model-dir "${MODEL_DIR}" \
        --batch_size 32 --num_workers 8 --diffusion_steps 100
done

echo; echo "=== [4/5] embed with TMR (both 20->30) ==="
for split in content repetition; do
    echo "--- embedding ${split}/text2motion ---"
    python benchmark/embed_folder.py "${GEN_ROOT}/${split}/text2motion" \
        --src-fps 20 --tgt-fps 30
done

echo; echo "=== [5/5] evaluate (both fps=20) ==="
for split in content repetition; do
    echo "--- evaluating ${split}/text2motion ---"
    python benchmark/evaluate_folder.py "${GEN_ROOT}/${split}/text2motion" --fps 20
done

echo; echo "=== aggregate ==="
python benchmark/parse_folder.py "${GEN_ROOT}" --format md || echo "parse partial (text2motion only)"

echo; echo "Job finished at $(date)"
