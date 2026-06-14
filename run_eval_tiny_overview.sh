#!/bin/bash
# Fast overview-only eval for the TINY bones_seed model (runs/bones_seed_small_tiny).
# Mirror of run_eval_bones_seed_step200k.sh: overview category, content+repetition,
# 50 diffusion steps, cached LLM2Vec text. Artifacts under the model's run dir:
# runs/bones_seed_small_tiny/eval/{models,gen,logs}/<label>.
#
# Usage:
#   sbatch run_eval_tiny_overview.sh                      # default latest.pt
#   sbatch run_eval_tiny_overview.sh ckpt_step0300000.pt  # any tiny checkpoint
#SBATCH --job-name=eval_tiny_ov
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny/eval/logs/%x_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny/eval/logs/%x_%j.err

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

# ---- config ----
RUN_DIR=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny
CKPT="${1:-latest.pt}"               # override as positional arg
TS=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite
TEXT_CACHE=/home/jungbin_cho/kimodo_caches/benchmark_llm2vec.pt   # "" -> live LLM2Vec
BATCH=256
DIFF_STEPS=50                        # fast iteration; use 100 for the full-quality number
SPLITS=(content repetition)
CATS=(overview)                      # fast: overview only. full: (overview timeline_single timeline_multi)

# label: ckpt_step0350000.pt -> step_350000 ; latest.pt -> resolve symlink target's step
resolved="$(readlink -f "${RUN_DIR}/${CKPT}" 2>/dev/null || echo "${CKPT}")"
num=$(echo "$(basename "${resolved}")" | grep -oP '[0-9]+' || true)
if [ -n "${num}" ]; then LABEL="step_$((10#${num}))"; else LABEL="${CKPT%.pt}"; fi

EVAL_DIR=${RUN_DIR}/eval
MODEL_DIR=${EVAL_DIR}/models/${LABEL}
GEN_ROOT=${EVAL_DIR}/gen/${LABEL}
mkdir -p "${EVAL_DIR}/logs"

# --text-cache flag (empty TEXT_CACHE -> omit -> live encoder)
CACHE_ARG=()
[ -n "${TEXT_CACHE}" ] && CACHE_ARG=(--text-cache "${TEXT_CACHE}")

echo "RUN_DIR=${RUN_DIR}  CKPT=${CKPT}  LABEL=${LABEL}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "GEN_ROOT=${GEN_ROOT}"
echo "TEXT_CACHE=${TEXT_CACHE:-<live LLM2Vec>}"

echo; echo "=== [1/5] build model folder (force fps=20) ==="
python build_eval_model_folder.py \
    --run-dir "${RUN_DIR}" --ckpt "${CKPT}" --out "${MODEL_DIR}" --fps 20

echo; echo "=== sanity: load model + check fps=20 (cache -> no 8B load) ==="
python - <<PY
from kimodo.model import load_model_from_dir
kw = {}
cache = "${TEXT_CACHE}"
if cache:
    from kimodo.model.cached_text import CachedTextEncoder
    kw["text_encoder"] = CachedTextEncoder(cache, device="cuda")
m = load_model_from_dir("${MODEL_DIR}", device="cuda", **kw)
print("loaded; fps=", m.fps, "skeleton=", type(m.skeleton).__name__)
assert abs(m.fps - 20) < 1e-6, f"expected fps=20, got {m.fps}"
PY

echo; echo "=== [2/5] verify TMR evaluator fps (expect 30) ==="
python - <<PY
from kimodo.model import load_model
m = load_model("tmr-soma-rp", default_family="TMR", device="cuda")
print("TMR motion_rep fps =", float(m.motion_rep.fps))
assert abs(float(m.motion_rep.fps) - 30) < 1e-6
PY

echo; echo "=== [3/5] generate motions (20fps, ${DIFF_STEPS} steps; cats: ${CATS[*]}) ==="
for split in "${SPLITS[@]}"; do
    for cat in "${CATS[@]}"; do
        echo "--- generating ${split}/text2motion/${cat} ---"
        python benchmark/generate_eval.py \
            --benchmark "${TS}/${split}/text2motion/${cat}" \
            --output "${GEN_ROOT}/${split}/text2motion/${cat}" \
            --model-dir "${MODEL_DIR}" \
            ${CACHE_ARG[@]+"${CACHE_ARG[@]}"} \
            --batch_size "${BATCH}" --num_workers 8 --diffusion_steps "${DIFF_STEPS}"
    done
done

echo; echo "=== [4/5] embed with TMR (resample 20->30) ==="
for split in "${SPLITS[@]}"; do
    for cat in "${CATS[@]}"; do
        echo "--- embedding ${split}/text2motion/${cat} ---"
        python benchmark/embed_folder.py "${GEN_ROOT}/${split}/text2motion/${cat}" --src-fps 20 --tgt-fps 30
    done
done

echo; echo "=== [5/5] evaluate (fps=20) ==="
for split in "${SPLITS[@]}"; do
    for cat in "${CATS[@]}"; do
        echo "--- evaluating ${split}/text2motion/${cat} ---"
        python benchmark/evaluate_folder.py "${GEN_ROOT}/${split}/text2motion/${cat}" --fps 20
    done
done

echo; echo "=== aggregate -> ${GEN_ROOT}/summary_rows.json ==="
python benchmark/parse_folder.py "${GEN_ROOT}" --format md || echo "parse partial (text2motion only)"

echo; echo "Job finished at $(date)"
