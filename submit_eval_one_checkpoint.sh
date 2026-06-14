#!/bin/bash
# Evaluate ONE bones_seed_small checkpoint. Submit one of these per checkpoint
# (independent single-GPU jobs schedule/backfill better than an array):
#   for s in 20000 40000 ... ; do sbatch --job-name=BSsm_$((s/1000))k submit_eval_one_checkpoint.sh $s; done
# Pipeline: build(fps20) -> generate(bs256) -> embed(20->30) -> evaluate(--fps20)
#           -> copy JSONs -> ERASE the ~12GB gen folder. Resumable: exits if results exist.
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/runs/bones_seed_small/eval/logs/%x_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/runs/bones_seed_small/eval/logs/%x_%j.err

set -uo pipefail
s="${1:?usage: sbatch submit_eval_one_checkpoint.sh <STEP>}"

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

SP=$(printf "%07d" "$s")
CKPT="ckpt_step${SP}.pt"
RUN_DIR=/home/jungbin_cho/kimodo_open/runs/bones_seed_small
TS=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite
# Eval artifacts live under the model's run dir: runs/<model>/eval/{models,gen,results,logs}/step_<N>
EVAL_DIR=${RUN_DIR}/eval
MODEL_DIR=${EVAL_DIR}/models/step_${s}
GEN_ROOT=${EVAL_DIR}/gen/step_${s}
RES_DIR=${EVAL_DIR}/results/step_${s}
SPLITS=(content repetition)
CATS=(overview timeline_single timeline_multi)

log() { echo "[$(date '+%F %T')] [step ${s}] $*"; }
log "job ${SLURM_JOB_ID:-local} on $(hostname), GPUs=${CUDA_VISIBLE_DEVICES:-none}, ckpt=${CKPT}"

# ---- resume guard ----
have=1
for split in "${SPLITS[@]}"; do for cat in "${CATS[@]}"; do
    [ -f "${RES_DIR}/${split}/${cat}.json" ] || have=0
done; done
if [ "$have" = "1" ]; then log "results already present, exiting"; exit 0; fi

t0=$(date +%s)

# ---- [1] build model folder (force fps=20) ----
python build_eval_model_folder.py --run-dir "${RUN_DIR}" --ckpt "${CKPT}" \
    --out "${MODEL_DIR}" --fps 20 || { log "BUILD FAILED"; exit 1; }

# ---- [2] generate (content + repetition), bs=256 ----
for split in "${SPLITS[@]}"; do
    log "generate ${split}"
    python benchmark/generate_eval.py \
        --benchmark "${TS}/${split}/text2motion" \
        --output "${GEN_ROOT}/${split}/text2motion" \
        --model-dir "${MODEL_DIR}" \
        --batch_size 256 --num_workers 8 --diffusion_steps 100 \
        || { log "GENERATE ${split} FAILED (leaving gen dir)"; exit 1; }
done

# ---- [3] embed (TMR, resample 20->30) ----
for split in "${SPLITS[@]}"; do
    log "embed ${split}"
    python benchmark/embed_folder.py "${GEN_ROOT}/${split}/text2motion" --src-fps 20 --tgt-fps 30
done

# ---- [4] evaluate (fps=20) ----
for split in "${SPLITS[@]}"; do
    log "evaluate ${split}"
    python benchmark/evaluate_folder.py "${GEN_ROOT}/${split}/text2motion" --fps 20
done

# ---- [5] copy out the small JSONs ----
copyok=1
for split in "${SPLITS[@]}"; do
    mkdir -p "${RES_DIR}/${split}"
    for cat in "${CATS[@]}"; do
        src="${GEN_ROOT}/${split}/text2motion/${cat}.json"
        if [ -f "$src" ]; then cp "$src" "${RES_DIR}/${split}/${cat}.json"; else log "MISSING ${src}"; copyok=0; fi
    done
done

# ---- [6] ERASE only if JSONs saved ----
if [ "$copyok" = "1" ]; then
    rm -rf "${GEN_ROOT}" "${MODEL_DIR}"
    log "erased gen+model dirs (results in ${RES_DIR})"
else
    log "NOT erasing (missing JSONs) -> ${GEN_ROOT}"; exit 1
fi

dt=$(( $(date +%s) - t0 ))
log "done in ${dt}s ($(( dt/60 ))m)"
