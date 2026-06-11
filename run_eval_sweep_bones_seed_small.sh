#!/bin/bash
# Multi-checkpoint eval sweep for bones_seed_small (save -> eval -> ERASE per ckpt).
# Reuses the validated 5-step pipeline (matches run_eval_bones_seed_text2motion.sh):
#   build_eval_model_folder (--fps 20) -> generate_eval -> embed_folder (20->30) -> evaluate_folder (--fps 20)
# Only the small per-category JSONs are persisted; the ~12GB gen folder is rm -rf'd after each ckpt.
# Resumable: a step whose results JSONs already exist is skipped.

set -uo pipefail   # NOT -e: handle per-step failures and keep going

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

# ---- config ----
STEPS=(20000 40000 60000 80000 100000 150000 200000 250000)
RUN_DIR=/home/jungbin_cho/kimodo_open/runs/bones_seed_small
TS=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite
SWEEP=/home/jungbin_cho/kimodo_eval_gen/BS-Small-sweep
RESULTS=${SWEEP}/results
MODELS=/home/jungbin_cho/kimodo_eval_models/BS-Small-sweep
SPLITS=(content repetition)
CATS=(overview timeline_single timeline_multi)
mkdir -p "${RESULTS}" "${MODELS}"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- one-time: verify TMR evaluator is 30fps (also triggers download) ----
log "verify TMR fps (expect 30) ..."
python - <<'PY' || { echo "TMR fps check failed"; exit 1; }
from kimodo.model import load_model
m = load_model("tmr-soma-rp", default_family="TMR", device="cuda")
fps = float(m.motion_rep.fps)
print("TMR motion_rep fps =", fps)
assert abs(fps - 30) < 1e-6, f"TMR fps {fps} != 30"
print("OK TMR fps")
PY

for s in "${STEPS[@]}"; do
    SP=$(printf "%07d" "$s")
    CKPT="ckpt_step${SP}.pt"
    MODEL_DIR="${MODELS}/step_${s}"
    GEN_ROOT="${SWEEP}/step_${s}"
    RES_DIR="${RESULTS}/step_${s}"

    # ---- resume: skip if all 6 result JSONs already present ----
    have=1
    for split in "${SPLITS[@]}"; do for cat in "${CATS[@]}"; do
        [ -f "${RES_DIR}/${split}/${cat}.json" ] || have=0
    done; done
    if [ "$have" = "1" ]; then log "step ${s}: results already present, skipping"; continue; fi

    log "================ step ${s} (${CKPT}) ================"
    t0=$(date +%s)

    # ---- [1] build model folder (force fps=20; EMA on by default) ----
    if ! python build_eval_model_folder.py --run-dir "${RUN_DIR}" --ckpt "${CKPT}" \
            --out "${MODEL_DIR}" --fps 20; then
        log "step ${s}: BUILD FAILED, skipping"; continue
    fi

    # ---- [2] generate (content + repetition), 20fps, 100 diffusion steps ----
    genok=1
    for split in "${SPLITS[@]}"; do
        log "step ${s}: generate ${split}"
        if ! python benchmark/generate_eval.py \
                --benchmark "${TS}/${split}/text2motion" \
                --output "${GEN_ROOT}/${split}/text2motion" \
                --model-dir "${MODEL_DIR}" \
                --batch_size 32 --num_workers 8 --diffusion_steps 100; then
            log "step ${s}: GENERATE ${split} FAILED"; genok=0; break
        fi
    done
    [ "$genok" = "1" ] || { log "step ${s}: skipping (gen failed); leaving gen dir for inspection"; continue; }

    # ---- [3] embed with TMR (resample 20->30) ----
    for split in "${SPLITS[@]}"; do
        log "step ${s}: embed ${split}"
        python benchmark/embed_folder.py "${GEN_ROOT}/${split}/text2motion" --src-fps 20 --tgt-fps 30
    done

    # ---- [4] evaluate (fps=20 for foot-skate/contact) ----
    for split in "${SPLITS[@]}"; do
        log "step ${s}: evaluate ${split}"
        python benchmark/evaluate_folder.py "${GEN_ROOT}/${split}/text2motion" --fps 20
    done

    # ---- [5] copy out the small per-category JSONs ----
    copyok=1
    for split in "${SPLITS[@]}"; do
        mkdir -p "${RES_DIR}/${split}"
        for cat in "${CATS[@]}"; do
            src="${GEN_ROOT}/${split}/text2motion/${cat}.json"
            if [ -f "$src" ]; then cp "$src" "${RES_DIR}/${split}/${cat}.json";
            else log "step ${s}: MISSING ${src}"; copyok=0; fi
        done
    done

    # ---- [6] ERASE the big gen folder + model dir ONLY if JSONs were saved ----
    if [ "$copyok" = "1" ]; then
        rm -rf "${GEN_ROOT}" "${MODEL_DIR}"
        log "step ${s}: erased gen+model dirs (results kept in ${RES_DIR})"
    else
        log "step ${s}: NOT erasing (some JSONs missing) -> ${GEN_ROOT}"
    fi

    dt=$(( $(date +%s) - t0 ))
    log "step ${s}: done in ${dt}s ($(( dt/60 ))m)"
done

log "================ all steps done; aggregating ================"
python aggregate_sweep_curve.py "${RESULTS}" --out "${SWEEP}/sweep_curve" || log "aggregation failed"
log "sweep complete."
