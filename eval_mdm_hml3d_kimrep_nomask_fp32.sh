#!/bin/bash
# Evaluate runs/mdm_hml3d_kimrep_nomask_fp32 with the MoMask/T2M evaluator.
#
# MDM one-stage on KIMODO 273-D representation, NO mask-concat input channel.
# Standard kimodo->humanml3d-263 conversion in evaluation/eval_hml3d.py.
#
# Outputs:
#   runs/mdm_hml3d_kimrep_nomask_fp32/eval_<ckpt-stem>.json
#   runs/mdm_hml3d_kimrep_nomask_fp32/eval_<ckpt-stem>.log

set -uo pipefail

# --------- knobs (env overridable) ---------
RUN_NAME="mdm_hml3d_kimrep_nomask_fp32"
CKPT_NAME="${CKPT_NAME:-ckpt_step0500000.pt}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REPEAT_TIMES="${REPEAT_TIMES:-5}"
CFG_SCALE="${CFG_SCALE:-2.5}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-50}"
USE_EMA="${USE_EMA:-1}"
DEVICE="${DEVICE:-}"
SEED="${SEED:-0}"
# -------------------------

cd /home/jungbin_cho/kimodo_open
source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo

run_dir="/home/jungbin_cho/kimodo_open/runs/${RUN_NAME}"
ckpt="${run_dir}/${CKPT_NAME}"
[[ -e "${ckpt}" ]] || { echo "missing checkpoint: ${ckpt}"; exit 1; }

ckpt_real="$(readlink -f "${ckpt}")"
ckpt_stem="$(basename "${ckpt_real}" .pt)"
log_path="${run_dir}/eval_${ckpt_stem}.log"
json_path="${run_dir}/eval_${ckpt_stem}.json"

echo "[$(date +%H:%M:%S)] Evaluating ${RUN_NAME}"
echo "  ckpt: ${ckpt_real}"
echo "  log:  ${log_path}"
echo "  json: ${json_path}"

args=(
    --ckpt "${ckpt_real}"
    --out "${json_path}"
    --batch-size "${BATCH_SIZE}"
    --repeat-times "${REPEAT_TIMES}"
    --cfg-scale "${CFG_SCALE}"
    --num-denoising-steps "${NUM_DENOISING_STEPS}"
    --seed "${SEED}"
)
[[ "${USE_EMA}" == "1" ]] && args+=( --use-ema )
[[ -n "${DEVICE}" ]]      && args+=( --device "${DEVICE}" )

python -m evaluation.eval_hml3d "${args[@]}" 2>&1 | tee "${log_path}"
