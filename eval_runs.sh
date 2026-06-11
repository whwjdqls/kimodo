#!/bin/bash
# Evaluate four HumanML3D runs with the MoMask/T2M evaluator.
#
#   1) mdm_hml3d_native_fp32        — MDM one-stage, native 263-D rep
#   2) mdm_hml3d_kimrep_nomask_fp32 — MDM one-stage, kimodo 273-D rep, no mask
#   3) mdm_hml3d_kimrep_fp32        — MDM one-stage, kimodo 273-D rep, mask=concat
#   4) kim_hml3d_nomask_fp32        — Kimodo two-stage, kimodo 273-D rep, no mask
#
# Each run's metrics + per-iteration breakdown land in
# ``<run_dir>/eval_<ckpt-stem>.json`` and an evaluation log in
# ``<run_dir>/eval_<ckpt-stem>.log``. The driver is
# ``evaluation.eval_hml3d`` which now dispatches on motion_rep_dim
# (263 = native, 273 = kimodo) so the same script handles both reps.
#
# Default knobs match the MoMask reporting convention:
#   --batch-size 32 --repeat-times 5 --cfg-scale 2.5 --use-ema --num-denoising-steps 50

set -uo pipefail   # NOT -e: we want to keep going if one run errors.

RUNS=(
    mdm_hml3d_native_fp32
    mdm_hml3d_kimrep_nomask_fp32
    mdm_hml3d_kimrep_fp32
    kim_hml3d_nomask_fp32
)

# --------- knobs (override via env: e.g. CKPT_NAME=ckpt_step0500000.pt ./eval_runs.sh) ---------
CKPT_NAME="${CKPT_NAME:-latest.pt}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REPEAT_TIMES="${REPEAT_TIMES:-5}"
CFG_SCALE="${CFG_SCALE:-2.5}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-50}"
USE_EMA="${USE_EMA:-1}"          # set USE_EMA=0 to disable
DEVICE="${DEVICE:-}"             # blank = auto (cuda if available)
SEED="${SEED:-0}"
# -------------------------

RUNS_ROOT="/home/jungbin_cho/kimodo_open/runs"
cd /home/jungbin_cho/kimodo_open

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo

declare -A STATUS

for name in "${RUNS[@]}"; do
    run_dir="${RUNS_ROOT}/${name}"
    ckpt="${run_dir}/${CKPT_NAME}"

    echo "=================================================================="
    echo "[$(date +%H:%M:%S)] Evaluating: ${name}"
    echo "  ckpt: ${ckpt}"

    if [[ ! -e "${ckpt}" ]]; then
        echo "  SKIP: checkpoint not found"
        STATUS[$name]="MISSING_CKPT"
        continue
    fi

    # Resolve symlink so the JSON/log filename uses the concrete ckpt stem.
    ckpt_real="$(readlink -f "${ckpt}")"
    ckpt_stem="$(basename "${ckpt_real}" .pt)"
    log_path="${run_dir}/eval_${ckpt_stem}.log"
    json_path="${run_dir}/eval_${ckpt_stem}.json"

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

    echo "  log:  ${log_path}"
    echo "  json: ${json_path}"

    if python -m evaluation.eval_hml3d "${args[@]}" 2>&1 | tee "${log_path}"; then
        STATUS[$name]="OK"
    else
        STATUS[$name]="FAILED"
    fi
done

echo
echo "=================================================================="
echo "Summary"
echo "=================================================================="
for name in "${RUNS[@]}"; do
    printf "  %-32s %s\n" "${name}" "${STATUS[$name]:-UNKNOWN}"
done
