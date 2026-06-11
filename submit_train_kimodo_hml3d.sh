#!/bin/bash
#SBATCH --job-name=kim_hml3d
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/runs/%x/slurm_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/runs/%x/slurm_%j.err

# IMPORTANT: keep RUN_NAME below in sync with --job-name above. The SBATCH
# lines are parsed by slurm BEFORE bash runs, so they can't read $RUN_NAME.
# %x in --output/--error expands to the job name; the log files therefore
# land in runs/<job-name>/.

# --------- knobs ---------
RUN_NAME="kim_hml3d_fp32"
CONFIG="/home/jungbin_cho/kimodo_open/configs/training/hml3d_clip.yaml"
RESUME_FROM=""   # set to "" to start fresh
# -------------------------

set -euo pipefail

RUN_DIR="/home/jungbin_cho/kimodo_open/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Run dir: ${RUN_DIR}"
echo "Config:  ${CONFIG}"
echo "Resume:  ${RESUME_FROM:-<from scratch>}"
echo

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo

cd /home/jungbin_cho/kimodo_open

EXTRA=()
[[ -n "${RESUME_FROM}" ]] && EXTRA+=( "trainer.resume_from=${RESUME_FROM}" )

python -m kimodo.scripts.train_w_hml3d \
    --config "${CONFIG}" \
    output_dir="${RUN_DIR}" \
    "${EXTRA[@]}"

echo
echo "Job finished at $(date)"
