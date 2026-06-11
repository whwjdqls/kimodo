#!/bin/bash
#SBATCH --job-name=kimodo_compare_pretrained
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/runs/compare_pretrained_v_ours/slurm_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/runs/compare_pretrained_v_ours/slurm_%j.err

set -euo pipefail

OUR_CKPT="${OUR_CKPT:-/home/jungbin_cho/kimodo_open/runs/bones_seed_small/ckpt_step0050000.pt}"
OUT_DIR="${OUT_DIR:-/home/jungbin_cho/kimodo_open/runs/compare_pretrained_v_ours}"
PROMPT="${PROMPT:-a person is walking forward.}"

mkdir -p "${OUT_DIR}"

echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}   CPUs: ${SLURM_CPUS_PER_TASK:-local}"
echo "Our ckpt:  ${OUR_CKPT}"
echo "Out dir:   ${OUT_DIR}"
echo "Prompt:    ${PROMPT}"
echo

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

python -m kimodo.scripts.compare_pretrained \
    --our-ckpt "${OUR_CKPT}" \
    --out      "${OUT_DIR}" \
    --prompt   "${PROMPT}"

echo
echo "Job finished at $(date)"
