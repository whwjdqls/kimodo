#!/bin/bash
#SBATCH --job-name=kim_BS_tiny_constr
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny_constraints/slurm_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny_constraints/slurm_%j.err

set -euo pipefail

mkdir -p /home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny_constraints

echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-local}"
echo

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo

cd /home/jungbin_cho/kimodo_open

# Tiny model (latent 512, 8 layers, 4 heads) + Phase 2 (constraint sampling
# from step 0). Same dataset / stats / text-encoder cache as the text-only
# tiny run; only trainer.phase and output_dir differ. Constraint mix is
# 40% 'none' (text-only batches) + 60% across full-body / root / EE
# constraints, so every epoch gets BOTH text-only and text+constraint
# supervision instead of staging them.
python -m kimodo.scripts.train \
    --config /home/jungbin_cho/kimodo_open/configs/training/bones_seed_small_tiny.yaml \
    data.train_split_path=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths_small.txt \
    data.cache_index=/home/jungbin_cho/kimodo_caches/seg_index_small.json \
    text_encoder.cache_path=/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt \
    trainer.phase=constraints \
    output_dir=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny_constraints

echo
echo "Job finished at $(date)"
