#!/bin/bash
#SBATCH --job-name=kim_bones_full
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=384G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/runs/bones_seed/slurm_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/runs/bones_seed/slurm_%j.err

set -euo pipefail

mkdir -p /home/jungbin_cho/kimodo_open/runs/bones_seed

echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-local}"
echo

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo

cd /home/jungbin_cho/kimodo_open

# 8-GPU DDP via torchrun. Effective batch = 8 ranks * 128 per-rank = 1024.
# Workers per rank reduced from 16 -> 8 so total workers (8*8 = 64) fits the
# 64 CPUs we allocated.
torchrun --standalone --nproc_per_node=8 -m kimodo.scripts.train \
    --config /home/jungbin_cho/kimodo_open/configs/training/bones_seed_full.yaml \
    trainer.batch_size=128 \
    data.num_workers=8 \
    data.train_split_path=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths.txt \
    data.cache_index=/home/jungbin_cho/kimodo_caches/seg_index.json \
    text_encoder.cache_path=/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt \
    output_dir=/home/jungbin_cho/kimodo_open/runs/bones_seed

echo
echo "Job finished at $(date)"
