#!/bin/bash
#SBATCH --job-name=BSti_c_warm
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny_constraints_warm/slurm_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny_constraints_warm/slurm_%j.err

set -euo pipefail
mkdir -p /home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny_constraints_warm

echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

# Phase 2 (text + constraints), WARM-STARTED from the phase-1 (text) checkpoint
# at step 190000 of runs/bones_seed_small_tiny (the BS-Tiny-190k model we evaluated).
# Tiny model (latent 512, 8 layers, 4 heads). init_from_safetensors loads only the
# EMA-overlaid denoiser weights (fresh optimizer/scheduler/step); trainer.phase=
# constraints turns on the constraint sampler. NOTE: differs from the earlier
# from-scratch runs/bones_seed_small_tiny_constraints run (job 6691, cancelled) --
# this one is warm-started, so it goes to a separate _warm dir.
python -m kimodo.scripts.train \
    --config /home/jungbin_cho/kimodo_open/configs/training/bones_seed_small_tiny.yaml \
    data.train_split_path=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths_small.txt \
    data.cache_index=/home/jungbin_cho/kimodo_caches/seg_index_small.json \
    text_encoder.cache_path=/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt \
    init_from_safetensors=/home/jungbin_cho/kimodo_eval_models/BS-Tiny-init190k/model.safetensors \
    trainer.phase=constraints \
    trainer.num_steps=150000 \
    output_dir=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_tiny_constraints_warm

echo
echo "Job finished at $(date)"
