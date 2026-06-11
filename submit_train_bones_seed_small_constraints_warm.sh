#!/bin/bash
#SBATCH --job-name=BSsm_c_warm
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_constraints_warm/slurm_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_constraints_warm/slurm_%j.err

set -euo pipefail
mkdir -p /home/jungbin_cho/kimodo_open/runs/bones_seed_small_constraints_warm

echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

# Phase 2 (text + constraints), WARM-STARTED from the phase-1 (text) checkpoint
# at step 190000 of runs/bones_seed_small. init_from_safetensors loads only the
# EMA-overlaid denoiser weights (fresh optimizer/scheduler/step); trainer.phase=
# constraints turns on the constraint sampler (constraint_weights: 40% text-only
# + 60% full-body/root/EE constraints). num_steps=150k (~90 epochs on the ~107k
# -segment small set; well past the paper text-phase equiv of ~50k, giving the
# harder constraint objective room). ckpt_every=5000 -> pick best-by-eval afterward.
python -m kimodo.scripts.train \
    --config /home/jungbin_cho/kimodo_open/configs/training/bones_seed_small.yaml \
    data.train_split_path=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths_small.txt \
    data.cache_index=/home/jungbin_cho/kimodo_caches/seg_index_small.json \
    text_encoder.cache_path=/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt \
    init_from_safetensors=/home/jungbin_cho/kimodo_eval_models/BS-Small-init190k/model.safetensors \
    trainer.phase=constraints \
    trainer.num_steps=150000 \
    output_dir=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_constraints_warm

echo
echo "Job finished at $(date)"
