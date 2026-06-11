#!/bin/bash
#SBATCH --job-name=kim_M
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_medium/slurm_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_medium/slurm_%j.err

set -euo pipefail

mkdir -p /home/jungbin_cho/kimodo_open/runs/bones_seed_small_medium

echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-local}"
echo

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo

cd /home/jungbin_cho/kimodo_open

# ---------- node-local motion pack ----------
# rsync one ~4 GB pack file (built by kimodo.scripts.pack_bones_seed_motions)
# instead of 13k individual NPZs. The pack stores raw local_rot_mats +
# root_positions in two big concat tensors; the dataset slices into them via
# mmap. Eliminates per-step NPZ decompression on top of the NFS-vs-local win.
SPLIT=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths_small.txt
PACK_SRC=/home/jungbin_cho/kimodo_caches/bones_seed_small_raw.pt
LOCAL_PACK=/tmp/${SLURM_JOB_ID:-local}/bones_seed_small_raw.pt
trap 'rm -rf /tmp/${SLURM_JOB_ID:-local}' EXIT

mkdir -p "$(dirname "${LOCAL_PACK}")"
echo "[$(date)] rsync pack $(du -h "${PACK_SRC}" | cut -f1) from NFS -> ${LOCAL_PACK}"
time rsync -a "${PACK_SRC}" "${LOCAL_PACK}"
echo "[$(date)] pack copy done"
# --------------------------------------------

python -m kimodo.scripts.train \
    --config /home/jungbin_cho/kimodo_open/configs/training/bones_seed_small_medium.yaml \
    data.packed_motions_path="${LOCAL_PACK}" \
    data.train_split_path="${SPLIT}" \
    data.cache_index=/home/jungbin_cho/kimodo_caches/seg_index_small.json \
    text_encoder.cache_path=/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt \
    output_dir=/home/jungbin_cho/kimodo_open/runs/bones_seed_small_medium
    # trainer.resume_from=/home/jungbin_cho/kimodo_open/runs/bones_seed_small/ckpt_step0230000.pt \
    # trainer.lr=1e-5 \
    # trainer.grad_clip=0.5 \
    # output_dir=/home/jungbin_cho/kimodo_open/runs/bones_seed_small

echo
echo "Job finished at $(date)"
