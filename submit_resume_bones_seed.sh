#!/bin/bash
#SBATCH --job-name=kim_bones_rsm
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --mem=192G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/runs/bones_seed/slurm_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/runs/bones_seed/slurm_%j.err

set -euo pipefail

mkdir -p /home/jungbin_cho/kimodo_open/runs/bones_seed

echo "Job started on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-local}"
echo "Resuming from: ckpt_step0225000.pt (clean; prior run crashed on NCCL viz timeout, now 60min timeout)"
echo

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo

cd /home/jungbin_cho/kimodo_open

# Batch 128 already fills ~93% of a 40 GB A100. expandable_segments lets the
# caching allocator reclaim fragmented blocks instead of stranding them, which
# otherwise leaves the first post-resume forward without headroom. Belt-and-
# suspenders with the CPU checkpoint-load fix in train.load_checkpoint.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- node-local features pack ----------
# The NFS NPZ path is the throughput bottleneck (sustained ~8 s/step regardless
# of worker count — NFS itself saturates). rsync the 26 GB precomputed-features
# pack to node-local disk ONCE, then the dataset mmap-slices it (no NFS opens,
# no per-step FK). Built + validated by submit_pack_feats_full.sh.
PACK_SRC=/home/jungbin_cho/kimodo_caches/bones_seed_feats.pt
LOCAL_PACK=/tmp/${SLURM_JOB_ID:-local}/bones_seed_feats.pt
trap 'rm -rf /tmp/${SLURM_JOB_ID:-local}' EXIT
mkdir -p "$(dirname "${LOCAL_PACK}")"
echo "[$(date)] /tmp free before copy:"; df -h /tmp | tail -1
echo "[$(date)] rsync pack $(du -h "${PACK_SRC}" | cut -f1) NFS -> ${LOCAL_PACK}"
time rsync -a "${PACK_SRC}" "${LOCAL_PACK}"
echo "[$(date)] pack copy done"
# ----------------------------------------------

# 8-GPU DDP via torchrun. Effective batch = 8 ranks * 128 per-rank = 1024.
# CPU/RAM footprint shrunk (64->24 CPU, 384->192G) to fit the cluster's idle
# slots and start now instead of waiting ~1.5 days for a 64-CPU node.
# num_workers=8: with the node-local features pack each worker just mmap-slices
# + canonicalize/heading/normalize (no NFS open, no FK), so even few workers on
# 24 CPUs easily feed the 8 GPUs. (Worker oversubscription on the NFS path did
# NOT help — NFS itself saturated; the pack is what fixes throughput.)
#
# trainer.resume_from restores denoiser/optimizer/scheduler/scaler/EMA and the
# step counter, so training continues from step 220000 toward num_steps=500000.
# Resuming from the last CLEAN checkpoint (220k); the run diverged at ~223k and
# 225k was deleted. grad_clip tightened 0.5->0.3 to reduce the late-training
# spike risk; LR left at 2e-5 (per request). The fp32-FK fix in train_one_step
# is picked up automatically.
torchrun --standalone --nproc_per_node=8 -m kimodo.scripts.train \
    --config /home/jungbin_cho/kimodo_open/configs/training/bones_seed_full.yaml \
    trainer.batch_size=128 \
    trainer.grad_clip=0.3 \
    trainer.resume_from=/home/jungbin_cho/kimodo_open/runs/bones_seed/ckpt_step0225000.pt \
    data.num_workers=8 \
    data.packed_features_path="${LOCAL_PACK}" \
    data.train_split_path=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths.txt \
    data.cache_index=/home/jungbin_cho/kimodo_caches/seg_index.json \
    text_encoder.cache_path=/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt \
    output_dir=/home/jungbin_cho/kimodo_open/runs/bones_seed

echo
echo "Job finished at $(date)"
