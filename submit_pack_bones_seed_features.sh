#!/bin/bash
#SBATCH --job-name=pack_feats
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/jungbin_cho/kimodo_caches/pack_feats_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_caches/pack_feats_%j.err

set -euo pipefail
source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

python -m kimodo.scripts.pack_bones_seed_features \
    --split      /home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths_small.txt \
    --data-root  /home/jungbin_cho/seed/soma_uniform_motions_20fps \
    --stats-path /home/jungbin_cho/Kimodo-SOMA-SEED-v1.1/stats/motion/ \
    --fps        20 \
    --out        /home/jungbin_cho/kimodo_caches/bones_seed_small_feats.pt
