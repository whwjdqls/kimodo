#!/bin/bash
#SBATCH -p a2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -J kim_bench_textcache
#SBATCH -o /home/jungbin_cho/kimodo_open/eval_logs/bench_textcache_%j.log
set -euo pipefail

source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

export PYTHONUNBUFFERED=1
python benchmark/precompute_benchmark_text.py \
    --benchmark /home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite \
    --out       /home/jungbin_cho/kimodo_caches/benchmark_llm2vec.pt \
    --batch-size 16 \
    --device cuda

echo "[$(date)] precompute done"
