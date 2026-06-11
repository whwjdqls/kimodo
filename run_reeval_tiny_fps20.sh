#!/bin/bash
#SBATCH --job-name=reeval_tiny_20
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/eval_logs/reeval_tiny_20_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/eval_logs/reeval_tiny_20_%j.err

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

GEN_ROOT=/home/jungbin_cho/kimodo_eval_gen/BS-Tiny-190k

# Recompute metrics at the NATIVE 20 fps. Foot-skate (cm/s) and velocity-
# thresholded contact metrics are fps-sensitive; the default fps=30 inflated
# skate ~1.5x on these 20fps motions. FID/R-precision are unchanged (they use
# the precomputed TMR embeddings). Reuses existing motion.npz + *_embedding.npy.
for split in content repetition; do
    echo "--- re-evaluating ${split}/text2motion at fps=20 ---"
    python benchmark/evaluate_folder.py "${GEN_ROOT}/${split}/text2motion" --fps 20
done

echo; echo "=== aggregate (fps=20 corrected) ==="
python benchmark/parse_folder.py "${GEN_ROOT}" --format md || \
    echo "parse_folder partial (text2motion only)"

echo; echo "Job finished at $(date)"
