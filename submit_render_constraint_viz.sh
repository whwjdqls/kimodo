#!/bin/bash
# Quick visual test of the constraint overlay + fixed-camera viz in
# kimodo/scripts/visualize.py. Renders GT motions (aligned with their
# constraints by construction) so markers should land exactly on the skeleton.
#SBATCH --job-name=render_cviz
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/jungbin_cho/kimodo_open/viz_constraint_test/render_%j.log
#SBATCH --error=/home/jungbin_cho/kimodo_open/viz_constraint_test/render_%j.log

set -euo pipefail
source /home/jungbin_cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /home/jungbin_cho/kimodo_open

TS=/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite
OUT=/home/jungbin_cho/kimodo_open/viz_constraint_test
mkdir -p "$OUT"

MIX="$TS/repetition/constraints_withtext/mixture/root_path_fullbody/0061"
EE="$TS/repetition/constraints_withtext/end-effectors/hands_posrot/0061"

echo "=== [1] mixture (root path + fullbody ghost), fixed camera, general+top ==="
python -m kimodo.scripts.visualize "$MIX/gt_motion.npz" \
    --constraints "$MIX/constraints.json" \
    --view general,top --fps 20 \
    -o "$OUT/mixture"

echo "=== [2] end-effector hands (2 target spheres), fixed camera, general ==="
python -m kimodo.scripts.visualize "$EE/gt_motion.npz" \
    --constraints "$EE/constraints.json" \
    --view general --fps 20 \
    -o "$OUT/ee_hands"

echo "=== done; outputs in $OUT ==="
ls -lh "$OUT"/*.mp4
