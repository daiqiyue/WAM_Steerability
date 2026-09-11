#!/usr/bin/env bash
#SBATCH --job-name=dit4dit-plot-alpha
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --chdir=/storage/scratch1/9/qdai41/steering_robustness_WAMs
#SBATCH --output=DiT4DiT_steering/lqr/logs/ideas_l40s_plot_alpha_%j.out
#SBATCH --error=DiT4DiT_steering/lqr/logs/ideas_l40s_plot_alpha_%j.err

set -euo pipefail

PROJECT_ROOT=/storage/scratch1/9/qdai41/steering_robustness_WAMs
source "$PROJECT_ROOT/DiT4DiT_steering/env/ideas_l40s.sh"

NOISE="$PROJECT_ROOT/DiT4DiT_steering/lqr/runs/ideas_l40s_noise_task01/eval/noise_task01/interpretability/steer_output_alpha_sweep_all_blocks_steps.pt"
GRIPPER="$PROJECT_ROOT/DiT4DiT_steering/lqr/runs/ideas_l40s_gripper_xyz_task01/eval/gripper_task01/interpretability/steer_output_alpha_sweep_all_blocks_steps.pt"
OUTPUT_DIR="$PROJECT_ROOT/DiT4DiT_steering/interpretability/runs/alpha_sweep"

test -f "$NOISE"
test -f "$GRIPPER"
mkdir -p "$OUTPUT_DIR"

"$DIT4DIT_PYTHON" -u \
  "$PROJECT_ROOT/DiT4DiT_steering/interpretability/plot_alpha_sweep.py" \
  --result gaussian_noise "$NOISE" \
  --result initial_gripper_position "$GRIPPER" \
  --output-dir "$OUTPUT_DIR" \
  --selected-step 0 \
  --selected-blocks 0,8,15

echo "complete output=$OUTPUT_DIR"
