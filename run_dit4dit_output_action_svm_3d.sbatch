#!/usr/bin/env bash
#SBATCH --job-name=dit4dit-action-svm
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --chdir=/storage/scratch1/9/qdai41/steering_robustness_WAMs
#SBATCH --output=DiT4DiT_steering/lqr/logs/ideas_l40s_action_svm_%j.out
#SBATCH --error=DiT4DiT_steering/lqr/logs/ideas_l40s_action_svm_%j.err

set -euo pipefail

PROJECT_ROOT=/storage/scratch1/9/qdai41/steering_robustness_WAMs
source "$PROJECT_ROOT/DiT4DiT_steering/env/ideas_l40s.sh"

NOISE="$PROJECT_ROOT/DiT4DiT_steering/lqr/runs/ideas_l40s_noise_task01/eval_repro/noise_task01_seed42/inference_snapshots"
GRIPPER="$PROJECT_ROOT/DiT4DiT_steering/lqr/runs/ideas_l40s_gripper_xyz_task01/eval_repro/gripper_task01_seed42/inference_snapshots"
OUTPUT_DIR="$PROJECT_ROOT/DiT4DiT_steering/interpretability/runs/output_action_svm_3d_seed42"

test -d "$NOISE"
test -d "$GRIPPER"
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/DiT4DiT_steering/lqr/logs"

"$DIT4DIT_PYTHON" -u \
  "$PROJECT_ROOT/DiT4DiT_steering/interpretability/analyze_output_action_svm_3d.py" \
  --dataset gaussian_noise "$NOISE" \
  --dataset initial_gripper_position "$GRIPPER" \
  --output-dir "$OUTPUT_DIR" \
  --c 1 \
  --seed 42

echo "complete output=$OUTPUT_DIR"
