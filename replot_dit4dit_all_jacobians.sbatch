#!/usr/bin/env bash
#SBATCH --job-name=dit4dit-replot-jac
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --chdir=/storage/scratch1/9/qdai41/steering_robustness_WAMs
#SBATCH --output=DiT4DiT_steering/lqr/logs/ideas_l40s_replot_jac_%j.out
#SBATCH --error=DiT4DiT_steering/lqr/logs/ideas_l40s_replot_jac_%j.err

set -euo pipefail

PROJECT_ROOT=/storage/scratch1/9/qdai41/steering_robustness_WAMs
PYTHON=/storage/scratch1/9/qdai41/.conda/envs/dit4dit/bin/python
export MPLCONFIGDIR="/tmp/dit4dit-mpl-$SLURM_JOB_ID"
export XDG_CACHE_HOME="/tmp/dit4dit-xdg-$SLURM_JOB_ID"
SCRIPT="$PROJECT_ROOT/DiT4DiT_steering/interpretability/replot_steer_output_jacobian.py"

NOISE="$PROJECT_ROOT/DiT4DiT_steering/lqr/runs/ideas_l40s_noise_task01/eval/noise_task01/interpretability/steer_output_jacobian_all_blocks_steps.pt"
GRIPPER="$PROJECT_ROOT/DiT4DiT_steering/lqr/runs/ideas_l40s_gripper_xyz_task01/eval/gripper_task01/interpretability/steer_output_jacobian_all_blocks_steps.pt"

"$PYTHON" -u "$SCRIPT" "$NOISE"
"$PYTHON" -u "$SCRIPT" "$GRIPPER"
echo "all Jacobian figures regenerated with a dedicated colorbar margin"
