#!/usr/bin/env bash
#SBATCH --job-name=dit4dit-jac-alpha
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --chdir=/storage/scratch1/9/qdai41/steering_robustness_WAMs
#SBATCH --output=DiT4DiT_steering/lqr/logs/ideas_l40s_jac_alpha_%A_%a.out
#SBATCH --error=DiT4DiT_steering/lqr/logs/ideas_l40s_jac_alpha_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/storage/scratch1/9/qdai41/steering_robustness_WAMs
source "$PROJECT_ROOT/DiT4DiT_steering/env/ideas_l40s.sh"

PROMPT="put both the cream cheese box and the butter in the basket"
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  KIND=gaussian_noise
  BASE="$PROJECT_ROOT/DiT4DiT_steering/lqr/runs/ideas_l40s_noise_task01"
  SVD_DIR="$BASE/svd/libero10_task01_noise_sigma75.0_pairs_no_action_dsall_N-1_k64_p10_ws1_tsall"
  INPUTS_NPZ="$BASE/inputs/libero_10__task01__noise_sigma75.0/negative.npz"
  ANALYSIS_DIR="$BASE/eval/noise_task01/interpretability"
else
  KIND=initial_gripper_position
  BASE="$PROJECT_ROOT/DiT4DiT_steering/lqr/runs/ideas_l40s_gripper_xyz_task01"
  SVD_DIR="$BASE/svd/libero10_task01_xyz_random_xlarge_3_seed42_paired_no_action_dsall_N-1_k64_p10_ws1_tsall"
  INPUTS_NPZ="$BASE/inputs/libero_10__task01__xyz_random_xlarge_3__seed42__pos_neg__paired/negative.npz"
  ANALYSIS_DIR="$BASE/eval/gripper_task01/interpretability"
fi

mkdir -p "$ANALYSIS_DIR" "$PROJECT_ROOT/DiT4DiT_steering/lqr/logs"
echo "kind=$KIND host=$(hostname) job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID"
nvidia-smi -L
test -f "$SVD_DIR/config.json"
test -f "$SVD_DIR/svd_summary.pt"
test -f "$INPUTS_NPZ"

"$DIT4DIT_PYTHON" -u \
  "$PROJECT_ROOT/DiT4DiT_steering/interpretability/compute_steer_output_jacobian.py" \
  --svd-dir "$SVD_DIR" \
  --inputs-npz "$INPUTS_NPZ" \
  --ckpt-path "$CKPT_PATH" \
  --prompt "$PROMPT" \
  --obs-index 0 \
  --seed 42 \
  --blocks all \
  --steps all \
  --derivative-method central \
  --fd-epsilon 0.01 \
  --fd-sweep-epsilons 0.01,0.1,0.5,1 \
  --out-path "$ANALYSIS_DIR/steer_output_alpha_sweep_all_blocks_steps.pt"

echo "complete kind=$KIND output=$ANALYSIS_DIR"
