#!/usr/bin/env bash
#SBATCH --job-name=dit4dit-grip-lqr-int
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/storage/scratch1/9/qdai41/steering_robustness_WAMs
#SBATCH --output=DiT4DiT_steering/lqr/logs/ideas_l40s_gripper_pipeline_%j.out
#SBATCH --error=DiT4DiT_steering/lqr/logs/ideas_l40s_gripper_pipeline_%j.err

set -euo pipefail

PROJECT_ROOT=/storage/scratch1/9/qdai41/steering_robustness_WAMs
source "$PROJECT_ROOT/DiT4DiT_steering/env/ideas_l40s.sh"

export PERTURBATION=gripper_xyz
export SUITE=libero_10
export COLLECT_TASK_ID=1
export EVAL_TASK_IDS=1
export N_EPISODES=30
export EVAL_NUM_EPISODES=20
export PRESET=xyz_random_xlarge_3
export BASE_SEED=42
export EVAL_BASE_SEED=42
export WORLD_SIZE=1
export SVD_WORLD_SIZE=1
export JAC_WORLD_SIZE=1
export EVAL_WORLD_SIZE=1
export START_AT="${START_AT:-1}"
export SKIP_EVAL=0
export RUN_BASELINE=1
export SAVE_ACTIVATIONS=1
export VL_BATCH_SIZE=1
export OUT_BASE="$PROJECT_ROOT/DiT4DiT_steering/lqr/runs/ideas_l40s_gripper_xyz_task01"

mkdir -p "$PROJECT_ROOT/DiT4DiT_steering/lqr/logs" "$OUT_BASE"

echo "host=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT partition=$SLURM_JOB_PARTITION"
nvidia-smi -L
test -x "$DIT4DIT_PYTHON"
test -f "$CKPT_PATH"
test -f "$DIT4DIT_CODE_ROOT/DiT4DiT/model/framework/base_framework.py"

bash "$PROJECT_ROOT/run_dit4dit_lqr_local.sh"

PROMPT="put both the cream cheese box and the butter in the basket"
SVD_DIR="$OUT_BASE/svd/libero10_task01_xyz_random_xlarge_3_seed42_paired_no_action_dsall_N-1_k64_p10_ws1_tsall"
ROLLOUT_DIR="$OUT_BASE/eval/gripper_task01"
INPUTS_NPZ="$OUT_BASE/inputs/libero_10__task01__xyz_random_xlarge_3__seed42__pos_neg__paired/negative.npz"
ANALYSIS_DIR="$ROLLOUT_DIR/interpretability"

set +e
"$DIT4DIT_PYTHON" "$PROJECT_ROOT/DiT4DiT_steering/interpretability/plot_steering_activations.py" \
  --rollout-dir "$ROLLOUT_DIR" \
  --svd-dir "$SVD_DIR" \
  --out-dir "$ANALYSIS_DIR" \
  --step 0
PLOT_STATUS=$?
set -e

"$DIT4DIT_PYTHON" "$PROJECT_ROOT/DiT4DiT_steering/interpretability/compute_steer_output_jacobian.py" \
  --svd-dir "$SVD_DIR" \
  --inputs-npz "$INPUTS_NPZ" \
  --ckpt-path "$CKPT_PATH" \
  --prompt "$PROMPT" \
  --obs-index 0 \
  --blocks 0,8,15 \
  --steps 0 \
  --out-path "$ANALYSIS_DIR/steer_output_jacobian.pt"

echo "pipeline complete; plot_status=$PLOT_STATUS"
echo "rollout=$ROLLOUT_DIR"
echo "analysis=$ANALYSIS_DIR"
