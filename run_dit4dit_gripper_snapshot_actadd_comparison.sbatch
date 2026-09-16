#!/usr/bin/env bash
#SBATCH --job-name=dit4dit-grip-actadd
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --chdir=/storage/scratch1/9/qdai41/steering_robustness_WAMs
#SBATCH --output=DiT4DiT_steering/lqr/logs/ideas_l40s_gripper_actadd_%j.out
#SBATCH --error=DiT4DiT_steering/lqr/logs/ideas_l40s_gripper_actadd_%j.err

set -euo pipefail

PROJECT_ROOT=/storage/scratch1/9/qdai41/steering_robustness_WAMs
source "$PROJECT_ROOT/DiT4DiT_steering/env/ideas_l40s.sh"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE="$PROJECT_ROOT/DiT4DiT_steering/lqr/runs/ideas_l40s_gripper_xyz_task01"
SNAPSHOT="$BASE/eval_repro/gripper_task01_seed42/inference_snapshots/ep06__steered.pt"
SVD_DIR="$BASE/svd/libero10_task01_xyz_random_xlarge_3_seed42_paired_no_action_dsall_N-1_k64_p10_ws1_tsall"
OUTPUT_DIR="$BASE/eval_repro/snapshot_actadd_ep06_inference010"

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/DiT4DiT_steering/lqr/logs"
echo "host=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT partition=$SLURM_JOB_PARTITION"
nvidia-smi -L
test -f "$SNAPSHOT"
test -f "$SVD_DIR/config.json"
test -f "$SVD_DIR/svd_summary.pt"

"$DIT4DIT_PYTHON" -u \
  "$PROJECT_ROOT/DiT4DiT_steering/interpretability/replay_snapshot_actadd.py" \
  --snapshot "$SNAPSHOT" \
  --svd-dir "$SVD_DIR" \
  --ckpt-path "$CKPT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --inference-index 10 \
  --max-inferences 55 \
  --alpha 1 \
  --blocks all \
  --steps all \
  --resolution 256 \
  --video-fps 30

echo "complete output=$OUTPUT_DIR"
