#!/usr/bin/env bash
#SBATCH --job-name=dit4dit-repro-video
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --chdir=/storage/scratch1/9/qdai41/steering_robustness_WAMs
#SBATCH --output=DiT4DiT_steering/lqr/logs/ideas_l40s_repro_video_%A_%a.out
#SBATCH --error=DiT4DiT_steering/lqr/logs/ideas_l40s_repro_video_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/storage/scratch1/9/qdai41/steering_robustness_WAMs
source "$PROJECT_ROOT/DiT4DiT_steering/env/ideas_l40s.sh"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

LQR_ROOT="$PROJECT_ROOT/DiT4DiT_steering/lqr"
PROMPT="put both the cream cheese box and the butter in the basket"
POLICY_SEED=42
ENV_SEED=42
N_EPISODES=20
COMMON=(
  --phase rollout --rank 0 --world-size 1
  --prompt "$PROMPT" --n-episodes "$N_EPISODES"
  --suite libero_10 --task-id 1 --resolution 256
  --max-env-steps 520 --num-steps-wait 10
  --ckpt-path "$CKPT_PATH" --seed "$POLICY_SEED"
  --save-video --save-inference-snapshots --no-baseline
)

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  KIND=gaussian_noise
  BASE="$LQR_ROOT/runs/ideas_l40s_noise_task01"
  SVD_DIR="$BASE/svd/libero10_task01_noise_sigma75.0_pairs_no_action_dsall_N-1_k64_p10_ws1_tsall"
  JAC_SUBDIR="A_tilde_full__put_both_the_cream_cheese_box_and_the_butter_in_th__vjp_no_retain__vbf16"
  OUT_DIR="${NOISE_OUT_DIR:-$BASE/eval_repro/noise_task01_seed42}"
  DRIVER="$LQR_ROOT/run_lqr_dit4dit_noised.py"
  EXTRA=(--noise-sigma 22.0 --noise-seed-base 42000)
else
  KIND=initial_gripper_position
  BASE="$LQR_ROOT/runs/ideas_l40s_gripper_xyz_task01"
  SVD_DIR="$BASE/svd/libero10_task01_xyz_random_xlarge_3_seed42_paired_no_action_dsall_N-1_k64_p10_ws1_tsall"
  JAC_SUBDIR="A_tilde_full__put_both_the_cream_cheese_box_and_the_butter_in_th__vjp_no_retain__vbf16"
  OUT_DIR="${GRIPPER_OUT_DIR:-$BASE/eval_repro/gripper_task01_seed42}"
  DRIVER="$LQR_ROOT/run_lqr_dit4dit_gripper_xyz.py"
  EXTRA=(--preset xyz_random_xlarge_3 --base-seed 42)
fi

mkdir -p "$OUT_DIR" "$LQR_ROOT/logs"
echo "kind=$KIND policy_seed=$POLICY_SEED env_seed=$ENV_SEED episodes=$N_EPISODES"
echo "host=$(hostname) job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID"
nvidia-smi -L
test -f "$SVD_DIR/config.json"
test -f "$SVD_DIR/$JAC_SUBDIR/A_tilde__full.pt"

"$DIT4DIT_PYTHON" "$DRIVER" \
  "${COMMON[@]}" \
  --svd-dir "$SVD_DIR" --jac-dir-act "$JAC_SUBDIR" --out-dir "$OUT_DIR" \
  "${EXTRA[@]}"

"$DIT4DIT_PYTHON" "$DRIVER" \
  --phase merge --world-size 1 \
  --prompt "$PROMPT" --n-episodes "$N_EPISODES" \
  --suite libero_10 --task-id 1 --resolution 256 \
  --max-env-steps 520 --num-steps-wait 10 \
  --ckpt-path "$CKPT_PATH" --seed "$POLICY_SEED" \
  --save-video --save-inference-snapshots --no-baseline \
  --svd-dir "$SVD_DIR" --jac-dir-act "$JAC_SUBDIR" --out-dir "$OUT_DIR" \
  "${EXTRA[@]}"

echo "complete kind=$KIND out=$OUT_DIR"
