#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LINGBOT_DIR="${LINGBOT_DIR:-$ROOT/LingBot-VA_steering/LingBot-VA-Modification-lingbot-clean}"
PERTURBATION="${PERTURBATION:-camera}"

case "$PERTURBATION" in
  camera|cam)
    export TASK_ID="${COLLECT_TASK_ID:-${TASK_ID:-0}}"
    export TASK_IDS="${EVAL_TASK_IDS:-${TASK_IDS:-0 2 4 5 9}}"
    SCRIPT="run_lqr_camera.sh"
    ;;
  gripper|gripper_xyz|init_pos|init|position)
    export TASK_ID="${COLLECT_TASK_ID:-${TASK_ID:-1}}"
    export TASK_IDS="${EVAL_TASK_IDS:-${TASK_IDS:-1 2 3 7 9}}"
    SCRIPT="run_lqr_init_pos.sh"
    ;;
  gaussian|noise)
    export TASK_ID="${COLLECT_TASK_ID:-${TASK_ID:-6}}"
    export TASK_IDS="${EVAL_TASK_IDS:-${TASK_IDS:-0 1 4 6 7}}"
    SCRIPT="run_lqr_gaussian.sh"
    ;;
  *)
    echo "Unsupported PERTURBATION=$PERTURBATION" >&2
    exit 2
    ;;
esac

export CONDA_ENV="${CONDA_ENV:-lingbot}"
export NUM_EPISODES="${NUM_EPISODES:-10}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
export SKIP_EVAL="${SKIP_EVAL:-1}"

echo "=== LingBot local LQR ==="
echo "PERTURBATION=$PERTURBATION"
echo "TASK_ID=$TASK_ID"
echo "TASK_IDS=$TASK_IDS"
echo "SCRIPT=$LINGBOT_DIR/$SCRIPT"
echo "SKIP_EVAL=$SKIP_EVAL"

cd "$LINGBOT_DIR"
exec bash "$SCRIPT"
