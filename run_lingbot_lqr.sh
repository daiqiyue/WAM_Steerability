#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LINGBOT_DIR="${LINGBOT_DIR:-$ROOT/LingBot-VA_steering/LingBot-VA-Modification-lingbot-clean}"

PERTURBATION="${PERTURBATION:-init_pos}"     # init_pos | gaussian | camera
case "$PERTURBATION" in
  camera|cam)
    DEFAULT_COLLECT_TASK_ID=0
    DEFAULT_EVAL_TASK_IDS="0 2 4 5 9"
    ;;
  init_pos|init|position|gripper|gripper_xyz)
    DEFAULT_COLLECT_TASK_ID=1
    DEFAULT_EVAL_TASK_IDS="1 2 3 7 9"
    ;;
  gaussian|noise)
    DEFAULT_COLLECT_TASK_ID=6
    DEFAULT_EVAL_TASK_IDS="0 1 4 6 7"
    ;;
  *)
    DEFAULT_COLLECT_TASK_ID=1
    DEFAULT_EVAL_TASK_IDS="1"
    ;;
esac
export TASK_ID="${COLLECT_TASK_ID:-${TASK_ID:-$DEFAULT_COLLECT_TASK_ID}}"
export TASK_IDS="${EVAL_TASK_IDS:-${TASK_IDS:-$DEFAULT_EVAL_TASK_IDS}}"
export LINGBOT_REPO_DIR="${LINGBOT_REPO_DIR:-$LINGBOT_DIR}"
export SKIP_EVAL="${SKIP_EVAL:-0}"

case "$PERTURBATION" in
  init_pos|init|position|gripper|gripper_xyz)
    exec bash "$LINGBOT_DIR/run_lqr_init_pos.sh"
    ;;
  gaussian|noise)
    exec bash "$LINGBOT_DIR/run_lqr_gaussian.sh"
    ;;
  camera|cam)
    exec bash "$LINGBOT_DIR/run_lqr_camera.sh"
    ;;
  *)
    echo "Unknown PERTURBATION=$PERTURBATION (use init_pos, gaussian, or camera)" >&2
    exit 2
    ;;
esac
