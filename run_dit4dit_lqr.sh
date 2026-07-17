#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DIT_DIR="${DIT_DIR:-$ROOT/DiT4DiT_steering}"
export DIT4DIT_ROOT="${DIT4DIT_ROOT:-$DIT_DIR}"
export LQR_ROOT="${DIT4DIT_LQR_ROOT:-${LQR_ROOT:-$DIT_DIR/lqr}}"

PERTURBATION="${PERTURBATION:-noise}"        # noise | gaussian | gripper
case "$PERTURBATION" in
  gripper|gripper_xyz)
    DEFAULT_COLLECT_TASK_ID=1
    DEFAULT_EVAL_TASK_IDS="1 2 3 7 9"
    ;;
  noise|gaussian)
    DEFAULT_COLLECT_TASK_ID=6
    DEFAULT_EVAL_TASK_IDS="0 1 4 6 7"
    ;;
  *)
    DEFAULT_COLLECT_TASK_ID=1
    DEFAULT_EVAL_TASK_IDS="1"
    ;;
esac
COLLECT_TASK_ID="${COLLECT_TASK_ID:-${TASK_ID:-$DEFAULT_COLLECT_TASK_ID}}"
EVAL_TASK_IDS="${EVAL_TASK_IDS:-${TASK_IDS:-${EVAL_TASK_ID:-${TASK2_ID:-$DEFAULT_EVAL_TASK_IDS}}}}"

export TASK_ID="$COLLECT_TASK_ID"
export START_AT="${START_AT:-1}"
export N_EPISODES="${N_EPISODES:-${NUM_EPISODES:-30}}"
export OUT_BASE="${OUT_BASE:-$LQR_ROOT/rollouts}"

run_one_eval_task() {
  local eval_task_id="$1"
  TASK_ID="$COLLECT_TASK_ID" TASK2_ID="$eval_task_id" \
    PERTURBATION="$PERTURBATION" START_AT="$START_AT" N_EPISODES="$N_EPISODES" \
    OUT_BASE="$OUT_BASE" LQR_ROOT="$LQR_ROOT" DIT4DIT_ROOT="$DIT4DIT_ROOT" \
    bash -c '
      set -euo pipefail
      case "$PERTURBATION" in
        noise|gaussian)
          if [[ "$TASK2_ID" == "$TASK_ID" ]]; then
            exec bash "$LQR_ROOT/e2e_scripts/run_noised_pipeline.sh"
          else
            exec bash "$LQR_ROOT/e2e_scripts/run_noise_pipeline_generalize.sh"
          fi
          ;;
        gripper|gripper_xyz)
          exec bash "$LQR_ROOT/e2e_scripts/run_gripper_xyz_pipeline_generalize.sh"
          ;;
        *)
          echo "Unknown PERTURBATION=$PERTURBATION (DiT4DiT supports noise/gaussian or gripper)" >&2
          exit 2
          ;;
      esac
    '
}

case "$PERTURBATION" in
  noise|gaussian)
    for eval_task_id in $EVAL_TASK_IDS; do
      echo "=== DiT4DiT $PERTURBATION: collect task $COLLECT_TASK_ID, eval task $eval_task_id ==="
      run_one_eval_task "$eval_task_id"
    done
    ;;
  gripper|gripper_xyz)
    for eval_task_id in $EVAL_TASK_IDS; do
      echo "=== DiT4DiT gripper: collect task $COLLECT_TASK_ID, eval task $eval_task_id ==="
      run_one_eval_task "$eval_task_id"
    done
    ;;
  *)
    echo "Unknown PERTURBATION=$PERTURBATION (DiT4DiT supports noise/gaussian or gripper)" >&2
    exit 2
    ;;
esac
