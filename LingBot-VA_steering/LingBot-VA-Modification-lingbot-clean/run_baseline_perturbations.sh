#!/bin/bash

source ~/.bashrc
set -euo pipefail

conda activate "${CONDA_ENV:-lingbot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${LINGBOT_REPO_DIR:-${SCRIPT_DIR}}"

export PYTHONPATH=.

export CONFIG_NAME="${CONFIG_NAME:-libero}"
export LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
export EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
export PROMPT="${PROMPT:-}"
export OUT_BASE="${OUT_BASE:-outputs/baseline}"
export CLIENT_EPISODE_BATCH_SIZE="${CLIENT_EPISODE_BATCH_SIZE:-1}"
export RESUME="${RESUME:-0}"

source scripts/lqr/slurm_port.sh

run_baseline() {
  local name="$1"
  local task_ids="$2"
  local perturb_spec="$3"
  local out_dir="${OUT_BASE}/${name}"
  local args=(
    python scripts/lqr/run_libero_policy_eval.py
    --config-name "${CONFIG_NAME}"
    --libero-benchmark "${LIBERO_BENCHMARK}"
    --task-ids ${task_ids}
    --num-episodes "${EVAL_NUM_EPISODES}"
    --startup-wait-sec "${EVAL_STARTUP_WAIT_SEC}"
    --port "${PORT}"
    --perturb-spec "${perturb_spec}"
    --out-dir "${out_dir}"
    --client-episode-batch-size "${CLIENT_EPISODE_BATCH_SIZE}"
  )

  if [[ -n "${PROMPT}" ]]; then
    args+=(--prompt "${PROMPT}")
  fi
  if [[ "${RESUME}" == "1" ]]; then
    args+=(--resume)
  fi

  echo "Running ${name} baseline:"
  echo "  TASK_IDS=${task_ids}"
  echo "  PERTURB_SPEC=${perturb_spec}"
  echo "  OUT_DIR=${out_dir}"
  echo "  PORT=${PORT} (job ${SLURM_JOB_ID:-local})"
  "${args[@]}"
}

run_baseline \
  "init_pos" \
  "${INIT_POS_TASK_IDS:-1 2 3 7 9}" \
  "${INIT_POS_PERTURB_SPEC:-scripts/lqr/configs/eval_init_pos.yaml}"

run_baseline \
  "gaussian" \
  "${GAUSSIAN_TASK_IDS:-6 0 1 4 7}" \
  "${GAUSSIAN_PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_gaussian_30.yaml}"

run_baseline \
  "camera" \
  "${CAMERA_TASK_IDS:-0 2 4 5 9}" \
  "${CAMERA_PERTURB_SPEC:-scripts/lqr/configs/eval_camera.yaml}"
