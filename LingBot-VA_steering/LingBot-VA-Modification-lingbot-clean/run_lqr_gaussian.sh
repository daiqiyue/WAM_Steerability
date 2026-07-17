#!/bin/bash

source ~/.bashrc
set -euo pipefail

conda activate "${CONDA_ENV:-lingbot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${LINGBOT_REPO_DIR:-${SCRIPT_DIR}}"

export PYTHONPATH=.

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
DEFAULT_MASTER_PORT="$((12000 + (${SLURM_JOB_ID:-0} % 20000)))"
export MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}"

export CONFIG_NAME="${CONFIG_NAME:-libero}"
export LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
export TASK_ID="${TASK_ID:-6}"
export NUM_EPISODES="${NUM_EPISODES:-20}"
export COLLECT_MODE="${COLLECT_MODE:-action}"
if [[ -z "${SELECTED_TIMESTEPS:-}" ]]; then
  if [[ "${COLLECT_MODE}" == "video" ]]; then
    export SELECTED_TIMESTEPS="0,4,9,14,19"
  else
    export SELECTED_TIMESTEPS="0,10,20,30,40"
  fi
else
  export SELECTED_TIMESTEPS
fi
export PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_gaussian_30.yaml}"
export PAIR_INIT_BY_SIMILARITY="${PAIR_INIT_BY_SIMILARITY:-0}"
export NUM_SAMPLES="${NUM_SAMPLES:-200}"
export K_TARGET="${K_TARGET:-64}"
export P_OVER="${P_OVER:-10}"
export PARTITIONS="${PARTITIONS:-}"
export JAC_METHOD="${JAC_METHOD:-vjp}"
export JAC_OBS_INDEX="${JAC_OBS_INDEX:-0}"
export JAC_NUM_SHARDS="${JAC_NUM_SHARDS:-1}"
export JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot}"
if [[ -z "${LQR_CONFIG:-}" ]]; then
  if [[ "${COLLECT_MODE}" == "video" ]]; then
    export LQR_CONFIG="scripts/lqr/configs/lqr_config_video.yaml"
  else
    export LQR_CONFIG="scripts/lqr/configs/lqr_config.yaml"
  fi
else
  export LQR_CONFIG
fi
export TASK_RANGE_START="${TASK_RANGE_START:-6}"
export TASK_RANGE_END="${TASK_RANGE_END:-7}"
export TASK_IDS="${TASK_IDS:-6 0 1 4 7}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
source scripts/lqr/slurm_port.sh
export EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
export INJECT_MODE="${INJECT_MODE:-auto}"
export SKIP_EVAL="${SKIP_EVAL:-0}"
export PROMPT="${PROMPT:-}"

echo "Running GAUSSIAN-LQR pipeline with:"
echo "  REPO=$(pwd)"
echo "  TASK_ID=${TASK_ID} (collect)"
echo "  PERTURB_SPEC=${PERTURB_SPEC}"
echo "  PARTITIONS=${PARTITIONS:-<auto 3-partition>}"
echo "  JAC_METHOD=${JAC_METHOD}"
echo "  TASK_IDS=${TASK_IDS}"
echo "  TASK_RANGE=[${TASK_RANGE_START}, ${TASK_RANGE_END})"
echo "  SKIP_EVAL=${SKIP_EVAL}"

export ACTIVATE_CONDA=0
bash scripts/lqr/run_lqr_pipeline.sh
