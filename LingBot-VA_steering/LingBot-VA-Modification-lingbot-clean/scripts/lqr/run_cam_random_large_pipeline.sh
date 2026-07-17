#!/usr/bin/env bash

# LingBot camera-view LQR pipeline aligned with
# the repo root's notebooks/lqr/e2e_scripts/run_cam_random_large_pipeline.sh.
#
# Usage:
#   bash scripts/lqr/run_cam_random_large_pipeline.sh
#   START_AT=3 bash scripts/lqr/run_cam_random_large_pipeline.sh
#   SUBMIT_EVAL=0 bash scripts/lqr/run_cam_random_large_pipeline.sh
#   N_POS=1 N_NEG=1 NUM_SAMPLES=-1 K_TARGET=1 SUBMIT_EVAL=0 bash scripts/lqr/run_cam_random_large_pipeline.sh

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

if [[ "${ACTIVATE_CONDA:-1}" == "1" ]]; then
  source ~/.bashrc
  conda activate "${CONDA_ENV:-lingbot}"
fi

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
START_AT="${START_AT:-1}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"

CONFIG_NAME="${CONFIG_NAME:-libero}"
LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
TASK_ID="${TASK_ID:-0}"
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"

PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_camera.yaml}"
CAM_BASE_SEED="${CAM_BASE_SEED:-42}"
EVAL_CAM_BASE_SEED="${EVAL_CAM_BASE_SEED:-99}"
N_POS="${N_POS:-10}"
N_NEG="${N_NEG:-10}"

SELECTED_TIMESTEPS="${SELECTED_TIMESTEPS:-0,10,20,30,40}"
COLLECT_MODE="${COLLECT_MODE:-action}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
K_TARGET="${K_TARGET:-64}"
P_OVER="${P_OVER:-10}"
PARTITIONS="${PARTITIONS:-}"
JAC_METHOD="${JAC_METHOD:-vjp}"
JAC_OBS_INDEX="${JAC_OBS_INDEX:-0}"
JAC_NUM_SHARDS="${JAC_NUM_SHARDS:-1}"
JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot}"

OUT_BASE="${OUT_BASE:-outputs/lqr}"
PAIRS_DIR="${PAIRS_DIR:-${OUT_BASE}/pairs_cam_random_large_seed${CAM_BASE_SEED}_${TS}}"
SVD_DIR="${SVD_DIR:-${OUT_BASE}/svd_cam_random_large_seed${CAM_BASE_SEED}_${TS}}"

TASK_RANGE_START="${TASK_RANGE_START:-0}"
TASK_RANGE_END="${TASK_RANGE_END:-1}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
EVAL_OUT_BASE="${EVAL_OUT_BASE:-outputs/lqr_eval_cam_random_large_seed${EVAL_CAM_BASE_SEED}_${TS}}"
LQR_CONFIG="${LQR_CONFIG:-scripts/lqr/configs/lqr_config.yaml}"
INJECT_MODE="${INJECT_MODE:-auto}"
EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
# shellcheck source=scripts/lqr/slurm_port.sh
source "${REPO_ROOT}/scripts/lqr/slurm_port.sh"

should_run_step() {
  local n="$1"
  (( n >= START_AT ))
}

echo "===== LingBot cam_random_large LQR pipeline ====="
echo "REPO_ROOT=${REPO_ROOT}"
echo "TASK=${LIBERO_BENCHMARK}/task${TASK_ID}"
echo "PROMPT=${PROMPT}"
echo "PERTURB_SPEC=${PERTURB_SPEC}"
echo "CAM_BASE_SEED=${CAM_BASE_SEED}"
echo "EVAL_CAM_BASE_SEED=${EVAL_CAM_BASE_SEED}"
echo "N_POS=${N_POS} N_NEG=${N_NEG}"
echo "PAIRS_DIR=${PAIRS_DIR}"
echo "SVD_DIR=${SVD_DIR}"
echo "JAC_SUBDIR=${JAC_SUBDIR}"
echo "SUBMIT_EVAL=${SUBMIT_EVAL}"
echo "START_AT=${START_AT}"
echo "==============================================="

if should_run_step 1; then
  echo "[1/4] collect repo-root-style camera positive/negative pairs"
  if [[ -z "${FORCE_STEP1:-}" && -f "${PAIRS_DIR}/positive.npz" && -f "${PAIRS_DIR}/negative.npz" ]]; then
    echo "[skip] pair NPZs already exist under ${PAIRS_DIR}"
  else
    COLLECT_CMD=(
      python scripts/lqr/run_collect_pairs.py
      --config-name "${CONFIG_NAME}" \
      --libero-benchmark "${LIBERO_BENCHMARK}" \
      --task-id "${TASK_ID}" \
      --num-episodes "$(( N_POS > N_NEG ? N_POS : N_NEG ))" \
      --n-pos-rollouts "${N_POS}" \
      --n-neg-rollouts "${N_NEG}" \
      --perturb-spec "${PERTURB_SPEC}" \
      --out-dir "${PAIRS_DIR}"
    )
    if ! "${COLLECT_CMD[@]}"; then
      if [[ -f "${PAIRS_DIR}/positive.npz" && -f "${PAIRS_DIR}/negative.npz" ]]; then
        echo "[warn] collect command exited non-zero, but pair NPZs exist; continuing with ${PAIRS_DIR}"
      else
        exit 1
      fi
    fi
  fi
fi

if should_run_step 2; then
  echo "[2/4] run SVD over all camera pairs"
  if [[ -z "${FORCE_STEP2:-}" && -f "${SVD_DIR}/svd_summary.pt" ]]; then
    echo "[skip] SVD summary already exists at ${SVD_DIR}/svd_summary.pt"
  else
    SVD_CMD=(
      python scripts/lqr/run_partition_svd.py
      --pairs-dir "${PAIRS_DIR}"
      --out-dir "${SVD_DIR}"
      --config-name "${CONFIG_NAME}"
      --mode "${COLLECT_MODE}"
      --selected-timesteps "${SELECTED_TIMESTEPS}"
      --num-samples "${NUM_SAMPLES}"
      --k-target "${K_TARGET}"
      --p-over "${P_OVER}"
      --prompt "${PROMPT}"
    )
    if [[ -n "${PARTITIONS}" ]]; then
      SVD_CMD+=(--partitions "${PARTITIONS}")
    fi
    "${SVD_CMD[@]}"
  fi
fi

if should_run_step 3; then
  echo "[3/4] compute projected jacobians from perturbed-camera row ${JAC_OBS_INDEX}"
  if [[ -z "${FORCE_STEP3:-}" && -f "${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt" ]]; then
    echo "[skip] jacobian already exists at ${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt"
  else
    python scripts/lqr/run_compute_jacobians.py \
      --svd-dir "${SVD_DIR}" \
      --out-subdir "${JAC_SUBDIR}" \
      --inputs-npz "${PAIRS_DIR}/negative.npz" \
      --obs-index "${JAC_OBS_INDEX}" \
      --config-name "${CONFIG_NAME}" \
      --num-shards "${JAC_NUM_SHARDS}" \
      --method "${JAC_METHOD}"
  fi
fi

if should_run_step 4; then
  if [[ "${SUBMIT_EVAL}" != "1" ]]; then
    echo "[4/4] skipped eval (SUBMIT_EVAL=${SUBMIT_EVAL})"
  else
    echo "[4/4] run camera-perturbed LQR eval with the original sweep seed ${EVAL_CAM_BASE_SEED}"
    EVAL_CMD=(
      python scripts/lqr/run_libero_lqr_eval.py
      --config-name "${CONFIG_NAME}"
      --libero-benchmark "${LIBERO_BENCHMARK}"
      --task-range "${TASK_RANGE_START}" "${TASK_RANGE_END}"
      --num-episodes "${EVAL_NUM_EPISODES}"
      --startup-wait-sec "${EVAL_STARTUP_WAIT_SEC}"
      --port "${PORT}"
      --svd-dir "${SVD_DIR}"
      --jac-dir-act "${JAC_SUBDIR}"
      --lqr-config "${LQR_CONFIG}"
      --inject-mode "${INJECT_MODE}"
      --perturb-spec "${PERTURB_SPEC}"
      --random-camera-base-seed "${EVAL_CAM_BASE_SEED}"
      --prompt "${PROMPT}"
      --out-dir "${EVAL_OUT_BASE}"
    )
    if [[ "${RESUME_EVAL:-0}" == "1" ]]; then
      EVAL_CMD+=(--resume)
    fi
    "${EVAL_CMD[@]}"
  fi
fi

echo ""
echo "Done."
echo "pairs_dir : ${PAIRS_DIR}"
echo "svd_dir   : ${SVD_DIR}"
echo "jac       : ${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt"
if [[ "${SUBMIT_EVAL}" == "1" ]]; then
  echo "eval_out  : ${EVAL_OUT_BASE}"
fi
