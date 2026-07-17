#!/usr/bin/env bash

# ===========================
# Lingbot LQR end-to-end runner
# ===========================
# Default usage:
#   bash scripts/lqr/run_lqr_pipeline.sh
#
# Common overrides:
#   NUM_EPISODES=5 TASK_RANGE_START=0 TASK_RANGE_END=2 bash scripts/lqr/run_lqr_pipeline.sh
#   PERTURB_SPEC=scripts/lqr/configs/perturb_spec_init_pos.yaml bash scripts/lqr/run_lqr_pipeline.sh
#   EXISTING_PAIRS_ALL_DIR=outputs/lqr/pairs_all_init_pos_4_20260525_015124 bash scripts/lqr/run_lqr_pipeline.sh
#   SKIP_EVAL=1 bash scripts/lqr/run_lqr_pipeline.sh

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

if [[ "${ACTIVATE_CONDA:-1}" == "1" ]]; then
  source ~/.bashrc
  conda activate "${CONDA_ENV:-lingbot}"
fi

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

# -------- Paired data collection / SVD --------
CONFIG_NAME="${CONFIG_NAME:-libero}"
LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
TASK_ID="${TASK_ID:-0}"
NUM_EPISODES="${NUM_EPISODES:-10}"
N_POS="${N_POS:-${NUM_EPISODES}}"
N_NEG="${N_NEG:-${NUM_EPISODES}}"
COLLECT_MODE="${COLLECT_MODE:-action}"
if [[ -z "${SELECTED_TIMESTEPS:-}" ]]; then
  if [[ "${COLLECT_MODE}" == "video" ]]; then
    SELECTED_TIMESTEPS="0,4,9,14,19"
  else
    SELECTED_TIMESTEPS="0,10,20,30,40"
  fi
fi
PERTURB_SPEC="${PERTURB_SPEC:-scripts/lqr/configs/perturb_spec_camera.yaml}"
EXISTING_COLLECT_DIR="${EXISTING_COLLECT_DIR:-}"
EXISTING_PAIRS_ALL_DIR="${EXISTING_PAIRS_ALL_DIR:-}"
EXISTING_SVD_DIR="${EXISTING_SVD_DIR:-}"
K_TARGET="${K_TARGET:-64}"
P_OVER="${P_OVER:-10}"
PARTITIONS="${PARTITIONS:-}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
JAC_METHOD="${JAC_METHOD:-vjp}"
JAC_OBS_INDEX="${JAC_OBS_INDEX:-0}"
JAC_NUM_SHARDS="${JAC_NUM_SHARDS:-1}"
JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_lingbot}"
JAC_RESUME="${JAC_RESUME:-0}"

OUT_BASE="${OUT_BASE:-outputs/lqr}"
PERTURB_SPEC_BASENAME="$(basename -- "${PERTURB_SPEC}")"
PERTURB_TAG="${PERTURB_SPEC_BASENAME%.*}"
PERTURB_TAG="${PERTURB_TAG#perturb_spec_}"
if [[ -z "${PERTURB_TAG}" ]]; then
  PERTURB_TAG="perturb"
fi
PAIRS_DIR="${PAIRS_DIR:-${OUT_BASE}/pairs_${PERTURB_TAG}_${TS}}"
PAIRED_DIR="${PAIRED_DIR:-${PAIRS_DIR}__paired}"
PAIR_INIT_BY_SIMILARITY="${PAIR_INIT_BY_SIMILARITY:-auto}"
PAIR_FEATURE="${PAIR_FEATURE:-proprio}"
PAIR_MATCH_MODE="${PAIR_MATCH_MODE:-nn-greedy}"
PAIR_MAX_ROWS="${PAIR_MAX_ROWS:--1}"
PAIR_MAX_DISTANCE="${PAIR_MAX_DISTANCE:--1}"
PAIRS_ALL_DIR="${PAIRS_ALL_DIR:-${PAIRS_DIR}}"
SVD_DIR="${SVD_DIR:-${OUT_BASE}/svd_all_perturb_${PERTURB_TAG}_${TS}}"

# -------- Evaluation --------
SKIP_EVAL="${SKIP_EVAL:-0}"
EVAL_OUT_BASE="${EVAL_OUT_BASE:-outputs/lqr_eval_all_perturb_${PERTURB_TAG}_${TS}}"
TASK_RANGE_START="${TASK_RANGE_START:-0}"
TASK_RANGE_END="${TASK_RANGE_END:-2}"
TASK_IDS="${TASK_IDS:-}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
# shellcheck source=scripts/lqr/slurm_port.sh
source "${REPO_ROOT}/scripts/lqr/slurm_port.sh"
EVAL_STARTUP_WAIT_SEC="${EVAL_STARTUP_WAIT_SEC:-1200}"
INJECT_MODE="${INJECT_MODE:-auto}"
PROMPT="${PROMPT:-}"
if [[ -z "${LQR_CONFIG:-}" ]]; then
  if [[ "${COLLECT_MODE}" == "video" ]]; then
    LQR_CONFIG="scripts/lqr/configs/lqr_config_video.yaml"
  else
    LQR_CONFIG="scripts/lqr/configs/lqr_config.yaml"
  fi
fi
AGENTVIEW_NOISE_SEED_BASE="${AGENTVIEW_NOISE_SEED_BASE:-}"
AGENTVIEW_NOISE_SIGMA="${AGENTVIEW_NOISE_SIGMA:-}"
GRIPPER_XYZ_BASE_SEED="${GRIPPER_XYZ_BASE_SEED:-}"
RANDOM_CAMERA_BASE_SEED="${RANDOM_CAMERA_BASE_SEED:-}"

echo "===== Lingbot LQR Pipeline ====="
echo "REPO_ROOT=${REPO_ROOT}"
echo "TASK_ID=${TASK_ID}"
echo "N_POS=${N_POS}"
echo "N_NEG=${N_NEG}"
echo "PERTURB_SPEC=${PERTURB_SPEC}"
echo "PERTURB_TAG=${PERTURB_TAG}"
echo "EXISTING_COLLECT_DIR=${EXISTING_COLLECT_DIR:-<none>}"
echo "EXISTING_PAIRS_ALL_DIR=${EXISTING_PAIRS_ALL_DIR:-<none>}"
echo "EXISTING_SVD_DIR=${EXISTING_SVD_DIR:-<none>}"
echo "PAIRS_DIR=${PAIRS_DIR}"
echo "PAIRS_ALL_DIR=${PAIRS_ALL_DIR}"
echo "PAIRED_DIR=${PAIRED_DIR}"
echo "PAIR_INIT_BY_SIMILARITY=${PAIR_INIT_BY_SIMILARITY}"
echo "PARTITIONS=${PARTITIONS:-<auto 3-partition>}"
echo "JAC_METHOD=${JAC_METHOD}"
echo "SVD_DIR=${SVD_DIR}"
echo "EVAL_OUT_BASE=${EVAL_OUT_BASE}"
echo "TASK_IDS=${TASK_IDS:-<unset>}"
echo "PORT=${PORT} (SLURM_JOB_ID=${SLURM_JOB_ID:-<none>})"
echo "INJECT_MODE=${INJECT_MODE}"
echo "EVAL_STARTUP_WAIT_SEC=${EVAL_STARTUP_WAIT_SEC}"
echo "AGENTVIEW_NOISE_SEED_BASE=${AGENTVIEW_NOISE_SEED_BASE:-<unset>}"
echo "AGENTVIEW_NOISE_SIGMA=${AGENTVIEW_NOISE_SIGMA:-<unset>}"
echo "GRIPPER_XYZ_BASE_SEED=${GRIPPER_XYZ_BASE_SEED:-<unset>}"
echo "RANDOM_CAMERA_BASE_SEED=${RANDOM_CAMERA_BASE_SEED:-<unset>}"
echo "TS=${TS}"
echo "================================"

PAIRS_DIR_FOR_SVD="${PAIRS_ALL_DIR}"
COLLECT_DIR_FOR_PAIRING="${PAIRS_DIR}"
if [[ -n "${EXISTING_PAIRS_ALL_DIR}" ]]; then
  if [[ ! -f "${EXISTING_PAIRS_ALL_DIR}/manifest.json" ]]; then
    echo "[error] EXISTING_PAIRS_ALL_DIR missing manifest.json: ${EXISTING_PAIRS_ALL_DIR}" >&2
    exit 2
  fi
  if [[ ! -f "${EXISTING_PAIRS_ALL_DIR}/positive.npz" ]]; then
    echo "[error] EXISTING_PAIRS_ALL_DIR missing positive.npz: ${EXISTING_PAIRS_ALL_DIR}" >&2
    exit 2
  fi
  if [[ ! -f "${EXISTING_PAIRS_ALL_DIR}/negative.npz" ]]; then
    echo "[error] EXISTING_PAIRS_ALL_DIR missing negative.npz: ${EXISTING_PAIRS_ALL_DIR}" >&2
    exit 2
  fi
  PAIRS_DIR_FOR_SVD="${EXISTING_PAIRS_ALL_DIR}"
  PAIRS_ALL_DIR="${EXISTING_PAIRS_ALL_DIR}"
  echo "[1/4] skip collect, reuse existing paired NPZs: ${PAIRS_DIR_FOR_SVD}"
else
  if [[ -n "${EXISTING_COLLECT_DIR}" ]]; then
    if [[ ! -f "${EXISTING_COLLECT_DIR}/manifest.json" ]]; then
      echo "[error] EXISTING_COLLECT_DIR missing manifest.json: ${EXISTING_COLLECT_DIR}" >&2
      exit 2
    fi
    if [[ ! -f "${EXISTING_COLLECT_DIR}/positive.npz" || ! -f "${EXISTING_COLLECT_DIR}/negative.npz" ]]; then
      echo "[error] EXISTING_COLLECT_DIR must contain positive.npz and negative.npz: ${EXISTING_COLLECT_DIR}" >&2
      exit 2
    fi
    COLLECT_DIR_FOR_PAIRING="${EXISTING_COLLECT_DIR}"
    PAIRS_DIR_FOR_SVD="${EXISTING_COLLECT_DIR}"
    PAIRS_ALL_DIR="${EXISTING_COLLECT_DIR}"
    echo "[1/4] skip collect, reuse existing paired collect dir: ${COLLECT_DIR_FOR_PAIRING}"
  else
    COLLECT_CMD=(
      python scripts/lqr/run_collect_pairs.py
      --config-name "${CONFIG_NAME}"
      --libero-benchmark "${LIBERO_BENCHMARK}"
      --task-id "${TASK_ID}"
      --num-episodes "${NUM_EPISODES}"
      --n-pos-rollouts "${N_POS}"
      --n-neg-rollouts "${N_NEG}"
      --perturb-spec "${PERTURB_SPEC}"
      --out-dir "${PAIRS_DIR}"
    )
    echo "[1/5] collect positive/negative observations"
    set +e
    "${COLLECT_CMD[@]}"
    status=$?
    set -e
    if [[ "${status}" -ne 0 ]]; then
      if [[ -f "${PAIRS_DIR}/positive.npz" && -f "${PAIRS_DIR}/negative.npz" ]]; then
        echo "[1/5] warning: collect exited with status ${status}, but positive.npz/negative.npz exist; continuing with checkpointed outputs"
      else
        exit "${status}"
      fi
    fi
  fi
fi

if [[ -z "${EXISTING_PAIRS_ALL_DIR}" ]]; then
  SHOULD_PAIR_INIT=0
  if [[ "${PAIR_INIT_BY_SIMILARITY}" == "1" ]]; then
    SHOULD_PAIR_INIT=1
  elif [[ "${PAIR_INIT_BY_SIMILARITY}" == "auto" && "${PERTURB_TAG}" == *"init"* ]]; then
    SHOULD_PAIR_INIT=1
  fi
  if [[ "${SHOULD_PAIR_INIT}" == "1" ]]; then
    echo "[2/5] pair init-position success/failure buckets by similarity"
    python scripts/lqr/pair_inputs_by_similarity.py \
      --in-dir "${COLLECT_DIR_FOR_PAIRING}" \
      --out-dir "${PAIRED_DIR}" \
      --feature "${PAIR_FEATURE}" \
      --match-mode "${PAIR_MATCH_MODE}" \
      --max-rows "${PAIR_MAX_ROWS}" \
      --max-distance "${PAIR_MAX_DISTANCE}"
    PAIRS_DIR_FOR_SVD="${PAIRED_DIR}"
    PAIRS_ALL_DIR="${PAIRED_DIR}"
  else
    echo "[2/5] no extra pairing needed for ${PERTURB_TAG}"
  fi
else
  echo "[2/5] skip similarity pairing because EXISTING_PAIRS_ALL_DIR was provided"
fi

echo "[3/5] run SVD / build contrastive vectors"
if [[ -n "${EXISTING_SVD_DIR}" ]]; then
  if [[ ! -f "${EXISTING_SVD_DIR}/config.json" ]]; then
    echo "[error] EXISTING_SVD_DIR missing config.json: ${EXISTING_SVD_DIR}" >&2
    exit 2
  fi
  if [[ ! -f "${EXISTING_SVD_DIR}/svd_summary.pt" ]]; then
    echo "[error] EXISTING_SVD_DIR missing svd_summary.pt: ${EXISTING_SVD_DIR}" >&2
    exit 2
  fi
  SVD_DIR="${EXISTING_SVD_DIR}"
  echo "[3/5] skip SVD, reuse existing artifacts: ${SVD_DIR}"
else
SVD_CMD=(
  python scripts/lqr/run_partition_svd.py
  --pairs-dir "${PAIRS_DIR_FOR_SVD}"
  --out-dir "${SVD_DIR}"
  --config-name "${CONFIG_NAME}"
  --mode "${COLLECT_MODE}"
  --selected-timesteps "${SELECTED_TIMESTEPS}"
  --num-samples "${NUM_SAMPLES}"
  --k-target "${K_TARGET}"
  --p-over "${P_OVER}"
)
if [[ -n "${PARTITIONS}" ]]; then
  SVD_CMD+=(--partitions "${PARTITIONS}")
fi
if [[ -n "${PROMPT}" ]]; then
  SVD_CMD+=(--prompt "${PROMPT}")
fi
"${SVD_CMD[@]}"
fi

echo "[4/5] compute projected jacobians - repo-root-aligned VJP"
if [[ -z "${FORCE_STEP4:-}" && -f "${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt" ]]; then
  echo "[4/5] skip jacobian, existing artifact found: ${SVD_DIR}/${JAC_SUBDIR}/A_tilde__full.pt"
else
  JAC_CMD=(
    python scripts/lqr/run_compute_jacobians.py
    --svd-dir "${SVD_DIR}"
    --out-subdir "${JAC_SUBDIR}"
    --inputs-npz "${PAIRS_DIR_FOR_SVD}/negative.npz"
    --obs-index "${JAC_OBS_INDEX:-0}"
    --config-name "${CONFIG_NAME}"
    --num-shards "${JAC_NUM_SHARDS:-1}"
    --method "${JAC_METHOD:-vjp}"
  )
  if [[ "${JAC_RESUME}" == "1" ]]; then
    JAC_CMD+=(--resume)
  fi
  "${JAC_CMD[@]}"
fi

if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[5/5] skipped eval (SKIP_EVAL=1)"
else
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
    --out-dir "${EVAL_OUT_BASE}"
  )
  if [[ -n "${TASK_IDS}" ]]; then
    TASK_IDS_NORMALIZED="${TASK_IDS//,/ }"
    read -r -a TASK_IDS_ARR <<< "${TASK_IDS_NORMALIZED}"
    EVAL_CMD+=(--task-ids "${TASK_IDS_ARR[@]}")
  fi
  if [[ -n "${PROMPT}" ]]; then
    EVAL_CMD+=(--prompt "${PROMPT}")
  fi
  if [[ -n "${AGENTVIEW_NOISE_SEED_BASE}" ]]; then
    EVAL_CMD+=(--agentview-noise-seed-base "${AGENTVIEW_NOISE_SEED_BASE}")
  fi
  if [[ -n "${AGENTVIEW_NOISE_SIGMA}" ]]; then
    EVAL_CMD+=(--agentview-noise-sigma "${AGENTVIEW_NOISE_SIGMA}")
  fi
  if [[ -n "${GRIPPER_XYZ_BASE_SEED}" ]]; then
    EVAL_CMD+=(--gripper-xyz-base-seed "${GRIPPER_XYZ_BASE_SEED}")
  fi
  if [[ -n "${RANDOM_CAMERA_BASE_SEED}" ]]; then
    EVAL_CMD+=(--random-camera-base-seed "${RANDOM_CAMERA_BASE_SEED}")
  fi
  if [[ "${RESUME_EVAL:-0}" == "1" ]]; then
    EVAL_CMD+=(--resume)
  fi
  echo "[5/5] run lqr eval"
  "${EVAL_CMD[@]}"
fi

echo ""
echo "Done."
echo "collect_dir : ${COLLECT_DIR_FOR_PAIRING}"
echo "pairs_npz   : ${PAIRS_DIR_FOR_SVD}"
echo "svd_dir     : ${SVD_DIR}"
if [[ "${SKIP_EVAL}" != "1" ]]; then
  echo "eval_out    : ${EVAL_OUT_BASE}"
fi
