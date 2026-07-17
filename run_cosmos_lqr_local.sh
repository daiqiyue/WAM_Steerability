#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
COSMOS_DIR="${COSMOS_DIR:-$ROOT/cosmos_steering}"
LQR_ROOT="${COSMOS_LQR_ROOT:-$COSMOS_DIR/notebooks/lqr}"
PYTHON="${PYTHON:-python}"

PERTURBATION="${PERTURBATION:-camera}"
SUITE="${SUITE:-libero_10}"
N_EPISODES="${N_EPISODES:-10}"
N_POS="${N_POS:-$N_EPISODES}"
N_NEG="${N_NEG:-$N_EPISODES}"
RESOLUTION="${RESOLUTION:-256}"
START_AT="${START_AT:-1}"
SKIP_EVAL="${SKIP_EVAL:-1}"

WORLD_SIZE="${WORLD_SIZE:-1}"
SVD_WORLD_SIZE="${SVD_WORLD_SIZE:-$WORLD_SIZE}"
JAC_WORLD_SIZE="${JAC_WORLD_SIZE:-$WORLD_SIZE}"
EVAL_WORLD_SIZE="${EVAL_WORLD_SIZE:-$WORLD_SIZE}"

K_TARGET="${K_TARGET:-64}"
P_OVER="${P_OVER:-10}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
TIMESTEPS="${TIMESTEPS:-all}"
SAMPLING_STEPS="${SAMPLING_STEPS:-5}"
PARTITIONS="${PARTITIONS:-0-9,10-18,19-27}"
DRIVE_SOURCE="${DRIVE_SOURCE:-all}"
JAC_MODE="${JAC_MODE:-vjp_no_retain}"
V_DTYPE="${V_DTYPE:-bf16}"
OBS_INDEX="${OBS_INDEX:-0}"
CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
OUT_BASE="${OUT_BASE:-$LQR_ROOT/local_runs}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

check_cosmos_env() {
  "$PYTHON" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys

def get_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None

def major_minor(ver):
    parts = []
    for chunk in ver.replace("-", ".").split("."):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            digits = ""
            for ch in chunk:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                parts.append(int(digits))
            break
    return tuple(parts + [0] * (2 - len(parts)))

robosuite_v = get_version("robosuite")
mujoco_v = get_version("mujoco")
if robosuite_v is None or mujoco_v is None:
    missing = []
    if robosuite_v is None:
        missing.append("robosuite")
    if mujoco_v is None:
        missing.append("mujoco")
    sys.exit(
        "Missing Cosmos/LIBERO simulation dependency: "
        + ", ".join(missing)
        + "\nInstall robosuite==1.4.1 and mujoco==2.3.7 in the active env."
    )

if major_minor(robosuite_v) < (1, 5) and major_minor(mujoco_v) >= (3, 0):
    sys.exit(
        f"Incompatible robosuite/mujoco versions: robosuite=={robosuite_v}, "
        f"mujoco=={mujoco_v}.\n"
        "LIBERO with robosuite==1.4.1 uses the MuJoCo 2.x mj_fullM API. "
        "Run: python -m pip install 'mujoco==2.3.7'"
    )
PY
}

if [[ "${SKIP_ENV_CHECK:-0}" != "1" ]]; then
  check_cosmos_env
fi

if [[ -n "${FORCE_STEP1:-}" ]]; then
  export FORCE_STEP2="${FORCE_STEP2:-1}"
  export FORCE_STEP3="${FORCE_STEP3:-1}"
fi

prompt_for_task() {
  case "$1" in
    0) echo "put both the alphabet soup and the tomato sauce in the basket" ;;
    1) echo "put both the cream cheese box and the butter in the basket" ;;
    2) echo "turn on the stove and put the moka pot on it" ;;
    3) echo "put the black bowl in the bottom drawer of the cabinet and close it" ;;
    4) echo "put the white mug on the left plate and put the yellow and white mug on the right plate" ;;
    5) echo "pick up the book and place it in the back compartment of the caddy" ;;
    6) echo "put the white mug on the plate and put the chocolate pudding to the right of the plate" ;;
    7) echo "put both the alphabet soup and the cream cheese box in the basket" ;;
    8) echo "put both moka pots on the stove" ;;
    9) echo "put the yellow and white mug in the microwave and close it" ;;
    *) echo "" ;;
  esac
}

slug() {
  printf '%s' "$1" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50
}

case "$PERTURBATION" in
  camera|cam)
    COLLECT_TASK_ID="${COLLECT_TASK_ID:-0}"
    EVAL_TASK_IDS="${EVAL_TASK_IDS:-0 2 4 5 9}"
    KIND="camera"
    PRESET_NAME="${PRESET_NAME:-cam_random_large}"
    CAM_BASE_SEED="${CAM_BASE_SEED:-42}"
    TAG="libero10_task$(printf '%02d' "$COLLECT_TASK_ID")_${PRESET_NAME}_seed${CAM_BASE_SEED}_pairs_no_action_ds${DRIVE_SOURCE}"
    INPUT_DIR="${INPUT_DIR:-$OUT_BASE/inputs/${SUITE}__task$(printf '%02d' "$COLLECT_TASK_ID")__${PRESET_NAME}__seed${CAM_BASE_SEED}__pos_neg}"
    ;;
  gripper|gripper_xyz)
    COLLECT_TASK_ID="${COLLECT_TASK_ID:-1}"
    EVAL_TASK_IDS="${EVAL_TASK_IDS:-1 2 3 7 9}"
    KIND="gripper"
    PRESET="${PRESET:-xyz_random_xlarge_3}"
    BASE_SEED="${BASE_SEED:-42}"
    TAG="libero10_task$(printf '%02d' "$COLLECT_TASK_ID")_${PRESET}_seed${BASE_SEED}_paired_no_action_ds${DRIVE_SOURCE}"
    INPUT_DIR="${INPUT_DIR:-$OUT_BASE/inputs/${SUITE}__task$(printf '%02d' "$COLLECT_TASK_ID")__${PRESET}__seed${BASE_SEED}__pos_neg}"
    PAIRED_DIR="${PAIRED_DIR:-${INPUT_DIR}__paired}"
    ;;
  gaussian|noise|noise_extreme)
    COLLECT_TASK_ID="${COLLECT_TASK_ID:-6}"
    EVAL_TASK_IDS="${EVAL_TASK_IDS:-0 1 4 6 7}"
    KIND="noise"
    NOISE_SIGMA="${NOISE_SIGMA:-90.0}"
    TAG="libero10_task$(printf '%02d' "$COLLECT_TASK_ID")_noise_sigma${NOISE_SIGMA}_pairs_no_action_ds${DRIVE_SOURCE}"
    INPUT_DIR="${INPUT_DIR:-$OUT_BASE/inputs/${SUITE}__task$(printf '%02d' "$COLLECT_TASK_ID")__noise_sigma${NOISE_SIGMA}}"
    ;;
  *)
    echo "Unsupported PERTURBATION=$PERTURBATION" >&2
    exit 2
    ;;
esac

PROMPT="${PROMPT:-$(prompt_for_task "$COLLECT_TASK_ID")}"
SVD_DIR="${SVD_DIR:-$OUT_BASE/svd/${TAG}_N${NUM_SAMPLES}_k${K_TARGET}_p${P_OVER}_ws${SVD_WORLD_SIZE}_ts${TIMESTEPS}}"
JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_full__$(slug "$PROMPT")__${JAC_MODE}__v${V_DTYPE}}"
JAC_DIR="${JAC_DIR:-$SVD_DIR/$JAC_SUBDIR}"
PAIR_DIR="$INPUT_DIR"
if [[ "$KIND" == "gripper" ]]; then
  PAIR_DIR="$PAIRED_DIR"
fi

should_run() {
  (( "$1" >= START_AT ))
}

run_svd() {
  local pos="$1" neg="$2"
  mkdir -p "$SVD_DIR"
  if [[ -z "${FORCE_STEP2:-}" && -f "$SVD_DIR/svd_summary.pt" ]]; then
    echo "[2/4] skip SVD: $SVD_DIR/svd_summary.pt exists"
    return
  fi
  for ((rank=0; rank<SVD_WORLD_SIZE; rank++)); do
    "$PYTHON" "$LQR_ROOT/svd/run_partition_svd_pairs_no_action.py" \
      --mode sketch --rank "$rank" --world-size "$SVD_WORLD_SIZE" \
      --prompt "$PROMPT" --pos-npz "$pos" --neg-npz "$neg" \
      --drive-source "$DRIVE_SOURCE" --N "$NUM_SAMPLES" \
      --k-target "$K_TARGET" --p-over "$P_OVER" --partitions "$PARTITIONS" \
      --sampling-steps "$SAMPLING_STEPS" --timesteps "$TIMESTEPS" \
      --ckpt-path "$CKPT_PATH" --out-dir "$SVD_DIR"
  done
  "$PYTHON" "$LQR_ROOT/svd/run_partition_svd_pairs_no_action.py" \
    --mode svd --world-size "$SVD_WORLD_SIZE" \
    --prompt "$PROMPT" --pos-npz "$pos" --neg-npz "$neg" \
    --drive-source "$DRIVE_SOURCE" --N "$NUM_SAMPLES" \
    --k-target "$K_TARGET" --p-over "$P_OVER" --partitions "$PARTITIONS" \
    --sampling-steps "$SAMPLING_STEPS" --timesteps "$TIMESTEPS" \
    --ckpt-path "$CKPT_PATH" --out-dir "$SVD_DIR"
}

run_jacobians() {
  local inputs_npz="$1"
  if [[ -z "${FORCE_STEP3:-}" && -f "$JAC_DIR/A_tilde__full.pt" ]]; then
    echo "[3/4] skip Jacobians: $JAC_DIR/A_tilde__full.pt exists"
    return
  fi
  mkdir -p "$JAC_DIR"
  for ((rank=0; rank<JAC_WORLD_SIZE; rank++)); do
    "$PYTHON" "$LQR_ROOT/jacobians/compute_jacobians_full.py" \
      --phase worker --rank "$rank" --num-shards "$JAC_WORLD_SIZE" \
      --svd-dir "$SVD_DIR" --out-dir "$JAC_DIR" \
      --prompt "$PROMPT" --inputs-npz "$inputs_npz" --obs-index "$OBS_INDEX" \
      --mode "$JAC_MODE" --v-dtype "$V_DTYPE" --ckpt-path "$CKPT_PATH" --resume
  done
  "$PYTHON" "$LQR_ROOT/jacobians/compute_jacobians_full.py" \
    --phase merge --num-shards "$JAC_WORLD_SIZE" \
    --svd-dir "$SVD_DIR" --out-dir "$JAC_DIR" \
    --prompt "$PROMPT" --inputs-npz "$inputs_npz" --obs-index "$OBS_INDEX" \
    --mode "$JAC_MODE" --v-dtype "$V_DTYPE" --ckpt-path "$CKPT_PATH"
}

echo "=== Cosmos local LQR ==="
echo "PERTURBATION=$PERTURBATION"
echo "COLLECT_TASK_ID=$COLLECT_TASK_ID"
echo "EVAL_TASK_IDS=$EVAL_TASK_IDS"
echo "INPUT_DIR=$INPUT_DIR"
echo "SVD_DIR=$SVD_DIR"
echo "JAC_DIR=$JAC_DIR"
echo "WORLD_SIZE=$WORLD_SIZE SVD_WORLD_SIZE=$SVD_WORLD_SIZE JAC_WORLD_SIZE=$JAC_WORLD_SIZE"

if should_run 1; then
  echo "[1/4] collect positive/negative inputs"
  mkdir -p "$INPUT_DIR"
  if [[ -z "${FORCE_STEP1:-}" && -f "$INPUT_DIR/positive.npz" && -f "$INPUT_DIR/negative.npz" ]]; then
    if [[ "$KIND" == "camera" && -f "$INPUT_DIR/manifest.json" ]]; then
      "$PYTHON" - "$INPUT_DIR/manifest.json" "$N_POS" "$N_NEG" <<'PY'
import json
import sys

manifest, n_pos, n_neg = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
data = json.load(open(manifest))
old_pos = data.get("n_pos_rollouts_requested")
old_neg = data.get("n_neg_rollouts_requested")
if old_pos != n_pos or old_neg != n_neg:
    raise SystemExit(
        f"Existing camera inputs were built with N_POS={old_pos}, N_NEG={old_neg}, "
        f"but this run requested N_POS={n_pos}, N_NEG={n_neg}.\n"
        "Use a fresh OUT_BASE, or rerun with FORCE_STEP1=1 FORCE_STEP2=1 FORCE_STEP3=1."
    )
PY
    fi
    echo "[1/4] skip collect: existing npz files found"
  elif [[ "$KIND" == "camera" ]]; then
    SCRATCH_DIR="${SCRATCH_DIR:-$INPUT_DIR/scratch}"
    for ((rank=0; rank<WORLD_SIZE; rank++)); do
      "$PYTHON" "$LQR_ROOT/inputs/collect_policy_inputs_camera_view_perturbation.py" \
        --mode collect --rank "$rank" --world-size "$WORLD_SIZE" \
        --out-dir "$INPUT_DIR" --scratch-dir "$SCRATCH_DIR" \
        --suite "$SUITE" --task-id "$COLLECT_TASK_ID" --resolution "$RESOLUTION" \
        --ckpt-path "$CKPT_PATH" --n-pos-rollouts "$N_POS" --n-neg-rollouts "$N_NEG" \
        --prompt "$PROMPT" --cam-base-seed "$CAM_BASE_SEED" \
        --preset-name "$PRESET_NAME"
    done
    "$PYTHON" "$LQR_ROOT/inputs/collect_policy_inputs_camera_view_perturbation.py" \
      --mode finalize --world-size "$WORLD_SIZE" \
      --out-dir "$INPUT_DIR" --scratch-dir "$SCRATCH_DIR" \
      --suite "$SUITE" --task-id "$COLLECT_TASK_ID" --resolution "$RESOLUTION" \
      --ckpt-path "$CKPT_PATH" --n-pos-rollouts "$N_POS" --n-neg-rollouts "$N_NEG" \
      --prompt "$PROMPT" --cam-base-seed "$CAM_BASE_SEED" \
      --preset-name "$PRESET_NAME"
  elif [[ "$KIND" == "gripper" ]]; then
    "$PYTHON" "$LQR_ROOT/inputs/collect_policy_inputs_gripper_xyz_perturbation.py" \
      --out-dir "$INPUT_DIR" --suite "$SUITE" --task-id "$COLLECT_TASK_ID" \
      --n-episodes "$N_EPISODES" --resolution "$RESOLUTION" \
      --preset "$PRESET" --base-seed "$BASE_SEED" --prompt "$PROMPT" \
      --ckpt-path "$CKPT_PATH"
    "$PYTHON" "$LQR_ROOT/svd/pair_inputs_by_similarity.py" \
      --in-dir "$INPUT_DIR" --out-dir "$PAIRED_DIR" \
      --feature "${PAIR_FEATURE:-proprio}" --match-mode "${PAIR_MATCH_MODE:-nn-greedy}"
  else
    "$PYTHON" "$LQR_ROOT/inputs/collect_policy_inputs_noise_extreme.py" \
      --out-dir "$INPUT_DIR" --suite "$SUITE" --task-id "$COLLECT_TASK_ID" \
      --n-episodes "$N_EPISODES" --resolution "$RESOLUTION" \
      --prompt "$PROMPT" --noise-sigma "$NOISE_SIGMA" --ckpt-path "$CKPT_PATH"
  fi
fi

if should_run 2; then
  run_svd "$PAIR_DIR/positive.npz" "$PAIR_DIR/negative.npz"
fi

if should_run 3; then
  run_jacobians "$PAIR_DIR/negative.npz"
fi

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "[4/4] skipped eval (SKIP_EVAL=1)"
  exit 0
fi

echo "[4/4] run local LQR rollouts"
read -r -a _eval_tasks <<< "$EVAL_TASK_IDS"
for task_id in "${_eval_tasks[@]}"; do
  eval_prompt="${EVAL_PROMPT:-$(prompt_for_task "$task_id")}"
  if [[ "$KIND" == "camera" ]]; then
    driver="$LQR_ROOT/run_lqr_decay_cosmos_policy_cam_perturb.py"
    cam_mode="${EVAL_CAM_MODE:-${CAM_MODE:-random}}"
    case "$cam_mode" in
      match) cam_base_seed="${EVAL_CAM_BASE_SEED:-$CAM_BASE_SEED}" ;;
      random) cam_base_seed="${EVAL_CAM_BASE_SEED:-99}" ;;
      off) cam_base_seed="${EVAL_CAM_BASE_SEED:-0}" ;;
      *)
        echo "Unsupported EVAL_CAM_MODE/CAM_MODE=$cam_mode (use match, random, or off)" >&2
        exit 2
        ;;
    esac
    cam_args=(
      --svd-dir "$SVD_DIR" --jac-dir-act "$JAC_SUBDIR"
      --world-size "$EVAL_WORLD_SIZE" --suite "$SUITE" --task-id "$task_id"
      --n-episodes "${EVAL_NUM_EPISODES:-$N_EPISODES}" --resolution "$RESOLUTION"
      --video-fps "${VIDEO_FPS:-30}" --prompt "$eval_prompt"
      --lambda "${LAMBDA:-10.0}" --q-scale "${Q_SCALE:-10.0}"
      --r-scale-init "${R_SCALE:-20.0}" --r-scale-tau "${R_SCALE_TAU:-7.0}"
      --r-scale-final "${R_SCALE_FINAL:-1e9}" --qf-scale "${QF_SCALE:-1.0}"
      --max-chunks "${MAX_CHUNKS:-50}"
      --cam-mode "$cam_mode" --cam-base-seed "$cam_base_seed"
      --cam-preset-name "$PRESET_NAME"
      --run-tag "${RUN_TAG:-cam_perturb_local}"
      --rollout-subdir "${ROLLOUT_SUBDIR:-local_runs/cam_perturb}"
    )
    if [[ "${SIDE_BY_SIDE_VIDEO:-0}" == "1" || "${SIDE_BY_SIDE_VIDEO:-}" == "true" ]]; then
      cam_args+=(--side-by-side-video)
    fi
    if [[ "${RUN_BASELINE:-0}" == "1" || "${RUN_BASELINE:-}" == "true" ]]; then
      cam_args+=(--run-baseline)
    fi
    for ((rank=0; rank<EVAL_WORLD_SIZE; rank++)); do
      "$PYTHON" "$driver" --mode rollouts --rank "$rank" "${cam_args[@]}"
    done
    "$PYTHON" "$driver" --mode finalize "${cam_args[@]}"
    continue
  fi

  eval_dir="${EVAL_OUT_BASE:-$OUT_BASE/eval}/${KIND}_task$(printf '%02d' "$task_id")"
  mkdir -p "$eval_dir"
  if [[ "$KIND" == "gripper" ]]; then
    driver="$LQR_ROOT/run_lqr_cosmos_policy_gripper_xyz.py"
    extra=(--preset "$PRESET" --base-seed "${EVAL_BASE_SEED:-$BASE_SEED}")
  else
    driver="$LQR_ROOT/run_lqr_cosmos_policy_noised.py"
    extra=(--pair-dir "$PAIR_DIR" --noise-sigma "${EVAL_NOISE_SIGMA:-$NOISE_SIGMA}" --noise-seed-base "${NOISE_SEED_BASE:-0}")
  fi
  for ((rank=0; rank<EVAL_WORLD_SIZE; rank++)); do
    "$PYTHON" "$driver" --phase rollout --rank "$rank" --world-size "$EVAL_WORLD_SIZE" \
      --svd-dir "$SVD_DIR" --jac-dir-act "$JAC_SUBDIR" \
      --prompt "$eval_prompt" --n-episodes "${EVAL_NUM_EPISODES:-$N_EPISODES}" \
      --suite "$SUITE" --task-id "$task_id" --resolution "$RESOLUTION" \
      --ckpt-path "$CKPT_PATH" --out-dir "$eval_dir" --no-save-video "${extra[@]}"
  done
  "$PYTHON" "$driver" --phase merge --world-size "$EVAL_WORLD_SIZE" \
    --svd-dir "$SVD_DIR" --jac-dir-act "$JAC_SUBDIR" \
    --prompt "$eval_prompt" --n-episodes "${EVAL_NUM_EPISODES:-$N_EPISODES}" \
    --suite "$SUITE" --task-id "$task_id" --resolution "$RESOLUTION" \
    --ckpt-path "$CKPT_PATH" --out-dir "$eval_dir" --no-save-video "${extra[@]}"
done
