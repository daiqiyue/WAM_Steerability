#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DIT4DIT_ROOT="${DIT4DIT_ROOT:-$ROOT/DiT4DiT_steering}"
LQR_ROOT="${DIT4DIT_LQR_ROOT:-$DIT4DIT_ROOT/lqr}"
PYTHON="${PYTHON:-python}"

PERTURBATION="${PERTURBATION:-noise}"
SUITE="${SUITE:-libero_10}"
N_EPISODES="${N_EPISODES:-10}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-$N_EPISODES}"
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
SAMPLING_STEPS="${SAMPLING_STEPS:-4}"
PARTITIONS="${PARTITIONS:-0-5,6-10,11-15}"
DRIVE_SOURCE="${DRIVE_SOURCE:-all}"
JAC_MODE="${JAC_MODE:-vjp_no_retain}"
V_DTYPE="${V_DTYPE:-bf16}"
OBS_INDEX="${OBS_INDEX:-0}"
CKPT_PATH="${CKPT_PATH:-$DIT4DIT_ROOT/checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt}"
OUT_BASE="${OUT_BASE:-$LQR_ROOT/local_runs}"

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
  gripper|gripper_xyz)
    KIND="gripper"
    COLLECT_TASK_ID="${COLLECT_TASK_ID:-1}"
    EVAL_TASK_IDS="${EVAL_TASK_IDS:-1 2 3 7 9}"
    PRESET="${PRESET:-xyz_random_xlarge_3}"
    BASE_SEED="${BASE_SEED:-42}"
    TAG="libero10_task$(printf '%02d' "$COLLECT_TASK_ID")_${PRESET}_seed${BASE_SEED}_paired_no_action_ds${DRIVE_SOURCE}"
    INPUT_DIR="${INPUT_DIR:-$OUT_BASE/inputs/${SUITE}__task$(printf '%02d' "$COLLECT_TASK_ID")__${PRESET}__seed${BASE_SEED}__pos_neg}"
    PAIRED_DIR="${PAIRED_DIR:-${INPUT_DIR}__paired}"
    ;;
  gaussian|noise)
    KIND="noise"
    COLLECT_TASK_ID="${COLLECT_TASK_ID:-6}"
    EVAL_TASK_IDS="${EVAL_TASK_IDS:-0 1 4 6 7}"
    NOISE_SIGMA="${NOISE_SIGMA:-75.0}"
    EVAL_NOISE_SIGMA="${EVAL_NOISE_SIGMA:-$NOISE_SIGMA}"
    TAG="libero10_task$(printf '%02d' "$COLLECT_TASK_ID")_noise_sigma${NOISE_SIGMA}_pairs_no_action_ds${DRIVE_SOURCE}"
    INPUT_DIR="${INPUT_DIR:-$OUT_BASE/inputs/${SUITE}__task$(printf '%02d' "$COLLECT_TASK_ID")__noise_sigma${NOISE_SIGMA}}"
    ;;
  camera|cam)
    echo "DiT4DiT LQR local runner supports gripper and gaussian/noise. No camera LQR pipeline is present under $LQR_ROOT." >&2
    exit 2
    ;;
  *)
    echo "Unsupported PERTURBATION=$PERTURBATION" >&2
    exit 2
    ;;
esac

PROMPT="${PROMPT:-$(prompt_for_task "$COLLECT_TASK_ID")}"
PAIR_DIR="$INPUT_DIR"
if [[ "$KIND" == "gripper" ]]; then
  PAIR_DIR="$PAIRED_DIR"
fi
VL_EMBS_PATH="${VL_EMBS_PATH:-$PAIR_DIR/vl_embs__$(slug "$PROMPT").pt}"
SVD_DIR="${SVD_DIR:-$OUT_BASE/svd/${TAG}_N${NUM_SAMPLES}_k${K_TARGET}_p${P_OVER}_ws${SVD_WORLD_SIZE}_ts${TIMESTEPS}}"
JAC_SUBDIR="${JAC_SUBDIR:-A_tilde_full__$(slug "$PROMPT")__${JAC_MODE}__v${V_DTYPE}}"
JAC_DIR="${JAC_DIR:-$SVD_DIR/$JAC_SUBDIR}"

should_run() {
  (( "$1" >= START_AT ))
}

run_svd() {
  local pos="$1" neg="$2"
  if [[ -z "${FORCE_STEP3:-}" && -f "$SVD_DIR/svd_summary.pt" ]]; then
    echo "[3/5] skip SVD: $SVD_DIR/svd_summary.pt exists"
    return
  fi
  mkdir -p "$SVD_DIR"
  for ((rank=0; rank<SVD_WORLD_SIZE; rank++)); do
    "$PYTHON" "$LQR_ROOT/svd/run_partition_svd_pairs_no_action.py" \
      --mode sketch --rank "$rank" --world-size "$SVD_WORLD_SIZE" \
      --prompt "$PROMPT" --pos-npz "$pos" --neg-npz "$neg" \
      --vl-embs-path "$VL_EMBS_PATH" --drive-source "$DRIVE_SOURCE" \
      --N "$NUM_SAMPLES" --k-target "$K_TARGET" --p-over "$P_OVER" \
      --partitions "$PARTITIONS" --sampling-steps "$SAMPLING_STEPS" \
      --timesteps "$TIMESTEPS" --ckpt-path "$CKPT_PATH" --out-dir "$SVD_DIR"
  done
  "$PYTHON" "$LQR_ROOT/svd/run_partition_svd_pairs_no_action.py" \
    --mode svd --world-size "$SVD_WORLD_SIZE" \
    --prompt "$PROMPT" --pos-npz "$pos" --neg-npz "$neg" \
    --vl-embs-path "$VL_EMBS_PATH" --drive-source "$DRIVE_SOURCE" \
    --N "$NUM_SAMPLES" --k-target "$K_TARGET" --p-over "$P_OVER" \
    --partitions "$PARTITIONS" --sampling-steps "$SAMPLING_STEPS" \
    --timesteps "$TIMESTEPS" --ckpt-path "$CKPT_PATH" --out-dir "$SVD_DIR"
}

run_jacobians() {
  local inputs_npz="$1"
  if [[ -z "${FORCE_STEP4:-}" && -f "$JAC_DIR/A_tilde__full.pt" ]]; then
    echo "[4/5] skip Jacobians: $JAC_DIR/A_tilde__full.pt exists"
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

echo "=== DiT4DiT local LQR ==="
echo "PERTURBATION=$PERTURBATION"
echo "COLLECT_TASK_ID=$COLLECT_TASK_ID"
echo "EVAL_TASK_IDS=$EVAL_TASK_IDS"
echo "PAIR_DIR=$PAIR_DIR"
echo "VL_EMBS_PATH=$VL_EMBS_PATH"
echo "SVD_DIR=$SVD_DIR"
echo "JAC_DIR=$JAC_DIR"

if should_run 1; then
  echo "[1/5] collect inputs"
  mkdir -p "$INPUT_DIR"
  if [[ -z "${FORCE_STEP1:-}" && -f "$INPUT_DIR/positive.npz" && -f "$INPUT_DIR/negative.npz" ]]; then
    echo "[1/5] skip collect: existing npz files found"
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
    "$PYTHON" "$LQR_ROOT/inputs/collect_policy_inputs_noise.py" \
      --out-dir "$INPUT_DIR" --suite "$SUITE" --task-id "$COLLECT_TASK_ID" \
      --n-episodes "$N_EPISODES" --resolution "$RESOLUTION" \
      --prompt "$PROMPT" --noise-sigma "$NOISE_SIGMA" --ckpt-path "$CKPT_PATH"
  fi
fi

if should_run 2; then
  echo "[2/5] precompute vision-language embeddings"
  if [[ -z "${FORCE_STEP2:-}" && -f "$VL_EMBS_PATH" ]]; then
    echo "[2/5] skip vl_embs: $VL_EMBS_PATH exists"
  else
    "$PYTHON" "$LQR_ROOT/inputs/precompute_vl_embs.py" \
      --pos-npz "$PAIR_DIR/positive.npz" --neg-npz "$PAIR_DIR/negative.npz" \
      --out-path "$VL_EMBS_PATH" --prompt "$PROMPT" --ckpt-path "$CKPT_PATH"
  fi
fi

if should_run 3; then
  run_svd "$PAIR_DIR/positive.npz" "$PAIR_DIR/negative.npz"
fi

if should_run 4; then
  run_jacobians "$PAIR_DIR/negative.npz"
fi

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "[5/5] skipped eval (SKIP_EVAL=1)"
  exit 0
fi

echo "[5/5] run local LQR rollouts"
read -r -a _eval_tasks <<< "$EVAL_TASK_IDS"
for task_id in "${_eval_tasks[@]}"; do
  eval_prompt="${EVAL_PROMPT:-$(prompt_for_task "$task_id")}"
  eval_dir="${EVAL_OUT_BASE:-$OUT_BASE/eval}/${KIND}_task$(printf '%02d' "$task_id")"
  mkdir -p "$eval_dir"
  if [[ "$KIND" == "gripper" ]]; then
    driver="$LQR_ROOT/run_lqr_dit4dit_gripper_xyz.py"
    extra=(--preset "$PRESET" --base-seed "${EVAL_BASE_SEED:-$BASE_SEED}")
  else
    driver="$LQR_ROOT/run_lqr_dit4dit_noised.py"
    extra=(--noise-sigma "$EVAL_NOISE_SIGMA" --noise-seed-base "${NOISE_SEED_BASE:-0}")
  fi
  for ((rank=0; rank<EVAL_WORLD_SIZE; rank++)); do
    "$PYTHON" "$driver" --phase rollout --rank "$rank" --world-size "$EVAL_WORLD_SIZE" \
      --svd-dir "$SVD_DIR" --jac-dir-act "$JAC_SUBDIR" \
      --prompt "$eval_prompt" --n-episodes "$EVAL_NUM_EPISODES" \
      --suite "$SUITE" --task-id "$task_id" --resolution "$RESOLUTION" \
      --ckpt-path "$CKPT_PATH" --out-dir "$eval_dir" --no-save-video "${extra[@]}"
  done
  "$PYTHON" "$driver" --phase merge --world-size "$EVAL_WORLD_SIZE" \
    --svd-dir "$SVD_DIR" --jac-dir-act "$JAC_SUBDIR" \
    --prompt "$eval_prompt" --n-episodes "$EVAL_NUM_EPISODES" \
    --suite "$SUITE" --task-id "$task_id" --resolution "$RESOLUTION" \
    --ckpt-path "$CKPT_PATH" --out-dir "$eval_dir" --no-save-video "${extra[@]}"
done
