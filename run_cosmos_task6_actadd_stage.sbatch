#!/usr/bin/env bash
#SBATCH --job-name=cosmos-t6-actadd
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --chdir=/storage/scratch1/9/qdai41/steering_robustness_WAMs/cosmos_steering

set -euo pipefail

ROOT=/storage/scratch1/9/qdai41/steering_robustness_WAMs/cosmos_steering
source "$ROOT/environment_ideas.sh"

: "${STAGE:?STAGE must be set}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${CKPT_PATH:?CKPT_PATH must be set}"

PROMPT="${PROMPT:-put the white mug on the plate and put the chocolate pudding to the right of the plate}"
NOISE_SIGMA="${NOISE_SIGMA:-90.0}"
TRAIN_EPISODES="${TRAIN_EPISODES:-0 1 2 3 4 5 6 7 8 9}"
EVAL_EPISODES="${EVAL_EPISODES:-20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39}"

PAIR_SHARDS="$RUN_ROOT/contrastive_pairs/shards"
PAIR_MERGED="$RUN_ROOT/contrastive_pairs/all_rows"
PAIR_SELECTED="$RUN_ROOT/contrastive_pairs/balanced_rows"
ACTIVATIONS="$RUN_ROOT/contrastive_vector/activations_task06.npz"
V_PATH="$RUN_ROOT/contrastive_vector/v_task06.pt"
EVAL_DIR="$RUN_ROOT/eval"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs"
nvidia-smi -L
python - <<'PY'
import torch
from importlib.metadata import version
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("mujoco", version("mujoco"), "robosuite", version("robosuite"))
PY

case "$STAGE" in
  smoke)
    python -u notebooks/actadd/smoke_task6_actadd.py \
      --ckpt-path "$CKPT_PATH" \
      --out-path "$RUN_ROOT/smoke.json"
    ;;
  collect_pairs)
    mapfile -t TRAIN_ARRAY < <(tr ' ' '\n' <<< "$TRAIN_EPISODES" | sed '/^$/d')
    EPISODE_ID="${TRAIN_ARRAY[$SLURM_ARRAY_TASK_ID]}"
    OUT_DIR="$PAIR_SHARDS/episode_$(printf '%02d' "$EPISODE_ID")"
    python -u notebooks/lqr/inputs/collect_policy_inputs_noise_extreme.py \
      --out-dir "$OUT_DIR" --suite libero_10 --task-id 6 \
      --n-episodes 50 --episode-ids "$EPISODE_ID" --resolution 256 \
      --prompt "$PROMPT" --noise-sigma "$NOISE_SIGMA" \
      --noise-per-episode-seed --ckpt-path "$CKPT_PATH"
    ;;
  merge_select)
    read -r -a TRAIN_ARRAY <<< "$TRAIN_EPISODES"
    python -u notebooks/actadd/merge_noise_pair_shards.py \
      --shard-root "$PAIR_SHARDS" --out-dir "$PAIR_MERGED" \
      --expected-episodes "${TRAIN_ARRAY[@]}"
    python -u notebooks/actadd/select_balanced_pair_rows.py \
      --pair-dir "$PAIR_MERGED" --out-dir "$PAIR_SELECTED" \
      --rows-per-episode "${ROWS_PER_EPISODE:-4}"
    ;;
  collect_activations)
    python -u notebooks/actadd/collect_acts_passive.py \
      --pair-dir "$PAIR_SELECTED" --prompt "$PROMPT" --task-id 6 \
      --out-path "$ACTIVATIONS" --ckpt-path "$CKPT_PATH" --suite libero_10
    ;;
  make_vector)
    python -u notebooks/actadd/make_contrastive_vec.py \
      --activations-path "$ACTIVATIONS" --out-path "$V_PATH"
    ;;
  eval)
    mapfile -t EVAL_ARRAY < <(tr ' ' '\n' <<< "$EVAL_EPISODES" | sed '/^$/d')
    EPISODE_ID="${EVAL_ARRAY[$SLURM_ARRAY_TASK_ID]}"
    python -u notebooks/actadd/evaluate_task6_actadd_sensitivity.py \
      --v-path "$V_PATH" --out-dir "$EVAL_DIR" --episode-id "$EPISODE_ID" \
      --suite libero_10 --task-id 6 --prompt "$PROMPT" \
      --ckpt-path "$CKPT_PATH" --resolution 256 \
      --sampling-steps "${SAMPLING_STEPS:-10}" \
      --max-env-steps 520 --num-steps-wait 10 \
      --anchor-chunk "${ANCHOR_CHUNK:-8}" \
      --branch-horizon-chunks "${BRANCH_HORIZON_CHUNKS:-10}" \
      --raw-repeat-chunks "${RAW_REPEAT_CHUNKS:-3}" \
      --rollout-alpha "${ROLLOUT_ALPHA:-0.1}" \
      --noise-sigma "$NOISE_SIGMA" --noise-seed-base "${NOISE_SEED_BASE:-99}" \
      --policy-seed "${POLICY_SEED:-1}" \
      --random-direction-seed "${RANDOM_DIRECTION_SEED:-20260917}"
    ;;
  analyze)
    read -r -a EVAL_ARRAY <<< "$EVAL_EPISODES"
    python -u notebooks/actadd/analyze_task6_actadd_sensitivity.py \
      --input-dir "$EVAL_DIR" --expected-episodes "${EVAL_ARRAY[@]}"
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
