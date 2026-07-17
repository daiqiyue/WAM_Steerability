#!/bin/bash
# Slurm launcher for the unsteered baseline pass that mirrors
# run_lqr_cosmos_policy_noised.sh:
#   - same noise sigma + per-episode seed (NOISE_SEED_BASE + ep)
#   - same suite/task/init_states
#   - LQR steering disabled (--baseline-only); no jacobians used
#
# Output goes to <OUT_BASE>/<TAG>/__baseline__/ so the aggregator (with
# --baseline-dir) can pick it up and report it as the no-LQR reference.
#
# Submits a sbatch array of WORLD_SIZE workers + a dependent merge job, same
# shape as the LQR sweep.
#
# Usage:
#   ./run_baseline_noised.sh                              # seed_base=99, 50 episodes, 4 workers
#   NOISE_SEED_BASE=42 ./run_baseline_noised.sh           # different seed
#   N_EPISODES=20 WORLD_SIZE=2 ./run_baseline_noised.sh   # smaller run

set -euo pipefail

# --- Parallelism ------------------------------------------------------------
WORLD_SIZE="${WORLD_SIZE:-4}"

# --- Rollout knobs ----------------------------------------------------------
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"
N_EPISODES="${N_EPISODES:-50}"
TASK_ID="${TASK_ID:-0}"
SUITE="${SUITE:-libero_10}"
RESOLUTION="${RESOLUTION:-256}"
VIDEO_FPS="${VIDEO_FPS:-30}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1000}"
SEED="${SEED:-1}"

# --- Noise (must match the LQR sweep this is the baseline for) -------------
NOISE_SIGMA="${NOISE_SIGMA:-75.0}"
NOISE_PER_EPISODE_SEED="${NOISE_PER_EPISODE_SEED:-1}"
NOISE_SEED_BASE="${NOISE_SEED_BASE:-99}"
OBS_INDEX="${OBS_INDEX:-0}"
NO_OVERRIDE_FIRST_CHUNK="${NO_OVERRIDE_FIRST_CHUNK:-1}"

# --- SVD / jacobian inputs (still required by the loader even in baseline) -
SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_noise_high_4_pairs_no_action_dsall_N-1_k64_p10_ws8_tsall}"
JAC_DIR_ACT="${JAC_DIR_ACT:-A_tilde_full__put_both_the_alphabet_soup_and_the_tomato_sauce_in__vjp_no_retain__vbf16}"
PAIR_DIR="${PAIR_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_high_4_pos_neg}"

# --- Slurm resources --------------------------------------------------------
WORKER_TIME="${WORKER_TIME:-00:40:00}"
MERGE_TIME="${MERGE_TIME:-00:10:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
# shellcheck disable=SC1091
. "$(dirname -- "${BASH_SOURCE[0]}")/../exclude_nodes.sh"
EXCLUDE_NODES="${EXCLUDE_NODES-$EXCLUDE_NODES_DEFAULT}"
# --- Output layout ----------------------------------------------------------
TAG="${TAG:-noise_seed${NOISE_SEED_BASE}}"
OUT_BASE="${OUT_BASE:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/rollouts}"
OUT_DIR="${OUT_DIR:-$OUT_BASE/$TAG/__baseline__}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_lqr_cosmos_policy_noised.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
[[ -d "$SVD_DIR" ]] || { echo "ERROR: SVD_DIR not found: $SVD_DIR" >&2; exit 1; }
[[ -f "$PAIR_DIR/negative.npz" ]] || { echo "ERROR: $PAIR_DIR/negative.npz not found" >&2; exit 1; }

echo "=== run_baseline_noised ==="
echo "  WORLD_SIZE     = $WORLD_SIZE"
echo "  N_EPISODES     = $N_EPISODES"
echo "  NOISE          = sigma=$NOISE_SIGMA  seed_base=$NOISE_SEED_BASE  per_ep=$NOISE_PER_EPISODE_SEED"
echo "  PROMPT         = $PROMPT"
echo "  SUITE / TASK   = $SUITE / $TASK_ID"
echo "  MAX_ENV_STEPS  = $MAX_ENV_STEPS"
echo "  OUT_DIR        = $OUT_DIR"
echo "  partition=$PARTITION_SLURM  worker_time=$WORKER_TIME  mem=$MEM"
echo "  exclude_nodes=${EXCLUDE_NODES:-<none>}"
echo

COMMON_ARGS=(
    --svd-dir "$SVD_DIR"
    --jac-dir-act "$JAC_DIR_ACT"
    --pair-dir "$PAIR_DIR"
    --obs-index "$OBS_INDEX"
    --noise-sigma "$NOISE_SIGMA"
    --noise-seed-base "$NOISE_SEED_BASE"
    --prompt "$PROMPT"
    --lambda-scale 1.0          # ignored when --baseline-only
    --q-scale     1.0           # ignored
    --r-scale     1.0           # ignored
    --r-scale-tau 1.0           # ignored
    --r-scale-final 1e9         # ignored
    --max-chunks 50             # ignored
    --qf-scale   1.0            # ignored
    --n-episodes "$N_EPISODES"
    --task-id "$TASK_ID"
    --suite "$SUITE"
    --resolution "$RESOLUTION"
    --video-fps "$VIDEO_FPS"
    --num-steps-wait "$NUM_STEPS_WAIT"
    --max-env-steps "$MAX_ENV_STEPS"
    --world-size "$WORLD_SIZE"
    --out-dir "$OUT_DIR"
    --baseline-only
    --no-baseline
    --tag "$TAG"
)
if [[ "$NOISE_PER_EPISODE_SEED" == "1" || "$NOISE_PER_EPISODE_SEED" == "true" ]]; then
    COMMON_ARGS+=( --noise-per-episode-seed )
else
    COMMON_ARGS+=( --no-noise-per-episode-seed )
fi
if [[ "$NO_OVERRIDE_FIRST_CHUNK" == "1" || "$NO_OVERRIDE_FIRST_CHUNK" == "true" ]]; then
    COMMON_ARGS+=( --no-override-first-chunk )
fi
if [[ -n "$SEED" ]]; then
    COMMON_ARGS+=( --seed "$SEED" )
fi

ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

T5_CLEAR='
python -u -c "
import os, pickle
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id=\"nvidia/Cosmos-Policy-LIBERO-Predict2-2B\",
    filename=\"libero_t5_embeddings.pkl\",
    cache_dir=os.environ.get(\"HF_HUB_CACHE\"),
)
real = os.path.realpath(p)
with open(real, \"wb\") as f:
    pickle.dump({}, f)
print(f\"[t5-cache] cleared (empty pickle at {p})\")
for suffix in (\".backup\", \".lock\"):
    aux = p + suffix
    if os.path.exists(aux):
        try: os.remove(aux); print(f\"[t5-cache] removed {aux}\")
        except OSError: pass
"'

LAST_RANK=$((WORLD_SIZE - 1))
ROLLOUT_JOB_NAME="lqrbaseline_${TAG}"

echo "=== submitting rollout array (0-$LAST_RANK) ==="
ROLLOUT_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$ROLLOUT_JOB_NAME" \
    --array=0-"$LAST_RANK" \
    --gpus-per-task=1 --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" --time="$WORKER_TIME" \
    --output="$LOG_DIR/rollout_rank%a_%A.out" \
    --error="$LOG_DIR/rollout_rank%a_%A.err" \
    --export=ALL \
    ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE\"
$T5_CLEAR
python -u '$PY_SCRIPT' --phase rollout --rank \"\$SLURM_ARRAY_TASK_ID\"$ARGS_QUOTED
")
echo "  rollout job id: $ROLLOUT_ID  (array 0-$LAST_RANK)"

echo "=== submitting merge afterok on rollout array ==="
MERGE_ID=$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$PARTITION_SLURM" \
    --job-name="lqrbaseline_merge_${TAG}" \
    --dependency=afterok:"$ROLLOUT_ID" \
    --gpus-per-task=1 --ntasks=1 --cpus-per-task=2 --mem=16G \
    --time="$MERGE_TIME" \
    --output="$LOG_DIR/merge_%j.out" \
    --error="$LOG_DIR/merge_%j.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
python -u '$PY_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --svd-dir '$SVD_DIR' --jac-dir-act '$JAC_DIR_ACT' \
    --pair-dir '$PAIR_DIR' --prompt '$PROMPT'
")
echo "  merge job id:   $MERGE_ID"

echo
echo "=== submitted ==="
echo "  rollout : $ROLLOUT_ID  (array 0-$LAST_RANK)"
echo "  merge   : $MERGE_ID    (afterok:$ROLLOUT_ID)"
echo "  out_dir : $OUT_DIR"
echo "  logs    : $LOG_DIR"
echo
echo "After it lands, refresh the sweep md to include the baseline reference:"
echo "  python notebooks/lqr/aggregate_lqr_noised.py \\"
echo "    --rollouts-root notebooks/lqr/rollouts/$TAG \\"
echo "    --summary-md   notebooks/lqr/${TAG}_sweep.md \\"
echo "    --noise-seed-base $NOISE_SEED_BASE \\"
echo "    --baseline-dir   $OUT_DIR"
