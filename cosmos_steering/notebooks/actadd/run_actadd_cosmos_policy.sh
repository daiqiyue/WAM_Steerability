#!/bin/bash
# Slurm launcher for run_actadd_cosmos_policy.py (image-noise variant).
#
# Submits a sbatch array of WORLD_SIZE workers (one GPU each), each handling
# its slice of N_EPISODES via --rank R --world-size W. After they all finish,
# a dependent merge job aggregates per-rank results into results.json and
# patches the top-level manifest.json.
#
# Each episode rollout:
#   - resets env, runs the policy under sigma=NOISE_SIGMA Gaussian image noise
#     on the primary + wrist cameras (per-episode-seeded by --noise-seed-base
#     + episode_idx, matching run_lqr_cosmos_policy_noised.py),
#   - applies activation-addition steering:
#       output_steered = output + ALPHA * v[layer]
#     where v is loaded from V_PATH and applied at every hooked DiT block.
#
# NOTE: This file is the SLURM launcher and is preserved for clusters that
# have slurm. For the local (no-slurm) workflow used in this workspace, the
# pipeline driver is notebooks/lqr/e2e_scripts/run_actadd_from_task06_local.sh,
# which fans out via notebooks/lqr/violet/gpu_pool.py.
#
# Usage:
#   V_PATH=/path/to/v.pt ALPHA=1.0 ./run_actadd_cosmos_policy.sh
#   WORLD_SIZE=4 N_EPISODES=50 V_PATH=... ALPHA=0.5 \
#     ./run_actadd_cosmos_policy.sh
#   MERGE_ONLY=1 OUT_DIR=/path/to/run ./run_actadd_cosmos_policy.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_actadd_cosmos_policy.py"
# notebooks/actadd -> repo root is two dirs up
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." &>/dev/null && pwd)}"
LQR_ROOT="${LQR_ROOT:-$REPO_ROOT/notebooks/lqr}"
ENV_SH="${ENV_SH:-$REPO_ROOT/environment.sh}"

# --- Parallelization --------------------------------------------------------
WORLD_SIZE="${WORLD_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"

# --- ActAdd hyperparameters -------------------------------------------------
V_PATH="${V_PATH:-}"          # required: path to the steering vectors .pt/.npy
ALPHA="${ALPHA:-1.0}"         # scalar multiplier applied at every hooked block
SAMPLING_STEPS="${SAMPLING_STEPS:-10}"

# --- Image noise ------------------------------------------------------------
NOISE_SIGMA="${NOISE_SIGMA:-90.0}"
NOISE_SEED_BASE="${NOISE_SEED_BASE:-99}"
NOISE_PER_EPISODE_SEED="${NOISE_PER_EPISODE_SEED:-1}"

# --- Rollout knobs ----------------------------------------------------------
PROMPT="${PROMPT:-put the white mug on the plate and put the chocolate pudding to the right of the plate}"
TASK_ID="${TASK_ID:-6}"
SUITE="${SUITE:-libero_10}"
RESOLUTION="${RESOLUTION:-256}"
VIDEO_FPS="${VIDEO_FPS:-30}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-520}"
RUN_BASELINE="${RUN_BASELINE:-0}"   # 1 = also run unsteered baseline per episode
SEED="${SEED:-1}"

# --- Slurm resources --------------------------------------------------------
WORKER_TIME="${WORKER_TIME:-00:35:00}"
MERGE_TIME="${MERGE_TIME:-00:10:00}"
ACCOUNT="${ACCOUNT:-}"
PARTITION_SLURM="${PARTITION_SLURM:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

# --- Output naming ----------------------------------------------------------
_slug() { local s="$1"; local n="${2:-24}"; echo "${s:0:$n}" | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_-'; }
PROMPT_SLUG="$(_slug "$PROMPT" 24)"
ACTADD_TAG="alpha${ALPHA}"
NOISE_TAG="s${NOISE_SIGMA}_sb${NOISE_SEED_BASE}"

DEFAULT_TAG="${SUITE}__task$(printf '%02d' "$TASK_ID")__actadd_noised__${NOISE_TAG}__${ACTADD_TAG}__${PROMPT_SLUG}"
RUN_TAG="${RUN_TAG:-}"
CONFIG_TAG="${CONFIG_TAG:-$DEFAULT_TAG}"
if [[ -n "$RUN_TAG" ]]; then
    CONFIG_TAG="${CONFIG_TAG}__${RUN_TAG}"
fi

TAG="${TAG:-seed${NOISE_SEED_BASE}}"
OUT_BASE="${OUT_BASE:-$LQR_ROOT/rollouts}"
OUT_DIR="${OUT_DIR:-$OUT_BASE${TAG:+/$TAG}/$CONFIG_TAG}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

MERGE_ONLY="${MERGE_ONLY:-0}"

[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
[[ -f "$ENV_SH" ]] || { echo "ERROR: missing $ENV_SH" >&2; exit 1; }
if [[ "$MERGE_ONLY" != "1" ]]; then
    [[ -n "$V_PATH" ]] || { echo "ERROR: V_PATH is not set" >&2; exit 1; }
    [[ -f "$V_PATH" ]] || { echo "ERROR: V_PATH not found: $V_PATH" >&2; exit 1; }
fi
if [[ -z "$ACCOUNT" || -z "$PARTITION_SLURM" ]]; then
    echo "ERROR: ACCOUNT and PARTITION_SLURM must be set for the slurm launcher" >&2
    echo "       (this script is the slurm variant; for the local workflow use" >&2
    echo "        notebooks/lqr/e2e_scripts/run_actadd_from_task06_local.sh)" >&2
    exit 1
fi

# --- Echo config ------------------------------------------------------------
echo "=== run_actadd_cosmos_policy (noised) ==="
echo "  WORLD_SIZE     = $WORLD_SIZE"
echo "  N_EPISODES     = $N_EPISODES"
echo "  NOISE_SIGMA    = $NOISE_SIGMA   seed_base=$NOISE_SEED_BASE per_ep=$NOISE_PER_EPISODE_SEED"
echo "  V_PATH         = $V_PATH"
echo "  ALPHA          = $ALPHA"
echo "  SAMPLING_STEPS = $SAMPLING_STEPS"
echo "  PROMPT         = $PROMPT"
echo "  TASK_ID        = $TASK_ID  SUITE = $SUITE  RESOLUTION = $RESOLUTION"
echo "  MAX_ENV_STEPS  = $MAX_ENV_STEPS  RUN_BASELINE = $RUN_BASELINE  SEED = $SEED"
echo "  OUT_DIR        = $OUT_DIR"
echo "  account=$ACCOUNT  partition=$PARTITION_SLURM  worker_time=$WORKER_TIME"
echo "  merge_only=$MERGE_ONLY  exclude_nodes=${EXCLUDE_NODES:-<none>}"
echo "  ENV_SH         = $ENV_SH"
echo

# --- Build common CLI args for the python script ---------------------------
COMMON_ARGS=(
    --v-path "$V_PATH"
    --alpha "$ALPHA"
    --sampling-steps "$SAMPLING_STEPS"
    --noise-sigma "$NOISE_SIGMA"
    --noise-seed-base "$NOISE_SEED_BASE"
    --prompt "$PROMPT"
    --n-episodes "$N_EPISODES"
    --task-id "$TASK_ID"
    --suite "$SUITE"
    --resolution "$RESOLUTION"
    --video-fps "$VIDEO_FPS"
    --num-steps-wait "$NUM_STEPS_WAIT"
    --max-env-steps "$MAX_ENV_STEPS"
    --world-size "$WORLD_SIZE"
    --out-dir "$OUT_DIR"
    --seed "$SEED"
)
if [[ "$NOISE_PER_EPISODE_SEED" == "1" || "$NOISE_PER_EPISODE_SEED" == "true" ]]; then
    COMMON_ARGS+=( --noise-per-episode-seed )
else
    COMMON_ARGS+=( --no-noise-per-episode-seed )
fi
if [[ "$RUN_BASELINE" == "1" || "$RUN_BASELINE" == "true" ]]; then
    COMMON_ARGS+=( --run-baseline )
else
    COMMON_ARGS+=( --no-baseline )
fi
if [[ -n "$TAG" ]]; then
    COMMON_ARGS+=( --tag "$TAG" )
fi

ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

ENV_ACTIVATE_BLOCK="
set +e
source $(printf '%q' "$ENV_SH")
set -e
"

# --- Stage 1: rollout array (one rank per task) -----------------------------
if [[ "$MERGE_ONLY" == "1" ]]; then
    echo "=== MERGE_ONLY=1; submitting merge-only job (no rollout array) ==="
    MERGE_ONLY_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION_SLURM" \
        --job-name="actadd_merge_only_${CONFIG_TAG:0:48}" \
        --gpus-per-task=1 \
        --ntasks=1 \
        --cpus-per-task=2 \
        --mem=16G \
        --time="$MERGE_TIME" \
        --output="$LOG_DIR/merge_only_%j.out" \
        --error="$LOG_DIR/merge_only_%j.err" \
        --wrap "
set -uo pipefail
$ENV_ACTIVATE_BLOCK
cd '$SCRIPT_DIR'
python -u '$PY_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --v-path '$V_PATH' --prompt '$PROMPT'
")
    echo "  merge-only job id: $MERGE_ONLY_ID"
    echo "  out_dir: $OUT_DIR"
    echo "  Tail log: tail -f $LOG_DIR/merge_only_${MERGE_ONLY_ID}.out"
    exit 0
fi

ROLLOUT_JOB_NAME="actadd_${CONFIG_TAG:0:48}"
LAST_RANK=$((WORLD_SIZE - 1))
echo "=== submitting rollout array (0-$LAST_RANK) ==="

ROLLOUT_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$ROLLOUT_JOB_NAME" \
    --array=0-"$LAST_RANK" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$WORKER_TIME" \
    --output="$LOG_DIR/rollout_rank%a_%A.out" \
    --error="$LOG_DIR/rollout_rank%a_%A.err" \
    --export=ALL \
    ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
    --wrap "
set -uo pipefail
$ENV_ACTIVATE_BLOCK
cd '$SCRIPT_DIR'
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE\"
python -u '$PY_SCRIPT' --phase rollout --rank \"\$SLURM_ARRAY_TASK_ID\"$ARGS_QUOTED
")
echo "  rollout job id: $ROLLOUT_ID  (array 0-$LAST_RANK)"

# --- Stage 2: merge (depends on all rollout ranks) -------------------------
MERGE_JOB_NAME="actadd_merge_${CONFIG_TAG:0:48}"
echo "=== submitting merge (depends on $ROLLOUT_ID) ==="

MERGE_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$MERGE_JOB_NAME" \
    --dependency=afterany:"$ROLLOUT_ID" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --time="$MERGE_TIME" \
    --output="$LOG_DIR/merge_%j.out" \
    --error="$LOG_DIR/merge_%j.err" \
    --wrap "
set -uo pipefail
$ENV_ACTIVATE_BLOCK
cd '$SCRIPT_DIR'
python -u '$PY_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --v-path '$V_PATH' --prompt '$PROMPT'
")
echo "  merge job id:   $MERGE_ID  (afterany:$ROLLOUT_ID)"

echo
echo "=== submitted ==="
echo "  rollout : sbatch $ROLLOUT_ID  (array 0-$LAST_RANK)"
echo "  merge   : sbatch $MERGE_ID    (afterany:$ROLLOUT_ID)"
echo "  out_dir : $OUT_DIR"
echo "  logs    : $LOG_DIR"
echo
echo "Monitor:        squeue -u \$USER --start"
echo "Tail rank 0:    tail -f $LOG_DIR/rollout_rank0_${ROLLOUT_ID}.out"
echo "Tail merge:     tail -f $LOG_DIR/merge_${MERGE_ID}.out"
