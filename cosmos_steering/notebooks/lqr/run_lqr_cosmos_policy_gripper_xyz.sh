#!/bin/bash
# Slurm launcher for run_lqr_cosmos_policy_gripper_xyz.py.
#
# Submits a sbatch array of WORLD_SIZE workers (one GPU each), each handling
# its slice of N_EPISODES via --rank R --world-size W. After they all finish,
# a dependent merge job aggregates per-rank results into results.json and
# patches the top-level manifest.json.
#
# Each episode rollout:
#   - resets env, applies a per-episode RANDOM gripper xyz perturbation
#     (default preset xyz_random_xlarge_3 = sigma=10cm isotropic, sampled
#     via SeedSequence([BASE_SEED, episode_idx])),
#   - lets the scene settle,
#   - runs the policy under closed-loop A-LQR steering using the SVD V
#     matrices in SVD_DIR and the jacobian A_tilde under
#     SVD_DIR/JAC_DIR_ACT/A_tilde__full.pt,
#   - applies the same exponentially-decaying R_SCALE schedule as
#     run_lqr_cosmos_policy_noised.sh.
#
# Usage:
#   ./run_lqr_cosmos_policy_gripper_xyz.sh
#   WORLD_SIZE=8 N_EPISODES=50 ./run_lqr_cosmos_policy_gripper_xyz.sh
#   PRESET=xyz_random_xlarge BASE_SEED=42 SVD_DIR=... \
#     ./run_lqr_cosmos_policy_gripper_xyz.sh
#   MERGE_ONLY=1 ./run_lqr_cosmos_policy_gripper_xyz.sh  # only run merge

set -euo pipefail

# --- Parallelization --------------------------------------------------------
WORLD_SIZE="${WORLD_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"

# --- Gripper perturbation ---------------------------------------------------
PRESET="${PRESET:-xyz_random_xlarge_3}"
BASE_SEED="${BASE_SEED:-42}"
GRIPPER_ACTION="${GRIPPER_ACTION:--1.0}"

# --- LQR cost hyperparameters ----------------------------------------------
LAMBDA="${LAMBDA:-15.0}"
Q_SCALE="${Q_SCALE:-1.0}"
R_SCALE="${R_SCALE:-5.0}"
R_SCALE_TAU="${R_SCALE_TAU:-5.0}"
R_SCALE_FINAL="${R_SCALE_FINAL:-1e9}"
MAX_CHUNKS="${MAX_CHUNKS:-50}"
QF_SCALE="${QF_SCALE:-1.0}"

# --- Rollout knobs ----------------------------------------------------------
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"
TASK_ID="${TASK_ID:-0}"
SUITE="${SUITE:-libero_10}"
RESOLUTION="${RESOLUTION:-256}"
VIDEO_FPS="${VIDEO_FPS:-30}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1000}"
RUN_BASELINE="${RUN_BASELINE:-0}"   # 1 = also run unsteered baseline per episode
SEED="${SEED:-1}"

# --- SVD / jacobian inputs --------------------------------------------------
# Default to the gripper-xyz paired SVD output that the upstream pipeline
# produces. Override these to point at any SVD+jacobian artifacts you want
# the LQR to use.
# SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_gripper_xyz_${PRESET}_seed${BASE_SEED}_paired_N-1_k64_p10_ws8_tsall}"
SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_gripper_xyz_${PRESET}_seed42_paired_N-1_k64_p10_ws8_tsall}"
JAC_DIR_ACT="${JAC_DIR_ACT:-A_tilde_full__put_both_the_alphabet_soup_and_the_tomato_sauce_in__vjp_no_retain__vbf16}"

# --- Slurm resources --------------------------------------------------------
WORKER_TIME="${WORKER_TIME:-01:00:00}"
MERGE_TIME="${MERGE_TIME:-00:10:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
# shellcheck disable=SC1091
. "$(dirname -- "${BASH_SOURCE[0]}")/../exclude_nodes.sh"
EXCLUDE_NODES="${EXCLUDE_NODES-$EXCLUDE_NODES_DEFAULT}"
# --- Output naming ----------------------------------------------------------
_slug() { local s="$1"; local n="${2:-24}"; echo "${s:0:$n}" | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_-'; }
PROMPT_SLUG="$(_slug "$PROMPT" 24)"
LQR_TAG="lam${LAMBDA}_q${Q_SCALE}_rinit${R_SCALE}_rfin${R_SCALE_FINAL}_tau${R_SCALE_TAU}_qf${QF_SCALE}"
PERT_TAG="${PRESET}_seed${BASE_SEED}"

DEFAULT_TAG="${SUITE}__task$(printf '%02d' "$TASK_ID")__lqr_grxyz__${PERT_TAG}__${LQR_TAG}__${PROMPT_SLUG}"
RUN_TAG="${RUN_TAG:-}"
CONFIG_TAG="${CONFIG_TAG:-$DEFAULT_TAG}"
if [[ -n "$RUN_TAG" ]]; then
    CONFIG_TAG="${CONFIG_TAG}__${RUN_TAG}"
fi

TAG="${TAG:-seed1}"
OUT_BASE="${OUT_BASE:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/rollouts}"
OUT_DIR="$OUT_BASE${TAG:+/$TAG}/$CONFIG_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_lqr_cosmos_policy_gripper_xyz.py"

MERGE_ONLY="${MERGE_ONLY:-0}"

[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
if [[ "$MERGE_ONLY" != "1" ]]; then
    [[ -d "$SVD_DIR" ]] || { echo "ERROR: SVD_DIR not found: $SVD_DIR" >&2; exit 1; }
    [[ -f "$SVD_DIR/config.json" ]] || {
        echo "ERROR: $SVD_DIR has no config.json -- finalize the SVD first" >&2
        exit 1
    }
    [[ -f "$SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" ]] || {
        echo "ERROR: A_tilde missing at $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" >&2
        exit 1
    }
fi

# --- Echo config ------------------------------------------------------------
echo "=== run_lqr_cosmos_policy_gripper_xyz ==="
echo "  WORLD_SIZE     = $WORLD_SIZE"
echo "  N_EPISODES     = $N_EPISODES"
echo "  PRESET         = $PRESET   (base_seed=$BASE_SEED  gripper_action=$GRIPPER_ACTION)"
echo "  SVD_DIR        = $SVD_DIR"
echo "  JAC_DIR_ACT    = $JAC_DIR_ACT"
echo "  PROMPT         = $PROMPT"
echo "  LAMBDA         = $LAMBDA  Q_SCALE = $Q_SCALE  QF_SCALE = $QF_SCALE"
echo "  R_SCALE (init) = $R_SCALE  R_SCALE_TAU = $R_SCALE_TAU  R_SCALE_FINAL = $R_SCALE_FINAL"
echo "  MAX_CHUNKS     = $MAX_CHUNKS"
echo "  TASK_ID        = $TASK_ID  SUITE = $SUITE  RESOLUTION = $RESOLUTION"
echo "  MAX_ENV_STEPS  = $MAX_ENV_STEPS  RUN_BASELINE = $RUN_BASELINE  SEED = $SEED"
echo "  OUT_DIR        = $OUT_DIR"
echo "  account=$ACCOUNT  partition=$PARTITION_SLURM  worker_time=$WORKER_TIME"
echo "  merge_only=$MERGE_ONLY  exclude_nodes=${EXCLUDE_NODES:-<none>}"
echo

# --- Build common CLI args for the python script ---------------------------
COMMON_ARGS=(
    --svd-dir "$SVD_DIR"
    --jac-dir-act "$JAC_DIR_ACT"
    --preset "$PRESET"
    --base-seed "$BASE_SEED"
    --gripper-action "$GRIPPER_ACTION"
    --prompt "$PROMPT"
    --lambda-scale "$LAMBDA"
    --q-scale "$Q_SCALE"
    --r-scale "$R_SCALE"
    --r-scale-tau "$R_SCALE_TAU"
    --r-scale-final "$R_SCALE_FINAL"
    --max-chunks "$MAX_CHUNKS"
    --qf-scale "$QF_SCALE"
    --n-episodes "$N_EPISODES"
    --task-id "$TASK_ID"
    --suite "$SUITE"
    --resolution "$RESOLUTION"
    --video-fps "$VIDEO_FPS"
    --num-steps-wait "$NUM_STEPS_WAIT"
    --max-env-steps "$MAX_ENV_STEPS"
    --world-size "$WORLD_SIZE"
    --out-dir "$OUT_DIR"
)
if [[ "$RUN_BASELINE" == "1" || "$RUN_BASELINE" == "true" ]]; then
    COMMON_ARGS+=( --run-baseline )
else
    COMMON_ARGS+=( --no-baseline )
fi
if [[ -n "$SEED" ]]; then
    COMMON_ARGS+=( --seed "$SEED" )
fi
if [[ -n "$TAG" ]]; then
    COMMON_ARGS+=( --tag "$TAG" )
fi

ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

# --- Stage 1: rollout array (one rank per task) -----------------------------
if [[ "$MERGE_ONLY" == "1" ]]; then
    echo "=== MERGE_ONLY=1; submitting merge-only job (no rollout array) ==="
    MERGE_ONLY_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION_SLURM" \
        --job-name="lqrgxyz_merge_only_${CONFIG_TAG:0:48}" \
        --gpus-per-task=1 \
        --ntasks=1 \
        --cpus-per-task=2 \
        --mem=16G \
        --time="$MERGE_TIME" \
        --output="$LOG_DIR/merge_only_%j.out" \
        --error="$LOG_DIR/merge_only_%j.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
python -u '$PY_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --svd-dir '$SVD_DIR' --jac-dir-act '$JAC_DIR_ACT' \
    --prompt '$PROMPT' --preset '$PRESET'
")
    echo "  merge-only job id: $MERGE_ONLY_ID"
    echo "  out_dir: $OUT_DIR"
    echo "  Tail log: tail -f $LOG_DIR/merge_only_${MERGE_ONLY_ID}.out"
    exit 0
fi

ROLLOUT_JOB_NAME="lqrgxyz_${CONFIG_TAG:0:48}"
LAST_RANK=$((WORLD_SIZE - 1))
echo "=== submitting rollout array (0-$LAST_RANK) ==="

# T5 cache wipe (matches sibling launchers).
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

# --- Stage 2: merge (depends on all rollout ranks) -------------------------
MERGE_JOB_NAME="lqrgxyz_merge_${CONFIG_TAG:0:48}"
echo "=== submitting merge (depends on $ROLLOUT_ID) ==="

MERGE_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$MERGE_JOB_NAME" \
    --dependency=afterok:"$ROLLOUT_ID" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=16G \
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
    --prompt '$PROMPT' --preset '$PRESET'
")
echo "  merge job id:   $MERGE_ID  (afterok:$ROLLOUT_ID)"

echo
echo "=== submitted ==="
echo "  rollout : sbatch $ROLLOUT_ID  (array 0-$LAST_RANK)"
echo "  merge   : sbatch $MERGE_ID    (afterok:$ROLLOUT_ID)"
echo "  out_dir : $OUT_DIR"
echo "  logs    : $LOG_DIR"
echo
echo "Monitor:        squeue -u \$USER --start"
echo "Tail rank 0:    tail -f $LOG_DIR/rollout_rank0_${ROLLOUT_ID}.out"
echo "Tail merge:     tail -f $LOG_DIR/merge_${MERGE_ID}.out"
