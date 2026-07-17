#!/bin/bash
# Slurm launcher for run_lqr_dit4dit_gripper_xyz.py.
#
# Analogue of the repo root's notebooks/lqr/run_lqr_cosmos_policy_gripper_xyz.sh
# adapted for DiT4DiT.
#
# Submits sbatch array of WORLD_SIZE workers (one GPU each), each handling
# its slice of N_EPISODES.  After all workers finish, a dependent merge job
# aggregates per-rank results into results.json + manifest.json.
#
# Usage:
#   ./run_lqr_dit4dit_gripper_xyz.sh
#   WORLD_SIZE=4 N_EPISODES=50 ./run_lqr_dit4dit_gripper_xyz.sh
#   MERGE_ONLY=1 ./run_lqr_dit4dit_gripper_xyz.sh

set -euo pipefail

DIT4DIT_ROOT=/work/hdd/bhde/jhong7/DiT4DiT
LQR_ROOT=$DIT4DIT_ROOT/notebooks/lqr

# --- Parallelization --------------------------------------------------------
WORLD_SIZE="${WORLD_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"

# --- Gripper perturbation ---------------------------------------------------
PRESET="${PRESET:-xyz_random_xlarge_3}"
BASE_SEED="${BASE_SEED:-42}"
GRIPPER_ACTION="${GRIPPER_ACTION:--1.0}"

# --- LQR cost hyperparameters ----------------------------------------------
LAMBDA="${LAMBDA:-10.0}"
Q_SCALE="${Q_SCALE:-1.0}"
R_SCALE="${R_SCALE:-10.0}"
R_SCALE_TAU="${R_SCALE_TAU:-7.0}"
R_SCALE_FINAL="${R_SCALE_FINAL:-1e9}"
MAX_CHUNKS="${MAX_CHUNKS:-50}"
QF_SCALE="${QF_SCALE:-1.0}"

# --- Rollout knobs ----------------------------------------------------------
PROMPT="${PROMPT:-put both the cream cheese box and the butter in the basket}"
TASK_ID="${TASK_ID:-1}"
SUITE="${SUITE:-libero_10}"
RESOLUTION="${RESOLUTION:-256}"
VIDEO_FPS="${VIDEO_FPS:-30}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1000}"
RUN_BASELINE="${RUN_BASELINE:-0}"
SEED="${SEED:-1}"

# --- SVD / jacobian inputs --------------------------------------------------
SVD_DIR="${SVD_DIR:-}"
JAC_DIR_ACT="${JAC_DIR_ACT:-}"

if [[ -z "$SVD_DIR" || -z "$JAC_DIR_ACT" ]]; then
    echo "ERROR: SVD_DIR and JAC_DIR_ACT must be set." >&2
    echo "  SVD_DIR:    path to the svd output directory (containing config.json, svd_summary.pt)" >&2
    echo "  JAC_DIR_ACT: subdir name under SVD_DIR containing A_tilde__full.pt" >&2
    exit 1
fi

# --- Slurm resources --------------------------------------------------------
WORKER_TIME="${WORKER_TIME:-01:00:00}"
MERGE_TIME="${MERGE_TIME:-00:10:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

# --- Output naming ----------------------------------------------------------
CKPT_PATH="${CKPT_PATH:-$DIT4DIT_ROOT/checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt}"
_slug() { local s="$1"; local n="${2:-24}"; echo "${s:0:$n}" | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_-'; }
PROMPT_SLUG="$(_slug "$PROMPT" 24)"
LQR_TAG="lam${LAMBDA}_q${Q_SCALE}_rinit${R_SCALE}_rfin${R_SCALE_FINAL}_tau${R_SCALE_TAU}_qf${QF_SCALE}"
PERT_TAG="${PRESET}_seed${BASE_SEED}"
DEFAULT_TAG="${SUITE}__task$(printf '%02d' "$TASK_ID")__lqr_grxyz__${PERT_TAG}__${LQR_TAG}__${PROMPT_SLUG}"
RUN_TAG="${RUN_TAG:-}"
CONFIG_TAG="${CONFIG_TAG:-$DEFAULT_TAG}"
[[ -n "$RUN_TAG" ]] && CONFIG_TAG="${CONFIG_TAG}__${RUN_TAG}"

TAG="${TAG:-seed1}"
OUT_BASE="${OUT_BASE:-$LQR_ROOT/rollouts}"
OUT_DIR="$OUT_BASE${TAG:+/$TAG}/$CONFIG_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_lqr_dit4dit_gripper_xyz.py"
MERGE_ONLY="${MERGE_ONLY:-0}"

[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
if [[ "$MERGE_ONLY" != "1" ]]; then
    [[ -d "$SVD_DIR" ]] || { echo "ERROR: SVD_DIR not found: $SVD_DIR" >&2; exit 1; }
    [[ -f "$SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" ]] || {
        echo "ERROR: A_tilde missing at $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" >&2
        exit 1
    }
fi

echo "=== run_lqr_dit4dit_gripper_xyz ==="
echo "  WORLD_SIZE     = $WORLD_SIZE"
echo "  N_EPISODES     = $N_EPISODES"
echo "  PRESET         = $PRESET   (base_seed=$BASE_SEED  gripper_action=$GRIPPER_ACTION)"
echo "  SVD_DIR        = $SVD_DIR"
echo "  JAC_DIR_ACT    = $JAC_DIR_ACT"
echo "  PROMPT         = $PROMPT"
echo "  LAMBDA         = $LAMBDA  Q_SCALE = $Q_SCALE  QF_SCALE = $QF_SCALE"
echo "  R_SCALE (init) = $R_SCALE  R_SCALE_TAU = $R_SCALE_TAU  R_SCALE_FINAL = $R_SCALE_FINAL"
echo "  MAX_CHUNKS     = $MAX_CHUNKS"
echo "  TASK_ID        = $TASK_ID  SUITE = $SUITE"
echo "  OUT_DIR        = $OUT_DIR"
echo "  account=$ACCOUNT  partition=$PARTITION_SLURM  worker_time=$WORKER_TIME"
echo "  merge_only=$MERGE_ONLY"
echo

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
    --ckpt-path "$CKPT_PATH"
)
if [[ "$RUN_BASELINE" == "1" || "$RUN_BASELINE" == "true" ]]; then
    COMMON_ARGS+=( --run-baseline )
else
    COMMON_ARGS+=( --no-baseline )
fi
[[ -n "$SEED" ]] && COMMON_ARGS+=( --seed "$SEED" )
[[ -n "$TAG"  ]] && COMMON_ARGS+=( --tag "$TAG" )

ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

VENV_ACTIVATE="
set -euo pipefail
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate /projects/bhde/jhong7/dit4dit-env/dit4dit
# These rollout array tasks are INDEPENDENT single-GPU episode shards, not a
# torch.distributed group. The sweep exports WORLD_SIZE/RANK for episode
# sharding and they leak in via --export=ALL; the script gets its sharding from
# the --world-size/--rank CLI args instead, so unset the env copies here to stop
# DiT4DiT's overwatch (accelerate PartialState) from trying to init multinode
# (which fails with 'MASTER_ADDR not set').
unset WORLD_SIZE RANK LOCAL_RANK
export PYTHONPATH=/work/nvme/bhde/jhong7/LIBERO_pkg:${PYTHONPATH:-}
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH='$DIT4DIT_ROOT':\${PYTHONPATH:-}
export LIBERO_HOME=/work/nvme/bhde/jhong7/LIBERO_pkg
export LIBERO_CONFIG_PATH=/work/nvme/bhde/jhong7/LIBERO_pkg/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
"

if [[ "$MERGE_ONLY" == "1" ]]; then
    echo "=== MERGE_ONLY=1; submitting merge-only job ==="
    MERGE_ONLY_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION_SLURM" \
        --job-name="dit4dit_lqr_merge_${CONFIG_TAG:0:48}" \
        --gpus-per-task=1 \
        --ntasks=1 \
        --cpus-per-task=2 \
        --mem=16G \
        --time="$MERGE_TIME" \
        --output="$LOG_DIR/merge_only_%j.out" \
        --error="$LOG_DIR/merge_only_%j.err" \
        --wrap "${VENV_ACTIVATE}
cd '$SCRIPT_DIR'
python -u '$PY_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --svd-dir '$SVD_DIR' --jac-dir-act '$JAC_DIR_ACT' \
    --prompt '$PROMPT' --preset '$PRESET' --ckpt-path '$CKPT_PATH'
")
    echo "  merge-only job id: $MERGE_ONLY_ID"
    exit 0
fi

# --- Stage 1: rollout array -------------------------------------------------
ROLLOUT_JOB_NAME="dit4dit_lqr_${CONFIG_TAG:0:48}"
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
    --wrap "${VENV_ACTIVATE}
cd '$SCRIPT_DIR'
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE\"
python -u '$PY_SCRIPT' --phase rollout --rank \"\$SLURM_ARRAY_TASK_ID\"$ARGS_QUOTED
")
echo "  rollout job id: $ROLLOUT_ID  (array 0-$LAST_RANK)"

# --- Stage 2: merge ---------------------------------------------------------
MERGE_JOB_NAME="dit4dit_lqr_merge_${CONFIG_TAG:0:44}"
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
    --wrap "${VENV_ACTIVATE}
cd '$SCRIPT_DIR'
python -u '$PY_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --svd-dir '$SVD_DIR' --jac-dir-act '$JAC_DIR_ACT' \
    --prompt '$PROMPT' --preset '$PRESET' --ckpt-path '$CKPT_PATH'
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
