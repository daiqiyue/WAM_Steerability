#!/bin/bash
# Slurm launcher for notebooks/actadd/run_actadd.py — runs first N episodes
# of libero_10 task 0 with the prompt "put both the milk and the tomato sauce
# in the basket" and an ActAdd hook installed on every DiT block. The
# contrastive direction comes from the partition-SVD output produced by
# notebooks/lqr/svd/run_partition_svd_pairs.sh.
#
# Usage:
#   ./run_actadd.sh
#   ALPHA=2.0 ./run_actadd.sh
#   ALPHA=1.0 RANK=1 LAYERS=10-18 TIMESTEPS=0,1,2 ./run_actadd.sh
#   SVD_DIR=/path/to/svd_run OUT_TAG=mytag ./run_actadd.sh

set -euo pipefail

# =========================================================================
# User-configurable (override via env)
# =========================================================================
SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_milk_pairs_dsall_N-1_k64_p10_ws8_tsall}"

ALPHA="${ALPHA:-1.5}"
RANK="${RANK:--1}"                # -1 = all k_target components
LAYERS="${LAYERS:-all}"           # 'all' | '10-18' | '0,5,12'
TIMESTEPS="${TIMESTEPS:-all}"     # 'all' | '0,2,4'

N_EPISODES="${N_EPISODES:-10}"
SUITE="${SUITE:-libero_10}"
TASK_ID="${TASK_ID:-0}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-5}"
PROMPT_OLD="${PROMPT_OLD:-alphabet soup}"
PROMPT_NEW="${PROMPT_NEW:-milk}"
RESOLUTION="${RESOLUTION:-256}"
VIDEO_FPS="${VIDEO_FPS:-30}"
SEED="${SEED:-1}"

CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
CONFIG_NAME="${CONFIG_NAME:-cosmos_predict2_2b_480p_libero__inference_only}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"

# Output naming.
SVD_TAG="$(basename "$SVD_DIR")"
ALPHA_TAG="alpha$(echo "$ALPHA" | tr '.' 'p' | tr -d '+')"
RANK_TAG="r${RANK}"
LAYERS_TAG="L$(echo "$LAYERS" | tr ',' '-')"
TIMESTEPS_TAG="ts$(echo "$TIMESTEPS" | tr ',' '-')"
DEFAULT_OUT_TAG="${SVD_TAG}__${ALPHA_TAG}_${RANK_TAG}_${LAYERS_TAG}_${TIMESTEPS_TAG}_N${N_EPISODES}"
OUT_TAG="${OUT_TAG:-$DEFAULT_OUT_TAG}"

OUT_BASE="${OUT_BASE:-/u/jhong7/Workspace/cosmos-policy/notebooks/actadd/rollouts}"
OUT_DIR="$OUT_BASE/$OUT_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Slurm resources
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4-interactive}"
TIME="${TIME:-01:30:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"

# =========================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PY_SCRIPT="$SCRIPT_DIR/run_actadd.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
[[ -d "$SVD_DIR"  ]] || { echo "ERROR: SVD_DIR not found: $SVD_DIR" >&2; exit 1; }
[[ -f "$SVD_DIR/config.json" ]] || {
    echo "ERROR: $SVD_DIR has no config.json — finalize the SVD first" >&2
    exit 1
}

echo "=== config ==="
echo "  svd_dir:         $SVD_DIR"
echo "  out_dir:         $OUT_DIR"
echo "  alpha:           $ALPHA"
echo "  rank:            $RANK  (-1 = all)"
echo "  layers:          $LAYERS"
echo "  timesteps:       $TIMESTEPS"
echo "  n_episodes:      $N_EPISODES"
echo "  suite / task_id: $SUITE / $TASK_ID"
echo "  denoise steps:   $NUM_DENOISING_STEPS"
echo "  prompt swap:     '$PROMPT_OLD' -> '$PROMPT_NEW'"
echo "  ckpt:            $CKPT_PATH"
echo "  account:         $ACCOUNT"
echo "  partition:       $PARTITION_SLURM"
echo

CMD_ARGS=(
    --svd-dir "$SVD_DIR"
    --alpha "$ALPHA"
    --rank "$RANK"
    --layers "$LAYERS"
    --timesteps "$TIMESTEPS"
    --n-episodes "$N_EPISODES"
    --suite "$SUITE"
    --task-id "$TASK_ID"
    --num-denoising-steps-action "$NUM_DENOISING_STEPS"
    --prompt-old "$PROMPT_OLD"
    --prompt-new "$PROMPT_NEW"
    --resolution "$RESOLUTION"
    --video-fps "$VIDEO_FPS"
    --seed "$SEED"
    --ckpt-path "$CKPT_PATH"
    --config-name "$CONFIG_NAME"
    --config-file "$CONFIG_FILE"
    --out-dir "$OUT_DIR"
)
ARGS_QUOTED=""
for a in "${CMD_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

JOB_NAME="actadd_${OUT_TAG}"
JOB_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$JOB_NAME" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$LOG_DIR/actadd_%j.out" \
    --error="$LOG_DIR/actadd_%j.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
echo 'launching: $PY_SCRIPT'
python -u '$PY_SCRIPT'$ARGS_QUOTED
")

echo "=== submitted ==="
echo "  job_id:   $JOB_ID"
echo "  out_dir:  $OUT_DIR"
echo "  logs:     $LOG_DIR"
echo
echo "Tail log: tail -f $LOG_DIR/actadd_${JOB_ID}.out"
