#!/bin/bash
# Slurm launcher for compute_contrastive_per_task.py — postprocess that adds
# per-TASK mean contrastive vectors (c_means_per_task.pt) to an existing
# multi-task SVD output dir produced by
# run_partition_svd_pairs_no_action_multitask.sh.
#
# Reuses the V matrices already on disk in $SVD_DIR (no re-SVD); re-runs the
# paired forward passes from positive.npz / negative.npz, accumulates the
# (pos - neg) projected activation difference separately per `pos_src_task_id`,
# and normalizes by per-task pair counts at finalize time.
#
# Stage 1: a sketch sbatch array (one rank per GPU) that runs the assigned
#          range(R, N, W) of pairs and dumps a small per-rank partial
#          (T_tasks, L_p, k_target) accumulator per (partition, timestep).
# Stage 2: a single-GPU finalize job (depends on sketch) that sums the
#          partials across ranks, divides by per-task counts, stitches the
#          per-partition tiles, and writes $SVD_DIR/c_means_per_task.pt.
#
# Output:  $SVD_DIR/c_means_per_task.pt
#
# Usage:
#   ./compute_contrastive_per_task.sh
#   SVD_DIR=/path/to/multitask_svd_dir ./compute_contrastive_per_task.sh
#   WORLD_SIZE=4 ./compute_contrastive_per_task.sh

set -euo pipefail

# =========================================================================
# User-configurable (override via env)
# =========================================================================
WORLD_SIZE="${WORLD_SIZE:-8}"

# Defaults to the most recently used multi-task SVD output. Override to
# point at the 0-1-7 dir or any other multitask run.
SVD_BASE="${SVD_BASE:-/work/hdd/bhde/jhong7/cosmos-policy/directions/svd}"
SVD_DIR_NAME="${SVD_DIR_NAME:-libero10_tasks0-1-7_xyz_random_xlarge_2_seed42_paired_multitask_dsall_N-1_k64_p10_ws8_tsall}"
SVD_DIR="${SVD_DIR:-$SVD_BASE/$SVD_DIR_NAME}"

# Empty -> use the value recorded in $SVD_DIR/config.json. The per-task
# prompts default to the unique_task_ids + unique_prompts already in
# config.json (no extra mapping file needed).
POS_NPZ="${POS_NPZ:-}"
NEG_NPZ="${NEG_NPZ:-}"
TASK_PROMPTS_JSON="${TASK_PROMPTS_JSON:-}"
DRIVE_SOURCE="${DRIVE_SOURCE:-}"
N="${N:-}"
OUTPUT_NAME="${OUTPUT_NAME:-c_means_per_task.pt}"
V_DEVICE="${V_DEVICE:-auto}"

# Scratch on /work/hdd (matches the multitask SVD launcher's reasoning).
SCRATCH_BASE="${SCRATCH_BASE:-/work/hdd/bhde/jhong7/cosmos-policy/directions/svd}"
SCRATCH_DIR="${SCRATCH_DIR:-$SCRATCH_BASE/$(basename "$SVD_DIR")/scratch_per_task}"
KEEP_SCRATCH_FLAG=""
[[ -n "${KEEP_SCRATCH:-}" ]] && KEEP_SCRATCH_FLAG="--keep-scratch"

# Slurm resources.
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
SKETCH_TIME="${SKETCH_TIME:-01:00:00}"
FINALIZE_TIME="${FINALIZE_TIME:-00:30:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
# Optional shared exclude list (best-effort; tolerate missing helper).
EXCLUDE_NODES_DEFAULT=""
if [[ -f "$(dirname -- "${BASH_SOURCE[0]}")/../../exclude_nodes.sh" ]]; then
    # shellcheck disable=SC1091
    . "$(dirname -- "${BASH_SOURCE[0]}")/../../exclude_nodes.sh"
fi
EXCLUDE_NODES="${EXCLUDE_NODES-$EXCLUDE_NODES_DEFAULT}"
# =========================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PY_SCRIPT="$SCRIPT_DIR/compute_contrastive_per_task.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
[[ -d "$SVD_DIR"   ]] || { echo "ERROR: missing SVD_DIR $SVD_DIR" >&2; exit 1; }
[[ -f "$SVD_DIR/config.json" ]] || { echo "ERROR: missing $SVD_DIR/config.json" >&2; exit 1; }
if [[ -n "$TASK_PROMPTS_JSON" && ! -f "$TASK_PROMPTS_JSON" ]]; then
    echo "ERROR: TASK_PROMPTS_JSON does not exist: $TASK_PROMPTS_JSON" >&2
    exit 1
fi

LOG_DIR="$SVD_DIR/logs"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
    --svd-dir    "$SVD_DIR"
    --scratch-dir "$SCRATCH_DIR"
    --output-name "$OUTPUT_NAME"
    --v-device    "$V_DEVICE"
    --world-size  "$WORLD_SIZE"
)
[[ -n "$POS_NPZ"           ]] && COMMON_ARGS+=( --pos-npz "$POS_NPZ" )
[[ -n "$NEG_NPZ"           ]] && COMMON_ARGS+=( --neg-npz "$NEG_NPZ" )
[[ -n "$TASK_PROMPTS_JSON" ]] && COMMON_ARGS+=( --task-prompts-json "$TASK_PROMPTS_JSON" )
[[ -n "$DRIVE_SOURCE"      ]] && COMMON_ARGS+=( --drive-source "$DRIVE_SOURCE" )
[[ -n "$N"                 ]] && COMMON_ARGS+=( --N "$N" )
[[ -n "$KEEP_SCRATCH_FLAG" ]] && COMMON_ARGS+=( "$KEEP_SCRATCH_FLAG" )

ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

echo "=== config ==="
echo "  svd_dir:           $SVD_DIR"
echo "  world_size:        $WORLD_SIZE"
echo "  pos_npz:           ${POS_NPZ:-<from svd-dir config.json>}"
echo "  neg_npz:           ${NEG_NPZ:-<from svd-dir config.json>}"
echo "  task_prompts_json: ${TASK_PROMPTS_JSON:-<from svd-dir config.json: unique_task_ids+unique_prompts>}"
echo "  drive_source:      ${DRIVE_SOURCE:-<from svd-dir config.json>}"
echo "  N:                 ${N:-<from svd-dir config.json>}"
echo "  output:            $SVD_DIR/$OUTPUT_NAME"
echo "  scratch_dir:       $SCRATCH_DIR"
echo "  v_device:          $V_DEVICE"
echo "  account=$ACCOUNT  partition=$PARTITION_SLURM  sketch_time=$SKETCH_TIME  finalize_time=$FINALIZE_TIME"
echo "  exclude_nodes=${EXCLUDE_NODES:-<none>}"
echo

# ---------- Stage 1: sketch (sbatch array, one rank per GPU) ----------
SKETCH_JOB_NAME="cpt_sketch_$(basename "$SVD_DIR")"
SKETCH_LAST=$((WORLD_SIZE - 1))
echo "=== submitting sketch array (0-$SKETCH_LAST) ==="

SKETCH_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$SKETCH_JOB_NAME" \
    --array=0-"$SKETCH_LAST" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --time="$SKETCH_TIME" \
    --output="$LOG_DIR/per_task_sketch_rank%a_%A.out" \
    --error="$LOG_DIR/per_task_sketch_rank%a_%A.err" \
    ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE\"
python -u '$PY_SCRIPT' --mode sketch --rank \"\$SLURM_ARRAY_TASK_ID\"$ARGS_QUOTED
")
echo "  sketch job id: $SKETCH_ID"

# ---------- Stage 2: finalize (single GPU, depends on sketch) ----------
FINAL_JOB_NAME="cpt_finalize_$(basename "$SVD_DIR")"
echo "=== submitting finalize (depends on $SKETCH_ID) ==="

FINAL_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$FINAL_JOB_NAME" \
    --dependency=afterok:"$SKETCH_ID" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --time="$FINALIZE_TIME" \
    --output="$LOG_DIR/per_task_finalize_%j.out" \
    --error="$LOG_DIR/per_task_finalize_%j.err" \
    ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
python -u '$PY_SCRIPT' --mode finalize$ARGS_QUOTED
")
echo "  finalize job id: $FINAL_ID"

echo
echo "=== submitted ==="
echo "  sketch:   sbatch $SKETCH_ID  (array 0-$SKETCH_LAST)"
echo "  finalize: sbatch $FINAL_ID   (afterok:$SKETCH_ID)"
echo "  output:   $SVD_DIR/$OUTPUT_NAME"
echo "  logs:     $LOG_DIR"
echo
echo "Monitor with:    squeue -u \$USER --start"
echo "Tail sketch log: tail -f $LOG_DIR/per_task_sketch_rank0_${SKETCH_ID}.out"
echo "Tail final log:  tail -f $LOG_DIR/per_task_finalize_${FINAL_ID}.out"
