#!/bin/bash
# Slurm launcher for run_partition_svd_pairs_no_action_multitask.py —
# multi-TASK variant of run_partition_svd_pairs_no_action_multi_prompt.sh.
#
# Consumes the OUTPUT of pair_inputs_by_similarity.sh applied to the
# multitask collector's pos_neg dir (e.g. libero_10__tasks0-2-4-5__...).
# The paired NPZs must carry `pos_src_task_id` / `neg_src_task_id` (preserved
# through pair_inputs_by_similarity.py). Per-row prompts are auto-discovered
# by walking the manifest chain (paired/manifest.json -> input_manifest.tasks)
# or supplied explicitly via TASK_PROMPTS_JSON.
#
# Recommended pairing flow:
#   WITHIN_TASK=1 IN_DIR=<multitask-collector-out> \
#     ./pair_inputs_by_similarity.sh
# then point this script at the resulting `__paired` dir.
#
# Output is plug-in compatible with notebooks/lqr/jacobians/compute_jacobians_*.sh
# and the LQR shells (same config.json shape fields; `prompt` -> `unique_prompts`
# + `prompt_rows_per_config` + `unique_task_ids` + `task_rows_per_task`).
#
# Usage:
#   ./run_partition_svd_pairs_no_action_multitask.sh
#   WORLD_SIZE=8 N=480 ./run_partition_svd_pairs_no_action_multitask.sh
#   DRIVE_SOURCE=0 ./run_partition_svd_pairs_no_action_multitask.sh

set -euo pipefail

# =========================================================================
# User-configurable (override via env)
# =========================================================================
WORLD_SIZE="${WORLD_SIZE:-8}"
N="${N:--1}"
K_TARGET="${K_TARGET:-64}"
P_OVER="${P_OVER:-10}"
PARTITIONS="${PARTITIONS:-0-9,10-18,19-27}"
NUM_LAYERS="${NUM_LAYERS:-28}"

SAMPLING_STEPS="${SAMPLING_STEPS:-5}"
TIMESTEPS="${TIMESTEPS:-all}"
GUIDE_SCALE="${GUIDE_SCALE:-1.0}"

# Paired NPZs from pair_inputs_by_similarity.sh (WITHIN_TASK=1 recommended).
PRESET="${PRESET:-xyz_random_xlarge_2}"
BASE_SEED="${BASE_SEED:-42}"
TASKS_TAG="${TASKS_TAG:-0-2-4-5}"
PAIRED_DIR="${PAIRED_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__tasks${TASKS_TAG}__${PRESET}__seed${BASE_SEED}__pos_neg__paired}"
POS_NPZ="${POS_NPZ:-$PAIRED_DIR/positive.npz}"
NEG_NPZ="${NEG_NPZ:-$PAIRED_DIR/negative.npz}"

# Optional explicit task->prompt map JSON; if unset, manifest auto-discovery.
TASK_PROMPTS_JSON="${TASK_PROMPTS_JSON:-}"

DRIVE_SOURCE="${DRIVE_SOURCE:-all}"

# Set INCLUDE_ACTION=1 to revert to the original behavior (action slot included).
INCLUDE_ACTION="${INCLUDE_ACTION:-0}"

TAG="${TAG:-libero10_tasks${TASKS_TAG}_${PRESET}_seed${BASE_SEED}_paired_multitask_ds${DRIVE_SOURCE}}"

CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
CONFIG_NAME="${CONFIG_NAME:-cosmos_predict2_2b_480p_libero__inference_only}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"

OUT_BASE="${OUT_BASE:-/projects/bhde/jhong7/cosmos-policy/directions/svd}"

ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
SKETCH_TIME="${SKETCH_TIME:-1:00:00}"
SVD_TIME="${SVD_TIME:-01:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

# =========================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PY_SCRIPT="$SCRIPT_DIR/run_partition_svd_pairs_no_action_multitask.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
[[ -f "$POS_NPZ"   ]] || { echo "ERROR: missing $POS_NPZ"   >&2; exit 1; }
[[ -f "$NEG_NPZ"   ]] || { echo "ERROR: missing $NEG_NPZ"   >&2; exit 1; }
if [[ -n "$TASK_PROMPTS_JSON" && ! -f "$TASK_PROMPTS_JSON" ]]; then
    echo "ERROR: TASK_PROMPTS_JSON does not exist: $TASK_PROMPTS_JSON" >&2
    exit 1
fi

TS_TAG="${TIMESTEPS//,/-}"
RUN_TAG="${TAG}_N${N}_k${K_TARGET}_p${P_OVER}_ws${WORLD_SIZE}_ts${TS_TAG}"
OUT_DIR="$OUT_BASE/$RUN_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Scratch on /work/hdd — co-locating scratch with final artifacts on
# /projects/bhde causes concurrent multi-rank writes to hang under IO
# contention when multiple ranks share a node; /work/nvme/bhde is over
# soft quota and rejects bulk writes. /work/hdd has ~800 GB headroom
# and handles the bursty per-rank ~4.4 GB dump (W.bin + Y/mu .npy)
# without contention.
SCRATCH_BASE="${SCRATCH_BASE:-/work/hdd/bhde/jhong7/cosmos-policy/directions/svd}"
SCRATCH_DIR="${SCRATCH_DIR:-$SCRATCH_BASE/$RUN_TAG/scratch}"
KEEP_SCRATCH_FLAG=""
[[ -n "${KEEP_SCRATCH:-}" ]] && KEEP_SCRATCH_FLAG="--keep-scratch"

ACTION_SLOT_FLAG="--exclude-action-slot"
if [[ "$INCLUDE_ACTION" == "1" || "$INCLUDE_ACTION" == "true" ]]; then
    ACTION_SLOT_FLAG="--include-action-slot"
fi

echo "=== config ==="
echo "  world_size:        $WORLD_SIZE"
echo "  N (after filt):    $N  (-1 = use all)"
echo "  k_target:          $K_TARGET"
echo "  p_over:            $P_OVER"
echo "  partitions:        $PARTITIONS"
echo "  num_layers:        $NUM_LAYERS"
echo "  sampling_steps:    $SAMPLING_STEPS"
echo "  timesteps:         $TIMESTEPS"
echo "  guide_scale:       $GUIDE_SCALE"
echo "  paired_dir:        $PAIRED_DIR"
echo "  pos_npz:           $POS_NPZ"
echo "  neg_npz:           $NEG_NPZ"
echo "  task_prompts_json: ${TASK_PROMPTS_JSON:-<auto-discover from manifest>}"
echo "  drive_source:      $DRIVE_SOURCE"
echo "  action_slot_flag:  $ACTION_SLOT_FLAG"
echo "  ckpt_path:         $CKPT_PATH"
echo "  out_dir:           $OUT_DIR"
echo "  scratch_dir:       $SCRATCH_DIR"
echo "  account:           $ACCOUNT"
echo "  partition:         $PARTITION_SLURM"
echo

COMMON_ARGS=(
    --N "$N"
    --k-target "$K_TARGET"
    --p-over "$P_OVER"
    --partitions "$PARTITIONS"
    --num-layers "$NUM_LAYERS"
    --sampling-steps "$SAMPLING_STEPS"
    --timesteps "$TIMESTEPS"
    --guide-scale "$GUIDE_SCALE"
    --pos-npz "$POS_NPZ"
    --neg-npz "$NEG_NPZ"
    --drive-source "$DRIVE_SOURCE"
    "$ACTION_SLOT_FLAG"
    --ckpt-path "$CKPT_PATH"
    --config-name "$CONFIG_NAME"
    --config-file "$CONFIG_FILE"
    --world-size "$WORLD_SIZE"
    --out-dir "$OUT_DIR"
    --scratch-dir "$SCRATCH_DIR"
)
[[ -n "$TASK_PROMPTS_JSON" ]] && COMMON_ARGS+=( --task-prompts-json "$TASK_PROMPTS_JSON" )
[[ -n "$KEEP_SCRATCH_FLAG" ]] && COMMON_ARGS+=("$KEEP_SCRATCH_FLAG")

ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

# ---------- Stage 1: sketch (sbatch array, one rank per task) ----------
SKETCH_JOB_NAME="cpspnamt_sketch_${RUN_TAG}"
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
    --output="$LOG_DIR/sketch_rank%a_%A.out" \
    --error="$LOG_DIR/sketch_rank%a_%A.err" \
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

# ---------- Stage 2: svd (single finalize, dependent on sketch) ----------
SVD_JOB_NAME="cpspnamt_svd_${RUN_TAG}"
echo "=== submitting svd finalize (depends on $SKETCH_ID) ==="

SVD_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$SVD_JOB_NAME" \
    --dependency=afterok:"$SKETCH_ID" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --time="$SVD_TIME" \
    --output="$LOG_DIR/svd_%j.out" \
    --error="$LOG_DIR/svd_%j.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
python -u '$PY_SCRIPT' --mode svd$ARGS_QUOTED
")
echo "  svd job id:    $SVD_ID"

echo
echo "=== submitted ==="
echo "  sketch:    sbatch $SKETCH_ID  (array 0-$SKETCH_LAST)"
echo "  svd:       sbatch $SVD_ID     (afterok:$SKETCH_ID)"
echo "  artifacts: $OUT_DIR"
echo "  logs:      $LOG_DIR"
echo
echo "Monitor with:    squeue -u \$USER --start"
echo "Tail sketch log: tail -f $LOG_DIR/sketch_rank0_${SKETCH_ID}.out"
echo "Tail svd log:    tail -f $LOG_DIR/svd_${SVD_ID}.out"
echo
echo "Next:"
echo "  # Per-inference-index contrastive vectors (c_means_per_j.pt) into the same dir:"
echo "  SVD_DIR='$OUT_DIR' $SCRIPT_DIR/compute_contrastive_per_j.sh"
