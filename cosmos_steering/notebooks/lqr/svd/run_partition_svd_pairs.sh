#!/bin/bash
# Slurm launcher for Cosmos-Policy-LIBERO-Predict2-2B partition-SVD with
# *paired observation* contrast (vs. the prompt-swap contrast of
# run_partition_svd_cosmos_policy.sh).
#
# Inputs: two paired npz files (positive.npz, negative.npz) produced by
#   notebooks/lqr/inputs/collect_policy_inputs_milk.ipynb. Each row in
#   positive.npz is a contrastive pair with the same row in negative.npz —
#   same proprio, different camera content (distractors absent vs present).
#
# Stage 1: an sbatch array with WORLD_SIZE tasks. Each array element pins
#          one GPU (gpus-per-task=1) and processes pairs `range(rank, N, ws)`.
#          Per-rank state is dumped under <scratch>/rank{rank}/.
# Stage 2: a single dependent sbatch finalize that sums across ranks and
#          runs Halko one-pass per (partition, timestep).
#
# Output is plug-in compatible with notebooks/lqr/jacobians/compute_jacobians_*.sh:
# point their SVD_DIR at OUT_DIR and the existing Jacobian scripts run unchanged.
#
# Usage:
#   ./run_partition_svd_pairs.sh
#   WORLD_SIZE=8 N=480 ./run_partition_svd_pairs.sh
#   DRIVE_SOURCE=0 ./run_partition_svd_pairs.sh           # only neg-drives rows
#   DRIVE_SOURCE=1 ./run_partition_svd_pairs.sh           # only pos-drives rows

set -euo pipefail

# =========================================================================
# User-configurable (override via env)
# =========================================================================
WORLD_SIZE="${WORLD_SIZE:-8}"            # number of parallel sketch ranks
N="${N:--1}"                              # paired rows to use after drive-source filter (-1 = all)
K_TARGET="${K_TARGET:-64}"
P_OVER="${P_OVER:-10}"
PARTITIONS="${PARTITIONS:-0-9,10-18,19-27}"
NUM_LAYERS="${NUM_LAYERS:-28}"

SAMPLING_STEPS="${SAMPLING_STEPS:-5}"
TIMESTEPS="${TIMESTEPS:-all}"            # subset, e.g. "0,2,4", or "all"
GUIDE_SCALE="${GUIDE_SCALE:-1.0}"

# Single prompt — used for BOTH the positive and negative forward pass.
PROMPT="${PROMPT:-put both the milk and the tomato sauce in the basket}"

# Paired-obs npz files produced by collect_policy_inputs_milk.ipynb.
POS_NPZ="${POS_NPZ:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__milk_pos_neg/positive.npz}"
NEG_NPZ="${NEG_NPZ:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__milk_pos_neg/negative.npz}"
DRIVE_SOURCE="${DRIVE_SOURCE:-all}"      # 'all' | '0' (neg-drives) | '1' (pos-drives)

TAG="${TAG:-libero10_task00_milk_pairs_ds${DRIVE_SOURCE}}"

CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
CONFIG_NAME="${CONFIG_NAME:-cosmos_predict2_2b_480p_libero__inference_only}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"

OUT_BASE="${OUT_BASE:-/work/nvme/bhde/jhong7/cosmos-policy/directions/svd}"

# Slurm resources
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
SKETCH_TIME="${SKETCH_TIME:-1:00:00}"
SVD_TIME="${SVD_TIME:-01:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

# =========================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PY_SCRIPT="$SCRIPT_DIR/run_partition_svd_pairs.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
[[ -f "$POS_NPZ" ]] || { echo "ERROR: missing $POS_NPZ" >&2; exit 1; }
[[ -f "$NEG_NPZ" ]] || { echo "ERROR: missing $NEG_NPZ" >&2; exit 1; }

# Sanitize TIMESTEPS for filenames: "0,2,4" -> "0-2-4", keep "all".
TS_TAG="${TIMESTEPS//,/-}"
RUN_TAG="${TAG}_N${N}_k${K_TARGET}_p${P_OVER}_ws${WORLD_SIZE}_ts${TS_TAG}"
OUT_DIR="$OUT_BASE/$RUN_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SCRATCH_DIR="${SCRATCH_DIR:-$OUT_DIR/scratch}"
KEEP_SCRATCH_FLAG=""
[[ -n "${KEEP_SCRATCH:-}" ]] && KEEP_SCRATCH_FLAG="--keep-scratch"

echo "=== config ==="
echo "  world_size:     $WORLD_SIZE"
echo "  N (after filt): $N  (-1 = use all)"
echo "  k_target:       $K_TARGET"
echo "  p_over:         $P_OVER"
echo "  partitions:     $PARTITIONS"
echo "  num_layers:     $NUM_LAYERS"
echo "  sampling_steps: $SAMPLING_STEPS"
echo "  timesteps:      $TIMESTEPS"
echo "  guide_scale:    $GUIDE_SCALE"
echo "  prompt:         \"$PROMPT\""
echo "  pos_npz:        $POS_NPZ"
echo "  neg_npz:        $NEG_NPZ"
echo "  drive_source:   $DRIVE_SOURCE"
echo "  ckpt_path:      $CKPT_PATH"
echo "  out_dir:        $OUT_DIR"
echo "  scratch_dir:    $SCRATCH_DIR"
echo "  account:        $ACCOUNT"
echo "  partition:      $PARTITION_SLURM"
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
    --prompt "$PROMPT"
    --pos-npz "$POS_NPZ"
    --neg-npz "$NEG_NPZ"
    --drive-source "$DRIVE_SOURCE"
    --ckpt-path "$CKPT_PATH"
    --config-name "$CONFIG_NAME"
    --config-file "$CONFIG_FILE"
    --world-size "$WORLD_SIZE"
    --out-dir "$OUT_DIR"
    --scratch-dir "$SCRATCH_DIR"
)
[[ -n "$KEEP_SCRATCH_FLAG" ]] && COMMON_ARGS+=("$KEEP_SCRATCH_FLAG")

# Serialize args into a single quoted string for the inner bash -c.
ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

# ---------- Stage 1: sketch (sbatch array, one rank per task) ----------
SKETCH_JOB_NAME="cpsp_sketch_${RUN_TAG}"
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
SVD_JOB_NAME="cpsp_svd_${RUN_TAG}"
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
echo "  sketch:   sbatch $SKETCH_ID  (array 0-$SKETCH_LAST)"
echo "  svd:      sbatch $SVD_ID     (afterok:$SKETCH_ID)"
echo "  artifacts: $OUT_DIR"
echo "  logs:      $LOG_DIR"
echo
echo "Monitor with: squeue -u \$USER --start"
echo "Tail sketch log: tail -f $LOG_DIR/sketch_rank0_${SKETCH_ID}.out"
echo "Tail svd log:    tail -f $LOG_DIR/svd_${SVD_ID}.out"
