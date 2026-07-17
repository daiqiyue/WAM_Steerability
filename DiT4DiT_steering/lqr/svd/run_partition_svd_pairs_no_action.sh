#!/bin/bash
# Slurm launcher for run_partition_svd_pairs_no_action.py — paired-obs SVD
# of DiT4DiT action-DiT activations with the action token subspace.
#
# Analogue of the repo root's notebooks/lqr/svd/run_partition_svd_pairs_no_action.sh
# but uses DiT4DiT instead of Cosmos-Policy.
#
# Usage:
#   ./run_partition_svd_pairs_no_action.sh
#   WORLD_SIZE=8 N=480 ./run_partition_svd_pairs_no_action.sh
#   DRIVE_SOURCE=0 ./run_partition_svd_pairs_no_action.sh

set -euo pipefail

DIT4DIT_ROOT=/work/hdd/bhde/jhong7/DiT4DiT

WORLD_SIZE="${WORLD_SIZE:-8}"
N="${N:--1}"
K_TARGET="${K_TARGET:-64}"
P_OVER="${P_OVER:-10}"
# Default partitions for the 16-block action DiT
PARTITIONS="${PARTITIONS:-0-5,6-10,11-15}"
NUM_LAYERS="${NUM_LAYERS:-16}"

SAMPLING_STEPS="${SAMPLING_STEPS:-4}"
TIMESTEPS="${TIMESTEPS:-all}"

PROMPT="${PROMPT:-put both the cream cheese box and the butter in the basket}"

POS_NPZ="${POS_NPZ:-$DIT4DIT_ROOT/notebooks/lqr/inputs/policy_inputs/libero_10__task01__xyz_random_xlarge_3__seed42__pos_neg__paired/positive.npz}"
NEG_NPZ="${NEG_NPZ:-$DIT4DIT_ROOT/notebooks/lqr/inputs/policy_inputs/libero_10__task01__xyz_random_xlarge_3__seed42__pos_neg__paired/negative.npz}"
DRIVE_SOURCE="${DRIVE_SOURCE:-all}"
# Pre-computed VLM embeddings from precompute_vl_embs.py.
# Set this to avoid each sketch rank loading the full 20 GB model from NFS.
VL_EMBS_PATH="${VL_EMBS_PATH:-}"

TAG="${TAG:-libero10_task01_gripper_xyz_xyz_random_xlarge_3_seed42_paired}"
CKPT_PATH="${CKPT_PATH:-$DIT4DIT_ROOT/checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt}"

OUT_BASE="${OUT_BASE:-$DIT4DIT_ROOT/notebooks/lqr/directions/svd}"

ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
SKETCH_TIME="${SKETCH_TIME:-4:00:00}"
SVD_TIME="${SVD_TIME:-01:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PY_SCRIPT="$SCRIPT_DIR/run_partition_svd_pairs_no_action.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
[[ -f "$POS_NPZ"   ]] || { echo "ERROR: missing $POS_NPZ" >&2; exit 1; }
[[ -f "$NEG_NPZ"   ]] || { echo "ERROR: missing $NEG_NPZ" >&2; exit 1; }

TS_TAG="${TIMESTEPS//,/-}"
RUN_TAG="${TAG}_N${N}_k${K_TARGET}_p${P_OVER}_ws${WORLD_SIZE}_ts${TS_TAG}"
OUT_DIR="$OUT_BASE/$RUN_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SCRATCH_DIR="${SCRATCH_DIR:-$OUT_DIR/scratch}"

echo "=== config ==="
echo "  world_size:     $WORLD_SIZE"
echo "  N:              $N  (-1 = use all)"
echo "  k_target:       $K_TARGET"
echo "  p_over:         $P_OVER"
echo "  partitions:     $PARTITIONS"
echo "  num_layers:     $NUM_LAYERS"
echo "  sampling_steps: $SAMPLING_STEPS"
echo "  timesteps:      $TIMESTEPS"
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
    --prompt "$PROMPT"
    --pos-npz "$POS_NPZ"
    --neg-npz "$NEG_NPZ"
    --drive-source "$DRIVE_SOURCE"
    --ckpt-path "$CKPT_PATH"
    --world-size "$WORLD_SIZE"
    --out-dir "$OUT_DIR"
    --scratch-dir "$SCRATCH_DIR"
)
[[ -n "$VL_EMBS_PATH" ]] && COMMON_ARGS+=(--vl-embs-path "$VL_EMBS_PATH")
[[ -n "${KEEP_SCRATCH:-}" ]] && COMMON_ARGS+=(--keep-scratch)

ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

VENV_ACTIVATE="
set -euo pipefail
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate /projects/bhde/jhong7/dit4dit-env/dit4dit
export PYTHONPATH=/work/nvme/bhde/jhong7/LIBERO_pkg:${PYTHONPATH:-}
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH='$DIT4DIT_ROOT':\${PYTHONPATH:-}
export LIBERO_HOME=/work/nvme/bhde/jhong7/LIBERO_pkg
export LIBERO_CONFIG_PATH=/work/nvme/bhde/jhong7/LIBERO_pkg/libero
"

# ---------- Stage 1: sketch (sbatch array, one rank per task) ----------
SKETCH_LAST=$((WORLD_SIZE - 1))
SKETCH_JOB_NAME="dit4dit_svdsk_${RUN_TAG:0:40}"
echo "=== submitting sketch array (0-$SKETCH_LAST) ==="

SKETCH_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$SKETCH_JOB_NAME" \
    --array=0-"$SKETCH_LAST" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="${SKETCH_MEM:-128G}" \
    --time="$SKETCH_TIME" \
    --output="$LOG_DIR/sketch_rank%a_%A.out" \
    --error="$LOG_DIR/sketch_rank%a_%A.err" \
    --wrap "${VENV_ACTIVATE}
unset WORLD_SIZE RANK LOCAL_RANK
cd '$SCRIPT_DIR'
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE\"
python -u '$PY_SCRIPT' --mode sketch --rank \"\$SLURM_ARRAY_TASK_ID\"$ARGS_QUOTED
")
echo "  sketch job id: $SKETCH_ID"

# ---------- Stage 2: svd finalize (dependent on sketch) ----------
SVD_JOB_NAME="dit4dit_svd_${RUN_TAG:0:44}"
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
    --wrap "${VENV_ACTIVATE}
unset WORLD_SIZE RANK LOCAL_RANK
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
