#!/bin/bash
# Launcher for compute_jacobians_full.py — sbatch-array workers, then a
# login-node merge once all workers have left the queue.
#
# Analogue of the repo root's notebooks/lqr/jacobians/run_jacobians_full.sh
# adapted for DiT4DiT.
#
# Stage 1: sbatch array with WORLD_SIZE tasks (--gpus-per-task=1).
# Stage 2: poll squeue until all array tasks are gone, then run --phase merge
#          directly in this shell (no GPU needed).
#
# This script BLOCKS while workers run. Run under tmux/screen/nohup.
# If interrupted, re-run with MERGE_ONLY=1 to skip workers and re-merge.
#
# Usage:
#   ./run_jacobians_full.sh
#   WORLD_SIZE=8 PROMPT="put both the cream cheese box and the butter in the basket" \
#       SVD_DIR=/path/to/svd_run ./run_jacobians_full.sh
#   MERGE_ONLY=1 ./run_jacobians_full.sh

set -euo pipefail

DIT4DIT_ROOT=/projects/bhhv/jskifstad/DiT4DiT

WORLD_SIZE="${WORLD_SIZE:-8}"

PROMPT="${PROMPT:-put both the cream cheese box and the butter in the basket}"

INPUTS_NPZ="${INPUTS_NPZ:-$DIT4DIT_ROOT/notebooks/lqr/inputs/policy_inputs/libero_10__task01__xyz_random_xlarge_3__seed42__pos_neg__paired/negative.npz}"
OBS_INDEX="${OBS_INDEX:-0}"

SVD_DIR="${SVD_DIR:-}"
[[ -n "$SVD_DIR" ]] || { echo "ERROR: SVD_DIR not set" >&2; exit 1; }

MODE="${MODE:-vjp_no_retain}"
V_DEVICE="${V_DEVICE:-cpu}"
V_DTYPE="${V_DTYPE:-bf16}"

CKPT_PATH="${CKPT_PATH:-$DIT4DIT_ROOT/checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt}"

ACCOUNT="${ACCOUNT:-bhhv-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
WORKER_TIME="${WORKER_TIME:-00:30:00}"

POLL_INTERVAL="${POLL_INTERVAL:-30}"
MERGE_ONLY="${MERGE_ONLY:-0}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PY_SCRIPT="$SCRIPT_DIR/compute_jacobians_full.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
[[ -d "$SVD_DIR"   ]] || { echo "ERROR: missing SVD_DIR $SVD_DIR" >&2; exit 1; }

PROMPT_TAG=$(echo "$PROMPT" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50)
RUN_TAG="A_tilde_full__${PROMPT_TAG}__${MODE}__v${V_DTYPE}"
OUT_DIR="$SVD_DIR/$RUN_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== config ==="
echo "  world_size:   $WORLD_SIZE"
echo "  prompt:       \"$PROMPT\""
echo "  inputs_npz:   $INPUTS_NPZ"
echo "  obs_index:    $OBS_INDEX"
echo "  mode:         $MODE"
echo "  v_device:     $V_DEVICE"
echo "  v_dtype:      $V_DTYPE"
echo "  svd_dir:      $SVD_DIR"
echo "  out_dir:      $OUT_DIR"
echo "  ckpt_path:    $CKPT_PATH"
echo "  account:      $ACCOUNT"
echo "  partition:    $PARTITION_SLURM"
echo "  worker_time:  $WORKER_TIME"
echo "  poll:         ${POLL_INTERVAL}s"
echo "  merge_only:   $MERGE_ONLY"
echo

WORKER_COMMON_ARGS=(
    --num-shards "$WORLD_SIZE"
    --svd-dir "$SVD_DIR"
    --out-dir "$OUT_DIR"
    --prompt "$PROMPT"
    --inputs-npz "$INPUTS_NPZ"
    --obs-index "$OBS_INDEX"
    --mode "$MODE"
    --v-device "$V_DEVICE"
    --v-dtype "$V_DTYPE"
    --ckpt-path "$CKPT_PATH"
)

WORKER_ARGS_QUOTED=""
for a in "${WORKER_COMMON_ARGS[@]}"; do
    WORKER_ARGS_QUOTED+=" $(printf '%q' "$a")"
done

VENV_ACTIVATE="
set -euo pipefail
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate /projects/bhhv/jskifstad/DiT4DiT/.conda/envs/dit4dit
export PYTHONPATH=/work/nvme/bhhv/jskifstad/LIBERO:${PYTHONPATH:-}
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH='$DIT4DIT_ROOT':\${PYTHONPATH:-}
export LIBERO_HOME=/work/nvme/bhhv/jskifstad/LIBERO
export LIBERO_CONFIG_PATH=/work/nvme/bhhv/jskifstad/LIBERO/libero
"

if [[ "$MERGE_ONLY" == "1" || "$MERGE_ONLY" == "true" ]]; then
    echo "=== MERGE_ONLY: skipping worker submission ==="
else
    WORKER_LAST=$((WORLD_SIZE - 1))
    WORKER_JOB_NAME="dit4dit_jacfull_${PROMPT_TAG:0:32}"
    echo "=== submitting worker array (0-$WORKER_LAST) ==="

    WORKER_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION_SLURM" \
        --job-name="$WORKER_JOB_NAME" \
        --array=0-"$WORKER_LAST" \
        --gpus-per-task=1 \
        --ntasks=1 \
        --time="$WORKER_TIME" \
        --output="$LOG_DIR/worker_rank%a_%A.out" \
        --error="$LOG_DIR/worker_rank%a_%A.err" \
        --wrap "${VENV_ACTIVATE}
unset WORLD_SIZE RANK LOCAL_RANK
cd '$SCRIPT_DIR'
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE\"
python -u '$PY_SCRIPT' --phase worker --rank \"\$SLURM_ARRAY_TASK_ID\"$WORKER_ARGS_QUOTED
")
    echo "  worker job id: $WORKER_ID"
    echo "  monitor:       squeue -u \$USER -j $WORKER_ID"
    echo "  tail rank 0:   tail -f $LOG_DIR/worker_rank0_${WORKER_ID}.out"

    echo
    echo "=== waiting for worker array $WORKER_ID (poll every ${POLL_INTERVAL}s) ==="
    while true; do
        ACTIVE=$(squeue -h -j "$WORKER_ID" -r -o '%A_%a %T' 2>/dev/null || true)
        if [[ -z "$ACTIVE" ]]; then
            break
        fi
        n_total=$(echo "$ACTIVE" | wc -l)
        n_pending=$(echo "$ACTIVE" | awk '$2=="PENDING"' | wc -l)
        n_running=$(echo "$ACTIVE" | awk '$2=="RUNNING"' | wc -l)
        n_other=$((n_total - n_pending - n_running))
        printf '  [%s] active=%d  pending=%d running=%d other=%d\n' \
            "$(date '+%H:%M:%S')" "$n_total" "$n_pending" "$n_running" "$n_other"
        sleep "$POLL_INTERVAL"
    done
    echo "  worker array drained from queue."
fi

echo
echo "=== running merge on login node ==="
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate /projects/bhhv/jskifstad/DiT4DiT/.conda/envs/dit4dit
export PYTHONPATH=/work/nvme/bhhv/jskifstad/LIBERO:${PYTHONPATH:-}
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$DIT4DIT_ROOT":${PYTHONPATH:-}
export LIBERO_HOME=/work/nvme/bhhv/jskifstad/LIBERO
export LIBERO_CONFIG_PATH=/work/nvme/bhhv/jskifstad/LIBERO/libero

unset WORLD_SIZE RANK LOCAL_RANK
cd "$SCRIPT_DIR"
python -u "$PY_SCRIPT" --phase merge "${WORKER_COMMON_ARGS[@]}"

echo
echo "=== done ==="
echo "  artifacts: $OUT_DIR"
echo "  logs:      $LOG_DIR"
