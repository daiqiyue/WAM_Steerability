#!/bin/bash
# Launcher for compute_jacobians_full.py — sbatch-array workers, then a
# login-node merge once all workers have left the queue.
#
# Stage 1: sbatch array with WORLD_SIZE tasks (--gpus-per-task=1). Each array
#          element derives its (t, l_in) slice from --rank/--num-shards and
#          writes shard_rank<R>.pt.
# Stage 2: poll squeue until all array tasks are gone, then run --phase merge
#          directly in this shell. Merge needs no GPU and no model load, so
#          login-node is fine. Login-node merge sidesteps the
#          DependencyNeverSatisfied trap that bites when a worker crashes
#          during interpreter teardown (non-zero exit AFTER shards saved).
#
# This script BLOCKS while workers run. To detach, launch it under tmux,
# screen, or nohup. If you Ctrl-C during the wait, the workers keep running;
# you can resume the merge later with:
#   MERGE_ONLY=1 PROMPT="<same prompt>" ./run_jacobians_full.sh
#
# Usage:
#   ./run_jacobians_full.sh
#   WORLD_SIZE=8 PROMPT="put both the milk and the tomato sauce in the basket" \
#       SVD_DIR=/path/to/svd_run ./run_jacobians_full.sh
#   INPUTS_NPZ=/path/to/other.npz OBS_INDEX=12 ./run_jacobians_full.sh
#   MERGE_ONLY=1 ./run_jacobians_full.sh   # skip workers, just re-merge shards

set -euo pipefail

# =========================================================================
# User-configurable (override via env)
# =========================================================================
# Number of parallel worker ranks (one GPU each, possibly on different nodes).
# Cosmos-Policy 2B has 28 DiT blocks; num_l_in = 27, so beyond ~27 ranks
# extra GPUs sit idle.
WORLD_SIZE="${WORLD_SIZE:-8}"

PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"

# Scene / observation feeding into get_action. Default is row 0 of the
# libero_10 task 0 capture = the very first frame of episode 0's
# original-prompt rollout (canonical initial scene of "put both the alphabet
# soup and the tomato sauce in the basket"). The Python script avoids zeroed
# inputs because the model is not trained on them, which makes Jacobians on
# zero obs unreliable. Override INPUTS_NPZ / OBS_INDEX to swap scenes.
# INPUTS_NPZ="${INPUTS_NPZ:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00/inputs.npz}"
# INPUTS_NPZ="${INPUTS_NPZ:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_high_4_pos_neg/negative.npz}"
# INPUTS_NPZ="${INPUTS_NPZ:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__object_pairs_pos_neg/negative.npz}"
INPUTS_NPZ="${INPUTS_NPZ:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__tasks0-2-4-5__xyz_random_xlarge_2__seed42__pos_neg__paired/negative.npz}"
OBS_INDEX="${OBS_INDEX:-0}"

# Defaults to the milk-pairs SVD on /u (1.8 PB free); /work/nvme/bhde is at 99%
# usage and was retired as the default output location.
# SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_milk_pairs_dsall_N-1_k64_p10_ws8_tsall}"
# SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_noise_high_4_pairs_no_action_dsall_N-1_k64_p10_ws8_tsall}"
# SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_object_pairs_pairs_no_action_multiprompt_dsall_N-1_k64_p10_ws8_tsall}"
SVD_DIR="${SVD_DIR:-/projects/bhde/jhong7/cosmos-policy/directions/svd/libero10_tasks0-2-4-5_xyz_random_xlarge_2_seed42_paired_multitask_dsall_N-1_k64_p10_ws8_tsall}"


MODE="${MODE:-vjp_no_retain}"          # 'jvp' | 'vjp' | 'vjp_no_retain'
# Default is vjp_no_retain because Cosmos-Policy's DiT wraps each block with
# selective activation checkpointing (SAC), which refuses multiple backward
# passes through the same checkpoint cycle. 'vjp' would crash with
# "Trying to backward an extra time" on any DiT compiled with SAC.
# vjp_no_retain rebuilds the autograd graph for each of the k backward passes
# (~2x slower per Jacobian than vjp, but works with SAC).
V_DEVICE="${V_DEVICE:-cpu}"  # 'cpu' (low GPU peak) or 'cuda'
V_DTYPE="${V_DTYPE:-bf16}"   # V storage dtype (Jacobian acc always fp32)

# Slurm resources
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
WORKER_TIME="${WORKER_TIME:-00:20:00}"

# How often (seconds) to poll squeue while waiting for the worker array.
POLL_INTERVAL="${POLL_INTERVAL:-30}"

# Set MERGE_ONLY=1 to skip worker submission and only run the login-node merge.
# Useful when the wait loop was interrupted and shards are already on disk.
MERGE_ONLY="${MERGE_ONLY:-0}"

EXTRA_WORKER_ARGS="${EXTRA_WORKER_ARGS:-}"  # extra flags forwarded to workers
# =========================================================================

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
echo "  world_size:      $WORLD_SIZE"
echo "  prompt:          \"$PROMPT\""
echo "  inputs_npz:      $INPUTS_NPZ"
echo "  obs_index:       $OBS_INDEX"
echo "  mode:            $MODE"
echo "  v_device:        $V_DEVICE"
echo "  v_dtype:         $V_DTYPE"
echo "  svd_dir:         $SVD_DIR"
echo "  out_dir:         $OUT_DIR"
echo "  account:         $ACCOUNT"
echo "  partition_slurm: $PARTITION_SLURM"
echo "  worker_time:     $WORKER_TIME"
echo "  poll_interval:   ${POLL_INTERVAL}s"
echo "  merge_only:      $MERGE_ONLY"
echo "  extra_worker:    $EXTRA_WORKER_ARGS"
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
)

WORKER_ARGS_QUOTED=""
for a in "${WORKER_COMMON_ARGS[@]}"; do
    WORKER_ARGS_QUOTED+=" $(printf '%q' "$a")"
done

if [[ "$MERGE_ONLY" == "1" || "$MERGE_ONLY" == "true" ]]; then
    echo "=== MERGE_ONLY: skipping worker submission ==="
else
    # ---------- Stage 1: workers (sbatch array, one rank per task) ----------
    WORKER_JOB_NAME="cps_jacfull_${PROMPT_TAG}"
    WORKER_LAST=$((WORLD_SIZE - 1))
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
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE\"
python -u '$PY_SCRIPT' --phase worker --rank \"\$SLURM_ARRAY_TASK_ID\"$WORKER_ARGS_QUOTED $EXTRA_WORKER_ARGS
")
    echo "  worker job id: $WORKER_ID"
    echo "  monitor:       squeue -u \$USER -j $WORKER_ID"
    echo "  tail rank 0:   tail -f $LOG_DIR/worker_rank0_${WORKER_ID}.out"

    # ---------- Stage 2a: wait for the worker array to leave the queue ----------
    # Ctrl-C only kills this loop — the sbatch jobs keep running. Re-run with
    # MERGE_ONLY=1 once they're done to finish the merge.
    echo
    echo "=== waiting for worker array $WORKER_ID (poll every ${POLL_INTERVAL}s) ==="
    while true; do
        # squeue -h -j <jobid> prints one line per active array task. Empty
        # output means every task has reached a terminal state.
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

# ---------- Stage 2b: merge on the login node ----------
echo
echo "=== running merge on login node ==="
# shellcheck disable=SC1091
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd "$SCRIPT_DIR"
# WORKER_COMMON_ARGS already contains everything --phase merge needs.
python -u "$PY_SCRIPT" --phase merge "${WORKER_COMMON_ARGS[@]}"

echo
echo "=== done ==="
echo "  artifacts: $OUT_DIR"
echo "  logs:      $LOG_DIR"
