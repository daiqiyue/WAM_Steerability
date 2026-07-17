#!/bin/bash
# Per-task Jacobian launcher.
#
# Loops over libero_10 tasks (default 0..9), and for each task submits the
# full compute_jacobians_full.py pipeline (sbatch-array workers + login-node
# merge) via run_jacobians_full.sh, with:
#     PROMPT     = that task's libero prompt
#     INPUTS_NPZ = that task's first-observation npz produced by
#                  collect_first_obs_per_task.py
#     OUT_DIR    = $SVD_DIR/A_tilde_full__task{TID:02d}__<prompt-slug>__...
#
# All tasks reuse the SAME SVD (the multitask one), so only the prompt and
# observation differ between per-task jacobians -- the V projection is
# unchanged.
#
# This script runs run_jacobians_full.sh SEQUENTIALLY per task, since each
# inner launcher itself blocks on its sbatch array. To parallelize across
# tasks (multiple inner pipelines submitted before any has finished), set
# PARALLEL_TASKS=1 -- the script will background each task and wait at the
# end. The shared cluster has plenty of nodes so this typically saves
# 10x wall time.
#
# Inputs root must already exist: produced by
# notebooks/lqr/inputs/collect_first_obs_per_task.sh
#
# Usage:
#   ./run_jacobians_per_task.sh                       # serial across tasks
#   PARALLEL_TASKS=1 ./run_jacobians_per_task.sh      # parallel (recommended)
#   TASK_IDS="0 1 2" ./run_jacobians_per_task.sh      # subset of tasks

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
INNER_SH="$SCRIPT_DIR/run_jacobians_full.sh"
[[ -x "$INNER_SH" ]] || { echo "ERROR: $INNER_SH not executable" >&2; exit 1; }

TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
SUITE="${SUITE:-libero_10}"
INPUTS_ROOT="${INPUTS_ROOT:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__first_obs__xyz_random_xlarge_2__seed42}"
SVD_DIR="${SVD_DIR:-/projects/bhde/jhong7/cosmos-policy/directions/svd/libero10_tasks0-2-4-5_xyz_random_xlarge_2_seed42_paired_multitask_dsall_N-1_k64_p10_ws8_tsall}"
WORLD_SIZE="${WORLD_SIZE:-8}"
MODE="${MODE:-vjp_no_retain}"
V_DEVICE="${V_DEVICE:-cpu}"
V_DTYPE="${V_DTYPE:-bf16}"
WORKER_TIME="${WORKER_TIME:-00:20:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
PARALLEL_TASKS="${PARALLEL_TASKS:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

MANIFEST="$INPUTS_ROOT/manifest.json"
[[ -f "$MANIFEST" ]] || {
    echo "ERROR: missing inputs manifest: $MANIFEST" >&2
    echo "  Run notebooks/lqr/inputs/collect_first_obs_per_task.sh first." >&2
    exit 1
}
[[ -d "$SVD_DIR" ]] || {
    echo "ERROR: SVD_DIR not found: $SVD_DIR" >&2
    exit 1
}

echo "=========================================================================="
echo "Per-task Jacobian sweep"
echo "  task_ids:      $TASK_IDS"
echo "  inputs_root:   $INPUTS_ROOT"
echo "  svd_dir:       $SVD_DIR"
echo "  world_size:    $WORLD_SIZE  per task ($(echo "$TASK_IDS" | wc -w | tr -d ' ') tasks)"
echo "  mode:          $MODE"
echo "  v_device:      $V_DEVICE  v_dtype: $V_DTYPE"
echo "  worker_time:   $WORKER_TIME"
echo "  parallel:      $PARALLEL_TASKS  (1 = submit all task pipelines in parallel)"
echo "=========================================================================="

mkdir -p "$SCRIPT_DIR/logs"

run_one_task() {
    local TID="$1"
    local INPUTS_NPZ="$INPUTS_ROOT/task$(printf '%02d' "$TID")/inputs.npz"
    [[ -f "$INPUTS_NPZ" ]] || {
        echo "[task$TID] ERROR: inputs not found: $INPUTS_NPZ" >&2
        return 1
    }
    # Pull this task's libero prompt out of the manifest. jq is on the cluster.
    local PROMPT
    PROMPT=$(python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))
for t in m['tasks']:
    if int(t['task_id']) == int($TID):
        print(t['prompt']); sys.exit()
sys.exit('no manifest entry for task $TID')
")
    [[ -n "$PROMPT" ]] || {
        echo "[task$TID] ERROR: no prompt for task $TID in manifest" >&2
        return 1
    }
    # Distinguish this task's output dir from the original via a 'task{TID:02d}'
    # prefix segment baked into the RUN_TAG. The inner script computes RUN_TAG
    # from PROMPT + MODE + V_DTYPE; to add a task tag we override RUN_TAG via
    # the inner script's PROMPT_TAG mechanism -- it derives the slug from
    # PROMPT, so just hand it a unique prompt prefix.
    #
    # Cleaner: capture the inner OUT_DIR from the inner script's printout and
    # rename. Simpler: just rely on PROMPT_TAG, which is the only thing that
    # disambiguates. Prepend "task<TID>__" to the prompt for the RUN_TAG
    # without changing what's sent to the model -- we cannot do that, since
    # PROMPT is also used at compute time.
    #
    # Workaround: capture the inner OUT_DIR after the inner script finishes,
    # then rename the directory to add the task prefix. The inner script
    # writes OUT_DIR=$SVD_DIR/A_tilde_full__<prompt_slug>__<mode>__v<dtype>.

    local INNER_LOG="$SCRIPT_DIR/logs/per_task_${TID}_inner.log"
    echo "[task$TID] launching inner pipeline (prompt=${PROMPT:0:60}...) -> $INNER_LOG"
    if WORLD_SIZE="$WORLD_SIZE" PROMPT="$PROMPT" INPUTS_NPZ="$INPUTS_NPZ" \
            SVD_DIR="$SVD_DIR" MODE="$MODE" V_DEVICE="$V_DEVICE" V_DTYPE="$V_DTYPE" \
            WORKER_TIME="$WORKER_TIME" ACCOUNT="$ACCOUNT" PARTITION_SLURM="$PARTITION_SLURM" \
            POLL_INTERVAL="$POLL_INTERVAL" \
            "$INNER_SH" >"$INNER_LOG" 2>&1; then
        local OUT_DIR
        OUT_DIR=$(grep -oE 'out_dir:[[:space:]]+/[^[:space:]]+' "$INNER_LOG" | head -1 | awk '{print $2}')
        if [[ -n "$OUT_DIR" && -d "$OUT_DIR" ]]; then
            # Rename to inject task tag and avoid collisions when two tasks
            # happen to share the same first-50-char prompt slug.
            local PARENT_DIR PARENT_BASENAME RENAMED
            PARENT_DIR="$(dirname -- "$OUT_DIR")"
            PARENT_BASENAME="$(basename -- "$OUT_DIR")"
            local TASK_TAG
            TASK_TAG="$(printf 'task%02d__' "$TID")"
            # Insert TASK_TAG right after the "A_tilde_full__" prefix.
            local NEW_NAME="${PARENT_BASENAME/A_tilde_full__/A_tilde_full__${TASK_TAG}}"
            RENAMED="$PARENT_DIR/$NEW_NAME"
            if [[ "$OUT_DIR" != "$RENAMED" ]]; then
                if [[ ! -d "$RENAMED" ]]; then
                    mv -- "$OUT_DIR" "$RENAMED"
                    echo "[task$TID] renamed -> $RENAMED"
                else
                    echo "[task$TID] WARN: destination $RENAMED already exists; left as $OUT_DIR"
                fi
            fi
        else
            echo "[task$TID] WARN: could not parse out_dir from inner log"
        fi
        echo "[task$TID] OK"
    else
        echo "[task$TID] FAILED  (see $INNER_LOG)" >&2
        return 1
    fi
}

if [[ "$PARALLEL_TASKS" == "1" || "$PARALLEL_TASKS" == "true" ]]; then
    pids=()
    for TID in $TASK_IDS; do
        run_one_task "$TID" &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=$((failed + 1))
        fi
    done
    echo "=========================================================================="
    echo "All tasks done.  failed=$failed / $(echo "$TASK_IDS" | wc -w | tr -d ' ')"
    [[ "$failed" -eq 0 ]] || exit 1
else
    for TID in $TASK_IDS; do
        run_one_task "$TID" || echo "(continuing despite task$TID failure)" >&2
    done
fi
