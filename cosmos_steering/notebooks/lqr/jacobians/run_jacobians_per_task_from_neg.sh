#!/bin/bash
# Per-task Jacobian launcher — variant of run_jacobians_per_task.sh that uses
# the FIRST ROW PER TASK of the multi-task SVD's recorded negative.npz as the
# observation, instead of the separately-collected first_obs_per_task inputs.
#
# For each task t in <SVD_DIR>/config.json:unique_task_ids:
#   PROMPT     = <task_id -> prompt> from unique_task_ids / unique_prompts
#   INPUTS_NPZ = <SVD_DIR>/config.json:neg_npz
#   OBS_INDEX  = first index i such that
#                negative.npz[neg_src_task_id][i] == t
#
# All tasks reuse the SAME SVD (the multitask one), so only the prompt and
# observation differ between per-task jacobians — V is shared. Outputs are
# named A_tilde_full__task<TID>__<prompt-slug>__<mode>__v<dtype>.
#
# Usage:
#   ./run_jacobians_per_task_from_neg.sh
#   SVD_DIR=/path/to/multitask_svd_dir ./run_jacobians_per_task_from_neg.sh
#   PARALLEL_TASKS=1 ./run_jacobians_per_task_from_neg.sh   # default
#   TASK_IDS="0 7" ./run_jacobians_per_task_from_neg.sh     # subset

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
INNER_SH="$SCRIPT_DIR/run_jacobians_full.sh"
[[ -x "$INNER_SH" ]] || { echo "ERROR: $INNER_SH not executable" >&2; exit 1; }

# =========================================================================
# User-configurable (override via env)
# =========================================================================
SVD_DIR="${SVD_DIR:-/work/hdd/bhde/jhong7/cosmos-policy/directions/svd/libero10_tasks0-1-7_xyz_random_xlarge_2_seed42_paired_multitask_dsall_N-1_k64_p10_ws8_tsall}"
TASK_IDS="${TASK_IDS:-}"          # empty = all unique_task_ids in config.json
NEG_NPZ="${NEG_NPZ:-}"            # empty = use neg_npz from <SVD_DIR>/config.json

WORLD_SIZE="${WORLD_SIZE:-8}"
MODE="${MODE:-vjp_no_retain}"
V_DEVICE="${V_DEVICE:-cpu}"
V_DTYPE="${V_DTYPE:-bf16}"
WORKER_TIME="${WORKER_TIME:-00:20:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
PARALLEL_TASKS="${PARALLEL_TASKS:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
# =========================================================================

[[ -d "$SVD_DIR" ]] || { echo "ERROR: SVD_DIR not found: $SVD_DIR" >&2; exit 1; }
[[ -f "$SVD_DIR/config.json" ]] || { echo "ERROR: missing $SVD_DIR/config.json" >&2; exit 1; }

# Resolve task_id -> (prompt, first negative row index) from config.json + NPZ.
# Emits 'tid<TAB>prompt<TAB>obs_idx' lines for the chosen tasks.
RESOLVED=$(python3 - "$SVD_DIR" "$NEG_NPZ" "$TASK_IDS" <<'PYEOF'
import sys, json
import numpy as np

svd_dir, neg_override, task_ids_str = sys.argv[1:4]
cfg = json.loads(open(f"{svd_dir}/config.json").read())
tids = [int(t) for t in cfg["unique_task_ids"]]
prompts = list(cfg["unique_prompts"])
if len(tids) != len(prompts):
    sys.exit("len(unique_task_ids) != len(unique_prompts) in config.json")
tid_to_prompt = dict(zip(tids, prompts))

neg_path = neg_override.strip() or cfg["neg_npz"]
neg = np.load(neg_path, allow_pickle=True)
if "neg_src_task_id" not in neg.files:
    sys.exit(f"{neg_path} has no `neg_src_task_id` column")
nt = neg["neg_src_task_id"].astype(int)

# Optional user-selected subset
sel = [int(x) for x in task_ids_str.split()] if task_ids_str.strip() else tids
for t in sel:
    if t not in tid_to_prompt:
        sys.exit(f"task_id {t} not in unique_task_ids {tids}")
    idxs = np.nonzero(nt == t)[0]
    if idxs.size == 0:
        sys.exit(f"no rows for task_id {t} in {neg_path}")
    obs_idx = int(idxs[0])
    print(f"{t}\t{tid_to_prompt[t]}\t{obs_idx}\t{neg_path}")
PYEOF
)

if [[ -z "$RESOLVED" ]]; then
    echo "ERROR: failed to resolve tasks/prompts/obs indices" >&2
    exit 1
fi

echo "=========================================================================="
echo "Per-task Jacobian sweep (neg-npz first-row obs)"
echo "  svd_dir:       $SVD_DIR"
echo "  world_size:    $WORLD_SIZE  per task"
echo "  mode:          $MODE  v_device=$V_DEVICE  v_dtype=$V_DTYPE"
echo "  worker_time:   $WORKER_TIME"
echo "  parallel:      $PARALLEL_TASKS"
echo "  tasks (tid / obs_idx / prompt):"
while IFS=$'\t' read -r TID PROMPT OBSIDX NEGPATH; do
    printf '    task%02d  obs_idx=%-4d  %s\n' "$TID" "$OBSIDX" "$PROMPT"
done <<< "$RESOLVED"
echo "  neg_npz:       $(echo "$RESOLVED" | head -1 | cut -f4)"
echo "=========================================================================="

mkdir -p "$SCRIPT_DIR/logs"

run_one_task() {
    local TID="$1" PROMPT="$2" OBSIDX="$3" NEGPATH="$4"
    local INNER_LOG
    INNER_LOG="$SCRIPT_DIR/logs/per_task_neg_${TID}_inner.log"
    echo "[task$TID] inner pipeline (obs=$OBSIDX, prompt=${PROMPT:0:60}...) -> $INNER_LOG"
    if WORLD_SIZE="$WORLD_SIZE" PROMPT="$PROMPT" INPUTS_NPZ="$NEGPATH" \
            OBS_INDEX="$OBSIDX" SVD_DIR="$SVD_DIR" MODE="$MODE" \
            V_DEVICE="$V_DEVICE" V_DTYPE="$V_DTYPE" \
            WORKER_TIME="$WORKER_TIME" ACCOUNT="$ACCOUNT" PARTITION_SLURM="$PARTITION_SLURM" \
            POLL_INTERVAL="$POLL_INTERVAL" \
            "$INNER_SH" >"$INNER_LOG" 2>&1; then
        local OUT_DIR
        OUT_DIR=$(grep -oE 'out_dir:[[:space:]]+/[^[:space:]]+' "$INNER_LOG" | head -1 | awk '{print $2}')
        if [[ -n "$OUT_DIR" && -d "$OUT_DIR" ]]; then
            local PARENT_DIR PARENT_BASENAME RENAMED TASK_TAG NEW_NAME
            PARENT_DIR="$(dirname -- "$OUT_DIR")"
            PARENT_BASENAME="$(basename -- "$OUT_DIR")"
            TASK_TAG="$(printf 'task%02d__' "$TID")"
            NEW_NAME="${PARENT_BASENAME/A_tilde_full__/A_tilde_full__${TASK_TAG}}"
            RENAMED="$PARENT_DIR/$NEW_NAME"
            if [[ "$OUT_DIR" != "$RENAMED" ]]; then
                if [[ ! -d "$RENAMED" ]]; then
                    mv -- "$OUT_DIR" "$RENAMED"
                    echo "[task$TID] renamed -> $RENAMED"
                else
                    echo "[task$TID] WARN: $RENAMED exists; left as $OUT_DIR"
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
    while IFS=$'\t' read -r TID PROMPT OBSIDX NEGPATH; do
        run_one_task "$TID" "$PROMPT" "$OBSIDX" "$NEGPATH" &
        pids+=("$!")
    done <<< "$RESOLVED"
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=$((failed + 1)); fi
    done
    echo "=========================================================================="
    echo "All tasks done.  failed=$failed / $(wc -l <<< "$RESOLVED")"
    [[ "$failed" -eq 0 ]] || exit 1
else
    while IFS=$'\t' read -r TID PROMPT OBSIDX NEGPATH; do
        run_one_task "$TID" "$PROMPT" "$OBSIDX" "$NEGPATH" \
            || echo "(continuing despite task$TID failure)" >&2
    done <<< "$RESOLVED"
fi
