#!/bin/bash
# End-to-end LQR-pipeline driver for the noise_extreme (sigma=90) experiment
# on libero_10 tasks 0 and 1.
#
# Pipeline (each step blocks on its sbatch jobs before moving to the next,
# except step 4 which only submits and returns):
#
#   1. inputs/collect_policy_inputs_noise_extreme.sh        (per-task)
#      - task 0: N_EPISODES=10 PARALLEL=1 WORLD_SIZE=5  (used for SVD + jac)
#      - other tasks: N_EPISODES=1 PARALLEL=0  (only row-0 negative.npz needed)
#
#   2. svd/run_partition_svd_pairs_noise_extreme.sh         (task 0 only)
#      - 8-GPU sketch array + dependent finalize
#      - no layer pooling: PARTITIONS=0-0,...,27-27  NUM_LAYERS=28
#      - shared across tasks
#
#   3. jacobians/run_jacobians_noise_extreme.sh             (per-task)
#      - 8-GPU array + login-node merge (script self-blocks)
#      - per-task PROMPT + negative.npz row 0
#      - artifacts share SVD_DIR but get prompt-slugged subdirs
#
#   4. run_sweep_noise_extreme_seed99.sh                    (multi-task)
#      - baseline + 108-combo sweep + final aggregator per task
#      - MEM=128G to accommodate the no-pool SVD's ~30 GB CPU resident
#      - rollouts at /u/.../rollouts/noise_extreme_seed99_task{NN}/
#
# Re-run safe: each step is skipped if its primary artifact exists. Force
# re-run with FORCE_STEP{1,2,3,4}=1.
#
# Usage:
#   ./run_noise_extreme_pipeline.sh                       # full pipeline
#   START_AT=3 ./run_noise_extreme_pipeline.sh            # resume at step 3
#   POLL_INTERVAL=15 ./run_noise_extreme_pipeline.sh
#   TASK_IDS="0 1" ./run_noise_extreme_pipeline.sh        # default
#   TASK_IDS="0"   ./run_noise_extreme_pipeline.sh        # task 0 only
#   FORCE_STEP4=1  ./run_noise_extreme_pipeline.sh        # resubmit sweep
#   SUBMIT_SWEEP=0 ./run_noise_extreme_pipeline.sh        # stop after step 3
#
# Artifacts:
#   inputs : notebooks/lqr/inputs/policy_inputs/libero_10__task{NN}__noise_extreme_pos_neg/
#   svd    : /projects/bhde/jhong7/cosmos-policy/directions/svd/libero10_task00_noise_extreme_pairs_no_action_no_pool_dsall_N-1_k64_p10_ws8_tsall/
#   jac    : $SVD_DIR/A_tilde_full__<prompt_slug>__vjp_no_retain__vbf16/A_tilde__full.pt
#   sweep  : notebooks/lqr/rollouts/noise_extreme_seed99_task{NN}/
#   md     : notebooks/lqr/noise_extreme_seed99_task{NN}_sweep.md

set -euo pipefail

# =========================================================================
# Common config
# =========================================================================
REPO_ROOT=/u/jhong7/Workspace/cosmos-policy
LQR_ROOT=$REPO_ROOT/notebooks/lqr

# Per-task prompts. Add a row here to extend the pipeline to a new task.
# Keys MUST match the TASK_PROMPTS map in run_sweep_noise_extreme_seed99.sh,
# otherwise step 4 will fail with "no entry for task_id=...".
declare -A TASK_PROMPTS=(
    [0]="put both the alphabet soup and the tomato sauce in the basket"
    [1]="put both the cream cheese box and the butter in the basket"
)

# Tasks to process. Default = all configured tasks (sorted ascending).
if [[ -n "${TASK_IDS:-}" ]]; then
    read -r -a TASK_IDS <<< "$TASK_IDS"
else
    TASK_IDS=()
    for k in "${!TASK_PROMPTS[@]}"; do TASK_IDS+=("$k"); done
    IFS=$'\n' TASK_IDS=($(sort -n <<< "${TASK_IDS[*]}")); unset IFS
fi

for tid in "${TASK_IDS[@]}"; do
    [[ -n "${TASK_PROMPTS[$tid]+x}" ]] || {
        echo "ERROR: no prompt defined for task_id=$tid (edit TASK_PROMPTS in this script)" >&2
        exit 1
    }
done

# Task 0 is the SVD source (and the only task whose inputs feed SVD). All
# tasks need a negative.npz so the per-task jacobian has a row-0 observation.
SVD_TASK=0
[[ " ${TASK_IDS[*]} " == *" $SVD_TASK "* ]] || {
    echo "ERROR: SVD task ($SVD_TASK) must be in TASK_IDS (${TASK_IDS[*]})" >&2
    exit 1
}

# How many episodes to collect per task in step 1. Task 0 is the SVD source
# so it gets a real corpus; other tasks only need row 0 of negative.npz.
SVD_TASK_N_EPISODES="${SVD_TASK_N_EPISODES:-10}"
OTHER_TASK_N_EPISODES="${OTHER_TASK_N_EPISODES:-1}"

# Per-rank memory for step 4 (no-pool SVD's 140 V_part files push CPU
# resident to ~30 GB, so 64 GB default OOMs).
export MEM="${MEM:-128G}"

# SVD output location. Must match run_partition_svd_pairs_noise_extreme.sh's
# OUT_BASE + RUN_TAG derivation:
#   RUN_TAG="${TAG}_N${N}_k${K_TARGET}_p${P_OVER}_ws${WORLD_SIZE}_ts${TS_TAG}"
# defaults: N=-1 K_TARGET=64 P_OVER=10 WORLD_SIZE=8 TIMESTEPS=all
SVD_BASE="${SVD_BASE:-/projects/bhde/jhong7/cosmos-policy/directions/svd}"
SVD_TAG="${SVD_TAG:-libero10_task00_noise_extreme_pairs_no_action_no_pool_dsall}"
SVD_DIR="${SVD_DIR:-$SVD_BASE/${SVD_TAG}_N-1_k64_p10_ws8_tsall}"

POLL_INTERVAL=${POLL_INTERVAL:-30}
START_AT=${START_AT:-1}
SUBMIT_SWEEP=${SUBMIT_SWEEP:-1}

# =========================================================================
# Helpers
# =========================================================================
banner() {
    echo
    echo "================================================================================"
    echo "  $*"
    echo "================================================================================"
}

# Replicate the PROMPT_TAG slugging in run_jacobians_full.sh so we can
# predict the jacobian output dir without parsing wrapper output:
#   PROMPT_TAG=$(echo "$PROMPT" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50)
prompt_slug() {
    echo "$1" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50
}

jac_dir_for_task() {
    local tid="$1"
    local slug
    slug=$(prompt_slug "${TASK_PROMPTS[$tid]}")
    echo "$SVD_DIR/A_tilde_full__${slug}__vjp_no_retain__vbf16"
}

input_dir_for_task() {
    local tid
    tid=$(printf "%02d" "$1")
    echo "$LQR_ROOT/inputs/policy_inputs/libero_10__task${tid}__noise_extreme_pos_neg"
}

# wait_for_jobs <job_id> [<job_id> ...]
wait_for_jobs() {
    local ids=()
    for j in "$@"; do [[ -n "$j" && "$j" != "?" ]] && ids+=("$j"); done
    [[ ${#ids[@]} -eq 0 ]] && return 0
    local joined
    joined=$(IFS=,; echo "${ids[*]}")
    echo "[wait] watching slurm jobs: $joined (poll every ${POLL_INTERVAL}s)"
    while true; do
        local active
        active=$(squeue -h -j "$joined" -r -o '%A %T' 2>/dev/null || true)
        [[ -z "$active" ]] && { echo "[wait] all drained: $joined"; return 0; }
        local n_total n_pending n_running
        n_total=$(echo "$active" | wc -l)
        n_pending=$(echo "$active" | awk '$2=="PENDING"' | wc -l)
        n_running=$(echo "$active" | awk '$2=="RUNNING"' | wc -l)
        printf '[wait %s] active=%d pending=%d running=%d\n' \
            "$(date '+%H:%M:%S')" "$n_total" "$n_pending" "$n_running"
        sleep "$POLL_INTERVAL"
    done
}

should_run_step() {
    local n="$1"
    if (( n < START_AT )); then
        echo "[skip] step $n is below START_AT=$START_AT; skipping."
        return 1
    fi
    return 0
}

# =========================================================================
# Banner
# =========================================================================
banner "noise_extreme pipeline: tasks ${TASK_IDS[*]}  (SVD task = $SVD_TASK)"
echo "  SVD_DIR              : $SVD_DIR"
echo "  MEM (step 4 per-rank): $MEM"
echo "  START_AT             : $START_AT"
echo "  SUBMIT_SWEEP         : $SUBMIT_SWEEP"
echo "  per-task plan:"
for tid in "${TASK_IDS[@]}"; do
    n_ep="$OTHER_TASK_N_EPISODES"
    [[ "$tid" == "$SVD_TASK" ]] && n_ep="$SVD_TASK_N_EPISODES"
    echo "    task $tid  n_episodes=$n_ep  prompt=\"${TASK_PROMPTS[$tid]}\""
done

# =========================================================================
# Step 1 — input collection (per-task; task 0 feeds SVD)
# =========================================================================
if should_run_step 1; then
    for tid in "${TASK_IDS[@]}"; do
        in_dir=$(input_dir_for_task "$tid")
        if [[ "$tid" == "$SVD_TASK" ]]; then
            n_ep="$SVD_TASK_N_EPISODES"
            parallel=1
            ws=5
            t_time=01:30:00
        else
            n_ep="$OTHER_TASK_N_EPISODES"
            parallel=0
            ws=1
            t_time=00:30:00
        fi
        banner "Step 1: collect inputs for task $tid  (N_EPISODES=$n_ep PARALLEL=$parallel)"
        if [[ -z "${FORCE_STEP1:-}" && -f "$in_dir/positive.npz" && -f "$in_dir/negative.npz" ]]; then
            echo "[skip] $in_dir/{positive,negative}.npz already exist."
            echo "       set FORCE_STEP1=1 to re-run."
            continue
        fi
        step1_out=$(mktemp)
        TASK_ID="$tid" \
        N_EPISODES="$n_ep" \
        PARALLEL="$parallel" \
        WORLD_SIZE="$ws" \
        PROMPT="${TASK_PROMPTS[$tid]}" \
        OUT_DIR="$in_dir" \
        TIME="$t_time" \
            "$LQR_ROOT/inputs/collect_policy_inputs_noise_extreme.sh" \
            | tee "$step1_out"
        # Parallel path submits an array + merge (afterok); the merge writes
        # the final positive/negative.npz, so wait on whichever id we capture.
        last_id=$(grep -oE 'Submitted batch job [0-9]+' "$step1_out" | tail -1 | awk '{print $NF}')
        rm -f "$step1_out"
        [[ -n "$last_id" ]] || {
            echo "ERROR: task $tid: failed to capture step 1 sbatch id" >&2
            exit 1
        }
        wait_for_jobs "$last_id"
        [[ -f "$in_dir/positive.npz" ]] || {
            echo "ERROR: task $tid: missing $in_dir/positive.npz" >&2; exit 1; }
        [[ -f "$in_dir/negative.npz" ]] || {
            echo "ERROR: task $tid: missing $in_dir/negative.npz" >&2; exit 1; }
    done
fi

# =========================================================================
# Step 2 — SVD on task-0 pairs (shared across tasks)
# =========================================================================
if should_run_step 2; then
    banner "Step 2: SVD on task $SVD_TASK pairs (8-GPU sketch + finalize)"
    if [[ -z "${FORCE_STEP2:-}" && -f "$SVD_DIR/svd_summary.pt" ]]; then
        echo "[skip] $SVD_DIR/svd_summary.pt already exists."
        echo "       set FORCE_STEP2=1 to re-run."
    else
        svd_in=$(input_dir_for_task "$SVD_TASK")
        [[ -f "$svd_in/positive.npz" && -f "$svd_in/negative.npz" ]] || {
            echo "ERROR: step 2 needs $svd_in/{positive,negative}.npz from step 1" >&2
            exit 1
        }
        step2_out=$(mktemp)
        PAIR_DIR="$svd_in" \
        POS_NPZ="$svd_in/positive.npz" \
        NEG_NPZ="$svd_in/negative.npz" \
        PROMPT="${TASK_PROMPTS[$SVD_TASK]}" \
        OUT_BASE="$SVD_BASE" \
        TAG="$SVD_TAG" \
            "$LQR_ROOT/svd/run_partition_svd_pairs_noise_extreme.sh" \
            | tee "$step2_out"
        sketch_id=$(grep -oE 'sketch job id: *[0-9]+' "$step2_out" | tail -1 | awk '{print $NF}')
        svd_id=$(grep -oE 'svd job id: *[0-9]+'       "$step2_out" | tail -1 | awk '{print $NF}')
        rm -f "$step2_out"
        [[ -n "$svd_id" ]] || {
            echo "ERROR: failed to capture SVD finalize job id" >&2; exit 1; }
        wait_for_jobs "$sketch_id" "$svd_id"
        [[ -f "$SVD_DIR/svd_summary.pt" ]] || {
            echo "ERROR: SVD did not produce $SVD_DIR/svd_summary.pt" >&2; exit 1; }
    fi
fi

# =========================================================================
# Step 3 — per-task jacobians (against shared SVD basis)
# =========================================================================
if should_run_step 3; then
    for tid in "${TASK_IDS[@]}"; do
        in_dir=$(input_dir_for_task "$tid")
        jac_dir=$(jac_dir_for_task "$tid")
        banner "Step 3: jacobians for task $tid  (PROMPT=${TASK_PROMPTS[$tid]})"
        echo "  pair_dir : $in_dir"
        echo "  jac_dir  : $jac_dir"
        if [[ -z "${FORCE_STEP3:-}" && -f "$jac_dir/A_tilde__full.pt" ]]; then
            echo "[skip] $jac_dir/A_tilde__full.pt already exists."
            echo "       set FORCE_STEP3=1 to re-run."
            continue
        fi
        [[ -f "$in_dir/negative.npz" ]] || {
            echo "ERROR: task $tid: missing $in_dir/negative.npz (need step 1 first)" >&2
            exit 1
        }
        [[ -f "$SVD_DIR/svd_summary.pt" ]] || {
            echo "ERROR: missing $SVD_DIR/svd_summary.pt (need step 2 first)" >&2
            exit 1
        }
        # run_jacobians_noise_extreme.sh self-blocks (polls its workers + runs
        # the login-node merge).
        PAIR_DIR="$in_dir" \
        PROMPT="${TASK_PROMPTS[$tid]}" \
        SVD_DIR="$SVD_DIR" \
            "$LQR_ROOT/jacobians/run_jacobians_noise_extreme.sh"
        [[ -f "$jac_dir/A_tilde__full.pt" ]] || {
            echo "ERROR: task $tid: jacobian not produced at $jac_dir/A_tilde__full.pt" >&2
            exit 1
        }
    done
fi

# =========================================================================
# Step 4 — multi-task LQR sweep (submits and returns; jobs run async)
# =========================================================================
if should_run_step 4; then
    if [[ "$SUBMIT_SWEEP" != "1" ]]; then
        banner "Step 4: SKIPPED (SUBMIT_SWEEP=$SUBMIT_SWEEP)"
    else
        banner "Step 4: LQR sweep (multi-task, 108 combos + baseline per task)"
        # Verify the jacobian per-task artifact is in place; the sweep
        # wrapper does its own sanity check but we double-check here so
        # the failure mode is obvious from the e2e log.
        for tid in "${TASK_IDS[@]}"; do
            jp=$(jac_dir_for_task "$tid")/A_tilde__full.pt
            [[ -f "$jp" ]] || {
                echo "ERROR: task $tid: missing jacobian $jp" >&2; exit 1; }
        done

        # Pass TASK_IDS through so the sweep wrapper only processes the
        # tasks we configured here (its default is also "0 1" but be explicit).
        TASK_IDS="${TASK_IDS[*]}" \
        SVD_BASE="$SVD_BASE" SVD_TAG="${SVD_TAG}_N-1_k64_p10_ws8_tsall" \
        SVD_DIR="$SVD_DIR" \
            "$LQR_ROOT/run_sweep_noise_extreme_seed99.sh"
    fi
fi

banner "PIPELINE COMPLETE"
cat <<EOF
Monitor sweep with:
    squeue -u \$USER

Rollouts will appear under:
    $LQR_ROOT/rollouts/noise_extreme_seed99_task{NN}/

Per-task summary mds will be refreshed after the final aggregator runs:
$(for tid in "${TASK_IDS[@]}"; do
    printf '    %s\n' "$LQR_ROOT/noise_extreme_seed99_task$(printf %02d "$tid")_sweep.md"
done)

Artifacts:
    inputs : $LQR_ROOT/inputs/policy_inputs/libero_10__task{NN}__noise_extreme_pos_neg/
    svd    : $SVD_DIR
    jac    : \$SVD_DIR/A_tilde_full__<prompt_slug>__vjp_no_retain__vbf16/A_tilde__full.pt
EOF
