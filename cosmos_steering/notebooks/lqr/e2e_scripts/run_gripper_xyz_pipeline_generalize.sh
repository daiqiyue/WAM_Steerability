#!/bin/bash
# Generalisation pipeline: use contrastive directions computed on libero_10
# task 01 ("put both the cream cheese box and the butter in the basket") to
# run the LQR sweep on task 02 ("turn on the stove and put the moka pot on it")
# — without recomputing SVD or jacobians for task 02.
#
# Pipeline:
#   1. inputs/collect_policy_inputs_gripper_xyz_perturbation.sh   (1 GPU)       [task 01]
#   2. svd/pair_inputs_by_similarity.sh                           (login CPU)   [task 01]
#   3. svd/run_partition_svd_pairs_no_action.sh                   (8-GPU sketch + finalize) [task 01]
#   4. jacobians/run_jacobians_full.sh                            (8-GPU array + login merge) [task 01]
#   5. sweep_lqr_gripper_xyz.sh + 5 extra-q points  [task 02 rollouts, task 01 artifacts]
#
# Steps 1-4 build (or reuse) the task-01 SVD/jacobian artifacts.
# Step 5 overrides TASK_ID and PROMPT to target task 02 while keeping
# SVD_DIR and JAC_DIR_ACT pointing at the task-01 artifacts, testing
# whether the learned directions generalise across tasks.
#
# Each step blocks until its sbatch jobs reach a terminal state before
# moving to the next. Step 5 only *submits* the sweep — it returns
# immediately and the LQR rollouts run asynchronously in slurm.
#
# All step launchers are reused unmodified; this script just sets env
# vars + waits on the right sbatch IDs in between. Re-run safe: each
# step is skipped if its primary output artifact already exists.
#
# Usage:
#   ./run_gripper_xyz_pipeline_generalize.sh                # full pipeline
#   START_AT=5 ./run_gripper_xyz_pipeline_generalize.sh     # skip to sweep only
#   POLL_INTERVAL=15 ./run_gripper_xyz_pipeline_generalize.sh
#   FORCE_STEP1=1 ./run_gripper_xyz_pipeline_generalize.sh  # re-run step 1 even if outputs exist
#
# SVD artifacts (steps 3 + 4) land under
#   /projects/bhhv/jskifstad/alt_wam/repo_root/directions/svd  (task-01 tag)
# Input + paired npzs (steps 1 + 2) stay under notebooks/lqr/inputs/policy_inputs/.
# LQR rollouts (step 5) stay under notebooks/lqr/rollouts/task01_to_task02_seed99/

set -euo pipefail

# =========================================================================
# Common config (exported so the inner launchers + sweep see them).
# =========================================================================
REPO_ROOT=/projects/bhhv/jskifstad/alt_wam/repo_root
LQR_ROOT=$REPO_ROOT/notebooks/lqr

export TASK_ID=1
export SUITE=libero_10
export PROMPT="put both the cream cheese box and the butter in the basket"
export PRESET=xyz_random_xlarge_3
export BASE_SEED=42
export N_EPISODES=30

INPUT_DIR=$LQR_ROOT/inputs/policy_inputs/libero_10__task01__${PRESET}__seed${BASE_SEED}__pos_neg
PAIRED_DIR=${INPUT_DIR}__paired
SVD_BASE=/projects/bhhv/jskifstad/alt_wam/repo_root/directions/svd
SVD_TAG=libero10_task01_gripper_xyz_${PRESET}_seed${BASE_SEED}_paired
# Matches the RUN_TAG derivation in run_partition_svd_pairs_no_action.sh:
#   RUN_TAG="${TAG}_N${N}_k${K_TARGET}_p${P_OVER}_ws${WORLD_SIZE}_ts${TS_TAG}"
# with defaults N=-1, K_TARGET=64, P_OVER=10, WORLD_SIZE=8, TIMESTEPS=all.
SVD_DIR=${SVD_BASE}/${SVD_TAG}_N-1_k64_p10_ws8_tsall
# Matches PROMPT_TAG in run_jacobians_full.sh:
#   PROMPT_TAG=$(echo "$PROMPT" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50)
#   RUN_TAG="A_tilde_full__${PROMPT_TAG}__${MODE}__v${V_DTYPE}"
# Computed the same way run_jacobians_full.sh does (echo appends a newline,
# which tr -c converts to '_', hence the trailing underscore before "vjp").
_PROMPT_TAG=$(echo "$PROMPT" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50)
JAC_DIR_ACT="A_tilde_full__${_PROMPT_TAG}__vjp_no_retain__vbf16"

# Inherited by sweep_lqr_gripper_xyz.sh -> run_lqr_cosmos_policy_gripper_xyz.sh
# (sweep only explicitly forwards LAMBDA/Q/RI/T/TAG/WORLD_SIZE/N_EPISODES/
# PRESET/BASE_SEED; everything else passes through the env.)
export SVD_DIR
export JAC_DIR_ACT

# =========================================================================
# Task 2 config — sweep target (step 5 only; SVD/jacobian artifacts stay
# fixed to task 01 above).
# =========================================================================
TASK2_ID=9
# TASK2_PROMPT="turn on the stove and put the moka pot on it" # 02
TASK2_PROMPT="put the yellow and white mug in the microwave and close it" # 09
# TASK2_PROMPT="put both the cream cheese box and the butter in the basket" # 01
# TASK2_PROMPT="put both the alphabet soup and the cream cheese box in the basket" # 07
# TASK2_PROMPT="put the black bowl in the bottom drawer of the cabinet and close it" # 03

# Sweep knobs — step 5 rolls out task 2 using task 01's artifacts.
# SWEEP_TAG is derived from TASK2_ID so changing the two vars above is enough
# to retarget the script at a different task.
SWEEP_TAG=task01_to_task$(printf '%02d' "$TASK2_ID")_seed99
SWEEP_BASE_SEED=99
SWEEP_WORLD_SIZE=4

POLL_INTERVAL=${POLL_INTERVAL:-30}
START_AT=${START_AT:-1}

# =========================================================================
# Helpers
# =========================================================================
banner() {
    echo
    echo "================================================================================"
    echo "  $*"
    echo "================================================================================"
}

# wait_for_jobs <job_id> [<job_id> ...]
# Polls squeue until every listed job leaves the queue. Safe with empty / "?" IDs.
wait_for_jobs() {
    local ids=()
    for j in "$@"; do
        [[ -n "$j" && "$j" != "?" ]] && ids+=("$j")
    done
    [[ ${#ids[@]} -eq 0 ]] && return 0
    local joined
    joined=$(IFS=,; echo "${ids[*]}")
    echo "[wait] watching slurm jobs: $joined (poll every ${POLL_INTERVAL}s)"
    while true; do
        local active
        active=$(squeue -h -j "$joined" -r -o '%A %T' 2>/dev/null || true)
        if [[ -z "$active" ]]; then
            echo "[wait] all drained: $joined"
            return 0
        fi
        local n_total n_pending n_running
        n_total=$(echo "$active" | wc -l)
        n_pending=$(echo "$active" | awk '$2=="PENDING"' | wc -l)
        n_running=$(echo "$active" | awk '$2=="RUNNING"' | wc -l)
        printf '[wait %s] active=%d pending=%d running=%d\n' \
            "$(date '+%H:%M:%S')" "$n_total" "$n_pending" "$n_running"
        sleep "$POLL_INTERVAL"
    done
}

# Run a step if step number >= START_AT, else announce skip.
should_run_step() {
    local n="$1"
    if (( n < START_AT )); then
        echo "[skip] step $n is below START_AT=$START_AT; skipping."
        return 1
    fi
    return 0
}

# =========================================================================
# Step 1 — collect inputs (1 GPU)
# =========================================================================
if should_run_step 1; then
    banner "Step 1/5: collect inputs (libero_10 task 01, $N_EPISODES eps, 1 GPU)"
    if [[ -z "${FORCE_STEP1:-}" && -f "$INPUT_DIR/positive.npz" && -f "$INPUT_DIR/negative.npz" ]]; then
        echo "[skip] $INPUT_DIR/{positive,negative}.npz already exist; skipping step 1."
        echo "       set FORCE_STEP1=1 to re-run."
    else
        step1_out=$(mktemp)
        TASK_ID=$TASK_ID SUITE=$SUITE \
        N_EPISODES=$N_EPISODES PRESET=$PRESET BASE_SEED=$BASE_SEED \
        OUT_DIR=$INPUT_DIR \
        TIME=00:30:00 \
            "$LQR_ROOT/inputs/collect_policy_inputs_gripper_xyz_perturbation.sh" \
            | tee "$step1_out"
        step1_id=$(grep -oE 'Submitted batch job [0-9]+' "$step1_out" | tail -1 | awk '{print $NF}')
        rm -f "$step1_out"
        [[ -n "$step1_id" ]] || { echo "ERROR: failed to capture step 1 sbatch id" >&2; exit 1; }
        wait_for_jobs "$step1_id"
        [[ -f "$INPUT_DIR/positive.npz" ]] || { echo "ERROR: missing $INPUT_DIR/positive.npz" >&2; exit 1; }
        [[ -f "$INPUT_DIR/negative.npz" ]] || { echo "ERROR: missing $INPUT_DIR/negative.npz" >&2; exit 1; }
    fi
fi

# =========================================================================
# Step 2 — pair inputs by similarity (CPU, login node, synchronous)
# =========================================================================
if should_run_step 2; then
    banner "Step 2/5: pair inputs by similarity (local CPU)"
    if [[ -z "${FORCE_STEP2:-}" && -f "$PAIRED_DIR/positive.npz" && -f "$PAIRED_DIR/negative.npz" ]]; then
        echo "[skip] $PAIRED_DIR/{positive,negative}.npz already exist; skipping step 2."
        echo "       set FORCE_STEP2=1 to re-run."
    else
        IN_DIR=$INPUT_DIR OUT_DIR=$PAIRED_DIR \
            "$LQR_ROOT/svd/pair_inputs_by_similarity.sh"
        [[ -f "$PAIRED_DIR/positive.npz" ]] || { echo "ERROR: missing $PAIRED_DIR/positive.npz" >&2; exit 1; }
        [[ -f "$PAIRED_DIR/negative.npz" ]] || { echo "ERROR: missing $PAIRED_DIR/negative.npz" >&2; exit 1; }
    fi
fi

# =========================================================================
# Step 3 — SVD on pairs (8-GPU sketch array + dependent finalize)
# =========================================================================
if should_run_step 3; then
    banner "Step 3/5: SVD on pairs (8-GPU sketch + finalize) -> $SVD_DIR"
    if [[ -z "${FORCE_STEP3:-}" && -f "$SVD_DIR/svd_summary.pt" ]]; then
        echo "[skip] $SVD_DIR/svd_summary.pt already exists; skipping step 3."
        echo "       set FORCE_STEP3=1 to re-run."
    else
        step3_out=$(mktemp)
        WORLD_SIZE=8 \
        POS_NPZ=$PAIRED_DIR/positive.npz \
        NEG_NPZ=$PAIRED_DIR/negative.npz \
        DRIVE_SOURCE=all \
        TAG=$SVD_TAG \
        PROMPT="$PROMPT" \
        OUT_BASE=$SVD_BASE \
            "$LQR_ROOT/svd/run_partition_svd_pairs_no_action.sh" \
            | tee "$step3_out"
        sketch_id=$(grep -oE 'sketch job id: [0-9]+' "$step3_out" | tail -1 | awk '{print $NF}')
        svd_id=$(grep -oE 'svd job id: *[0-9]+'    "$step3_out" | tail -1 | awk '{print $NF}')
        rm -f "$step3_out"
        [[ -n "$svd_id" ]] || { echo "ERROR: failed to capture SVD finalize job id" >&2; exit 1; }
        wait_for_jobs "$sketch_id" "$svd_id"
        [[ -f "$SVD_DIR/svd_summary.pt" ]] || {
            echo "ERROR: SVD did not produce $SVD_DIR/svd_summary.pt" >&2
            exit 1
        }
    fi
fi

# =========================================================================
# Step 4 — full-rank jacobian (8-GPU array, run_jacobians_full.sh self-blocks)
# =========================================================================
if should_run_step 4; then
    banner "Step 4/5: jacobians (8-GPU array; blocks until login-merge done)"
    if [[ -z "${FORCE_STEP4:-}" && -f "$SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" ]]; then
        echo "[skip] $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt already exists; skipping step 4."
        echo "       set FORCE_STEP4=1 to re-run."
    else
        WORLD_SIZE=8 \
        SVD_DIR=$SVD_DIR \
        PROMPT="$PROMPT" \
        INPUTS_NPZ=$PAIRED_DIR/negative.npz \
        OBS_INDEX=0 \
            "$LQR_ROOT/jacobians/run_jacobians_full.sh"
        [[ -f "$SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" ]] || {
            echo "ERROR: jacobian not produced at $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" >&2
            exit 1
        }
    fi
fi

# =========================================================================
# Step 5 — LQR sweep on task 02 using task 01 artifacts
# (submits and returns; jobs run async in slurm)
# =========================================================================
if should_run_step 5; then
    banner "Step 5/5: LQR sweep on task 02 with task 01 artifacts (72 base combos + 5 extra-q points)"

    # Override rollout target to task 02; SVD_DIR and JAC_DIR_ACT stay fixed
    # to the task-01 artifacts exported above.
    export TASK_ID=$TASK2_ID
    export PROMPT="$TASK2_PROMPT"

    echo
    echo "--- dry-run preview (DRY=1, BASE_SEED=$SWEEP_BASE_SEED) ---"
    DRY=1 FORCE=1 TAG=$SWEEP_TAG WORLD_SIZE=$SWEEP_WORLD_SIZE N_EPISODES=$N_EPISODES \
    PRESET=$PRESET BASE_SEED=$SWEEP_BASE_SEED \
        "$LQR_ROOT/sweep_lqr_gripper_xyz.sh"

    echo
    echo "--- submitting sweep (DRY=0, BASE_SEED=$SWEEP_BASE_SEED) ---"
    DRY=0 FORCE=1 TAG=$SWEEP_TAG WORLD_SIZE=$SWEEP_WORLD_SIZE N_EPISODES=$N_EPISODES \
    PRESET=$PRESET BASE_SEED=$SWEEP_BASE_SEED \
        "$LQR_ROOT/sweep_lqr_gripper_xyz.sh"

    # Extra q points the gripper_xyz_sweep_summary.md reports outside the
    # base grid, all at (lam=15, rinit=5, tau=3).
    # echo
    # echo "--- submitting extra-q points (lam=15, rinit=5, tau=3, BASE_SEED=$SWEEP_BASE_SEED) ---"
    # for L in 0.5 5; do
    #     for Q in 1 10; do
    #         echo "--- Q_SCALE=$Q ---"
    #         TAG=$SWEEP_TAG WORLD_SIZE=$SWEEP_WORLD_SIZE N_EPISODES=$N_EPISODES \
    #         PRESET=$PRESET BASE_SEED=$SWEEP_BASE_SEED \
    #         LAMBDA=$L Q_SCALE=$Q R_SCALE=5.0 R_SCALE_TAU=3.0 \
    #             "$LQR_ROOT/run_lqr_cosmos_policy_gripper_xyz.sh"
    #     done
    # done

    # Outlier from gripper_xyz_sweep_summary.md: lam=20 is outside the
    # base sweep grid (which tops out at lam=15) but the summary has a
    # single (20, 1, 10, 5) entry at seed=42. Include it for coverage.
    # echo
    # echo "--- submitting lambda=20 outlier (lam=20, q=1, rinit=10, tau=5, BASE_SEED=$SWEEP_BASE_SEED) ---"
    # TAG=$SWEEP_TAG WORLD_SIZE=$SWEEP_WORLD_SIZE N_EPISODES=$N_EPISODES \
    # PRESET=$PRESET BASE_SEED=$SWEEP_BASE_SEED \
    # LAMBDA=20.0 Q_SCALE=1.0 R_SCALE=10.0 R_SCALE_TAU=5.0 \
    #     "$LQR_ROOT/run_lqr_cosmos_policy_gripper_xyz.sh"
fi

banner "PIPELINE COMPLETE — sweep is running asynchronously in slurm."
cat <<EOF
Monitor sweep with:
    squeue -u \$USER

Rollouts will appear under:
    $LQR_ROOT/rollouts/$SWEEP_TAG/

Once all rollouts finish, summarize successes with:
    python $LQR_ROOT/summarize_lqr_gripper_xyz_sweep.py

Artifacts (task 01 — source of SVD/jacobian):
    inputs  : $INPUT_DIR
    paired  : $PAIRED_DIR
    svd     : $SVD_DIR
    jac     : $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt

Sweep target: task 02 ("$TASK2_PROMPT")
EOF
