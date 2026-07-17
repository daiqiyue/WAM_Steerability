#!/bin/bash
# Gripper-xyz generalisation pipeline for DiT4DiT: use contrastive directions
# computed on libero_10 task index 0 ("put both the alphabet soup and the tomato
# sauce in the basket") to run the LQR sweep on task index 3 ("put the black bowl
# in the bottom drawer of the cabinet and close it") — without recomputing SVD or
# jacobians for task 3.
#
# NAMING NOTE: the directory/tag strings below use "task01" / "task01_to_task03"
# as 1-INDEXED labels for the 0-indexed LIBERO tasks actually run here, i.e.
# "task01" == LIBERO index 0 (alphabet soup) and "task03" == LIBERO index 3
# (black bowl). The source is index 0, set by TASK_ID/PROMPT below — NOT LIBERO
# index 1 (the cream-cheese task). The dir names keep "task01" purely for path
# continuity with artifacts already on disk; the real source is alphabet soup.
#
# Analogue of the repo root's notebooks/lqr/e2e_scripts/run_gripper_xyz_pipeline_generalize.sh
# adapted for DiT4DiT's action-generation model.
#
# Pipeline (source = LIBERO index 0; target = LIBERO index 3):
#   1. inputs/collect_policy_inputs_gripper_xyz_perturbation.sh  (1 GPU)       [src idx 0]
#   2. svd/pair_inputs_by_similarity.sh                          (login CPU)   [src idx 0]
#   3. svd/run_partition_svd_pairs_no_action.sh                  (8-GPU sketch + finalize) [src idx 0]
#   4. jacobians/run_jacobians_full.sh                           (8-GPU array + login merge) [src idx 0]
#   5. sweep_lqr_gripper_xyz.sh + combos  [idx-3 rollouts, idx-0 artifacts]
#
# Steps 1-4 build (or reuse) the index-0 SVD/jacobian artifacts.
# Step 5 overrides TASK_ID and PROMPT to target index 3 while keeping
# SVD_DIR and JAC_DIR_ACT pointing at the index-0 artifacts.
#
# Re-run safe: each step is skipped if its primary output artifact already exists.
#
# Usage:
#   ./run_gripper_xyz_pipeline_generalize.sh
#   START_AT=5 ./run_gripper_xyz_pipeline_generalize.sh
#   FORCE_STEP1=1 ./run_gripper_xyz_pipeline_generalize.sh
#   POLL_INTERVAL=15 ./run_gripper_xyz_pipeline_generalize.sh

set -euo pipefail

# =========================================================================
# Common config (exported so inner launchers see them)
# =========================================================================
DIT4DIT_ROOT=/work/hdd/bhde/jhong7/DiT4DiT
LQR_ROOT=$DIT4DIT_ROOT/notebooks/lqr

export TASK_ID=0
export SUITE=libero_10
export PROMPT="put both the alphabet soup and the tomato sauce in the basket"
export PRESET=xyz_random_xlarge_3
export BASE_SEED=42
export N_EPISODES=50

# NOTE: "task01" in the path strings below is a 1-indexed label for LIBERO
# index 0 (alphabet soup), the actual source set by TASK_ID/PROMPT above.
INPUT_DIR=$LQR_ROOT/inputs/policy_inputs/libero_10__task01__${PRESET}__seed${BASE_SEED}__pos_neg
PAIRED_DIR=${INPUT_DIR}__paired
SVD_BASE=$LQR_ROOT/directions/svd
SVD_TAG=libero10_task01_gripper_xyz_${PRESET}_seed${BASE_SEED}_paired

# Matches the RUN_TAG derivation in run_partition_svd_pairs_no_action.sh:
#   RUN_TAG="${TAG}_N${N}_k${K_TARGET}_p${P_OVER}_ws${WORLD_SIZE}_ts${TS_TAG}"
# with defaults N=-1, K_TARGET=64, P_OVER=10, WORLD_SIZE=8, TIMESTEPS=all.
SVD_DIR=${SVD_BASE}/${SVD_TAG}_N-1_k64_p10_ws8_tsall

# Matches PROMPT_TAG in run_jacobians_full.sh:
#   PROMPT_TAG=$(echo "$PROMPT" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50)
#   RUN_TAG="A_tilde_full__${PROMPT_TAG}__${MODE}__v${V_DTYPE}"
_PROMPT_TAG=$(echo "$PROMPT" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50)
JAC_DIR_ACT="A_tilde_full__${_PROMPT_TAG}__vjp_no_retain__vbf16"
VL_EMBS_PATH="${PAIRED_DIR}/vl_embs__${_PROMPT_TAG:0:30}.pt"

export SVD_DIR
export JAC_DIR_ACT
export VL_EMBS_PATH

# =========================================================================
# Task 2 config — sweep target (step 5 only)
# =========================================================================
TASK2_ID=3
TASK2_PROMPT="put the black bowl in the bottom drawer of the cabinet and close it"

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
# Step 2 — pair inputs by similarity (CPU, login node)
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
# Step 3 — precompute VLM embeddings (1 GPU)
# =========================================================================
if should_run_step 3; then
    banner "Step 3/6: precompute vl_embs (1 GPU, loads full model once)"
    if [[ -z "${FORCE_STEP3:-}" && -f "$VL_EMBS_PATH" ]]; then
        echo "[skip] $VL_EMBS_PATH already exists; skipping step 3."
        echo "       set FORCE_STEP3=1 to re-run."
    else
        step3_out=$(mktemp)
        POS_NPZ=$PAIRED_DIR/positive.npz \
        NEG_NPZ=$PAIRED_DIR/negative.npz \
        PROMPT="$PROMPT" \
        OUT_PATH="$VL_EMBS_PATH" \
        TIME=02:00:00 \
            "$LQR_ROOT/inputs/precompute_vl_embs.sh" \
            | tee "$step3_out"
        step3_id=$(grep -oE 'Submitted batch job [0-9]+' "$step3_out" | tail -1 | awk '{print $NF}')
        rm -f "$step3_out"
        [[ -n "$step3_id" ]] || { echo "ERROR: failed to capture step 3 job id" >&2; exit 1; }
        wait_for_jobs "$step3_id"
        [[ -f "$VL_EMBS_PATH" ]] || { echo "ERROR: missing $VL_EMBS_PATH" >&2; exit 1; }
    fi
fi

# =========================================================================
# Step 4 — SVD on pairs (8-GPU sketch array + dependent finalize)
# =========================================================================
if should_run_step 4; then
    banner "Step 4/6: SVD on pairs -> $SVD_DIR"
    if [[ -z "${FORCE_STEP4:-}" && -f "$SVD_DIR/svd_summary.pt" ]]; then
        echo "[skip] $SVD_DIR/svd_summary.pt already exists; skipping step 4."
        echo "       set FORCE_STEP4=1 to re-run."
    else
        step4_out=$(mktemp)
        WORLD_SIZE=8 \
        POS_NPZ=$PAIRED_DIR/positive.npz \
        NEG_NPZ=$PAIRED_DIR/negative.npz \
        VL_EMBS_PATH="$VL_EMBS_PATH" \
        DRIVE_SOURCE=all \
        TAG=$SVD_TAG \
        PROMPT="$PROMPT" \
        OUT_BASE=$SVD_BASE \
            "$LQR_ROOT/svd/run_partition_svd_pairs_no_action.sh" \
            | tee "$step4_out"
        sketch_id=$(grep -oE 'sketch job id: [0-9]+' "$step4_out" | tail -1 | awk '{print $NF}')
        svd_id=$(grep -oE 'svd job id: *[0-9]+'    "$step4_out" | tail -1 | awk '{print $NF}')
        rm -f "$step4_out"
        [[ -n "$svd_id" ]] || { echo "ERROR: failed to capture SVD finalize job id" >&2; exit 1; }
        wait_for_jobs "$sketch_id" "$svd_id"
        [[ -f "$SVD_DIR/svd_summary.pt" ]] || {
            echo "ERROR: SVD did not produce $SVD_DIR/svd_summary.pt" >&2
            exit 1
        }
    fi
fi

# =========================================================================
# Step 5 — full-rank jacobian (8-GPU array; blocks until login-merge done)
# =========================================================================
if should_run_step 5; then
    banner "Step 5/6: jacobians (8-GPU array; blocks until login-merge done)"
    if [[ -z "${FORCE_STEP5:-}" && -f "$SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" ]]; then
        echo "[skip] $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt already exists; skipping step 5."
        echo "       set FORCE_STEP5=1 to re-run."
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
# Step 6 — LQR sweep on task 03 using task 01 artifacts
# =========================================================================
if should_run_step 6; then
    banner "Step 6/6: LQR sweep on task 03 with task 01 artifacts"

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
fi

banner "PIPELINE COMPLETE — sweep is running asynchronously in slurm."
cat <<EOF
Monitor sweep with:
    squeue -u \$USER

Rollouts will appear under:
    $LQR_ROOT/rollouts/$SWEEP_TAG/

Artifacts (task 01 — source of SVD/jacobian):
    inputs  : $INPUT_DIR
    paired  : $PAIRED_DIR
    svd     : $SVD_DIR
    jac     : $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt

Sweep target: task 03 ("$TASK2_PROMPT")
EOF
