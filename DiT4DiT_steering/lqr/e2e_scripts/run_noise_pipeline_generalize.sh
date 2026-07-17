#!/bin/bash
# Generalisation pipeline for DiT4DiT (Gaussian-noise variant): use the
# contrastive directions / SVD / Jacobians computed on libero_10 task 01
# ("put both the cream cheese box and the butter in the basket") to run the
# LQR noise sweep on a DIFFERENT libero_10 task — without recomputing SVD or
# Jacobians for the target task.
#
# Analogue of the cosmos pipeline
#   the repo root's notebooks/lqr/e2e_scripts/run_gripper_xyz_pipeline_generalize.sh
# but built on the noised pipeline (run_noised_pipeline.sh) instead of the
# gripper_xyz one: it reuses run_noised_pipeline.sh steps 1-4 verbatim to
# build (or reuse) the task-01 artifacts, then runs only step 5 (the noise
# sweep) against the task you specify.
#
# Pipeline:
#   1. inputs/collect_policy_inputs_noise.sh   (1 GPU)              [task 01]
#   2. inputs/precompute_vl_embs.sh            (1 GPU)              [task 01]
#   3. svd/run_partition_svd_pairs_no_action.sh(8-GPU sketch + finalize) [task 01]
#   4. jacobians/run_jacobians_full.sh         (8-GPU array + login merge)  [task 01]
#   5. sweep_lqr_noised.sh                     [TARGET task rollouts, task-01 artifacts]
#
# Steps 1-4 build (or reuse) the task-01 SVD/Jacobian artifacts. They write to
# the SAME paths as run_noised_pipeline.sh, so if you have already run the base
# pipeline they are all skipped and only step 5 runs.
#
# Step 5 overrides TASK_ID (and PROMPT, used for merge metadata) to target a
# different task while keeping SVD_DIR / JAC_DIR_ACT pointing at the task-01
# artifacts — testing whether the learned directions generalise across tasks.
#
# NOTE: the eval task is selected entirely by TASK_ID. The client calls
#   task_suite.get_task(TASK_ID).language
# and sends that language string to the server as the steering prompt, so
# TASK2_PROMPT below is purely descriptive/metadata — set TASK2_ID to retarget.
#
# Re-run safe: each step is skipped if its primary output artifact exists.
#
# Usage:
#   ./run_noise_pipeline_generalize.sh                  # full pipeline
#   START_AT=5 ./run_noise_pipeline_generalize.sh       # submit sweep only (reuse task-01 artifacts)
#   POLL_INTERVAL=15 ./run_noise_pipeline_generalize.sh
#   FORCE_STEP1=1 ./run_noise_pipeline_generalize.sh    # re-run step 1 even if outputs exist
#
#   sbatch --account=bhhv-dtai-gh --partition=ghx4 \
#     --ntasks=1 --cpus-per-task=2 --mem=4G --time=06:00:00 \
#     --output=pipeline_generalize_%j.log \
#     ./run_noise_pipeline_generalize.sh

set -euo pipefail

DIT4DIT_ROOT=/projects/bhhv/jskifstad/DiT4DiT
LQR_ROOT=$DIT4DIT_ROOT/notebooks/lqr

# =========================================================================
# Source task config — SVD/Jacobian artifacts (steps 1-4). Keep identical to
# run_noised_pipeline.sh so the artifacts are shared and reused.
# =========================================================================
export TASK_ID=1
export SUITE=libero_10
export PROMPT="put both the cream cheese box and the butter in the basket"
export SVD_NOISE_SIGMA=75.0   # sigma used to collect data / build SVD+Jacobians
export EVAL_NOISE_SIGMA=22.0  # sigma applied during rollout evaluation
export NOISE_SIGMA=$SVD_NOISE_SIGMA  # used by steps 1-4 (data, SVD, Jacobians)
export N_EPISODES=30

INPUT_DIR=$LQR_ROOT/inputs/policy_inputs/libero_10__task$(printf '%02d' "$TASK_ID")__noise_sigma${SVD_NOISE_SIGMA}
SVD_BASE=$LQR_ROOT/directions/svd
SVD_TAG=libero10_task$(printf '%02d' "$TASK_ID")_noise_sigma${SVD_NOISE_SIGMA}

# Matches the RUN_TAG derivation in run_partition_svd_pairs_no_action.sh:
#   RUN_TAG="${TAG}_N${N}_k${K_TARGET}_p${P_OVER}_ws${WORLD_SIZE}_ts${TS_TAG}"
# with defaults N=-1, K_TARGET=64, P_OVER=10, WORLD_SIZE=8, TIMESTEPS=all.
SVD_DIR=${SVD_BASE}/${SVD_TAG}_N-1_k64_p10_ws8_tsall

# Matches PROMPT_TAG in run_jacobians_full.sh:
#   PROMPT_TAG=$(echo "$PROMPT" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50)
#   RUN_TAG="A_tilde_full__${PROMPT_TAG}__${MODE}__v${V_DTYPE}"
_PROMPT_TAG=$(echo "$PROMPT" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50)
JAC_DIR_ACT="A_tilde_full__${_PROMPT_TAG}__vjp_no_retain__vbf16"
VL_EMBS_PATH="$INPUT_DIR/vl_embs__${_PROMPT_TAG:0:30}.pt"

export SVD_DIR
export JAC_DIR_ACT
export VL_EMBS_PATH

# =========================================================================
# Target task config — sweep target (step 5 only; SVD/Jacobian artifacts stay
# fixed to task 01 above). Set TASK2_ID to retarget.
#
# libero_10 task_id -> prompt:
#   0  put both the alphabet soup and the tomato sauce in the basket
#   1  put both the cream cheese box and the butter in the basket   (source task)
#   2  turn on the stove and put the moka pot on it
#   3  put the black bowl in the bottom drawer of the cabinet and close it
#   4  put the white mug on the left plate and put the yellow and white mug on the right plate
#   5  pick up the book and place it in the back compartment of the caddy
#   6  put the white mug on the plate and put the chocolate pudding to the right of the plate
#   7  put both the alphabet soup and the cream cheese box in the basket
#   8  put both moka pots on the stove
#   9  put the yellow and white mug in the microwave and close it
# =========================================================================
# TASK2_ID=0
# TASK2_PROMPT="put both the alphabet soup and the tomato sauce in the basket"

# TASK2_ID=2
# TASK2_PROMPT="turn on the stove and put the moka pot on it"

TASK2_ID=3
TASK2_PROMPT="put the black bowl in the bottom drawer of the cabinet and close it"


# SWEEP_TAG is the top-level rollouts/ subdirectory and is derived from the
# source + target task ids, so it is always a SEPARATE directory from the
# non-generalized pipeline (which uses TAG=noised_sigma<sigma>_seed1). Changing
# the two vars above is enough to retarget the script at a different task.
# Videos + results land under rollouts/$SWEEP_TAG/<config_tag>/.
SWEEP_TAG=rollout_task_${TASK_ID}_to_task${TASK2_ID}_noised_sigma${EVAL_NOISE_SIGMA}_seed1
SWEEP_WORLD_SIZE=4

POLL_INTERVAL=${POLL_INTERVAL:-30}
START_AT=${START_AT:-1}

# =========================================================================
# Helpers
# =========================================================================
banner() { echo; echo "================================================================================"; echo "  $*"; echo "================================================================================"; }

wait_for_jobs() {
    local ids=()
    for j in "$@"; do [[ -n "$j" && "$j" != "?" ]] && ids+=("$j"); done
    [[ ${#ids[@]} -eq 0 ]] && return 0
    local joined; joined=$(IFS=,; echo "${ids[*]}")
    echo "[wait] watching: $joined (poll every ${POLL_INTERVAL}s)"
    while true; do
        local active; active=$(squeue -h -j "$joined" -r -o '%A %T' 2>/dev/null || true)
        [[ -z "$active" ]] && { echo "[wait] all drained: $joined"; return 0; }
        local n_total n_pending n_running
        n_total=$(echo "$active" | wc -l)
        n_pending=$(echo "$active" | awk '$2=="PENDING"' | wc -l)
        n_running=$(echo "$active" | awk '$2=="RUNNING"' | wc -l)
        printf '[wait %s] active=%d pending=%d running=%d\n' "$(date '+%H:%M:%S')" "$n_total" "$n_pending" "$n_running"
        sleep "$POLL_INTERVAL"
    done
}

should_run_step() {
    local n="$1"
    if (( n < START_AT )); then echo "[skip] step $n below START_AT=$START_AT."; return 1; fi
    return 0
}

# =========================================================================
# Step 1 — collect inputs with noise (1 GPU)   [task 01]
# =========================================================================
if should_run_step 1; then
    banner "Step 1/5: collect inputs (task $(printf '%02d' "$TASK_ID"), $N_EPISODES eps, σ=${NOISE_SIGMA})"
    if [[ -z "${FORCE_STEP1:-}" && -f "$INPUT_DIR/positive.npz" && -f "$INPUT_DIR/negative.npz" ]]; then
        echo "[skip] ${INPUT_DIR}/{positive,negative}.npz already exist. Set FORCE_STEP1=1 to re-run."
    else
        step1_out=$(mktemp)
        TASK_ID=$TASK_ID SUITE=$SUITE N_EPISODES=$N_EPISODES \
        NOISE_SIGMA=$NOISE_SIGMA OUT_DIR=$INPUT_DIR \
        TIME=01:00:00 \
            "$LQR_ROOT/inputs/collect_policy_inputs_noise.sh" \
            | tee "$step1_out"
        step1_id=$(grep -oE 'Submitted batch job [0-9]+' "$step1_out" | tail -1 | awk '{print $NF}')
        rm -f "$step1_out"
        [[ -n "$step1_id" ]] || { echo "ERROR: failed to capture step 1 job id" >&2; exit 1; }
        wait_for_jobs "$step1_id"
        [[ -f "$INPUT_DIR/positive.npz" && -f "$INPUT_DIR/negative.npz" ]] || {
            echo "ERROR: step 1 did not produce positive.npz / negative.npz" >&2; exit 1
        }
    fi
fi

# =========================================================================
# Step 2 — precompute VLM embeddings (1 GPU, runs model once for all obs)
# =========================================================================
if should_run_step 2; then
    banner "Step 2/5: precompute vl_embs (1 GPU, loads full model once)"
    if [[ -z "${FORCE_STEP2:-}" && -f "$VL_EMBS_PATH" ]]; then
        echo "[skip] $VL_EMBS_PATH already exists. Set FORCE_STEP2=1 to re-run."
    else
        step2_out=$(mktemp)
        POS_NPZ=$INPUT_DIR/positive.npz \
        NEG_NPZ=$INPUT_DIR/negative.npz \
        PROMPT="$PROMPT" \
        OUT_PATH="$VL_EMBS_PATH" \
        TIME=02:00:00 \
            "$LQR_ROOT/inputs/precompute_vl_embs.sh" \
            | tee "$step2_out"
        step2_id=$(grep -oE 'Submitted batch job [0-9]+' "$step2_out" | tail -1 | awk '{print $NF}')
        rm -f "$step2_out"
        [[ -n "$step2_id" ]] || { echo "ERROR: failed to capture step 2 job id" >&2; exit 1; }
        wait_for_jobs "$step2_id"
        [[ -f "$VL_EMBS_PATH" ]] || { echo "ERROR: step 2 did not produce $VL_EMBS_PATH" >&2; exit 1; }
        echo "[ok] vl_embs: $VL_EMBS_PATH"
    fi
fi

# =========================================================================
# Step 3 — SVD (npzs are already aligned — no pairing step needed)
# =========================================================================
if should_run_step 3; then
    banner "Step 3/5: SVD -> $SVD_DIR"
    if [[ -z "${FORCE_STEP3:-}" && -f "$SVD_DIR/svd_summary.pt" ]]; then
        echo "[skip] $SVD_DIR/svd_summary.pt already exists. Set FORCE_STEP3=1 to re-run."
    else
        step3_out=$(mktemp)
        WORLD_SIZE=8 \
        POS_NPZ=$INPUT_DIR/positive.npz \
        NEG_NPZ=$INPUT_DIR/negative.npz \
        VL_EMBS_PATH="$VL_EMBS_PATH" \
        DRIVE_SOURCE=all \
        TAG=$SVD_TAG \
        PROMPT="$PROMPT" \
        OUT_BASE=$SVD_BASE \
            "$LQR_ROOT/svd/run_partition_svd_pairs_no_action.sh" \
            | tee "$step3_out"
        sketch_id=$(grep -oE 'sketch job id: [0-9]+' "$step3_out" | tail -1 | awk '{print $NF}')
        svd_id=$(grep -oE 'svd job id: *[0-9]+'    "$step3_out" | tail -1 | awk '{print $NF}')
        rm -f "$step3_out"
        [[ -n "$svd_id" ]] || { echo "ERROR: failed to capture SVD job id" >&2; exit 1; }
        wait_for_jobs "$sketch_id" "$svd_id"
        [[ -f "$SVD_DIR/svd_summary.pt" ]] || {
            echo "ERROR: SVD did not produce $SVD_DIR/svd_summary.pt" >&2; exit 1
        }
    fi
fi

# =========================================================================
# Step 4 — Jacobians (8-GPU array; self-blocks until login-merge done)
# =========================================================================
if should_run_step 4; then
    banner "Step 4/5: jacobians"
    if [[ -z "${FORCE_STEP4:-}" && -f "$SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" ]]; then
        echo "[skip] $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt already exists. Set FORCE_STEP4=1 to re-run."
    else
        WORLD_SIZE=8 \
        SVD_DIR=$SVD_DIR \
        PROMPT="$PROMPT" \
        INPUTS_NPZ=$INPUT_DIR/negative.npz \
        OBS_INDEX=0 \
            "$LQR_ROOT/jacobians/run_jacobians_full.sh"
        [[ -f "$SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" ]] || {
            echo "ERROR: jacobian not produced at $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" >&2; exit 1
        }
    fi
fi

# =========================================================================
# Step 5 — LQR noise sweep on the TARGET task using task-01 artifacts
# (submits async, returns immediately)
# =========================================================================
if should_run_step 5; then
    banner "Step 5/5: LQR noise sweep on task $(printf '%02d' "$TASK2_ID") with task $(printf '%02d' "$TASK_ID") artifacts (async)"

    # Override rollout target to the target task; SVD_DIR / JAC_DIR_ACT stay
    # fixed to the task-01 artifacts exported above. TASK_ID selects the eval
    # task (the client derives the steering prompt from it); PROMPT here is
    # only forwarded to the merge step as metadata.
    export TASK_ID=$TASK2_ID
    export PROMPT="$TASK2_PROMPT"

    echo "--- dry-run preview ---"
    DRY=1 FORCE=1 TAG=$SWEEP_TAG WORLD_SIZE=$SWEEP_WORLD_SIZE N_EPISODES=$N_EPISODES \
        EVAL_NOISE_SIGMA=$EVAL_NOISE_SIGMA \
        "$LQR_ROOT/sweep_lqr_noised.sh"

    echo "--- submitting ---"
    DRY=0 FORCE=1 TAG=$SWEEP_TAG WORLD_SIZE=$SWEEP_WORLD_SIZE N_EPISODES=$N_EPISODES \
        EVAL_NOISE_SIGMA=$EVAL_NOISE_SIGMA \
        "$LQR_ROOT/sweep_lqr_noised.sh"
fi

banner "PIPELINE COMPLETE — sweep is running asynchronously in slurm."
cat <<EOF
Monitor:
    squeue -u \$USER

Rollouts will appear under:
    $LQR_ROOT/rollouts/$SWEEP_TAG/

Artifacts (task $(printf '%02d' "$TASK_ID") — source of SVD/Jacobian):
    inputs  : $INPUT_DIR
    svd     : $SVD_DIR
    jac     : $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt

Sweep target: task $(printf '%02d' "$TASK2_ID") ("$TASK2_PROMPT")
EOF
