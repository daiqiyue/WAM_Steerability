#!/bin/bash
# End-to-end LQR-pipeline driver for the cam_random_large camera-view
# perturbation experiment on libero_10 task 0. Models the pipeline written
# up in notebooks/lqr/camera_position.md.
#
# Pipeline (each step blocks on its sbatch jobs before moving to the next,
# except step 4 which only submits and returns):
#
#   1. inputs/collect_policy_inputs_camera_view_perturbation.sh
#      - WORLD_SIZE=4 parallel ranks + dependent finalize
#      - N_POS / N_NEG default 10 each (clean-drive + perturbed-drive rollouts)
#      - perturbation: cam_random_large preset (pos_sigma=0.10 m,
#        rot_sigma=8°, fov_sigma=5°), visibility-constrained sampling
#
#   2. svd/run_partition_svd_pairs_no_action.sh
#      - 8-GPU sketch array + dependent finalize, action slot excluded
#      - SVD output shared by jacobians + LQR sweep
#
#   3. jacobians/run_jacobians_full.sh
#      - 8-GPU array + login-node merge (script self-blocks)
#      - inputs from negative.npz row 0 (perturbed-cam obs)
#
#   4. sweep_lqr_decay_cam_perturb.sh
#      - 108-combo λ × Q × R_init × τ grid + final aggregator
#      - rollouts at notebooks/lqr/rollouts/cam_perturb/<config_tag>/
#      - markdown summary at notebooks/lqr/camera_position_sweep.md
#
# Re-run safe: each step is skipped if its primary artifact exists. Force
# re-run with FORCE_STEP{1,2,3,4}=1.
#
# Usage:
#   ./run_cam_random_large_pipeline.sh                    # full pipeline
#   START_AT=3 ./run_cam_random_large_pipeline.sh         # resume at step 3
#   POLL_INTERVAL=15 ./run_cam_random_large_pipeline.sh
#   FORCE_STEP4=1 ./run_cam_random_large_pipeline.sh      # resubmit sweep
#   SUBMIT_SWEEP=0 ./run_cam_random_large_pipeline.sh     # stop after step 3
#   N_POS=20 N_NEG=20 ./run_cam_random_large_pipeline.sh  # larger corpus
#   CAM_BASE_SEED=7  ./run_cam_random_large_pipeline.sh   # different seed
#
# Artifacts:
#   inputs : notebooks/lqr/inputs/policy_inputs/libero_10__task00__cam_random_large__seed${SEED}__pos_neg/
#   svd    : /u/jhong7/cosmos-policy/directions/svd/libero10_task00_cam_random_large_seed${SEED}_pairs_no_action_dsall_N-1_k64_p10_ws8_tsall/
#   jac    : $SVD_DIR/A_tilde_full__<prompt_slug>__vjp_no_retain__vbf16/A_tilde__full.pt
#   sweep  : notebooks/lqr/rollouts/cam_perturb/
#   md     : notebooks/lqr/camera_position_sweep.md

set -euo pipefail

# =========================================================================
# Common config (override via env)
# =========================================================================
REPO_ROOT=/u/jhong7/Workspace/cosmos-policy
LQR_ROOT=$REPO_ROOT/notebooks/lqr

# Single-task pipeline; libero_10 task 0 is the canonical alphabet-soup +
# tomato-sauce task this perturbation regime was validated against.
TASK_ID="${TASK_ID:-0}"
SUITE_NAME="${SUITE_NAME:-libero_10}"
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"
RESOLUTION="${RESOLUTION:-256}"

# cam_random_large preset defaults (must match
# notebooks/stress_test/08_camera_view_perturbation.ipynb).
PRESET_NAME="${PRESET_NAME:-cam_random_large}"
CAM_BASE_SEED="${CAM_BASE_SEED:-42}"
POS_SIGMA="${POS_SIGMA:-0.10}"
ROT_SIGMA_DEG="${ROT_SIGMA_DEG:-8.0}"
FOV_SIGMA="${FOV_SIGMA:-5.0}"
WORKSPACE_TABLE_Z="${WORKSPACE_TABLE_Z:-0.90}"
WORKSPACE_VISIBLE_FRACTION="${WORKSPACE_VISIBLE_FRACTION:-0.55}"
VISIBILITY_MARGIN_PX="${VISIBILITY_MARGIN_PX:-8}"
MAX_REJECTION_ATTEMPTS="${MAX_REJECTION_ATTEMPTS:-2000}"

# Step 1 input corpus size (clean-drive + perturbed-drive rollout counts).
N_POS="${N_POS:-10}"
N_NEG="${N_NEG:-10}"
STEP1_WORLD_SIZE="${STEP1_WORLD_SIZE:-4}"
STEP1_TIME="${STEP1_TIME:-02:00:00}"

# Step 2 SVD knobs (matches run_partition_svd_pairs_no_action.sh defaults).
SVD_BASE="${SVD_BASE:-/u/jhong7/cosmos-policy/directions/svd}"
SVD_TAG="${SVD_TAG:-libero10_task00_${PRESET_NAME}_seed${CAM_BASE_SEED}_pairs_no_action_dsall}"
SVD_DIR="${SVD_DIR:-$SVD_BASE/${SVD_TAG}_N-1_k64_p10_ws8_tsall}"

# Step 3 jacobian knobs (single anchor obs at row 0).
OBS_INDEX="${OBS_INDEX:-0}"

# Step 4 sweep knobs (sweep_lqr_decay_cam_perturb.sh defaults: cam_mode=random
# seed=99). Tracks the v1 sweep that produced camera_position_sweep.md.
SWEEP_CAM_MODE="${SWEEP_CAM_MODE:-random}"
SWEEP_CAM_BASE_SEED="${SWEEP_CAM_BASE_SEED:-99}"
SWEEP_N_EPISODES="${SWEEP_N_EPISODES:-50}"
SWEEP_TIME="${SWEEP_TIME:-04:00:00}"
SUMMARY_MD="${SUMMARY_MD:-$LQR_ROOT/camera_position_sweep.md}"

POLL_INTERVAL=${POLL_INTERVAL:-30}
START_AT=${START_AT:-1}
SUBMIT_SWEEP=${SUBMIT_SWEEP:-1}

INPUT_DIR="${INPUT_DIR:-$LQR_ROOT/inputs/policy_inputs/${SUITE_NAME}__task$(printf '%02d' "$TASK_ID")__${PRESET_NAME}__seed${CAM_BASE_SEED}__pos_neg}"

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

JAC_PROMPT_SLUG=$(prompt_slug "$PROMPT")
JAC_DIR="$SVD_DIR/A_tilde_full__${JAC_PROMPT_SLUG}__vjp_no_retain__vbf16"

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
banner "${PRESET_NAME} pipeline: task ${TASK_ID}  seed=${CAM_BASE_SEED}"
echo "  PROMPT       : \"$PROMPT\""
echo "  INPUT_DIR    : $INPUT_DIR"
echo "  SVD_DIR      : $SVD_DIR"
echo "  JAC_DIR      : $JAC_DIR"
echo "  SUMMARY_MD   : $SUMMARY_MD"
echo "  N_POS / N_NEG: $N_POS / $N_NEG  (step 1 input collection)"
echo "  SWEEP grid   : λ×Q×R_init×τ = 4×3×3×3 = 108 combos, N_EPISODES=$SWEEP_N_EPISODES"
echo "                 cam_mode=$SWEEP_CAM_MODE  cam_base_seed=$SWEEP_CAM_BASE_SEED"
echo "  START_AT     : $START_AT"
echo "  SUBMIT_SWEEP : $SUBMIT_SWEEP"

# =========================================================================
# Step 1 — input collection (camera-perturbation pos/neg pairs)
# =========================================================================
if should_run_step 1; then
    banner "Step 1: collect cam_random_large pos/neg input pairs (WORLD_SIZE=$STEP1_WORLD_SIZE)"
    if [[ -z "${FORCE_STEP1:-}" && -f "$INPUT_DIR/positive.npz" && -f "$INPUT_DIR/negative.npz" ]]; then
        echo "[skip] $INPUT_DIR/{positive,negative}.npz already exist."
        echo "       set FORCE_STEP1=1 to re-run."
    else
        step1_out=$(mktemp)
        WORLD_SIZE="$STEP1_WORLD_SIZE" \
        N_POS="$N_POS" N_NEG="$N_NEG" \
        CAM_BASE_SEED="$CAM_BASE_SEED" \
        POS_SIGMA="$POS_SIGMA" ROT_SIGMA_DEG="$ROT_SIGMA_DEG" FOV_SIGMA="$FOV_SIGMA" \
        WORKSPACE_TABLE_Z="$WORKSPACE_TABLE_Z" \
        WORKSPACE_VISIBLE_FRACTION="$WORKSPACE_VISIBLE_FRACTION" \
        VISIBILITY_MARGIN_PX="$VISIBILITY_MARGIN_PX" \
        MAX_REJECTION_ATTEMPTS="$MAX_REJECTION_ATTEMPTS" \
        PRESET_NAME="$PRESET_NAME" \
        SUITE="$SUITE_NAME" TASK_ID="$TASK_ID" RESOLUTION="$RESOLUTION" \
        OUT_DIR="$INPUT_DIR" \
        TIME="$STEP1_TIME" \
            "$LQR_ROOT/inputs/collect_policy_inputs_camera_view_perturbation.sh" \
            | tee "$step1_out"
        # Wrapper prints "collect job id: N" (the array) + "finalize job id: N".
        collect_id=$(grep -oE 'collect job id: *[0-9]+' "$step1_out" | tail -1 | awk '{print $NF}')
        finalize_id=$(grep -oE 'finalize job id: *[0-9]+' "$step1_out" | tail -1 | awk '{print $NF}')
        rm -f "$step1_out"
        [[ -n "$finalize_id" ]] || {
            echo "ERROR: failed to capture step 1 finalize job id" >&2; exit 1; }
        wait_for_jobs "$collect_id" "$finalize_id"
        [[ -f "$INPUT_DIR/positive.npz" ]] || {
            echo "ERROR: missing $INPUT_DIR/positive.npz" >&2; exit 1; }
        [[ -f "$INPUT_DIR/negative.npz" ]] || {
            echo "ERROR: missing $INPUT_DIR/negative.npz" >&2; exit 1; }
    fi
fi

# =========================================================================
# Step 2 — SVD on the pos/neg pairs
# =========================================================================
if should_run_step 2; then
    banner "Step 2: partition SVD (action slot excluded; 8-GPU sketch + finalize)"
    if [[ -z "${FORCE_STEP2:-}" && -f "$SVD_DIR/svd_summary.pt" ]]; then
        echo "[skip] $SVD_DIR/svd_summary.pt already exists."
        echo "       set FORCE_STEP2=1 to re-run."
    else
        [[ -f "$INPUT_DIR/positive.npz" && -f "$INPUT_DIR/negative.npz" ]] || {
            echo "ERROR: step 2 needs $INPUT_DIR/{positive,negative}.npz from step 1" >&2
            exit 1
        }
        step2_out=$(mktemp)
        POS_NPZ="$INPUT_DIR/positive.npz" \
        NEG_NPZ="$INPUT_DIR/negative.npz" \
        TAG="$SVD_TAG" \
        PROMPT="$PROMPT" \
        OUT_BASE="$SVD_BASE" \
        WORLD_SIZE=8 N=-1 \
            "$LQR_ROOT/svd/run_partition_svd_pairs_no_action.sh" \
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
# Step 3 — jacobians against the SVD basis (self-blocking wrapper)
# =========================================================================
if should_run_step 3; then
    banner "Step 3: jacobians  (PROMPT=$PROMPT  OBS_INDEX=$OBS_INDEX)"
    echo "  inputs_npz : $INPUT_DIR/negative.npz"
    echo "  jac_dir    : $JAC_DIR"
    if [[ -z "${FORCE_STEP3:-}" && -f "$JAC_DIR/A_tilde__full.pt" ]]; then
        echo "[skip] $JAC_DIR/A_tilde__full.pt already exists."
        echo "       set FORCE_STEP3=1 to re-run."
    else
        [[ -f "$INPUT_DIR/negative.npz" ]] || {
            echo "ERROR: missing $INPUT_DIR/negative.npz (need step 1 first)" >&2; exit 1; }
        [[ -f "$SVD_DIR/svd_summary.pt" ]] || {
            echo "ERROR: missing $SVD_DIR/svd_summary.pt (need step 2 first)" >&2; exit 1; }
        # run_jacobians_full.sh self-blocks (polls its workers + runs the
        # login-node merge). No extra wait_for_jobs needed.
        SVD_DIR="$SVD_DIR" \
        INPUTS_NPZ="$INPUT_DIR/negative.npz" \
        OBS_INDEX="$OBS_INDEX" \
        PROMPT="$PROMPT" \
        WORLD_SIZE=8 \
            "$LQR_ROOT/jacobians/run_jacobians_full.sh"
        [[ -f "$JAC_DIR/A_tilde__full.pt" ]] || {
            echo "ERROR: jacobian not produced at $JAC_DIR/A_tilde__full.pt" >&2
            exit 1
        }
    fi
fi

# =========================================================================
# Step 4 — LQR hyperparameter sweep (submits and returns; jobs run async)
# =========================================================================
if should_run_step 4; then
    if [[ "$SUBMIT_SWEEP" != "1" ]]; then
        banner "Step 4: SKIPPED (SUBMIT_SWEEP=$SUBMIT_SWEEP)"
    else
        banner "Step 4: LQR sweep (108-combo λ × Q × R_init × τ + aggregator)"
        # Sanity-check the jacobian path the inner driver will read.
        [[ -f "$JAC_DIR/A_tilde__full.pt" ]] || {
            echo "ERROR: missing jacobian $JAC_DIR/A_tilde__full.pt" >&2; exit 1; }

        # The sweep launcher submits 108 (rollout array + finalize) pairs plus
        # one final aggregator with afterany on every finalize. It does NOT
        # block — it returns once the last sbatch is queued.
        SVD_DIR="$SVD_DIR" \
        JAC_DIR_ACT="$(basename "$JAC_DIR")" \
        PROMPT="$PROMPT" \
        SUITE_NAME="$SUITE_NAME" TASK_ID="$TASK_ID" RESOLUTION="$RESOLUTION" \
        CAM_MODE="$SWEEP_CAM_MODE" CAM_BASE_SEED="$SWEEP_CAM_BASE_SEED" \
        N_EPISODES="$SWEEP_N_EPISODES" \
        TIME="$SWEEP_TIME" \
        SUMMARY_MD="$SUMMARY_MD" \
            "$LQR_ROOT/sweep_lqr_decay_cam_perturb.sh"
    fi
fi

banner "PIPELINE COMPLETE"
cat <<EOF
Monitor sweep with:
    squeue -u \$USER --name=cps_lqr_cam_roll,cps_lqr_cam_finalize,cps_lqr_sweep_agg

Rollouts will appear under:
    $LQR_ROOT/rollouts/cam_perturb/<config_tag>/

Summary md (auto-refreshed by final aggregator):
    $SUMMARY_MD

Artifacts:
    inputs : $INPUT_DIR/
    svd    : $SVD_DIR
    jac    : $JAC_DIR/A_tilde__full.pt
    sweep  : $LQR_ROOT/rollouts/cam_perturb/
EOF
