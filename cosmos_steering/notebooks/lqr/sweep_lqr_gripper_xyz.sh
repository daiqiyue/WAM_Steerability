#!/bin/bash
# Hyperparameter sweep over (lambda, q, rinit, tau) for run_lqr_cosmos_policy_gripper_xyz.sh.
#
# Grid:
#   LAMBDA in {0.5, 5.0, 10.0, 15.0}    (4)
#   Q      in {1.0, 10.0}               (2)
#   RINIT  in {5.0, 10.0, 20.0}         (3)
#   TAU    in {3.0, 5.0, 7.0}           (3)
# = 72 combos total. 8 are already done (auto-detected from existing dirs
#   under rollouts/seed1/ + rollouts/position/), leaving 64 to submit.
#
# Each combo invokes run_lqr_cosmos_policy_gripper_xyz.sh, which itself
# submits an sbatch array of WORLD_SIZE workers + a dependent merge job.
# So a full sweep submits 64 * (WORLD_SIZE + 1) slurm jobs.
#
# Usage:
#   ./sweep_lqr_gripper_xyz.sh                     # DRY run (default)
#   DRY=0 ./sweep_lqr_gripper_xyz.sh               # actually submit
#   DRY=0 TAG=position ./sweep_lqr_gripper_xyz.sh  # override output subdir
#   DRY=0 WORLD_SIZE=8 N_EPISODES=20 ./sweep_lqr_gripper_xyz.sh
#   FORCE=1 DRY=0 ./sweep_lqr_gripper_xyz.sh       # ignore "already-done" skip list

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
INNER_SH="$SCRIPT_DIR/run_lqr_cosmos_policy_gripper_xyz.sh"
[[ -x "$INNER_SH" ]] || { echo "ERROR: $INNER_SH not executable" >&2; exit 1; }

# --- Sweep grid --------------------------------------------------------------
# Note: float-formatted (.0 suffix) to match the LQR_TAG naming convention
# used in run_lqr_cosmos_policy_gripper_xyz.sh, so OUT_DIR collisions with
# existing dirs are detected correctly.
LAMBDAS=(0.5 5.0 10.0 15.0)
QS=(1.0 10.0)
RINITS=(5.0 10.0 20.0)
TAUS=(3.0 5.0 7.0)

# --- Knobs (passthrough to the inner launcher) -------------------------------
DRY="${DRY:-1}"
FORCE="${FORCE:-0}"
TAG="${TAG:-position}"          # OUT_BASE/$TAG/ — most existing grxyz dirs are under position/
WORLD_SIZE="${WORLD_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"
# Fixed defaults that match the existing dir naming. Override if needed.
PRESET="${PRESET:-xyz_random_xlarge_3}"
BASE_SEED="${BASE_SEED:-42}"

# --- Skip list ---------------------------------------------------------------
# (lambda:q:rinit:tau) keys for combos whose OUT_DIR already exists with a
# results.json.
ALREADY_DONE=()
is_done() {
    local key="$1"
    for d in "${ALREADY_DONE[@]}"; do
        [[ "$d" == "$key" ]] && return 0
    done
    return 1
}

TOTAL=$(( ${#LAMBDAS[@]} * ${#QS[@]} * ${#RINITS[@]} * ${#TAUS[@]} ))
SKIPPED=0
SUBMITTED=0

echo "=========================================================================="
echo "Sweep grid:"
echo "  LAMBDA in [${LAMBDAS[*]}]"
echo "  Q      in [${QS[*]}]"
echo "  RINIT  in [${RINITS[*]}]"
echo "  TAU    in [${TAUS[*]}]"
echo "  Total combos: $TOTAL  (${#ALREADY_DONE[@]} already done -> will skip unless FORCE=1)"
echo ""
echo "Submission knobs:"
echo "  TAG        = $TAG       (OUT_BASE/\$TAG/<config_tag>)"
echo "  WORLD_SIZE = $WORLD_SIZE"
echo "  N_EPISODES = $N_EPISODES"
echo "  PRESET     = $PRESET"
echo "  BASE_SEED  = $BASE_SEED"
echo "  DRY        = $DRY  (1 = print only; 0 = actually submit)"
echo "  FORCE      = $FORCE (1 = ignore skip list)"
echo "=========================================================================="
echo ""

i=0
for L in "${LAMBDAS[@]}"; do
for Q in "${QS[@]}"; do
for RI in "${RINITS[@]}"; do
for T in "${TAUS[@]}"; do
    i=$((i + 1))
    KEY="${L}:${Q}:${RI}:${T}"
    LABEL=$(printf "lam=%-4s q=%-4s rinit=%-4s tau=%-3s" "$L" "$Q" "$RI" "$T")
    if [[ "$FORCE" != "1" ]] && is_done "$KEY"; then
        SKIPPED=$((SKIPPED + 1))
        printf '[%2d/%d]  SKIP    %s   (already done)\n' "$i" "$TOTAL" "$LABEL"
        continue
    fi
    SUBMITTED=$((SUBMITTED + 1))
    if [[ "$DRY" == "1" ]]; then
        printf '[%2d/%d]  WOULD   %s\n' "$i" "$TOTAL" "$LABEL"
    else
        printf '[%2d/%d]  SUBMIT  %s   ...' "$i" "$TOTAL" "$LABEL"
        INNER_LOG="$SCRIPT_DIR/logs/sweep_inner_lam${L}_q${Q}_rinit${RI}_tau${T}.log"
        mkdir -p "$SCRIPT_DIR/logs"
        if LAMBDA="$L" Q_SCALE="$Q" R_SCALE="$RI" R_SCALE_TAU="$T" \
                TAG="$TAG" WORLD_SIZE="$WORLD_SIZE" N_EPISODES="$N_EPISODES" \
                PRESET="$PRESET" BASE_SEED="$BASE_SEED" \
                "$INNER_SH" >"$INNER_LOG" 2>&1; then
            # Extract the rollout + merge job IDs from the inner launcher's log.
            ROLL_ID=$(grep -oE 'rollout job id: [0-9]+' "$INNER_LOG" | tail -1 | awk '{print $NF}')
            MERGE_ID=$(grep -oE 'merge job id:   [0-9]+' "$INNER_LOG" | tail -1 | awk '{print $NF}')
            printf '  rollout=%s merge=%s\n' "${ROLL_ID:-?}" "${MERGE_ID:-?}"
        else
            printf '  FAILED (see %s)\n' "$INNER_LOG"
        fi
    fi
done; done; done; done

echo ""
echo "=========================================================================="
echo "Summary:"
echo "  Total combos in grid: $TOTAL"
echo "  Skipped (already done): $SKIPPED"
if [[ "$DRY" == "1" ]]; then
    echo "  Would submit: $SUBMITTED"
    echo ""
    echo "DRY=1; nothing was actually submitted. Re-run with DRY=0 to launch."
else
    echo "  Submitted: $SUBMITTED  (each spawns a rollout array of $WORLD_SIZE + 1 merge job)"
    echo "  -> ~$((SUBMITTED * (WORLD_SIZE + 1))) total slurm jobs queued."
    echo ""
    echo "Monitor:        squeue -u \$USER"
    echo "After it all finishes, summarize successes:"
    echo "  python $SCRIPT_DIR/summarize_lqr_gripper_xyz_sweep.py"
fi
echo "=========================================================================="
