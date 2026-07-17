#!/bin/bash
# Hyperparameter sweep over ALPHA for run_actadd_cosmos_policy.sh (image-noise
# variant).
#
# For each alpha value the inner launcher submits an sbatch array of
# WORLD_SIZE workers + a dependent merge job. A full sweep therefore
# submits len(ALPHAS) * (WORLD_SIZE + 1) slurm jobs.
#
# NOTE: This file is the SLURM sweep launcher and is preserved for clusters
# that have slurm. For the local (no-slurm) workflow used in this workspace,
# the per-task alpha sweep is notebooks/lqr/violet/actadd_sweep.sh, driven
# end-to-end by notebooks/lqr/e2e_scripts/run_actadd_from_task06_local.sh.
#
# Usage:
#   V_PATH=/path/to/v.pt ./sweep_actadd_cosmos_policy.sh   # dry run
#   V_PATH=/path/to/v.pt DRY=0 ./sweep_actadd_cosmos_policy.sh
#   V_PATH=... DRY=0 TAG=sweep1 WORLD_SIZE=4 \
#     ./sweep_actadd_cosmos_policy.sh
#   FORCE=1 DRY=0 V_PATH=... ./sweep_actadd_cosmos_policy.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
INNER_SH="$SCRIPT_DIR/run_actadd_cosmos_policy.sh"
[[ -x "$INNER_SH" ]] || { echo "ERROR: $INNER_SH not executable" >&2; exit 1; }

# notebooks/actadd -> repo root is two dirs up
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." &>/dev/null && pwd)}"
LQR_ROOT="${LQR_ROOT:-$REPO_ROOT/notebooks/lqr}"

# --- Required inputs --------------------------------------------------------
V_PATH="${V_PATH:-}"

[[ -n "$V_PATH" ]] || { echo "ERROR: V_PATH is not set" >&2; exit 1; }

# --- Sweep grid: alpha values -----------------------------------------------
read -r -a ALPHAS <<< "${ALPHAS:-0.0 0.25 0.5 1.0 2.0 5.0 10.0 20.0}"

# --- Knobs (passthrough to the inner launcher) ------------------------------
DRY="${DRY:-1}"
FORCE="${FORCE:-0}"
TAG="${TAG:-actadd_alpha_sweep}"
WORLD_SIZE="${WORLD_SIZE:-4}"
N_EPISODES="${N_EPISODES:-30}"
NOISE_SIGMA="${NOISE_SIGMA:-90.0}"
NOISE_SEED_BASE="${NOISE_SEED_BASE:-99}"
NOISE_PER_EPISODE_SEED="${NOISE_PER_EPISODE_SEED:-1}"
PROMPT="${PROMPT:-put the white mug on the plate and put the chocolate pudding to the right of the plate}"
TASK_ID="${TASK_ID:-6}"
SUITE="${SUITE:-libero_10}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-520}"
SEED="${SEED:-1}"
SAMPLING_STEPS="${SAMPLING_STEPS:-10}"
OUT_BASE="${OUT_BASE:-$LQR_ROOT/rollouts}"

TOTAL="${#ALPHAS[@]}"
SKIPPED=0
SUBMITTED=0

echo "=========================================================================="
echo "ActAdd alpha sweep (image-noise variant):"
echo "  V_PATH     = $V_PATH"
echo "  ALPHAS     = [${ALPHAS[*]}]  ($TOTAL values)"
echo ""
echo "Submission knobs:"
echo "  TAG        = $TAG       (OUT_BASE/\$TAG/<config_tag>)"
echo "  WORLD_SIZE = $WORLD_SIZE"
echo "  N_EPISODES = $N_EPISODES"
echo "  NOISE_SIGMA= $NOISE_SIGMA  seed_base=$NOISE_SEED_BASE  per_ep=$NOISE_PER_EPISODE_SEED"
echo "  SUITE      = $SUITE   TASK_ID = $TASK_ID"
echo "  SEED       = $SEED"
echo "  DRY        = $DRY  (1 = print only; 0 = actually submit)"
echo "  FORCE      = $FORCE (1 = ignore already-done check)"
echo "=========================================================================="
echo ""

i=0
for A in "${ALPHAS[@]}"; do
    i=$((i + 1))
    LABEL=$(printf "alpha=%-6s" "$A")

    # Check if this alpha already has a completed results.json.
    _slug() { local s="$1"; local n="${2:-24}"; echo "${s:0:$n}" | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_-'; }
    PROMPT_SLUG="$(_slug "$PROMPT" 24)"
    ACTADD_TAG="alpha${A}"
    NOISE_TAG="s${NOISE_SIGMA}_sb${NOISE_SEED_BASE}"
    CONFIG_TAG="${SUITE}__task$(printf '%02d' "$TASK_ID")__actadd_noised__${NOISE_TAG}__${ACTADD_TAG}__${PROMPT_SLUG}"
    EXPECTED_OUT="$OUT_BASE${TAG:+/$TAG}/$CONFIG_TAG"

    if [[ "$FORCE" != "1" && -f "$EXPECTED_OUT/results.json" ]]; then
        SKIPPED=$((SKIPPED + 1))
        printf '[%2d/%d]  SKIP    %s   (results.json exists)\n' "$i" "$TOTAL" "$LABEL"
        continue
    fi

    SUBMITTED=$((SUBMITTED + 1))
    if [[ "$DRY" == "1" ]]; then
        printf '[%2d/%d]  WOULD   %s\n' "$i" "$TOTAL" "$LABEL"
    else
        printf '[%2d/%d]  SUBMIT  %s   ...' "$i" "$TOTAL" "$LABEL"
        INNER_LOG="$SCRIPT_DIR/logs/sweep_actadd_alpha${A}.log"
        mkdir -p "$SCRIPT_DIR/logs"
        if V_PATH="$V_PATH" ALPHA="$A" \
                SAMPLING_STEPS="$SAMPLING_STEPS" \
                TAG="$TAG" WORLD_SIZE="$WORLD_SIZE" N_EPISODES="$N_EPISODES" \
                NOISE_SIGMA="$NOISE_SIGMA" NOISE_SEED_BASE="$NOISE_SEED_BASE" \
                NOISE_PER_EPISODE_SEED="$NOISE_PER_EPISODE_SEED" \
                PROMPT="$PROMPT" TASK_ID="$TASK_ID" SUITE="$SUITE" \
                MAX_ENV_STEPS="$MAX_ENV_STEPS" SEED="$SEED" \
                OUT_BASE="$OUT_BASE" \
                "$INNER_SH" >"$INNER_LOG" 2>&1; then
            ROLL_ID=$(grep -oE 'rollout job id: [0-9]+' "$INNER_LOG" | tail -1 | awk '{print $NF}')
            MERGE_ID=$(grep -oE 'merge job id:   [0-9]+' "$INNER_LOG" | tail -1 | awk '{print $NF}')
            printf '  rollout=%s merge=%s\n' "${ROLL_ID:-?}" "${MERGE_ID:-?}"
        else
            printf '  FAILED (see %s)\n' "$INNER_LOG"
        fi
    fi
done

echo ""
echo "=========================================================================="
echo "Summary:"
echo "  Total alpha values: $TOTAL"
echo "  Skipped (already done): $SKIPPED"
if [[ "$DRY" == "1" ]]; then
    echo "  Would submit: $SUBMITTED"
    echo ""
    echo "DRY=1; nothing was actually submitted. Re-run with DRY=0 to launch."
else
    echo "  Submitted: $SUBMITTED  (each spawns a rollout array of $WORLD_SIZE + 1 merge job)"
    echo "  -> ~$((SUBMITTED * (WORLD_SIZE + 1))) total slurm jobs queued."
    echo ""
    echo "Monitor:  squeue -u \$USER"
fi
echo "=========================================================================="
