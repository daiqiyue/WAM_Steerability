#!/bin/bash
# End-to-end LQR sweep driver for the noise_extreme pipeline. Runs the
# exact same (λ, Q, r_init, τ) hyperparameter sweep on multiple libero_10
# tasks. The contrastive vectors (c_means in svd_summary.pt) and the
# projection matrices (V_part*_*.pt) are SHARED across tasks via a single
# SVD_DIR; only the per-task jacobian sub-dir (JAC_DIR_ACT) and the
# prompt change.
#
# Per task (default TASK_IDS=(0 1)):
#   1. Submit the unsteered baseline (run_baseline_noised.sh)
#   2. Submit the LQR hyperparameter sweep (sweep_lqr_noised.sh) over the
#      same (λ, Q, r_init, τ) grid as noise_seed99_sweep.md
#      (4 × 3 × 3 × 3 = 108 combos), at noise_seed_base=99 so all
#      perturbations are unseen w.r.t. the contrastive collection (seed 0)
#   3. Submit a final aggregator job that runs aggregate_lqr_noised.py with
#      --baseline-dir set, dependent on both the baseline merge job and
#      the sweep's built-in aggregator, so the final markdown contains
#      the baseline row + all sweep results
#
# Per-task output:
#   rollouts root: $OUT_BASE/${BASE_TAG}_task{NN}/
#   baseline dir : $OUT_BASE/${BASE_TAG}_task{NN}/__baseline__/
#   summary md   : notebooks/lqr/${BASE_TAG}_task{NN}_sweep.md
#
# Sweep grid matches sweep_lqr_noised.sh:
#   LAMBDAS=(0.5 5.0 10.0 15.0)
#   Q_SCALES=(1 10 100)
#   R_INITS=(5 10 20)
#   TAUS=(3 5 7)
#   fixed: r_fin=1e9, qf=1, obs_index=0
#
# Usage:
#   ./run_sweep_noise_extreme_seed99.sh                       # tasks 0 + 1
#   TASK_IDS="0"   ./run_sweep_noise_extreme_seed99.sh        # task 0 only
#   TASK_IDS="1"   ./run_sweep_noise_extreme_seed99.sh        # task 1 only
#   TASK_IDS="0 1" ./run_sweep_noise_extreme_seed99.sh        # both (default)
#   N_EPISODES=50 WORLD_SIZE=4 ./run_sweep_noise_extreme_seed99.sh
#   DRY_RUN=1 ./run_sweep_noise_extreme_seed99.sh             # enumerate only
#   SKIP_BASELINE=1 ./run_sweep_noise_extreme_seed99.sh
#   SKIP_SWEEP=1    ./run_sweep_noise_extreme_seed99.sh
#   AGGREGATOR_ONLY=1 ./run_sweep_noise_extreme_seed99.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASELINE_DRIVER="$SCRIPT_DIR/run_baseline_noised.sh"
SWEEP_DRIVER="$SCRIPT_DIR/sweep_lqr_noised.sh"
AGGREGATOR_PY="$SCRIPT_DIR/aggregate_lqr_noised.py"
[[ -f "$BASELINE_DRIVER" ]] || { echo "ERROR: missing $BASELINE_DRIVER" >&2; exit 1; }
[[ -f "$SWEEP_DRIVER"    ]] || { echo "ERROR: missing $SWEEP_DRIVER" >&2; exit 1; }
[[ -f "$AGGREGATOR_PY"   ]] || { echo "ERROR: missing $AGGREGATOR_PY" >&2; exit 1; }

# -------------------------- shared SVD (c_means + V) -----------------------
SVD_BASE="${SVD_BASE:-/projects/bhde/jhong7/cosmos-policy/directions/svd}"
SVD_TAG="${SVD_TAG:-libero10_task00_noise_extreme_pairs_no_action_no_pool_dsall_N-1_k64_p10_ws8_tsall}"
SVD_DIR="${SVD_DIR:-$SVD_BASE/$SVD_TAG}"

# -------------------------- per-task spec ----------------------------------
# task_id -> (prompt, jac_dir_act, pair_dir). JAC_DIR_ACT is the
# prompt-slugged dir under SVD_DIR that contains A_tilde__full.pt.
declare -A TASK_PROMPTS=(
    [0]="put both the alphabet soup and the tomato sauce in the basket"
    [1]="put both the cream cheese box and the butter in the basket"
)
declare -A TASK_JAC_DIRS=(
    [0]="A_tilde_full__put_both_the_alphabet_soup_and_the_tomato_sauce_in__vjp_no_retain__vbf16"
    [1]="A_tilde_full__put_both_the_cream_cheese_box_and_the_butter_in_th__vjp_no_retain__vbf16"
)
declare -A TASK_PAIR_DIRS=(
    [0]="/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_extreme_pos_neg"
    [1]="/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task01__noise_extreme_pos_neg"
)

# TASK_IDS env override (space-separated list); default = both tasks.
read -r -a TASK_IDS <<< "${TASK_IDS:-0 1}"

for tid in "${TASK_IDS[@]}"; do
    if [[ -z "${TASK_PROMPTS[$tid]+x}" ]]; then
        echo "ERROR: no entry for task_id=$tid in TASK_PROMPTS/TASK_JAC_DIRS/TASK_PAIR_DIRS" >&2
        exit 1
    fi
done

# -------------------------- runtime knobs ----------------------------------
SUITE="${SUITE:-libero_10}"
N_EPISODES="${N_EPISODES:-50}"
WORLD_SIZE="${WORLD_SIZE:-4}"
RESOLUTION="${RESOLUTION:-256}"
NOISE_SIGMA="${NOISE_SIGMA:-90.0}"      # noise_extreme
NOISE_SEED_BASE="${NOISE_SEED_BASE:-99}"
NOISE_PER_EPISODE_SEED="${NOISE_PER_EPISODE_SEED:-1}"
NO_OVERRIDE_FIRST_CHUNK="${NO_OVERRIDE_FIRST_CHUNK:-1}"
OBS_INDEX="${OBS_INDEX:-0}"
RUN_BASELINE="${RUN_BASELINE:-0}"

# Per-rank memory. Bumped from the inner driver's 64G default because the
# no-pool SVD has 28 × 5 = 140 V_part files (~30 GB CPU resident) which
# pushed peak past 64 GB and OOM-killed the prior submission. Export so
# both sweep_lqr_noised.sh -> run_lqr_cosmos_policy_noised.sh and
# run_baseline_noised.sh pick it up via env inheritance.
export MEM="${MEM:-128G}"

# -------------------------- output layout ----------------------------------
BASE_TAG="${BASE_TAG:-noise_extreme_seed${NOISE_SEED_BASE}}"
OUT_BASE="${OUT_BASE:-$SCRIPT_DIR/rollouts}"

# -------------------------- control flags ----------------------------------
DRY_RUN="${DRY_RUN:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
AGGREGATOR_ONLY="${AGGREGATOR_ONLY:-0}"
RUN_TAG="${RUN_TAG:-}"

# -------------------------- slurm knobs (for the final aggregator) --------
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
AGG_TIME="${AGG_TIME:-00:20:00}"

mkdir -p "$SCRIPT_DIR/logs"

echo "=== noise_extreme LQR sweep driver (multi-task) ==="
echo "  SVD_DIR (shared)      : $SVD_DIR"
echo "  TASK_IDS              : ${TASK_IDS[*]}"
echo "  NOISE_SIGMA           : $NOISE_SIGMA  (noise_extreme = 90)"
echo "  NOISE_SEED_BASE       : $NOISE_SEED_BASE"
echo "  N_EPISODES            : $N_EPISODES"
echo "  WORLD_SIZE            : $WORLD_SIZE  (per-config)"
echo "  BASE_TAG              : $BASE_TAG"
echo "  OUT_BASE              : $OUT_BASE"
echo "  MEM (per-rank)        : $MEM"
echo "  flags                 : DRY_RUN=$DRY_RUN  SKIP_BASELINE=$SKIP_BASELINE  SKIP_SWEEP=$SKIP_SWEEP  AGGREGATOR_ONLY=$AGGREGATOR_ONLY"
echo

# Sanity checks (only when actually submitting work).
if [[ "$DRY_RUN" != "1" && "$AGGREGATOR_ONLY" != "1" ]]; then
    [[ -d "$SVD_DIR" ]] || { echo "ERROR: SVD_DIR not found: $SVD_DIR" >&2; exit 1; }
    for tid in "${TASK_IDS[@]}"; do
        jac_path="$SVD_DIR/${TASK_JAC_DIRS[$tid]}/A_tilde__full.pt"
        pair_neg="${TASK_PAIR_DIRS[$tid]}/negative.npz"
        [[ -f "$jac_path" ]] || {
            echo "ERROR: task $tid: missing $jac_path" >&2
            echo "       (run run_jacobians_noise_extreme.sh for that task first)" >&2
            exit 1
        }
        [[ -f "$pair_neg" ]] || {
            echo "ERROR: task $tid: missing $pair_neg" >&2
            exit 1
        }
    done
fi

# Returns 0 if a numeric job id was captured, 1 otherwise.
_extract_job_id() {
    local label="$1" file="$2"
    grep -oE "${label}:[[:space:]]+[0-9]+" "$file" | tail -1 | awk '{print $NF}'
}

# ==================== per-task loop ========================================
for tid in "${TASK_IDS[@]}"; do
    PROMPT="${TASK_PROMPTS[$tid]}"
    JAC_DIR_ACT="${TASK_JAC_DIRS[$tid]}"
    PAIR_DIR="${TASK_PAIR_DIRS[$tid]}"
    TASK_ID="$tid"
    TASK_STR=$(printf "task%02d" "$tid")
    TAG="${BASE_TAG}_${TASK_STR}"
    ROLLOUTS_ROOT="$OUT_BASE/$TAG"
    BASELINE_DIR="$ROLLOUTS_ROOT/__baseline__"
    SUMMARY_MD="$SCRIPT_DIR/${TAG}_sweep.md"

    echo "================================================================"
    echo "=== TASK $tid ($TASK_STR) ==="
    echo "================================================================"
    echo "  PROMPT                : $PROMPT"
    echo "  JAC_DIR_ACT           : $JAC_DIR_ACT"
    echo "  PAIR_DIR              : $PAIR_DIR"
    echo "  TAG                   : $TAG"
    echo "  ROLLOUTS_ROOT         : $ROLLOUTS_ROOT"
    echo "  BASELINE_DIR          : $BASELINE_DIR"
    echo "  SUMMARY_MD            : $SUMMARY_MD"
    echo

    baseline_merge_id=""
    sweep_agg_id=""

    # ---------------- baseline ----------------
    if [[ "$SKIP_BASELINE" == "1" || "$AGGREGATOR_ONLY" == "1" ]]; then
        echo "[skip baseline]"
    elif [[ -f "$BASELINE_DIR/results.json" ]]; then
        echo "[skip baseline] $BASELINE_DIR/results.json already exists"
    elif [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY] would submit baseline -> $BASELINE_DIR"
    else
        echo "=== submitting baseline (run_baseline_noised.sh) ==="
        baseline_out=$(mktemp)
        WORLD_SIZE="$WORLD_SIZE" N_EPISODES="$N_EPISODES" \
        PROMPT="$PROMPT" SUITE="$SUITE" TASK_ID="$TASK_ID" \
        RESOLUTION="$RESOLUTION" \
        NOISE_SIGMA="$NOISE_SIGMA" NOISE_SEED_BASE="$NOISE_SEED_BASE" \
        NOISE_PER_EPISODE_SEED="$NOISE_PER_EPISODE_SEED" \
        NO_OVERRIDE_FIRST_CHUNK="$NO_OVERRIDE_FIRST_CHUNK" OBS_INDEX="$OBS_INDEX" \
        SVD_DIR="$SVD_DIR" JAC_DIR_ACT="$JAC_DIR_ACT" PAIR_DIR="$PAIR_DIR" \
        TAG="$TAG" OUT_BASE="$OUT_BASE" OUT_DIR="$BASELINE_DIR" \
            bash "$BASELINE_DRIVER" 2>&1 | tee "$baseline_out"
        baseline_merge_id=$(_extract_job_id "merge job id" "$baseline_out")
        rm -f "$baseline_out"
        if [[ -n "$baseline_merge_id" ]]; then
            echo "  baseline merge job id: $baseline_merge_id"
        else
            echo "  WARN: could not capture baseline merge job id (continuing)"
        fi
        echo
    fi

    # ---------------- sweep ----------------
    if [[ "$SKIP_SWEEP" == "1" || "$AGGREGATOR_ONLY" == "1" ]]; then
        echo "[skip sweep]"
    elif [[ "$DRY_RUN" == "1" ]]; then
        echo "=== sweep dry run (DRY_RUN=1) ==="
        DRY_RUN=1 \
        PROMPT="$PROMPT" SUITE="$SUITE" TASK_ID="$TASK_ID" \
        N_EPISODES="$N_EPISODES" WORLD_SIZE="$WORLD_SIZE" \
        RESOLUTION="$RESOLUTION" \
        NOISE_SIGMA="$NOISE_SIGMA" NOISE_SEED_BASE="$NOISE_SEED_BASE" \
        NOISE_PER_EPISODE_SEED="$NOISE_PER_EPISODE_SEED" \
        NO_OVERRIDE_FIRST_CHUNK="$NO_OVERRIDE_FIRST_CHUNK" OBS_INDEX="$OBS_INDEX" \
        RUN_BASELINE="$RUN_BASELINE" \
        PAIR_DIR="$PAIR_DIR" SVD_DIR="$SVD_DIR" JAC_DIR_ACT="$JAC_DIR_ACT" \
        TAG="$TAG" RUN_TAG="$RUN_TAG" OUT_BASE="$OUT_BASE" \
        SUMMARY_MD="$SUMMARY_MD" \
            bash "$SWEEP_DRIVER"
    else
        echo "=== submitting sweep (sweep_lqr_noised.sh) ==="
        sweep_out=$(mktemp)
        PROMPT="$PROMPT" SUITE="$SUITE" TASK_ID="$TASK_ID" \
        N_EPISODES="$N_EPISODES" WORLD_SIZE="$WORLD_SIZE" \
        RESOLUTION="$RESOLUTION" \
        NOISE_SIGMA="$NOISE_SIGMA" NOISE_SEED_BASE="$NOISE_SEED_BASE" \
        NOISE_PER_EPISODE_SEED="$NOISE_PER_EPISODE_SEED" \
        NO_OVERRIDE_FIRST_CHUNK="$NO_OVERRIDE_FIRST_CHUNK" OBS_INDEX="$OBS_INDEX" \
        RUN_BASELINE="$RUN_BASELINE" \
        PAIR_DIR="$PAIR_DIR" SVD_DIR="$SVD_DIR" JAC_DIR_ACT="$JAC_DIR_ACT" \
        TAG="$TAG" RUN_TAG="$RUN_TAG" OUT_BASE="$OUT_BASE" \
        SUMMARY_MD="$SUMMARY_MD" \
            bash "$SWEEP_DRIVER" 2>&1 | tee "$sweep_out"
        sweep_agg_id=$(_extract_job_id "aggregator job id" "$sweep_out")
        rm -f "$sweep_out"
        if [[ -n "$sweep_agg_id" ]]; then
            echo "  sweep aggregator job id: $sweep_agg_id"
        else
            echo "  WARN: could not capture sweep aggregator job id (continuing)"
        fi
        echo
    fi

    # ---------------- final aggregator with --baseline-dir ----------------
    # Re-runs aggregate_lqr_noised.py with --baseline-dir so the final
    # summary md gets the baseline row. The sweep's own internal aggregator
    # already produced a copy without --baseline-dir; this one overwrites
    # it. Even on AGGREGATOR_ONLY/SKIP_*, we still submit (no deps) so the
    # md is refreshed from whatever results.json files already exist.
    deps=()
    [[ -n "$baseline_merge_id" ]] && deps+=("afterany:$baseline_merge_id")
    [[ -n "$sweep_agg_id"      ]] && deps+=("afterany:$sweep_agg_id")
    dep_arg=""
    if [[ ${#deps[@]} -gt 0 ]]; then
        dep_arg="--dependency=$(IFS=,; printf '%s' "${deps[*]}")"
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY] would submit final aggregator with $dep_arg"
        echo "      python $AGGREGATOR_PY --rollouts-root $ROLLOUTS_ROOT --summary-md $SUMMARY_MD --noise-seed-base $NOISE_SEED_BASE --baseline-dir $BASELINE_DIR"
        echo
        continue
    fi

    echo "=== submitting final aggregator for $TASK_STR (with --baseline-dir) ==="
    final_agg_id=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION_SLURM" \
        --gpus-per-task=1 \
        --ntasks=1 --cpus-per-task=2 --mem=8G --time="$AGG_TIME" \
        $dep_arg \
        --job-name="agg_${TAG:0:48}" \
        --output="$SCRIPT_DIR/logs/agg_noise_extreme_${TASK_STR}_%j.out" \
        --error="$SCRIPT_DIR/logs/agg_noise_extreme_${TASK_STR}_%j.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
python -u '$AGGREGATOR_PY' \
    --rollouts-root '$ROLLOUTS_ROOT' \
    --summary-md   '$SUMMARY_MD' \
    --noise-seed-base '$NOISE_SEED_BASE' \
    --baseline-dir '$BASELINE_DIR' \
    ${RUN_TAG:+--run-tag '$RUN_TAG'}
")
    echo "  final aggregator job id: $final_agg_id"
    echo "  deps                   : ${dep_arg:-<none>}"
    echo "  rollouts root          : $ROLLOUTS_ROOT"
    echo "  summary md             : $SUMMARY_MD"
    echo "  tail log               : tail -f $SCRIPT_DIR/logs/agg_noise_extreme_${TASK_STR}_${final_agg_id}.out"
    echo
done

echo "=== all tasks submitted ==="
echo "Monitor: squeue -u \$USER"
