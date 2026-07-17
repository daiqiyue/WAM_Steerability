#!/bin/bash
# Sweep driver for run_lqr_cosmos_policy_noised.sh.
#
# Enumerates the LAMBDA x Q_SCALE x R_SCALE x R_SCALE_TAU grid for the noised
# A-LQR rollout, computes the expected output config tag, and:
#   - skips combos whose <ROLLOUTS_ROOT>/<tag>/results.json already exists,
#   - submits the missing ones via the inner driver (which submits its own
#     sbatch array + merge job pair),
#   - queues a final aggregator sbatch dependent on every merge job in this
#     sweep; the aggregator writes/refreshes the summary markdown.
#
# Usage:
#   ./sweep_lqr_noised.sh
#   DRY_RUN=1 ./sweep_lqr_noised.sh                 # enumerate only, no sbatch
#   NOISE_SEED_BASE=99 ./sweep_lqr_noised.sh        # decorrelate from contrastive seed
#   N_EPISODES=10 ./sweep_lqr_noised.sh             # forwarded to every job
#   WORLD_SIZE=8 ./sweep_lqr_noised.sh              # forwarded to every job
#
# The sweep grid is defined by the LAMBDAS / Q_SCALES / R_INITS / TAUS arrays
# below — edit them in-place to widen / shrink the sweep.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DRIVER="$SCRIPT_DIR/run_lqr_cosmos_policy_noised.sh"
AGGREGATOR_PY="$SCRIPT_DIR/aggregate_lqr_noised.py"
[[ -x "$DRIVER" ]] || chmod +x "$DRIVER" 2>/dev/null || true
[[ -f "$DRIVER" ]] || { echo "ERROR: driver missing: $DRIVER" >&2; exit 1; }
[[ -f "$AGGREGATOR_PY" ]] || { echo "ERROR: aggregator .py missing: $AGGREGATOR_PY" >&2; exit 1; }

DRY_RUN="${DRY_RUN:-0}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-1}"

# ----- sweep grid (edit here) ---------------------------------------------
LAMBDAS=(0.5 5.0 10.0 15.0)
Q_SCALES=(1 10 100)
R_INITS=(5 10 20)
TAUS=(3 5 7)
# --------------------------------------------------------------------------

# Fixed knobs (override via env; everything else falls through to the driver).
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"
SUITE="${SUITE:-libero_10}"
TASK_ID="${TASK_ID:-0}"
N_EPISODES="${N_EPISODES:-50}"
WORLD_SIZE="${WORLD_SIZE:-4}"
RESOLUTION="${RESOLUTION:-256}"
R_SCALE_FINAL="${R_SCALE_FINAL:-1e9}"
QF_SCALE="${QF_SCALE:-1.0}"
NOISE_SIGMA="${NOISE_SIGMA:-75.0}"
NOISE_SEED_BASE="${NOISE_SEED_BASE:-99}"   # default for this sweep: seed 99
NOISE_PER_EPISODE_SEED="${NOISE_PER_EPISODE_SEED:-1}"
NO_OVERRIDE_FIRST_CHUNK="${NO_OVERRIDE_FIRST_CHUNK:-1}"
OBS_INDEX="${OBS_INDEX:-0}"
RUN_BASELINE="${RUN_BASELINE:-0}"
RUN_TAG="${RUN_TAG:-}"
TAG="${TAG:-noise_seed${NOISE_SEED_BASE}}"
SUMMARY_MD="${SUMMARY_MD:-$SCRIPT_DIR/noise_seed${NOISE_SEED_BASE}_sweep.md}"

# Default PAIR_DIR / SVD_DIR / JAC_DIR_ACT match the inner driver. Override here
# if you want a different jacobian linearization point.
PAIR_DIR="${PAIR_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_high_4_pos_neg}"
SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_noise_high_4_pairs_no_action_dsall_N-1_k64_p10_ws8_tsall}"
JAC_DIR_ACT="${JAC_DIR_ACT:-A_tilde_full__put_both_the_alphabet_soup_and_the_tomato_sauce_in__vjp_no_retain__vbf16}"

OUT_BASE="${OUT_BASE:-$SCRIPT_DIR/rollouts}"
ROLLOUTS_ROOT="$OUT_BASE/$TAG"

# _slug(): mirror the inner sh _slug exactly (lowercase, ' ' -> '_', drop chars
# not in [a-z0-9_-]).
_slug() {
    local s="${1:0:24}"
    s=$(printf '%s' "$s" | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_-')
    printf '%s' "$s"
}
PROMPT_SLUG=$(_slug "$PROMPT")

# _config_tag(): replicate CONFIG_TAG from run_lqr_cosmos_policy_noised.sh:
#   NOISE_TAG = "s${NOISE_SIGMA}_obs${OBS_INDEX}_sb${NOISE_SEED_BASE}[_noov]"
#   LQR_TAG   = "lam${LAMBDA}_q${Q_SCALE}_rinit${R_SCALE}_rfin${R_SCALE_FINAL}_tau${R_SCALE_TAU}_qf${QF_SCALE}"
#   CONFIG    = "${SUITE}__task00__lqr_noised_decay__${NOISE_TAG}__${LQR_TAG}__${PROMPT_SLUG}[__${RUN_TAG}]"
_config_tag() {
    local lam="$1" q="$2" ri="$3" tau="$4"
    local noov=""
    [[ "$NO_OVERRIDE_FIRST_CHUNK" == "1" ]] && noov="_noov"
    local noise_tag="s${NOISE_SIGMA}_obs${OBS_INDEX}_sb${NOISE_SEED_BASE}${noov}"
    local lqr_tag="lam${lam}_q${q}_rinit${ri}_rfin${R_SCALE_FINAL}_tau${tau}_qf${QF_SCALE}"
    local task_str
    task_str=$(printf 'task%02d' "$TASK_ID")
    local base="${SUITE}__${task_str}__lqr_noised_decay__${noise_tag}__${lqr_tag}__${PROMPT_SLUG}"
    if [[ -n "$RUN_TAG" ]]; then
        printf '%s__%s' "$base" "$RUN_TAG"
    else
        printf '%s' "$base"
    fi
}

n_total=$(( ${#LAMBDAS[@]} * ${#Q_SCALES[@]} * ${#R_INITS[@]} * ${#TAUS[@]} ))
echo "=== sweep configuration ==="
echo "  grid: ${#LAMBDAS[@]} lambdas × ${#Q_SCALES[@]} q × ${#R_INITS[@]} rinit × ${#TAUS[@]} tau = $n_total combinations"
echo "  fixed: r_final=$R_SCALE_FINAL  qf=$QF_SCALE  N_EPISODES=$N_EPISODES  WORLD_SIZE=$WORLD_SIZE"
echo "         sigma=$NOISE_SIGMA  seed_base=$NOISE_SEED_BASE  obs_index=$OBS_INDEX  no_override=$NO_OVERRIDE_FIRST_CHUNK"
echo "         prompt_slug=$PROMPT_SLUG  tag=$TAG  run_tag=${RUN_TAG:-<none>}"
echo "  rollouts_root=$ROLLOUTS_ROOT"
echo "  summary_md=$SUMMARY_MD"
echo "  dry_run=$DRY_RUN"
echo

mkdir -p "$SCRIPT_DIR/logs"

submitted=0
skipped=0
failed_submit=0
merge_ids=()

for lam in "${LAMBDAS[@]}"; do
  for q in "${Q_SCALES[@]}"; do
    for ri in "${R_INITS[@]}"; do
      for tau in "${TAUS[@]}"; do
        tag=$(_config_tag "$lam" "$q" "$ri" "$tau")
        target="$ROLLOUTS_ROOT/$tag/results.json"
        label="lam=$lam q=$q rinit=$ri tau=$tau"
        if [[ -f "$target" ]]; then
            echo "  SKIP    $label  (results.json exists at $target)"
            ((skipped += 1))
            continue
        fi
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  WOULD   $label  ->  $tag"
            ((submitted += 1))
            continue
        fi
        echo "  SUBMIT  $label  ->  $tag"
        # Inner driver prints "rollout job id: N  (array ...)" and
        # "merge job id:   N  (afterok:...)". We parse the merge id for chaining.
        out=$(LAMBDA="$lam" Q_SCALE="$q" R_SCALE="$ri" R_SCALE_TAU="$tau" \
              R_SCALE_FINAL="$R_SCALE_FINAL" QF_SCALE="$QF_SCALE" \
              N_EPISODES="$N_EPISODES" WORLD_SIZE="$WORLD_SIZE" \
              RESOLUTION="$RESOLUTION" \
              NOISE_SIGMA="$NOISE_SIGMA" NOISE_SEED_BASE="$NOISE_SEED_BASE" \
              NOISE_PER_EPISODE_SEED="$NOISE_PER_EPISODE_SEED" \
              NO_OVERRIDE_FIRST_CHUNK="$NO_OVERRIDE_FIRST_CHUNK" \
              OBS_INDEX="$OBS_INDEX" \
              PROMPT="$PROMPT" SUITE="$SUITE" TASK_ID="$TASK_ID" \
              RUN_BASELINE="$RUN_BASELINE" \
              PAIR_DIR="$PAIR_DIR" SVD_DIR="$SVD_DIR" JAC_DIR_ACT="$JAC_DIR_ACT" \
              TAG="$TAG" RUN_TAG="$RUN_TAG" OUT_BASE="$OUT_BASE" \
              bash "$DRIVER" 2>&1) || { echo "$out"; echo "  ERROR  inner driver failed for $label"; ((failed_submit += 1)); continue; }
        mid=$(printf '%s\n' "$out" | grep -oE 'merge job id:[[:space:]]+[0-9]+' | awk '{print $NF}' | tail -1)
        rid=$(printf '%s\n' "$out" | grep -oE 'rollout job id:[[:space:]]+[0-9]+' | awk '{print $NF}' | tail -1)
        if [[ -n "$mid" ]]; then
            merge_ids+=("$mid")
            echo "    rollout=$rid  merge=$mid"
            ((submitted += 1))
        else
            echo "$out"
            echo "  WARN   could not parse merge job id for $label"
            ((failed_submit += 1))
        fi
        sleep "$SLEEP_BETWEEN"
      done
    done
  done
done

echo
echo "=== sweep summary ==="
echo "  submitted:     $submitted"
echo "  skipped:       $skipped"
echo "  failed submit: $failed_submit"

if [[ "$DRY_RUN" == "1" ]]; then
    echo
    echo "DRY_RUN=1; nothing submitted."
    echo "After a real run, regenerate the summary md manually with:"
    echo "  python $AGGREGATOR_PY --rollouts-root $ROLLOUTS_ROOT --summary-md $SUMMARY_MD --noise-seed-base $NOISE_SEED_BASE"
    exit 0
fi

# Queue the aggregator. Even with zero new submissions we still kick it off
# so the markdown stays in sync with whatever's already complete.
AGG_NAME="lqrnoise_sweep_agg_sb${NOISE_SEED_BASE}"
if [[ ${#merge_ids[@]} -gt 0 ]]; then
    dep="afterany:$(IFS=:; printf '%s' "${merge_ids[*]}")"
    echo
    echo "=== submitting aggregator dependent on ${#merge_ids[@]} merge job(s) ==="
    agg_id=$(sbatch --parsable \
        --account=bhde-dtai-gh \
        --partition=ghx4 \
        --gpus-per-task=1 \
        --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:20:00 \
        --dependency="$dep" \
        --job-name="$AGG_NAME" \
        --output="$SCRIPT_DIR/logs/sweep_noised_agg_%j.out" \
        --error="$SCRIPT_DIR/logs/sweep_noised_agg_%j.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
python -u '$AGGREGATOR_PY' \
    --rollouts-root '$ROLLOUTS_ROOT' \
    --summary-md '$SUMMARY_MD' \
    --noise-seed-base '$NOISE_SEED_BASE' \
    ${RUN_TAG:+--run-tag '$RUN_TAG'}
")
    echo "  aggregator job id: $agg_id"
    echo "  summary md will land at: $SUMMARY_MD"
else
    echo
    echo "=== no new submissions; running aggregator immediately on existing results ==="
    set +e
    python -u "$AGGREGATOR_PY" \
        --rollouts-root "$ROLLOUTS_ROOT" \
        --summary-md "$SUMMARY_MD" \
        --noise-seed-base "$NOISE_SEED_BASE" \
        ${RUN_TAG:+--run-tag "$RUN_TAG"}
    rc=$?
    set -e
    [[ $rc -ne 0 ]] && echo "  WARN: aggregator exited with $rc"
fi
