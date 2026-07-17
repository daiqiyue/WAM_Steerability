#!/bin/bash
# Refined v2 sweep around the best-of-v1 region (λ=10, Q=10, R_init=20, τ=7).
# Same skip-if-done / sbatch-array / aggregator scaffolding as
# sweep_lqr_decay_cam_perturb.sh, just with a tighter grid centered on the
# v1 top performer (and pushed past the upper edges of v1 on R_init / τ
# because the v1 best sat right at the corner of the grid).
#
# v1 was 4×3×3×3=108 combos covering λ∈{0.5,5,10,15} Q∈{1,10,100} R_init∈{5,10,20}
# τ∈{3,5,7}. v2 is 3×3×3×3=81 combos at λ∈{7,10,13} Q∈{5,10,20} R_init∈{20,30,50}
# τ∈{7,10,15}; the (10,10,20,7) corner overlaps v1 and is auto-skipped.
#
# Outputs share the cam_perturb rollouts dir with v1, and the shared aggregator
# regenerates camera_position_sweep.md (v1 + v2 combined) when this sweep's
# finalize jobs all complete.
#
# Usage / overrides identical to v1:
#   ./sweep_lqr_decay_cam_perturb_v2.sh
#   DRY_RUN=1 ./sweep_lqr_decay_cam_perturb_v2.sh
#   TIME=05:00:00 ./sweep_lqr_decay_cam_perturb_v2.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DRIVER="$SCRIPT_DIR/run_lqr_decay_cosmos_policy_cam_perturb.sh"
AGGREGATOR_PY="$SCRIPT_DIR/aggregate_lqr_decay_cam_perturb.py"
[[ -x "$DRIVER" ]] || { echo "ERROR: driver missing/not executable: $DRIVER" >&2; exit 1; }
[[ -f "$AGGREGATOR_PY" ]] || { echo "ERROR: aggregator .py missing: $AGGREGATOR_PY" >&2; exit 1; }

DRY_RUN="${DRY_RUN:-0}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-1}"
SUMMARY_MD="${SUMMARY_MD:-$SCRIPT_DIR/camera_position_sweep.md}"

# ----- sweep grid (refined v2 — centered on v1 top performer) -------------
# v1 best was (λ=10, Q=10, R_init=20, τ=7) at 68% success. R_init and τ both
# sat at the v1 upper edge, so this grid pushes past them and adds finer
# samples around the λ / Q peak. Adjust TIME=05:00:00 if you see TIMEOUTs.
LAMBDAS=(7 10 13)
Q_SCALES=(5 10 20)
R_INITS=(20 30 50)
TAUS=(7 10 15)
# --------------------------------------------------------------------------

# Fixed knobs (override via env; everything else falls through to the driver).
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"
SUITE_NAME="${SUITE_NAME:-libero_10}"
TASK_ID="${TASK_ID:-0}"
N_EPISODES="${N_EPISODES:-50}"
RESOLUTION="${RESOLUTION:-256}"
CAM_MODE="${CAM_MODE:-random}"
CAM_BASE_SEED_DEFAULT_FOR_MODE() {
    case "$1" in
        match)  echo 42 ;;
        random) echo 99 ;;
        off)    echo 0  ;;
        *)      echo "ERROR: unknown CAM_MODE=$1" >&2; exit 1 ;;
    esac
}
CAM_BASE_SEED="${CAM_BASE_SEED:-$(CAM_BASE_SEED_DEFAULT_FOR_MODE "$CAM_MODE")}"
R_SCALE_FINAL="${R_SCALE_FINAL:-1e9}"
QF_SCALE="${QF_SCALE:-1.0}"
RUN_TAG="${RUN_TAG:-cam_perturb_decay}"
ROLLOUT_SUBDIR="${ROLLOUT_SUBDIR:-cam_perturb}"
SIDE_BY_SIDE_VIDEO="${SIDE_BY_SIDE_VIDEO:-1}"
RUN_BASELINE="${RUN_BASELINE:-0}"

ROLLOUTS_ROOT="$SCRIPT_DIR/rollouts/$ROLLOUT_SUBDIR"

# _slug(): replicate the Python _slug in run_lqr_decay_cosmos_policy_cam_perturb.py
# (first 24 chars, lowercased, spaces -> "_", "/" -> "_", "." -> "_").
_slug() {
    local s="${1:0:24}"
    s=$(printf '%s' "$s" | tr '[:upper:]' '[:lower:]' \
        | tr ' ' '_' | tr '/' '_' | tr '.' '_')
    printf '%s' "$s"
}
PROMPT_SLUG=$(_slug "$PROMPT")

# _config_tag(): replicate _compute_config_tag in the Python script.
_config_tag() {
    local lam_fmt q_fmt ri_fmt rf_fmt tau_fmt qf_fmt
    lam_fmt=$(printf '%.2f' "$1")
    q_fmt=$(printf '%g' "$2")
    ri_fmt=$(printf '%g' "$3")
    rf_fmt=$(printf '%g' "$R_SCALE_FINAL")
    tau_fmt=$(printf '%g' "$4")
    qf_fmt=$(printf '%g' "$QF_SCALE")
    local base="${SUITE_NAME}__task$(printf '%02d' "$TASK_ID")__lqr_decay_camperturb__"
    base+="lam${lam_fmt}_q${q_fmt}_rinit${ri_fmt}_rfin${rf_fmt}_tau${tau_fmt}_qf${qf_fmt}__"
    base+="cam${CAM_MODE}_seed${CAM_BASE_SEED}__${PROMPT_SLUG}"
    if [[ -n "$RUN_TAG" ]]; then
        printf '%s__%s' "$base" "$RUN_TAG"
    else
        printf '%s' "$base"
    fi
}

n_total=$(( ${#LAMBDAS[@]} * ${#Q_SCALES[@]} * ${#R_INITS[@]} * ${#TAUS[@]} ))
echo "=== sweep configuration ==="
echo "  grid: ${#LAMBDAS[@]} lambdas × ${#Q_SCALES[@]} q × ${#R_INITS[@]} rinit × ${#TAUS[@]} tau = $n_total combinations"
echo "  fixed: r_final=$R_SCALE_FINAL  qf=$QF_SCALE  N_EPISODES=$N_EPISODES"
echo "         cam_mode=$CAM_MODE  cam_base_seed=$CAM_BASE_SEED  prompt_slug=$PROMPT_SLUG"
echo "         rollouts_root=$ROLLOUTS_ROOT  run_tag=$RUN_TAG  side_by_side=$SIDE_BY_SIDE_VIDEO"
echo "  dry_run=$DRY_RUN  summary_md=$SUMMARY_MD"
echo

mkdir -p "$SCRIPT_DIR/logs"

submitted=0
skipped=0
failed_submit=0
finalize_ids=()

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
        # The inner driver prints "rollouts job id: N" and "finalize job id: N";
        # we parse the latter so we can chain the aggregator after all finalizes.
        out=$(LAMBDA="$lam" Q_SCALE="$q" R_SCALE="$ri" R_SCALE_TAU="$tau" \
              R_SCALE_FINAL="$R_SCALE_FINAL" QF_SCALE="$QF_SCALE" \
              N_EPISODES="$N_EPISODES" RESOLUTION="$RESOLUTION" \
              RUN_TAG="$RUN_TAG" ROLLOUT_SUBDIR="$ROLLOUT_SUBDIR" \
              CAM_MODE="$CAM_MODE" CAM_BASE_SEED="$CAM_BASE_SEED" \
              PROMPT="$PROMPT" SUITE_NAME="$SUITE_NAME" TASK_ID="$TASK_ID" \
              SIDE_BY_SIDE_VIDEO="$SIDE_BY_SIDE_VIDEO" RUN_BASELINE="$RUN_BASELINE" \
              "$DRIVER" 2>&1) || { echo "$out"; echo "  ERROR  inner driver failed for $label"; ((failed_submit += 1)); continue; }
        fid=$(printf '%s\n' "$out" | grep -oE 'finalize job id: [0-9]+' | awk '{print $NF}' | tail -1)
        rid=$(printf '%s\n' "$out" | grep -oE 'rollouts job id: [0-9]+'  | awk '{print $NF}' | tail -1)
        if [[ -n "$fid" ]]; then
            finalize_ids+=("$fid")
            echo "    rollout=$rid  finalize=$fid"
            ((submitted += 1))
        else
            echo "$out"
            echo "  WARN   could not parse finalize job id for $label"
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

# Queue the aggregator. Even with zero new submissions we still kick it off
# (un-conditional `if [ -z $DRY_RUN ]`) so the markdown stays in sync with
# whatever's already complete.
if [[ "$DRY_RUN" == "1" ]]; then
    echo
    echo "DRY_RUN=1; nothing submitted."
    echo "After a real run, regenerate the summary md manually with:"
    echo "  python $AGGREGATOR_PY --rollouts-root $ROLLOUTS_ROOT --summary-md $SUMMARY_MD"
    exit 0
fi

mkdir -p "$SCRIPT_DIR/logs"
AGG_NAME="cps_lqr_sweep_agg"
if [[ ${#finalize_ids[@]} -gt 0 ]]; then
    dep="afterany:$(IFS=:; printf '%s' "${finalize_ids[*]}")"
    echo
    echo "=== submitting aggregator dependent on ${#finalize_ids[@]} finalize job(s) ==="
    agg_id=$(sbatch --parsable \
        --account=bhde-dtai-gh \
        --partition=ghx4 \
        --gpus-per-task=1 \
        --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 \
        --dependency="$dep" \
        --job-name="$AGG_NAME" \
        --output="$SCRIPT_DIR/logs/sweep_agg_%j.out" \
        --error="$SCRIPT_DIR/logs/sweep_agg_%j.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
python -u '$AGGREGATOR_PY' \
    --rollouts-root '$ROLLOUTS_ROOT' \
    --summary-md '$SUMMARY_MD' \
    --cam-mode '$CAM_MODE' --cam-seed '$CAM_BASE_SEED' --run-tag '$RUN_TAG'
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
        --cam-mode "$CAM_MODE" --cam-seed "$CAM_BASE_SEED" --run-tag "$RUN_TAG"
    rc=$?
    set -e
    [[ $rc -ne 0 ]] && echo "  WARN: aggregator exited with $rc"
fi
