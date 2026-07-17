#!/bin/bash
# Run pair_inputs_by_similarity.py to convert unpaired positive.npz /
# negative.npz (e.g. from
# notebooks/lqr/inputs/collect_policy_inputs_gripper_xyz_perturbation.py)
# into a paired pair compatible with run_partition_svd_pairs*.py.
#
# CPU-only and fast (a few seconds for a few hundred rows). Runs directly
# on the login node by default; pass --sbatch to submit instead.
#
# Usage:
#   IN_DIR=/path/to/<collector-output> ./pair_inputs_by_similarity.sh
#   IN_DIR=... OUT_DIR=... MATCH_MODE=optimal FEATURE=proprio+wrist \
#     ./pair_inputs_by_similarity.sh
#   ./pair_inputs_by_similarity.sh --sbatch   # submit through slurm
#
# Defaults assume the gripper_xyz collection output dir.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY="$SCRIPT_DIR/pair_inputs_by_similarity.py"
[[ -f "$PY" ]] || { echo "ERROR: missing $PY" >&2; exit 1; }

# Default input dir mirrors collect_policy_inputs_gripper_xyz_perturbation.sh
PRESET="${PRESET:-xyz_random_xlarge_3}"
BASE_SEED="${BASE_SEED:-42}"
IN_DIR="${IN_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__${PRESET}__seed${BASE_SEED}__pos_neg}"
OUT_DIR="${OUT_DIR:-${IN_DIR%/}__paired}"

FEATURE="${FEATURE:-proprio}"          # proprio | proprio_raw | proprio+wrist
IMAGE_BLOCK="${IMAGE_BLOCK:-16}"
MATCH_MODE="${MATCH_MODE:-nn-greedy}"  # nn-replace | nn-greedy | optimal
MAX_ROWS="${MAX_ROWS:--1}"
MAX_DISTANCE="${MAX_DISTANCE:--1.0}"
# Set WITHIN_TASK=1 to constrain matching to within-task (requires task_id
# column in both npz; from collect_policy_inputs_*_multitask.py).
WITHIN_TASK="${WITHIN_TASK:-0}"

USE_SBATCH=0
if [[ "${1:-}" == "--sbatch" ]]; then
    USE_SBATCH=1
fi

echo "=== config ==="
echo "  in_dir:       $IN_DIR"
echo "  out_dir:      $OUT_DIR"
echo "  feature:      $FEATURE"
echo "  image_block:  $IMAGE_BLOCK"
echo "  match_mode:   $MATCH_MODE"
echo "  max_rows:     $MAX_ROWS"
echo "  max_distance: $MAX_DISTANCE"
echo "  within_task:  $WITHIN_TASK"
echo

[[ -f "$IN_DIR/positive.npz" ]] || { echo "ERROR: $IN_DIR/positive.npz not found" >&2; exit 1; }
[[ -f "$IN_DIR/negative.npz" ]] || { echo "ERROR: $IN_DIR/negative.npz not found" >&2; exit 1; }

CMD=(
    python -u "$PY"
    --in-dir "$IN_DIR"
    --out-dir "$OUT_DIR"
    --feature "$FEATURE"
    --image-block "$IMAGE_BLOCK"
    --match-mode "$MATCH_MODE"
    --max-rows "$MAX_ROWS"
    --max-distance "$MAX_DISTANCE"
)
if [[ "$WITHIN_TASK" == "1" || "$WITHIN_TASK" == "true" ]]; then
    CMD+=( --within-task )
fi

if [[ "$USE_SBATCH" == "1" ]]; then
    ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
    PARTITION="${PARTITION:-ghx4}"
    TIME="${TIME:-00:15:00}"
    mkdir -p "$SCRIPT_DIR/logs"
    ARGS_QUOTED=""
    for a in "${CMD[@]:1}"; do
        ARGS_QUOTED+=" $(printf '%q' "$a")"
    done
    sbatch \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time="$TIME" \
        --job-name="pair_inputs_by_similarity" \
        --output="$SCRIPT_DIR/logs/pair_%j.out" \
        --error="$SCRIPT_DIR/logs/pair_%j.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
python -u '$PY'$ARGS_QUOTED
"
else
    echo "=== running locally ==="
    "${CMD[@]}"
fi

echo
echo "Next: drive the SVD pipeline against the paired output:"
echo "  POS_NPZ='$OUT_DIR/positive.npz' \\"
echo "  NEG_NPZ='$OUT_DIR/negative.npz' \\"
echo "  TAG='libero10_task00_gripper_xyz_${PRESET}_seed${BASE_SEED}_paired' \\"
echo "    $SCRIPT_DIR/run_partition_svd_pairs_no_action.sh"
