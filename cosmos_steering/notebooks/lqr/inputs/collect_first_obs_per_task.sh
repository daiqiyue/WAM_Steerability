#!/bin/bash
# Submit collect_first_obs_per_task.py for headless execution on a single GH200.
# Captures the first observation of episode 0 for each libero_10 task under
# the gripper-xyz perturbation. No model load -- pure env init -- ~30 s/task.
#
# Output dir defaults to:
#   notebooks/lqr/inputs/policy_inputs/libero_10__first_obs__<PRESET>__seed<BASE_SEED>
#
# Usage:
#   ./collect_first_obs_per_task.sh
#   PRESET=xyz_random_xlarge_2 BASE_SEED=42 ./collect_first_obs_per_task.sh

set -euo pipefail

PY_NAME="${PY_NAME:-collect_first_obs_per_task.py}"
TIME="${TIME:-00:20:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY="$SCRIPT_DIR/$PY_NAME"
[[ -f "$PY" ]] || { echo "ERROR: .py not found: $PY" >&2; exit 1; }

SUITE="${SUITE:-libero_10}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
EPISODE_IDX="${EPISODE_IDX:-0}"
RESOLUTION="${RESOLUTION:-256}"
PRESET="${PRESET:-xyz_random_xlarge_2}"
BASE_SEED="${BASE_SEED:-42}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
OUT_DIR="${OUT_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__first_obs__${PRESET}__seed${BASE_SEED}}"

mkdir -p "$SCRIPT_DIR/logs"

echo "=== config ==="
echo "  out_dir:        $OUT_DIR"
echo "  suite:          $SUITE"
echo "  task_ids:       $TASK_IDS"
echo "  episode_idx:    $EPISODE_IDX"
echo "  resolution:     $RESOLUTION"
echo "  preset:         $PRESET"
echo "  base_seed:      $BASE_SEED"
echo "  num_steps_wait: $NUM_STEPS_WAIT"
echo "  time:           $TIME  partition: $PARTITION"
echo

NB_BASE="$(basename "$PY" .py)"

# NOTE: $TASK_IDS is intentionally UNQUOTED below so the inner shell splits it
# into N separate --task-ids arguments (argparse nargs="+").
JOB_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --gpus-per-node=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time="$TIME" \
    --job-name="py_${NB_BASE}_${PRESET}" \
    --output="$SCRIPT_DIR/logs/first_obs_%j.out" \
    --error="$SCRIPT_DIR/logs/first_obs_%j.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
python -u '$PY' \
    --out-dir '$OUT_DIR' \
    --suite '$SUITE' \
    --task-ids $TASK_IDS \
    --episode-idx '$EPISODE_IDX' \
    --resolution '$RESOLUTION' \
    --preset '$PRESET' \
    --base-seed '$BASE_SEED' \
    --num-steps-wait '$NUM_STEPS_WAIT'
echo 'done; outputs under: $OUT_DIR'
")

echo "submitted job: $JOB_ID"
echo "tail log: tail -f $SCRIPT_DIR/logs/first_obs_${JOB_ID}.out"
