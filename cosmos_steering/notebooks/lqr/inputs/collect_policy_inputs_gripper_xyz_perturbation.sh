#!/bin/bash
# Submit collect_policy_inputs_gripper_xyz_perturbation.py for headless
# execution on a single GH200 GPU. The .py runs N gripper-xyz-perturbed
# rollouts of libero_10 task 0 under the original task prompt and buckets
# every captured policy input by rollout outcome:
#   positive.npz  = rows from successful rollouts
#   negative.npz  = rows from failed rollouts
#
# These outputs are UNPAIRED (different row counts, rows not 1-1 matched).
# Run notebooks/lqr/svd/pair_inputs_by_similarity.sh on the resulting
# directory to produce a paired, SVD-compatible version.
#
# Usage:
#   ./collect_policy_inputs_gripper_xyz_perturbation.sh
#   N_EPISODES=50 PRESET=xyz_random_xlarge_3 BASE_SEED=42 \
#     ./collect_policy_inputs_gripper_xyz_perturbation.sh
#   OUT_DIR=/path/to/custom_dir ./collect_policy_inputs_gripper_xyz_perturbation.sh
#
# sbatch stdout/stderr land under this directory as logs/nb_<jobid>.{out,err}.

set -euo pipefail

PY_NAME="${PY_NAME:-collect_policy_inputs_gripper_xyz_perturbation.py}"

# Budget: 50 episodes; each xyz_random_xlarge_3 rollout is ~15-40 s under
# get_action (failures hit the 530-step cap, ~25 s; successes ~250 steps,
# ~15 s). So expect roughly 50 * 30 s ~ 25 min of rollouts + ~3 min model
# load + env build. 1h leaves comfortable headroom; bump TIME for larger N.
TIME="${TIME:-01:00:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY="$SCRIPT_DIR/$PY_NAME"
[[ -f "$PY" ]] || { echo "ERROR: .py not found: $PY" >&2; exit 1; }

NB_BASE="$(basename "$PY" .py)"

SUITE="${SUITE:-libero_10}"
TASK_ID="${TASK_ID:-0}"
N_EPISODES="${N_EPISODES:-50}"
RESOLUTION="${RESOLUTION:-256}"
PRESET="${PRESET:-xyz_random_xlarge_3}"
BASE_SEED="${BASE_SEED:-42}"
CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"

# Optional explicit prompt; default uses the libero env's built-in description.
PROMPT="${PROMPT:-}"

# Default output dir mirrors the existing pos_neg dir naming.
OUT_DIR="${OUT_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__${PRESET}__seed${BASE_SEED}__pos_neg}"

mkdir -p "$SCRIPT_DIR/logs"

echo "=== config ==="
echo "  out_dir:      $OUT_DIR"
echo "  suite:        $SUITE"
echo "  task_id:      $TASK_ID"
echo "  n_episodes:   $N_EPISODES"
echo "  resolution:   $RESOLUTION"
echo "  preset:       $PRESET"
echo "  base_seed:    $BASE_SEED"
echo "  prompt:       ${PROMPT:-<env default>}"
echo "  ckpt_path:    $CKPT_PATH"
echo "  time:         $TIME"
echo "  partition:    $PARTITION"
echo

PROMPT_ARG=()
if [[ -n "$PROMPT" ]]; then
    PROMPT_ARG=(--prompt "$PROMPT")
fi
PROMPT_QUOTED=""
for a in "${PROMPT_ARG[@]}"; do
    PROMPT_QUOTED+=" $(printf '%q' "$a")"
done

sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --gpus-per-node=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time="$TIME" \
    --job-name="py_${NB_BASE}_${PRESET}" \
    --output="$SCRIPT_DIR/logs/nb_%j.out" \
    --error="$SCRIPT_DIR/logs/nb_%j.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
echo 'executing: $PY'
python -u '$PY' \
    --out-dir '$OUT_DIR' \
    --suite '$SUITE' \
    --task-id '$TASK_ID' \
    --n-episodes '$N_EPISODES' \
    --resolution '$RESOLUTION' \
    --preset '$PRESET' \
    --base-seed '$BASE_SEED' \
    --ckpt-path '$CKPT_PATH'$PROMPT_QUOTED
echo 'done; outputs under: $OUT_DIR'
echo 'Next: pair_inputs_by_similarity.sh IN_DIR=$OUT_DIR'
"
