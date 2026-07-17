#!/bin/bash
# Submit collect_policy_inputs_milk.ipynb for headless execution on a single
# GH200 GPU. The notebook runs two paired-rollout campaigns (neg-drives +
# pos-drives) for libero_10 task 0 under the milk-prompt, producing a paired
# positive.npz / negative.npz for contrastive-direction analysis.
#
# Usage:
#   ./collect_policy_inputs_milk.sh
#   TIME=06:00:00 ./collect_policy_inputs_milk.sh
#
# The executed notebook is written next to the input as
# collect_policy_inputs_milk.executed.ipynb. sbatch stdout/stderr land under
# this directory as logs/nb_<jobid>.{out,err}.

set -euo pipefail

NB_NAME="collect_policy_inputs_milk.ipynb"
# Default to 4h on ghx4 (non-interactive): two drive campaigns, N_EPISODES=10
# each, ~10-15 min per rollout in the cluttered scene (neg-drives runs to
# max_env_steps since the original BDDL goal won't fire for the milk prompt).
# ghx4-interactive caps below 4h, so we land on ghx4 by default — override to
# ghx4-interactive if you trim TIME under that cap.
TIME="${TIME:-04:00:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
NB="$SCRIPT_DIR/$NB_NAME"
[[ -f "$NB" ]] || { echo "ERROR: notebook not found: $NB" >&2; exit 1; }

NB_DIR="$(dirname "$NB")"
NB_BASE="$(basename "$NB" .ipynb)"
OUT_NB="${NB_BASE}.executed.ipynb"

mkdir -p "$SCRIPT_DIR/logs"

sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --gpus-per-node=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time="$TIME" \
    --job-name="nb_${NB_BASE}" \
    --output="$SCRIPT_DIR/logs/nb_%j.out" \
    --error="$SCRIPT_DIR/logs/nb_%j.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$NB_DIR'
nvidia-smi -L
echo 'executing: $NB'
jupyter nbconvert --to notebook --execute '$NB' \
    --ExecutePreprocessor.kernel_name=cosmos-policy \
    --ExecutePreprocessor.timeout=14400 \
    --output '$OUT_NB'
echo 'wrote: $NB_DIR/$OUT_NB'
"
