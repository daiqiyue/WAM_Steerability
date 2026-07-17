#!/bin/bash
# Submit collect_policy_inputs_object_pairs.ipynb for headless execution on a
# single GH200 GPU. The notebook sweeps the 3 successful uncluttered scenes
# from notebooks/stress_test/rollouts/10_object_pair_selected (C01, C03, C05),
# running paired neg-drives + pos-drives campaigns per config on each
# scene's SUCCESS.mp4 episode indices.
#
# Usage:
#   ./collect_policy_inputs_object_pairs.sh
#   TIME=06:00:00 ./collect_policy_inputs_object_pairs.sh
#
# The executed notebook is written next to the input as
# collect_policy_inputs_object_pairs.executed.ipynb. sbatch stdout/stderr land
# under this directory as logs/nb_<jobid>.{out,err}.

set -euo pipefail

NB_NAME="collect_policy_inputs_object_pairs.ipynb"
# Budget: 2-pass per config to reproduce 10_object_pair_selected's exact
# pos-drives trajectories. get_action is per-call deterministic (every call
# uses seed=1 by default, fresh np.RandomState for noise + fresh
# torch.Generator for scheduler steps; no global RNG carries between calls),
# so pass 1 only needs to re-run the success-list episodes (~14 total across
# 3 configs, not all 30 per config). Pass 1: ~3 min/config. Pass 2 (env_neg
# replay + neg-drives): ~7 min/config. With 3 configs that's ~30 min
# compute + ~5 min model load + env builds. 2h is generous; drop to 1h
# (ghx4-interactive) if you want a quicker queue.
TIME="${TIME:-02:00:00}"
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
