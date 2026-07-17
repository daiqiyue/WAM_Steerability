#!/bin/bash
# Submit collect_policy_inputs_milk_first_chunks.ipynb for headless execution
# on a single GH200 GPU. The notebook runs INDEPENDENT rollouts in env_pos
# (uncluttered, milk_only) and env_neg (cluttered, full 8-object) for
# libero_10 task 0 under the milk prompt, captures the first J inferences of
# each rollout, and pairs them by (episode i, inference j) -> (I(u,i,j),
# I(c,i,j)).
#
# Usage:
#   ./collect_policy_inputs_milk_first_chunks.sh
#   TIME=02:00:00 ./collect_policy_inputs_milk_first_chunks.sh
#
# The executed notebook is written next to the input as
# collect_policy_inputs_milk_first_chunks.executed.ipynb. sbatch stdout/stderr
# land under this directory as logs/nb_<jobid>.{out,err}.

set -euo pipefail

NB_NAME="collect_policy_inputs_milk_first_chunks.ipynb"
# 1h is plenty: J_THRESHOLD=8 per rollout, 2 rollouts per episode, 10
# episodes => 20 rollouts. Each rollout stops after ~8 inferences (~128 sim
# steps) so wall time is bounded well below the original milk notebook's
# 4-hour budget, which runs both rollouts to env-step cap.
TIME="${TIME:-01:00:00}"
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
