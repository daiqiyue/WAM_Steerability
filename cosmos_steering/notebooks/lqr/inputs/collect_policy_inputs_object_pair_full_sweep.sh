#!/bin/bash
# Submit collect_policy_inputs_object_pair_full_sweep.py for headless
# execution on a single GH200 GPU. The .py sweeps the
# 10_object_pair_full_sweep stress-test rollouts and keeps only episodes
# that succeeded in the uncluttered case AND failed in the cluttered case
# (per scene pair), then runs the 2-pass contrastive collection on those.
#
# Usage:
#   ./collect_policy_inputs_object_pair_full_sweep.sh
#   TIME=03:00:00 ./collect_policy_inputs_object_pair_full_sweep.sh
#   OUT_DIR=/path/to/custom_dir MAX_CONFIGS=3 ./collect_policy_inputs_object_pair_full_sweep.sh
#
# sbatch stdout/stderr land under this directory as logs/nb_<jobid>.{out,err}.
# Live per-episode progress streams to the .err via the .py's _log helper.

set -euo pipefail

PY_NAME="${PY_NAME:-collect_policy_inputs_object_pair_full_sweep.py}"

# Budget: 10 pairs x 30 episodes total. Pass 1 + pass 2b are both ~one
# rollout per episode (~25-50 s each in the uncluttered/cluttered scenes
# respectively); pass 2a (replay) is ~3 s/episode. So expect roughly
# 30 ep * (25+50)s/ep ≈ 40 min of rollouts + ~3 min model load + per-pair
# env build/close overhead. Keep 2h headroom; ghx4 allows it.
TIME="${TIME:-02:00:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY="$SCRIPT_DIR/$PY_NAME"
[[ -f "$PY" ]] || { echo "ERROR: .py not found: $PY" >&2; exit 1; }

NB_BASE="$(basename "$PY" .py)"

# Default output dir mirrors the object_pairs naming with an extra suffix.
OUT_DIR="${OUT_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__object_pair_full_sweep__unc_succ_clu_fail}"

# Stress-test rollouts root (must contain all_results.json + per-config subdirs).
ROLLOUTS_ROOT="${ROLLOUTS_ROOT:-/u/jhong7/Workspace/cosmos-policy/notebooks/stress_test/rollouts/10_object_pair_full_sweep}"

SUITE="${SUITE:-libero_10}"
TASK_ID="${TASK_ID:-0}"
RESOLUTION="${RESOLUTION:-256}"
CONTAINER_KEY="${CONTAINER_KEY:-basket_1}"

# Optional caps (-1 = no cap). MAX_CONFIGS limits how many pairs we sweep;
# MAX_EPISODES_PER_CONFIG caps episodes within each pair. Useful for smoke
# tests before committing to the full ~40 min run.
MAX_CONFIGS="${MAX_CONFIGS:--1}"
MAX_EPISODES_PER_CONFIG="${MAX_EPISODES_PER_CONFIG:--1}"

CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"

mkdir -p "$SCRIPT_DIR/logs"

echo "=== config ==="
echo "  out_dir:                 $OUT_DIR"
echo "  rollouts_root:           $ROLLOUTS_ROOT"
echo "  suite:                   $SUITE"
echo "  task_id:                 $TASK_ID"
echo "  resolution:              $RESOLUTION"
echo "  container_key:           $CONTAINER_KEY"
echo "  max_configs:             $MAX_CONFIGS"
echo "  max_episodes_per_config: $MAX_EPISODES_PER_CONFIG"
echo "  ckpt_path:               $CKPT_PATH"
echo "  time:                    $TIME"
echo "  partition:               $PARTITION"
echo

sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --gpus-per-node=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time="$TIME" \
    --job-name="py_${NB_BASE}" \
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
    --rollouts-root '$ROLLOUTS_ROOT' \
    --out-dir '$OUT_DIR' \
    --suite '$SUITE' \
    --task-id '$TASK_ID' \
    --resolution '$RESOLUTION' \
    --container-key '$CONTAINER_KEY' \
    --max-configs '$MAX_CONFIGS' \
    --max-episodes-per-config '$MAX_EPISODES_PER_CONFIG' \
    --ckpt-path '$CKPT_PATH'
echo 'done; outputs under: $OUT_DIR'
"
