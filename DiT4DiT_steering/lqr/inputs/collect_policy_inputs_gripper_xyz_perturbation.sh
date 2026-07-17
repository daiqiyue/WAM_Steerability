#!/bin/bash
# Submit collect_policy_inputs_gripper_xyz_perturbation.py for headless
# execution on a single GH200 GPU with the DiT4DiT model.
#
# Analogue of the repo root's notebooks/lqr/inputs/collect_policy_inputs_gripper_xyz_perturbation.sh
# but uses DiT4DiT instead of Cosmos-Policy.
#
# Usage:
#   ./collect_policy_inputs_gripper_xyz_perturbation.sh
#   N_EPISODES=50 PRESET=xyz_random_xlarge_3 BASE_SEED=42 \
#     ./collect_policy_inputs_gripper_xyz_perturbation.sh
#   OUT_DIR=/path/to/custom_dir ./collect_policy_inputs_gripper_xyz_perturbation.sh

set -euo pipefail

DIT4DIT_ROOT=/projects/bhhv/jskifstad/DiT4DiT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_NAME="${PY_NAME:-collect_policy_inputs_gripper_xyz_perturbation.py}"
PY="$SCRIPT_DIR/$PY_NAME"
[[ -f "$PY" ]] || { echo "ERROR: .py not found: $PY" >&2; exit 1; }

NB_BASE="$(basename "$PY" .py)"

SUITE="${SUITE:-libero_10}"
TASK_ID="${TASK_ID:-0}"
N_EPISODES="${N_EPISODES:-50}"
RESOLUTION="${RESOLUTION:-256}"
PRESET="${PRESET:-xyz_random_xlarge_3}"
BASE_SEED="${BASE_SEED:-42}"
CKPT_PATH="${CKPT_PATH:-/projects/bhhv/jskifstad/DiT4DiT/checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt}"
PROMPT="${PROMPT:-}"
TIME="${TIME:-01:00:00}"
ACCOUNT="${ACCOUNT:-bhhv-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"

OUT_DIR="${OUT_DIR:-/projects/bhhv/jskifstad/DiT4DiT/notebooks/lqr/inputs/policy_inputs/libero_10__task00__${PRESET}__seed${BASE_SEED}__pos_neg}"

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
[[ -n "$PROMPT" ]] && PROMPT_ARG=(--prompt "$PROMPT")
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
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate /projects/bhhv/jskifstad/DiT4DiT/.conda/envs/dit4dit
export PYTHONPATH=/work/nvme/bhhv/jskifstad/LIBERO:${PYTHONPATH:-}
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH='$DIT4DIT_ROOT':\${PYTHONPATH:-}
export LIBERO_HOME=/work/nvme/bhhv/jskifstad/LIBERO
export LIBERO_CONFIG_PATH=/work/nvme/bhhv/jskifstad/LIBERO/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
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
echo 'Next: pair_inputs_by_similarity.sh IN_DIR=\$OUT_DIR'
"
