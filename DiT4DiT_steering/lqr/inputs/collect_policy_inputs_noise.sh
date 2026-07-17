#!/bin/bash
# Submit collect_policy_inputs_noise.py on a single GH200 GPU.
#
# Runs CLEAN DiT4DiT rollouts and saves positive.npz (clean) + negative.npz
# (post-hoc Gaussian noise on both cameras). The two npzs are already 1-to-1
# paired — no separate pairing step is needed before SVD.
#
# Usage:
#   ./collect_policy_inputs_noise.sh
#   TASK_ID=1 N_EPISODES=50 NOISE_SIGMA=75 ./collect_policy_inputs_noise.sh
#   OUT_DIR=/custom/path ./collect_policy_inputs_noise.sh

set -euo pipefail

DIT4DIT_ROOT=/projects/bhhv/jskifstad/DiT4DiT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY="$SCRIPT_DIR/collect_policy_inputs_noise.py"
[[ -f "$PY" ]] || { echo "ERROR: .py not found: $PY" >&2; exit 1; }

SUITE="${SUITE:-libero_10}"
TASK_ID="${TASK_ID:-1}"
N_EPISODES="${N_EPISODES:-50}"
RESOLUTION="${RESOLUTION:-256}"
NOISE_SIGMA="${NOISE_SIGMA:-75.0}"
PROMPT="${PROMPT:-}"
CKPT_PATH="${CKPT_PATH:-$DIT4DIT_ROOT/checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt}"
TIME="${TIME:-01:00:00}"
ACCOUNT="${ACCOUNT:-bhhv-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"

OUT_DIR="${OUT_DIR:-$DIT4DIT_ROOT/notebooks/lqr/inputs/policy_inputs/libero_10__task$(printf '%02d' "$TASK_ID")__noise_sigma${NOISE_SIGMA}}"

mkdir -p "$SCRIPT_DIR/logs"

echo "=== config ==="
echo "  out_dir:      $OUT_DIR"
echo "  suite:        $SUITE"
echo "  task_id:      $TASK_ID"
echo "  n_episodes:   $N_EPISODES"
echo "  noise_sigma:  $NOISE_SIGMA"
echo "  prompt:       ${PROMPT:-<env default>}"
echo "  ckpt_path:    $CKPT_PATH"
echo "  time:         $TIME"
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
    --job-name="dit4dit_collect_noise_t${TASK_ID}" \
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
python -u '$PY' \
    --out-dir '$OUT_DIR' \
    --suite '$SUITE' \
    --task-id '$TASK_ID' \
    --n-episodes '$N_EPISODES' \
    --resolution '$RESOLUTION' \
    --noise-sigma '$NOISE_SIGMA' \
    --ckpt-path '$CKPT_PATH'$PROMPT_QUOTED
echo 'done; outputs under: $OUT_DIR'
echo 'Next: run_partition_svd_pairs_no_action.sh POS_NPZ=$OUT_DIR/positive.npz NEG_NPZ=$OUT_DIR/negative.npz'
"
