#!/bin/bash
# Precompute VLM backbone embeddings for all observations in a paired NPZ.
# Run this ONCE before the SVD sketch so sketch ranks don't each load the 20 GB VLM.
#
# Usage:
#   POS_NPZ=.../positive.npz NEG_NPZ=.../negative.npz PROMPT="..." \
#     ./precompute_vl_embs.sh

set -euo pipefail

DIT4DIT_ROOT=/projects/bhhv/jskifstad/DiT4DiT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY="$SCRIPT_DIR/precompute_vl_embs.py"
[[ -f "$PY" ]] || { echo "ERROR: missing $PY" >&2; exit 1; }

POS_NPZ="${POS_NPZ:-}"
NEG_NPZ="${NEG_NPZ:-}"
[[ -n "$POS_NPZ" ]] || { echo "ERROR: POS_NPZ must be set" >&2; exit 1; }
[[ -n "$NEG_NPZ" ]] || { echo "ERROR: NEG_NPZ must be set" >&2; exit 1; }

PROMPT="${PROMPT:-put both the cream cheese box and the butter in the basket}"
CKPT_PATH="${CKPT_PATH:-$DIT4DIT_ROOT/checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt}"
# Default: save alongside the NPZ files
OUT_PATH="${OUT_PATH:-$(dirname "$POS_NPZ")/vl_embs__$(echo "$PROMPT" | tr ' ' '_' | cut -c1-30).pt}"
BATCH_SIZE="${BATCH_SIZE:-5}"
TIME="${TIME:-06:00:00}"
ACCOUNT="${ACCOUNT:-bhhv-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"

mkdir -p "$SCRIPT_DIR/logs"

echo "=== config ==="
echo "  pos_npz:    $POS_NPZ"
echo "  neg_npz:    $NEG_NPZ"
echo "  out_path:   $OUT_PATH"
echo "  prompt:     $PROMPT"
echo "  batch_size: $BATCH_SIZE"
echo "  time:       $TIME"
echo

sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --gpus-per-node=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time="$TIME" \
    --job-name="dit4dit_vlembs" \
    --output="$SCRIPT_DIR/logs/vlembs_%j.out" \
    --error="$SCRIPT_DIR/logs/vlembs_%j.err" \
    --wrap "
set -euo pipefail
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate /projects/bhhv/jskifstad/DiT4DiT/.conda/envs/dit4dit
export PYTHONPATH=/work/nvme/bhhv/jskifstad/LIBERO:\${PYTHONPATH:-}
export PYTHONPATH='$DIT4DIT_ROOT':\${PYTHONPATH:-}
export LIBERO_HOME=/work/nvme/bhhv/jskifstad/LIBERO
export LIBERO_CONFIG_PATH=/work/nvme/bhhv/jskifstad/LIBERO/libero
export PYTHONUTF8=1; export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
python -u '$PY' \
    --pos-npz '$POS_NPZ' \
    --neg-npz '$NEG_NPZ' \
    --out-path '$OUT_PATH' \
    --prompt '$PROMPT' \
    --ckpt-path '$CKPT_PATH' \
    --batch-size '$BATCH_SIZE'
echo 'done: $OUT_PATH'
"
