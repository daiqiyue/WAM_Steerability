#!/usr/bin/env bash
# Submit precompute_vl_embs in parallel across N_WORKERS GPUs, then merge.
# Usage:
#   POS_NPZ=... NEG_NPZ=... OUT_PATH=... PROMPT="..." bash submit_precompute_parallel.sh
#   N_WORKERS=4 ... bash submit_precompute_parallel.sh   # override worker count

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/precompute_vl_embs.py"

DIT4DIT_ROOT="${DIT4DIT_ROOT:-/projects/bhhv/jskifstad/DiT4DiT}"
MODEL_PYTHON="${MODEL_PYTHON:-$DIT4DIT_ROOT/.conda/envs/dit4dit/bin/python}"
CKPT_PATH="${CKPT_PATH:-$DIT4DIT_ROOT/checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt}"

# Match the pipeline's path construction (run_noised_pipeline.sh)
TASK_ID="${TASK_ID:-1}"
NOISE_SIGMA="${NOISE_SIGMA:-75.0}"
PROMPT="${PROMPT:-put both the cream cheese box and the butter in the basket}"

LQR_ROOT="$DIT4DIT_ROOT/notebooks/lqr"
_INPUT_DIR="$LQR_ROOT/inputs/policy_inputs/libero_10__task$(printf '%02d' "$TASK_ID")__noise_sigma${NOISE_SIGMA}"
_PROMPT_TAG="$(echo "$PROMPT" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-30)"

POS_NPZ="${POS_NPZ:-$_INPUT_DIR/positive.npz}"
NEG_NPZ="${NEG_NPZ:-$_INPUT_DIR/negative.npz}"
OUT_PATH="${OUT_PATH:-$_INPUT_DIR/vl_embs__${_PROMPT_TAG}.pt}"
BATCH_SIZE="${BATCH_SIZE:-5}"
N_WORKERS="${N_WORKERS:-8}"
ACCOUNT="${ACCOUNT:-bhhv-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
WORKER_TIME="${WORKER_TIME:-01:00:00}"   # 1 h per worker; ~488 obs @ batch=5
MERGE_TIME="${MERGE_TIME:-00:10:00}"
MERGE_MEM="${MERGE_MEM:-64G}"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== parallel precompute ==="
echo "  n_workers:  $N_WORKERS"
echo "  batch_size: $BATCH_SIZE"
echo "  pos_npz:    $POS_NPZ"
echo "  neg_npz:    $NEG_NPZ"
echo "  out_path:   $OUT_PATH"
echo "  prompt:     $PROMPT"

# ── Submit worker jobs ────────────────────────────────────────────────────────
WORKER_IDS=()
for (( w=0; w<N_WORKERS; w++ )); do
    job_out=$(sbatch \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --qos="${ACCOUNT}" \
        --gpus-per-node=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=128G \
        --time="$WORKER_TIME" \
        --job-name="vlembs_w${w}of${N_WORKERS}" \
        --output="$LOG_DIR/vlembs_w${w}of${N_WORKERS}_%j.out" \
        --error="$LOG_DIR/vlembs_w${w}of${N_WORKERS}_%j.err" \
        --wrap="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH='$DIT4DIT_ROOT':\${PYTHONPATH:-}
export LIBERO_HOME=/work/nvme/bhhv/jskifstad/LIBERO
export LIBERO_CONFIG_PATH=\$LIBERO_HOME/libero
export PYTHONUTF8=1; export PYTHONIOENCODING=utf-8
nvidia-smi -L
'$MODEL_PYTHON' -u '$PY' \
    --pos-npz '$POS_NPZ' \
    --neg-npz '$NEG_NPZ' \
    --out-path '$OUT_PATH' \
    --prompt '$PROMPT' \
    --ckpt-path '$CKPT_PATH' \
    --batch-size '$BATCH_SIZE' \
    --worker-id $w \
    --n-workers '$N_WORKERS'
")
    job_id=$(echo "$job_out" | grep -oE '[0-9]+$')
    WORKER_IDS+=("$job_id")
    echo "  submitted worker $w → job $job_id"
done

# ── Submit merge job (runs after all workers succeed) ─────────────────────────
DEPENDENCY=$(IFS=:; echo "afterok:${WORKER_IDS[*]}")
merge_out=$(sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --qos="${ACCOUNT}" \
    --gpus-per-node=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="$MERGE_MEM" \
    --time="$MERGE_TIME" \
    --job-name="vlembs_merge" \
    --output="$LOG_DIR/vlembs_merge_%j.out" \
    --error="$LOG_DIR/vlembs_merge_%j.err" \
    --dependency="$DEPENDENCY" \
    --wrap="
export PYTHONPATH='$DIT4DIT_ROOT':\${PYTHONPATH:-}
export PYTHONUTF8=1; export PYTHONIOENCODING=utf-8
'$MODEL_PYTHON' -u '$PY' \
    --pos-npz '$POS_NPZ' \
    --neg-npz '$NEG_NPZ' \
    --out-path '$OUT_PATH' \
    --prompt '$PROMPT' \
    --ckpt-path '$CKPT_PATH' \
    --n-workers '$N_WORKERS' \
    --merge && echo 'merge done: $OUT_PATH'
")
merge_id=$(echo "$merge_out" | grep -oE '[0-9]+$')
echo "  submitted merge → job $merge_id (depends on ${WORKER_IDS[*]})"
echo ""
echo "Final output will be at: $OUT_PATH"
echo "Monitor: squeue --me"
