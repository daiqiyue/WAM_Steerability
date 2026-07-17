#!/bin/bash
# Slurm launcher for compute_contrastive_per_j.py — postprocess that adds
# per-inference-index contrastive vectors to an existing SVD output dir
# produced by run_partition_svd_pairs.sh.
#
# Reuses the V matrices on disk (no re-SVD); just re-runs the paired forward
# passes from positive.npz / negative.npz and accumulates the (pos - neg)
# activation difference separately per inference_idx j, then projects through
# V once at the end.
#
# Output:  $SVD_DIR/c_means_per_j.pt
#
# Defaults target the milk_first_chunks SVD dir that already exists.
#
# Usage:
#   ./compute_contrastive_per_j.sh
#   SVD_DIR=/path/to/other_svd ./compute_contrastive_per_j.sh
#   PROMPT="..." DRIVE_SOURCE=0 ./compute_contrastive_per_j.sh

set -euo pipefail

# =========================================================================
# User-configurable (override via env)
# =========================================================================
SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_milk_first_chunks_dsall_N-1_k64_p10_ws8_tsall}"

# Empty -> use the value recorded in $SVD_DIR/config.json.
POS_NPZ="${POS_NPZ:-}"
NEG_NPZ="${NEG_NPZ:-}"
PROMPT="${PROMPT:-}"
DRIVE_SOURCE="${DRIVE_SOURCE:-}"
N="${N:-}"
OUTPUT_NAME="${OUTPUT_NAME:-c_means_per_j.pt}"
V_DEVICE="${V_DEVICE:-auto}"

# Slurm resources
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4-interactive}"
TIME="${TIME:-00:30:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
# shellcheck disable=SC1091
. "$(dirname -- "${BASH_SOURCE[0]}")/../../exclude_nodes.sh"
EXCLUDE_NODES="${EXCLUDE_NODES-$EXCLUDE_NODES_DEFAULT}"
# =========================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PY_SCRIPT="$SCRIPT_DIR/compute_contrastive_per_j.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
[[ -d "$SVD_DIR"   ]] || { echo "ERROR: missing SVD_DIR $SVD_DIR" >&2; exit 1; }

LOG_DIR="$SVD_DIR/logs"
mkdir -p "$LOG_DIR"

PY_ARGS=( --svd-dir "$SVD_DIR" --output-name "$OUTPUT_NAME" --v-device "$V_DEVICE" )
[[ -n "$POS_NPZ"      ]] && PY_ARGS+=( --pos-npz "$POS_NPZ" )
[[ -n "$NEG_NPZ"      ]] && PY_ARGS+=( --neg-npz "$NEG_NPZ" )
[[ -n "$PROMPT"       ]] && PY_ARGS+=( --prompt "$PROMPT" )
[[ -n "$DRIVE_SOURCE" ]] && PY_ARGS+=( --drive-source "$DRIVE_SOURCE" )
[[ -n "$N"            ]] && PY_ARGS+=( --N "$N" )

ARGS_QUOTED=""
for a in "${PY_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

echo "=== config ==="
echo "  svd_dir:     $SVD_DIR"
echo "  pos_npz:     ${POS_NPZ:-<from svd-dir config.json>}"
echo "  neg_npz:     ${NEG_NPZ:-<from svd-dir config.json>}"
echo "  prompt:      ${PROMPT:-<from svd-dir config.json>}"
echo "  drive:       ${DRIVE_SOURCE:-<from svd-dir config.json>}"
echo "  N:           ${N:-<from svd-dir config.json>}"
echo "  output:      $SVD_DIR/$OUTPUT_NAME"
echo "  v_device:    $V_DEVICE"
echo "  account=$ACCOUNT  partition=$PARTITION_SLURM  time=$TIME"
echo "  exclude_nodes=${EXCLUDE_NODES:-<none>}"
echo

# Forward POS_NPZ etc. through the env so the wrapped bash -c can see them
# (sbatch --export=ALL inherits the calling shell's environment).
export SVD_DIR POS_NPZ NEG_NPZ PROMPT DRIVE_SOURCE N OUTPUT_NAME V_DEVICE

sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --gpus-per-node=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --time="$TIME" \
    --job-name="cpj_$(basename "$SVD_DIR")" \
    --output="$LOG_DIR/per_j_%j.out" \
    --error="$LOG_DIR/per_j_%j.err" \
    --export=ALL \
    ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
python -u '$PY_SCRIPT'$ARGS_QUOTED
"
