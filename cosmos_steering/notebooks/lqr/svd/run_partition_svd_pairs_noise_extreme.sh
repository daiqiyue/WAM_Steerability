#!/bin/bash
# Wrapper for run_partition_svd_pairs_no_action.sh that points the SVD
# pipeline at the noise_extreme (sigma=90) pos/neg pair and requests
# *no* layer pooling — i.e. 28 single-layer partitions, so every DiT
# block gets its own projection matrix V and (downstream) its own
# contrastive vector c.
#
# Default PARTITIONS = "0-0,1-1,...,27-27" (one layer per partition).
#
# Memory note
# -----------
# Going from the default 3-partition layout to 28 single-layer partitions
# multiplies the sketch-phase host-RAM footprint by ~9x (each partition
# pre-allocates its own W bf16 sketch of shape (k+p, D_flat) ~= 74 * 2M
# = ~300MB plus mu, Y, e_buf). With D_flat from the LIBERO 2B config this
# pushes the sketch ranks into the ~40-60 GB host-RAM range. SBATCH_MEM
# below requests 128G per rank to leave comfortable headroom; bump
# SBATCH_MEM if a sketch rank OOMs.
#
# Usage:
#   ./run_partition_svd_pairs_noise_extreme.sh
#   PAIR_DIR=/path/to/other_pos_neg ./run_partition_svd_pairs_noise_extreme.sh
#   WORLD_SIZE=4 ./run_partition_svd_pairs_noise_extreme.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
INNER="$SCRIPT_DIR/run_partition_svd_pairs_no_action.sh"
[[ -f "$INNER" ]] || { echo "ERROR: missing inner driver: $INNER" >&2; exit 1; }

# -------------------------- noise_extreme specifics -----------------------
PAIR_DIR="${PAIR_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_extreme_pos_neg}"
POS_NPZ="${POS_NPZ:-$PAIR_DIR/positive.npz}"
NEG_NPZ="${NEG_NPZ:-$PAIR_DIR/negative.npz}"
[[ -f "$POS_NPZ" ]] || { echo "ERROR: missing $POS_NPZ" >&2; exit 1; }
[[ -f "$NEG_NPZ" ]] || { echo "ERROR: missing $NEG_NPZ" >&2; exit 1; }

PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"
TAG="${TAG:-libero10_task00_noise_extreme_pairs_no_action_no_pool_ds${DRIVE_SOURCE:-all}}"

# -------------------------- no layer pooling (28 single-layer partitions) -
NUM_LAYERS="${NUM_LAYERS:-28}"
_default_partitions() {
    local n="$1"
    local parts=""
    for ((i = 0; i < n; i++)); do
        if [[ -z "$parts" ]]; then
            parts="${i}-${i}"
        else
            parts="${parts},${i}-${i}"
        fi
    done
    printf '%s' "$parts"
}
PARTITIONS="${PARTITIONS:-$(_default_partitions "$NUM_LAYERS")}"

# -------------------------- forwarded knobs -------------------------------
WORLD_SIZE="${WORLD_SIZE:-8}"
N="${N:--1}"
K_TARGET="${K_TARGET:-64}"
P_OVER="${P_OVER:-10}"
DRIVE_SOURCE="${DRIVE_SOURCE:-all}"
SAMPLING_STEPS="${SAMPLING_STEPS:-5}"
TIMESTEPS="${TIMESTEPS:-all}"
GUIDE_SCALE="${GUIDE_SCALE:-1.0}"
CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
SKETCH_TIME="${SKETCH_TIME:-2:00:00}"
SVD_TIME="${SVD_TIME:-02:00:00}"
SBATCH_MEM="${SBATCH_MEM:-128G}"

# -------------------------- output storage (override via env) -------------
# Default OUT_BASE is /projects/bhde/jhong7 (367 GB free, 417 GB allocation
# headroom). /u is over-quota; /work is filesystem-level full. The inner
# script writes SCRATCH_DIR at $OUT_DIR/scratch by default, which keeps
# both final artifacts and intermediate ~240 GB sketch dumps on /projects.
OUT_BASE="${OUT_BASE:-/projects/bhde/jhong7/cosmos-policy/directions/svd}"
SCRATCH_DIR="${SCRATCH_DIR:-}"   # empty -> inner script falls back to OUT_DIR/scratch

echo "=== noise_extreme SVD wrapper ==="
echo "  PAIR_DIR     : $PAIR_DIR"
echo "  POS_NPZ      : $POS_NPZ"
echo "  NEG_NPZ      : $NEG_NPZ"
echo "  PROMPT       : $PROMPT"
echo "  TAG          : $TAG"
echo "  PARTITIONS   : $PARTITIONS"
echo "  NUM_LAYERS   : $NUM_LAYERS  (no layer pooling -> 1 layer / partition)"
echo "  WORLD_SIZE   : $WORLD_SIZE"
echo "  DRIVE_SOURCE : $DRIVE_SOURCE"
echo "  OUT_BASE     : $OUT_BASE"
echo "  SCRATCH_DIR  : ${SCRATCH_DIR:-<OUT_DIR/scratch>}"
echo "  SBATCH_MEM   : $SBATCH_MEM"
echo "  SKETCH_TIME  : $SKETCH_TIME"
echo "  SVD_TIME     : $SVD_TIME"
echo

mkdir -p "$OUT_BASE"

# The inner script doesn't have a --mem knob, so we patch sbatch via the
# SBATCH_MEM env var that slurm picks up (SBATCH_MEM applies to all sbatch
# calls in this shell session unless overridden inline).
export SBATCH_MEM

WORLD_SIZE="$WORLD_SIZE" N="$N" K_TARGET="$K_TARGET" P_OVER="$P_OVER" \
PARTITIONS="$PARTITIONS" NUM_LAYERS="$NUM_LAYERS" \
SAMPLING_STEPS="$SAMPLING_STEPS" TIMESTEPS="$TIMESTEPS" GUIDE_SCALE="$GUIDE_SCALE" \
PROMPT="$PROMPT" \
POS_NPZ="$POS_NPZ" NEG_NPZ="$NEG_NPZ" DRIVE_SOURCE="$DRIVE_SOURCE" \
TAG="$TAG" CKPT_PATH="$CKPT_PATH" \
OUT_BASE="$OUT_BASE" SCRATCH_DIR="$SCRATCH_DIR" \
SKETCH_TIME="$SKETCH_TIME" SVD_TIME="$SVD_TIME" \
    bash "$INNER"
