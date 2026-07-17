#!/bin/bash
# Wrapper for run_jacobians_full.sh that points at the noise_extreme SVD
# output and uses row 0 of negative.npz (first inference of the first
# clean rollout, with Gaussian σ=90 noise applied) as the linearization
# obs. The prompt is task 0's original instruction.
#
# Usage:
#   ./run_jacobians_noise_extreme.sh
#   WORLD_SIZE=8 ./run_jacobians_noise_extreme.sh
#   MERGE_ONLY=1 ./run_jacobians_noise_extreme.sh   # re-merge existing shards

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
INNER="$SCRIPT_DIR/run_jacobians_full.sh"
[[ -f "$INNER" ]] || { echo "ERROR: missing inner driver: $INNER" >&2; exit 1; }

# -------------------------- noise_extreme specifics -----------------------
PAIR_DIR="${PAIR_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_extreme_pos_neg}"
INPUTS_NPZ="${INPUTS_NPZ:-$PAIR_DIR/negative.npz}"
OBS_INDEX="${OBS_INDEX:-0}"
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"

# SVD_DIR mirrors what run_partition_svd_pairs_noise_extreme.sh produces.
# RUN_TAG composition (from inner SVD driver):
#   ${TAG}_N${N}_k${K_TARGET}_p${P_OVER}_ws${WORLD_SIZE}_ts${TS_TAG}
# with our defaults TAG=libero10_task00_noise_extreme_pairs_no_action_no_pool_dsall,
# N=-1, K_TARGET=64, P_OVER=10, WORLD_SIZE=8, TIMESTEPS=all.
SVD_BASE="${SVD_BASE:-/projects/bhde/jhong7/cosmos-policy/directions/svd}"
SVD_TAG="${SVD_TAG:-libero10_task00_noise_extreme_pairs_no_action_no_pool_dsall_N-1_k64_p10_ws8_tsall}"
SVD_DIR="${SVD_DIR:-$SVD_BASE/$SVD_TAG}"

[[ -f "$INPUTS_NPZ" ]] || { echo "ERROR: missing $INPUTS_NPZ" >&2; exit 1; }
[[ -d "$SVD_DIR"    ]] || { echo "ERROR: missing SVD_DIR $SVD_DIR" >&2; exit 1; }

# -------------------------- forwarded knobs -------------------------------
WORLD_SIZE="${WORLD_SIZE:-8}"
MODE="${MODE:-vjp_no_retain}"
V_DEVICE="${V_DEVICE:-cpu}"
V_DTYPE="${V_DTYPE:-bf16}"
WORKER_TIME="${WORKER_TIME:-00:40:00}"
MERGE_ONLY="${MERGE_ONLY:-0}"

echo "=== noise_extreme jacobians wrapper ==="
echo "  PAIR_DIR     : $PAIR_DIR"
echo "  INPUTS_NPZ   : $INPUTS_NPZ (row $OBS_INDEX)"
echo "  PROMPT       : $PROMPT"
echo "  SVD_DIR      : $SVD_DIR"
echo "  WORLD_SIZE   : $WORLD_SIZE"
echo "  MODE         : $MODE"
echo "  V_DEVICE     : $V_DEVICE  V_DTYPE: $V_DTYPE"
echo "  WORKER_TIME  : $WORKER_TIME"
echo "  MERGE_ONLY   : $MERGE_ONLY"
echo

WORLD_SIZE="$WORLD_SIZE" PROMPT="$PROMPT" INPUTS_NPZ="$INPUTS_NPZ" OBS_INDEX="$OBS_INDEX" \
SVD_DIR="$SVD_DIR" MODE="$MODE" V_DEVICE="$V_DEVICE" V_DTYPE="$V_DTYPE" \
WORKER_TIME="$WORKER_TIME" MERGE_ONLY="$MERGE_ONLY" \
    bash "$INNER"
