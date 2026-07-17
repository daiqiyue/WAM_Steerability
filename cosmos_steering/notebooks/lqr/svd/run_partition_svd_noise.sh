#!/bin/bash
# Slurm launcher for Cosmos-Policy-LIBERO-Predict2-2B partition-SVD with
# *clean vs gaussian-noised* paired observation contrast.
#
# Inputs: two paired npz files (positive.npz, negative.npz) produced by
#   notebooks/lqr/inputs/collect_policy_inputs_noise.ipynb. Each row in
#   positive.npz is a contrastive pair with the same row in negative.npz —
#   same proprio, different camera content (clean vs noise_extreme σ=90).
#
# This is a thin wrapper around run_partition_svd_pairs.py with defaults
# pointing at the noise pair and the original libero_10 task 0 prompt.
# All other knobs (WORLD_SIZE, K_TARGET, PARTITIONS, ...) are inherited
# unchanged from run_partition_svd_pairs.sh — override via env vars.
#
# Usage:
#   ./run_partition_svd_noise.sh
#   WORLD_SIZE=8 N=480 ./run_partition_svd_noise.sh
#
# Output is plug-in compatible with notebooks/lqr/jacobians/compute_jacobians_*.sh:
# point their SVD_DIR at OUT_DIR and the existing Jacobian scripts run unchanged.

set -euo pipefail

# =========================================================================
# Pair-specific defaults (everything else inherits run_partition_svd_pairs.sh)
# =========================================================================
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"

POS_NPZ="${POS_NPZ:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_extreme_pos_neg/positive.npz}"
NEG_NPZ="${NEG_NPZ:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_extreme_pos_neg/negative.npz}"
DRIVE_SOURCE="${DRIVE_SOURCE:-all}"   # only one drive campaign in this pair; 'all' uses every row

# Tag is encoded into RUN_TAG by run_partition_svd_pairs.sh (along with N/k/p/ws/ts).
TAG="${TAG:-libero10_task00_noise_extreme_pairs_ds${DRIVE_SOURCE}}"

# Output base: /u has space; /work/nvme/bhde was retired (see
# notebooks/lqr/jacobians/run_jacobians_full.sh:48-49).
OUT_BASE="${OUT_BASE:-/u/jhong7/cosmos-policy/directions/svd}"

# =========================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
INNER="$SCRIPT_DIR/run_partition_svd_pairs.sh"
[[ -x "$INNER" ]] || { echo "ERROR: missing $INNER" >&2; exit 1; }

export PROMPT POS_NPZ NEG_NPZ DRIVE_SOURCE TAG OUT_BASE
exec "$INNER" "$@"
