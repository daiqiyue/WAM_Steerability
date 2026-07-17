#!/bin/bash
# Submit a single LQR run with RUN_BASELINE=1 so the unsteered (policy-only)
# baseline rate gets recorded for the task08 seed=99 sweep. The 78 configs
# launched by ./run_task08_gripper_xyz_pipeline.sh ran with the default
# RUN_BASELINE=0, so their `results.json` has `"baseline": null` and the
# summarizer's "Baseline (unsteered) success" section is empty.
#
# This run:
#   - reuses one (lambda, q, rinit, tau) tuple — the same (15, 10, 5, 3)
#     point that gripper_xyz_sweep_summary.md used for its seed=99 baseline row
#   - writes to a NEW dir with a __baseline suffix (via RUN_TAG=baseline),
#     so the existing un-suffixed dir for the same config is untouched
#   - exposes the per-episode baseline outcome as the `baseline` field
#     in results.json; summarize_lqr_gripper_xyz_sweep.py will then surface
#     it in the "Baseline (unsteered) success" section.
#
# Runtime: ~50 episodes × ~60s (≈2× a normal rollout because every episode
# runs steered + baseline) / WORLD_SIZE=4 ≈ 15 min on ghx4.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LQR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Match the upstream artifacts the e2e pipeline produced.
export TASK_ID=8
export SUITE=libero_10
export PROMPT="put both moka pots on the stove"
export PRESET=xyz_random_xlarge_3
export SVD_DIR="/work/hdd/bhde/jhong7/cosmos-policy/directions/svd/libero10_task08_gripper_xyz_xyz_random_xlarge_3_seed42_paired_N-1_k64_p10_ws8_tsall"
# Derived the same way run_jacobians_full.sh does (trailing _ from echo \n).
_PROMPT_TAG=$(echo "$PROMPT" | tr ' ' '_' | tr -c 'A-Za-z0-9_' '_' | cut -c1-50)
export JAC_DIR_ACT="A_tilde_full__${_PROMPT_TAG}__vjp_no_retain__vbf16"

TAG=task08_seed99 \
RUN_TAG=baseline \
RUN_BASELINE=1 \
WORLD_SIZE=4 \
N_EPISODES=50 \
PRESET=$PRESET \
BASE_SEED=99 \
LAMBDA=15.0 \
Q_SCALE=10.0 \
R_SCALE=5.0 \
R_SCALE_TAU=3.0 \
    "$LQR_ROOT/run_lqr_cosmos_policy_gripper_xyz.sh"
