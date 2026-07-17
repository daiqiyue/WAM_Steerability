#!/bin/bash
# Submit run_lqr_decay_cosmos_policy_cam_perturb.py for headless execution.
# Same LQR-decay rollouts as run_lqr_decay_cosmos_policy.sh, but the agentview
# camera is perturbed per-episode by the visibility-constrained
# RandomCameraViewPerturbation (cam_random_large defaults from the stress test).
#
# Episodes are sharded across WORLD_SIZE GPUs via an sbatch array; a dependent
# finalize job concatenates per-rank result shards into the unified
# results.json + baseline/results.json.
#
# Usage:
#   ./run_lqr_decay_cosmos_policy_cam_perturb.sh
#   WORLD_SIZE=4 N_EPISODES=10 ./run_lqr_decay_cosmos_policy_cam_perturb.sh
#   CAM_MODE=random CAM_BASE_SEED=99 ./run_lqr_decay_cosmos_policy_cam_perturb.sh
#   CAM_MODE=match ./run_lqr_decay_cosmos_policy_cam_perturb.sh  # default
#
# CAM_MODE values:
#   match  - use CAM_BASE_SEED=42 (matches the negative-example collection
#            seed, so the LQR rollout sees the exact same perturbations as
#            the inputs the SVD was fit on).
#   random - use a different CAM_BASE_SEED (default 99) for out-of-distribution
#            evaluation.
#   off    - no perturbation; equivalent to the original run_lqr_decay_*.sh.

set -euo pipefail

# =========================================================================
# User-configurable (override via env)
# =========================================================================
WORLD_SIZE="${WORLD_SIZE:-4}"

# --- SVD / jacobian inputs (same defaults as run_lqr_decay_cosmos_policy.sh).
# Point these at the SVD run + jacobian output produced from the camera-
# perturbation pos/neg inputs.
SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_cam_random_large_seed42_pairs_no_action_dsall_N-1_k64_p10_ws8_tsall}"
JAC_DIR_ACT="${JAC_DIR_ACT:-A_tilde_full__put_both_the_alphabet_soup_and_the_tomato_sauce_in__vjp_no_retain__vbf16}"

# --- LQR / rollout knobs.
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"
LAMBDA="${LAMBDA:-10.0}"
Q_SCALE="${Q_SCALE:-1.0}"
R_SCALE="${R_SCALE:-10.0}"
R_SCALE_TAU="${R_SCALE_TAU:-3.0}"
R_SCALE_FINAL="${R_SCALE_FINAL:-1e9}"
QF_SCALE="${QF_SCALE:-1.0}"
MAX_CHUNKS="${MAX_CHUNKS:-50}"
N_EPISODES="${N_EPISODES:-50}"

TASK_ID="${TASK_ID:-0}"
SUITE_NAME="${SUITE_NAME:-libero_10}"
RESOLUTION="${RESOLUTION:-256}"
VIDEO_FPS="${VIDEO_FPS:-30}"

RUN_TAG="${RUN_TAG:-cam_perturb_decay}"
ROLLOUT_SUBDIR="${ROLLOUT_SUBDIR:-cam_perturb}"
RUN_BASELINE="${RUN_BASELINE:-0}"
# When 1, each episode's video stacks the perturbed (policy-driving) view
# and a clean unperturbed view side-by-side. The clean view is rendered by a
# second LIBERO env that gets state-injected after every step, so it never
# affects what the policy sees. Skipped automatically when CAM_MODE=off.
SIDE_BY_SIDE_VIDEO="${SIDE_BY_SIDE_VIDEO:-1}"

# --- Camera perturbation knobs (defaults match cam_random_large preset).
CAM_MODE="${CAM_MODE:-random}"           # match | random | off
# In 'match' mode, use the inputs-collection seed (default 42). In 'random'
# mode use a different default (99) so the rollout sees fresh OOD samples.
case "$CAM_MODE" in
    match)  CAM_BASE_SEED="${CAM_BASE_SEED:-42}"  ;;
    random) CAM_BASE_SEED="${CAM_BASE_SEED:-99}"  ;;
    off)    CAM_BASE_SEED="${CAM_BASE_SEED:-0}"   ;;
    *)      echo "ERROR: unknown CAM_MODE=$CAM_MODE (match|random|off)" >&2; exit 1 ;;
esac
CAM_PRESET_NAME="${CAM_PRESET_NAME:-cam_random_large}"
CAM_POS_SIGMA="${CAM_POS_SIGMA:-0.10}"
CAM_ROT_SIGMA_DEG="${CAM_ROT_SIGMA_DEG:-8.0}"
CAM_FOV_SIGMA="${CAM_FOV_SIGMA:-5.0}"
CAM_WORKSPACE_TABLE_Z="${CAM_WORKSPACE_TABLE_Z:-0.90}"
CAM_WORKSPACE_VISIBLE_FRACTION="${CAM_WORKSPACE_VISIBLE_FRACTION:-0.55}"
CAM_VISIBILITY_MARGIN_PX="${CAM_VISIBILITY_MARGIN_PX:-8}"
CAM_MAX_REJECTION_ATTEMPTS="${CAM_MAX_REJECTION_ATTEMPTS:-2000}"

# --- Slurm.
TIME="${TIME:-00:30:00}"
FINALIZE_TIME="${FINALIZE_TIME:-00:15:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
# shellcheck disable=SC1091
. "$(dirname -- "${BASH_SOURCE[0]}")/../exclude_nodes.sh"
EXCLUDE_NODES="${EXCLUDE_NODES-$EXCLUDE_NODES_DEFAULT}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

# =========================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY="$SCRIPT_DIR/run_lqr_decay_cosmos_policy_cam_perturb.py"
[[ -f "$PY" ]] || { echo "ERROR: .py not found: $PY" >&2; exit 1; }

mkdir -p "$SCRIPT_DIR/logs"

COMMON_ARGS=(
    --svd-dir "$SVD_DIR"
    --jac-dir-act "$JAC_DIR_ACT"
    --world-size "$WORLD_SIZE"
    --suite "$SUITE_NAME"
    --task-id "$TASK_ID"
    --n-episodes "$N_EPISODES"
    --resolution "$RESOLUTION"
    --video-fps "$VIDEO_FPS"
    --prompt "$PROMPT"
    --lambda "$LAMBDA"
    --q-scale "$Q_SCALE"
    --r-scale-init "$R_SCALE"
    --r-scale-tau "$R_SCALE_TAU"
    --r-scale-final "$R_SCALE_FINAL"
    --qf-scale "$QF_SCALE"
    --max-chunks "$MAX_CHUNKS"
    --cam-mode "$CAM_MODE"
    --cam-base-seed "$CAM_BASE_SEED"
    --cam-preset-name "$CAM_PRESET_NAME"
    --cam-pos-sigma "$CAM_POS_SIGMA"
    --cam-rot-sigma-deg "$CAM_ROT_SIGMA_DEG"
    --cam-fov-sigma "$CAM_FOV_SIGMA"
    --cam-workspace-table-z "$CAM_WORKSPACE_TABLE_Z"
    --cam-workspace-visible-fraction "$CAM_WORKSPACE_VISIBLE_FRACTION"
    --cam-visibility-margin-px "$CAM_VISIBILITY_MARGIN_PX"
    --cam-max-rejection-attempts "$CAM_MAX_REJECTION_ATTEMPTS"
    --run-tag "$RUN_TAG"
    --rollout-subdir "$ROLLOUT_SUBDIR"
)
if [[ "$RUN_BASELINE" == "1" || "$RUN_BASELINE" == "true" ]]; then
    COMMON_ARGS+=(--run-baseline)
fi
if [[ "$SIDE_BY_SIDE_VIDEO" == "1" || "$SIDE_BY_SIDE_VIDEO" == "true" ]]; then
    COMMON_ARGS+=(--side-by-side-video)
fi

ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

echo "=== config ==="
echo "  world_size:        $WORLD_SIZE"
echo "  SVD_DIR:           $SVD_DIR"
echo "  JAC_DIR_ACT:       $JAC_DIR_ACT"
echo "  PROMPT:            \"$PROMPT\""
echo "  N_EPISODES:        $N_EPISODES"
echo "  LAMBDA:            $LAMBDA"
echo "  Q/Rinit/tau/final: $Q_SCALE / $R_SCALE / $R_SCALE_TAU / $R_SCALE_FINAL"
echo "  CAM_MODE:          $CAM_MODE   (base_seed=$CAM_BASE_SEED)"
echo "  CAM preset:        $CAM_PRESET_NAME"
echo "  ROLLOUT_SUBDIR:    $ROLLOUT_SUBDIR"
echo "  RUN_TAG:           $RUN_TAG"
echo "  RUN_BASELINE:      $RUN_BASELINE"
echo "  SIDE_BY_SIDE:      $SIDE_BY_SIDE_VIDEO  (clean view rendered alongside perturbed)"
echo "  account/part/time: $ACCOUNT / $PARTITION_SLURM / $TIME"
echo "  exclude_nodes:     ${EXCLUDE_NODES:-<none>}"
echo

# ---------- Stage 1: rollouts array (one rank per task) ----------
ROLL_JOB="cps_lqr_cam_roll"
ARRAY_LAST=$((WORLD_SIZE - 1))
echo "=== submitting rollouts array (0-$ARRAY_LAST) ==="

ROLL_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$ROLL_JOB" \
    --array=0-"$ARRAY_LAST" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem=64G \
    --time="$TIME" \
    ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
    --output="$SCRIPT_DIR/logs/lqr_cam_rank%a_%A.out" \
    --error="$SCRIPT_DIR/logs/lqr_cam_rank%a_%A.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR/..'   # repo-root cwd so config_file path resolves
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE\"

# Wipe the T5 prompt-embedding cache to recover from corruption left by a
# crashed prior run. The notebook variant does the same thing.
python -u -c '
import os, pickle
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id=\"nvidia/Cosmos-Policy-LIBERO-Predict2-2B\",
    filename=\"libero_t5_embeddings.pkl\",
    cache_dir=os.environ.get(\"HF_HUB_CACHE\"),
)
real = os.path.realpath(p)
with open(real, \"wb\") as f:
    pickle.dump({}, f)
print(f\"[t5-cache] cleared (empty pickle at {p})\")
for suffix in (\".backup\", \".lock\"):
    aux = p + suffix
    if os.path.exists(aux):
        try: os.remove(aux); print(f\"[t5-cache] removed {aux}\")
        except OSError: pass
'

python -u '$PY' --mode rollouts --rank \"\$SLURM_ARRAY_TASK_ID\"$ARGS_QUOTED
")
echo "  rollouts job id: $ROLL_ID"

# ---------- Stage 2: finalize (single, dependent on array) ----------
FIN_JOB="cps_lqr_cam_finalize"
echo "=== submitting finalize (depends on $ROLL_ID) ==="

FIN_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$FIN_JOB" \
    --dependency=afterany:"$ROLL_ID" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time="$FINALIZE_TIME" \
    --output="$SCRIPT_DIR/logs/lqr_cam_finalize_%j.out" \
    --error="$SCRIPT_DIR/logs/lqr_cam_finalize_%j.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR/..'
python -u '$PY' --mode finalize$ARGS_QUOTED
")
echo "  finalize job id: $FIN_ID"

echo
echo "=== submitted ==="
echo "  rollouts:  sbatch $ROLL_ID  (array 0-$ARRAY_LAST)"
echo "  finalize:  sbatch $FIN_ID   (afterany:$ROLL_ID)"
echo "  logs:      $SCRIPT_DIR/logs/"
echo
echo "Monitor:        squeue -u \$USER --start"
echo "Tail rank 0:    tail -f $SCRIPT_DIR/logs/lqr_cam_rank0_${ROLL_ID}.out"
echo "Tail finalize:  tail -f $SCRIPT_DIR/logs/lqr_cam_finalize_${FIN_ID}.out"
