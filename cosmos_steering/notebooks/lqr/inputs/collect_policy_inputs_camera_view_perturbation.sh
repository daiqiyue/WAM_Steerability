#!/bin/bash
# Submit collect_policy_inputs_camera_view_perturbation.py for headless
# execution on the GH200 cluster. WORLD_SIZE parallel sbatch-array workers
# each load the model and process a round-robin slice of episodes, writing
# per-rank shards under <SCRATCH_DIR>/rank{R}/. A dependent finalize job
# concatenates the shards into the unified positive.npz / negative.npz +
# per-config subdir + manifest.json under $OUT_DIR.
#
# Usage:
#   ./collect_policy_inputs_camera_view_perturbation.sh
#   WORLD_SIZE=4 N_POS=10 N_NEG=10 ./collect_policy_inputs_camera_view_perturbation.sh
#   CAM_BASE_SEED=42 ./collect_policy_inputs_camera_view_perturbation.sh
#   N_POS=20 N_NEG=20 TIME=03:00:00 ./collect_policy_inputs_camera_view_perturbation.sh
#
# Outputs land under
#   /u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/
#       libero_10__task00__cam_random_large__seed${CAM_BASE_SEED}__pos_neg/
# unless OUT_DIR is overridden. The naming mirrors the xyz_random / object_pairs
# variants so the SVD shells default-glob this pattern.

set -euo pipefail

# =========================================================================
# User-configurable (override via env)
# =========================================================================
WORLD_SIZE="${WORLD_SIZE:-4}"
N_POS="${N_POS:-10}"
N_NEG="${N_NEG:-10}"

# Camera perturbation knobs (defaults match cam_random_large from the stress test).
# Set CAM_BASE_SEED=42 to reproduce the perturbations used by
# 08_camera_view_perturbation__cam_random_large__seed42.executed.ipynb.
CAM_BASE_SEED="${CAM_BASE_SEED:-42}"
POS_SIGMA="${POS_SIGMA:-0.10}"
ROT_SIGMA_DEG="${ROT_SIGMA_DEG:-8.0}"
FOV_SIGMA="${FOV_SIGMA:-5.0}"
WORKSPACE_TABLE_Z="${WORKSPACE_TABLE_Z:-0.90}"
WORKSPACE_VISIBLE_FRACTION="${WORKSPACE_VISIBLE_FRACTION:-0.55}"
VISIBILITY_MARGIN_PX="${VISIBILITY_MARGIN_PX:-8}"
MAX_REJECTION_ATTEMPTS="${MAX_REJECTION_ATTEMPTS:-2000}"
PRESET_NAME="${PRESET_NAME:-cam_random_large}"

# LIBERO config.
SUITE="${SUITE:-libero_10}"
TASK_ID="${TASK_ID:-0}"
RESOLUTION="${RESOLUTION:-256}"
CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"

# Output locations. Default OUT_DIR follows the existing _pos_neg convention
# under notebooks/lqr/inputs/policy_inputs/ so the SVD shell scripts find it.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DEFAULT_OUT_DIR="/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/${SUITE}__task$(printf '%02d' "$TASK_ID")__${PRESET_NAME}__seed${CAM_BASE_SEED}__pos_neg"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
SCRATCH_DIR="${SCRATCH_DIR:-$OUT_DIR/scratch}"
KEEP_SCRATCH="${KEEP_SCRATCH:-}"

# Slurm.
TIME="${TIME:-02:00:00}"
FINALIZE_TIME="${FINALIZE_TIME:-00:30:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
# =========================================================================

PY_NAME="collect_policy_inputs_camera_view_perturbation.py"
PY="$SCRIPT_DIR/$PY_NAME"
[[ -f "$PY" ]] || { echo "ERROR: .py not found: $PY" >&2; exit 1; }

mkdir -p "$SCRIPT_DIR/logs" "$OUT_DIR" "$SCRATCH_DIR"

COMMON_ARGS=(
    --out-dir "$OUT_DIR"
    --scratch-dir "$SCRATCH_DIR"
    --world-size "$WORLD_SIZE"
    --suite "$SUITE"
    --task-id "$TASK_ID"
    --resolution "$RESOLUTION"
    --ckpt-path "$CKPT_PATH"
    --n-pos-rollouts "$N_POS"
    --n-neg-rollouts "$N_NEG"
    --cam-base-seed "$CAM_BASE_SEED"
    --pos-sigma "$POS_SIGMA"
    --rot-sigma-deg "$ROT_SIGMA_DEG"
    --fov-sigma "$FOV_SIGMA"
    --workspace-table-z "$WORKSPACE_TABLE_Z"
    --workspace-visible-fraction "$WORKSPACE_VISIBLE_FRACTION"
    --visibility-margin-px "$VISIBILITY_MARGIN_PX"
    --max-rejection-attempts "$MAX_REJECTION_ATTEMPTS"
    --preset-name "$PRESET_NAME"
)
[[ -n "$KEEP_SCRATCH" ]] && COMMON_ARGS+=(--keep-scratch)

ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

echo "=== config ==="
echo "  world_size:               $WORLD_SIZE"
echo "  n_pos_rollouts:           $N_POS"
echo "  n_neg_rollouts:           $N_NEG"
echo "  cam_base_seed:            $CAM_BASE_SEED"
echo "  preset_name:              $PRESET_NAME  (cam_random_large defaults)"
echo "  pos_sigma:                $POS_SIGMA m"
echo "  rot_sigma_deg:            $ROT_SIGMA_DEG deg"
echo "  fov_sigma_deg:            $FOV_SIGMA deg"
echo "  workspace_table_z:        $WORKSPACE_TABLE_Z m"
echo "  workspace_visible_frac:   $WORKSPACE_VISIBLE_FRACTION"
echo "  out_dir:                  $OUT_DIR"
echo "  scratch_dir:              $SCRATCH_DIR"
echo "  account/partition/time:   $ACCOUNT / $PARTITION_SLURM / $TIME"
echo

# ---------- Stage 1: collect (sbatch array, one rank per task) ----------
COLLECT_JOB_NAME="cps_cam_collect"
ARRAY_LAST=$((WORLD_SIZE - 1))
echo "=== submitting collect array (0-$ARRAY_LAST) ==="

COLLECT_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$COLLECT_JOB_NAME" \
    --array=0-"$ARRAY_LAST" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem=64G \
    --time="$TIME" \
    --output="$SCRIPT_DIR/logs/cam_collect_rank%a_%A.out" \
    --error="$SCRIPT_DIR/logs/cam_collect_rank%a_%A.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE\"
python -u '$PY' --mode collect --rank \"\$SLURM_ARRAY_TASK_ID\"$ARGS_QUOTED
")
echo "  collect job id: $COLLECT_ID"

# ---------- Stage 2: finalize (single job, dependent on collect array) ----------
FINALIZE_JOB_NAME="cps_cam_finalize"
echo "=== submitting finalize (depends on $COLLECT_ID) ==="

FINALIZE_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$FINALIZE_JOB_NAME" \
    --dependency=afterok:"$COLLECT_ID" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem=64G \
    --time="$FINALIZE_TIME" \
    --output="$SCRIPT_DIR/logs/cam_finalize_%j.out" \
    --error="$SCRIPT_DIR/logs/cam_finalize_%j.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
python -u '$PY' --mode finalize$ARGS_QUOTED
echo 'done; outputs under: $OUT_DIR'
")
echo "  finalize job id: $FINALIZE_ID"

echo
echo "=== submitted ==="
echo "  collect:  sbatch $COLLECT_ID  (array 0-$ARRAY_LAST)"
echo "  finalize: sbatch $FINALIZE_ID (afterok:$COLLECT_ID)"
echo "  outputs:  $OUT_DIR"
echo "  logs:     $SCRIPT_DIR/logs/"
echo
echo "Monitor:        squeue -u \$USER --start"
echo "Tail rank 0:    tail -f $SCRIPT_DIR/logs/cam_collect_rank0_${COLLECT_ID}.out"
echo "Tail finalize:  tail -f $SCRIPT_DIR/logs/cam_finalize_${FINALIZE_ID}.out"
