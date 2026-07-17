#!/bin/bash
# Submit collect_policy_inputs_gripper_xyz_perturbation_multitask.py for
# headless execution on a single GH200 GPU. The .py runs N gripper-xyz-
# perturbed rollouts per task across MULTIPLE libero_10 tasks (default:
# task 0, 2, 4, 5) under each task's prompt and buckets every captured
# policy input by rollout outcome:
#   positive.npz  = rows from successful rollouts (all tasks combined)
#   negative.npz  = rows from failed rollouts    (all tasks combined)
# Each row carries a `task_id` field so rows can be re-segregated later.
#
# These outputs are UNPAIRED (different row counts, rows not 1-1 matched).
# Run notebooks/lqr/svd/pair_inputs_by_similarity.sh on the resulting
# directory to produce a paired, SVD-compatible version.
#
# Usage:
#   ./collect_policy_inputs_gripper_xyz_perturbation_multitask.sh
#   TASK_IDS="0 2 4 5" N_EPISODES=50 PRESET=xyz_random_xlarge_2 BASE_SEED=42 \
#     ./collect_policy_inputs_gripper_xyz_perturbation_multitask.sh
#   OUT_DIR=/path/to/custom_dir ./collect_policy_inputs_gripper_xyz_perturbation_multitask.sh
#
#   # Multi-GPU: fan out one sbatch array task per task_id (each gets 1 GPU)
#   # and submit a dependent CPU-only merge job that concatenates the per-
#   # task shards into <OUT_DIR>/{positive,negative,manifest}.{npz,json}.
#   PARALLEL=1 N_EPISODES=50 \
#     ./collect_policy_inputs_gripper_xyz_perturbation_multitask.sh
#
# sbatch stdout/stderr land under this directory as logs/nb_<jobid>.{out,err}
# (serial mode) or logs/nb_<arrayid>_<rank>.{out,err} + logs/merge_<jobid>.*
# (PARALLEL=1).

set -euo pipefail

PY_NAME="${PY_NAME:-collect_policy_inputs_gripper_xyz_perturbation_multitask.py}"

# Budget: 4 tasks * 50 episodes = 200 rollouts; each xyz_random_xlarge_2
# rollout is ~15-40 s under get_action (failures hit the 530-step cap,
# ~25 s; successes ~250 steps, ~15 s). So expect roughly 200 * 30 s ~
# 100 min of rollouts + ~3 min model load + ~4 min env build (one per
# task). 3h leaves comfortable headroom; bump TIME for larger N or more
# tasks.
TIME="${TIME:-01:00:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY="$SCRIPT_DIR/$PY_NAME"
[[ -f "$PY" ]] || { echo "ERROR: .py not found: $PY" >&2; exit 1; }

NB_BASE="$(basename "$PY" .py)"

SUITE="${SUITE:-libero_10}"
TASK_IDS="${TASK_IDS:-0 2 4 5}"
N_EPISODES="${N_EPISODES:-20}"
RESOLUTION="${RESOLUTION:-256}"
PRESET="${PRESET:-xyz_random_xlarge_2}"
BASE_SEED="${BASE_SEED:-42}"
CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"

# PARALLEL=1 fans the tasks out across separate sbatch jobs (one per task_id,
# 1 GPU each, run in parallel) and merges the resulting per-task shards into
# the unified OUT_DIR via a dependent CPU job. TIME is interpreted as the
# PER-SHARD budget in this mode (each shard runs N_EPISODES rollouts of one
# task), so a tighter TIME than the serial budget is fine.
PARALLEL="${PARALLEL:-1}"
MERGE_TIME="${MERGE_TIME:-00:30:00}"

# Optional explicit prompt; default uses each libero env's built-in description.
PROMPT="${PROMPT:-}"

# Tag like "0-2-4-5" derived from TASK_IDS for filenames / job names.
TASKS_TAG="$(echo "$TASK_IDS" | tr -s ' ' '-')"

# Default output dir mirrors the existing pos_neg dir naming.
OUT_DIR="${OUT_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__tasks${TASKS_TAG}__${PRESET}__seed${BASE_SEED}__pos_neg}"

mkdir -p "$SCRIPT_DIR/logs"

echo "=== config ==="
echo "  out_dir:      $OUT_DIR"
echo "  suite:        $SUITE"
echo "  task_ids:     $TASK_IDS"
echo "  n_episodes:   $N_EPISODES   (per task)"
echo "  resolution:   $RESOLUTION"
echo "  preset:       $PRESET"
echo "  base_seed:    $BASE_SEED"
echo "  prompt:       ${PROMPT:-<env default per task>}"
echo "  ckpt_path:    $CKPT_PATH"
echo "  parallel:     $PARALLEL"
echo "  time:         $TIME  (per-shard if PARALLEL=1)"
echo "  partition:    $PARTITION"
echo

PROMPT_ARG=()
if [[ -n "$PROMPT" ]]; then
    PROMPT_ARG=(--prompt "$PROMPT")
fi
PROMPT_QUOTED=""
for a in "${PROMPT_ARG[@]}"; do
    PROMPT_QUOTED+=" $(printf '%q' "$a")"
done

if [[ "$PARALLEL" == "1" || "$PARALLEL" == "true" ]]; then
    # ------------------------------------------------------------------
    # Multi-GPU mode: sbatch array (1 rank per task_id) + dependent merge
    # ------------------------------------------------------------------
    MERGE_PY="$SCRIPT_DIR/merge_per_task_inputs.py"
    [[ -f "$MERGE_PY" ]] || {
        echo "ERROR: missing $MERGE_PY (required for PARALLEL=1)" >&2
        exit 1
    }

    # Bash array form of TASK_IDS so the array rank can index it.
    read -r -a TASK_IDS_ARR <<< "$TASK_IDS"
    N_TASKS="${#TASK_IDS_ARR[@]}"
    LAST_IDX=$((N_TASKS - 1))

    # Comma-separated, then bash-array literal for embedding in the wrap.
    TASK_LIST_LITERAL="${TASK_IDS_ARR[*]}"

    # Per-shard out dirs sit under $OUT_DIR/per_task/task<tid>.
    PER_TASK_DIRS_QUOTED=""
    for t in "${TASK_IDS_ARR[@]}"; do
        PER_TASK_DIRS_QUOTED+=" $(printf '%q' "$OUT_DIR/per_task/task$t")"
    done

    echo "=== submitting array of $N_TASKS shards (0-$LAST_IDX) ==="
    ARR_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --array=0-"$LAST_IDX" \
        --gpus-per-task=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time="$TIME" \
        --job-name="py_${NB_BASE}_${PRESET}_arr" \
        --output="$SCRIPT_DIR/logs/nb_%A_%a.out" \
        --error="$SCRIPT_DIR/logs/nb_%A_%a.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
TASK_IDS_ARR=($TASK_LIST_LITERAL)
TID=\${TASK_IDS_ARR[\$SLURM_ARRAY_TASK_ID]}
PER_TASK_OUT='$OUT_DIR'/per_task/task\$TID
echo \"rank=\$SLURM_ARRAY_TASK_ID  task_id=\$TID  per_task_out=\$PER_TASK_OUT\"
python -u '$PY' \
    --out-dir \"\$PER_TASK_OUT\" \
    --suite '$SUITE' \
    --task-ids \"\$TID\" \
    --n-episodes '$N_EPISODES' \
    --resolution '$RESOLUTION' \
    --preset '$PRESET' \
    --base-seed '$BASE_SEED' \
    --ckpt-path '$CKPT_PATH'$PROMPT_QUOTED
echo \"shard done: \$PER_TASK_OUT\"
")
    echo "  array job id: $ARR_ID"

    echo "=== submitting merge (afterok:$ARR_ID) ==="
    # NOTE: --gpus-per-task=1 is required even though the merge is pure CPU
    # numpy work — DeltaAI rejects sbatch jobs that don't request a GPU.
    MERGE_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --dependency=afterok:"$ARR_ID" \
        --gpus-per-task=1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time="$MERGE_TIME" \
        --job-name="py_${NB_BASE}_${PRESET}_merge" \
        --output="$SCRIPT_DIR/logs/merge_%j.out" \
        --error="$SCRIPT_DIR/logs/merge_%j.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
python -u '$MERGE_PY' \
    --in-dirs$PER_TASK_DIRS_QUOTED \
    --out-dir '$OUT_DIR'
echo 'merged; outputs under: $OUT_DIR'
echo 'Next: pair_inputs_by_similarity.sh IN_DIR=$OUT_DIR'
")
    echo "  merge job id: $MERGE_ID"
    echo
    echo "=== submitted ==="
    echo "  array:  sbatch $ARR_ID   (0-$LAST_IDX, one task per rank)"
    echo "  merge:  sbatch $MERGE_ID (afterok:$ARR_ID)"
    echo "  out:    $OUT_DIR"
    echo "  shards: $OUT_DIR/per_task/task<tid>/"
    echo
    echo "Monitor:  squeue -u \$USER"
    echo "Tail:     tail -f $SCRIPT_DIR/logs/nb_${ARR_ID}_0.err"
else
    # ------------------------------------------------------------------
    # Serial mode (original): one sbatch, .py loops over all task_ids
    # ------------------------------------------------------------------
    # NOTE: $TASK_IDS is intentionally UNQUOTED below so the inner shell
    # splits it into N separate --task-ids arguments (argparse nargs="+").
    sbatch \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --gpus-per-node=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time="$TIME" \
        --job-name="py_${NB_BASE}_${PRESET}_t${TASKS_TAG}" \
        --output="$SCRIPT_DIR/logs/nb_%j.out" \
        --error="$SCRIPT_DIR/logs/nb_%j.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
echo 'executing: $PY'
python -u '$PY' \
    --out-dir '$OUT_DIR' \
    --suite '$SUITE' \
    --task-ids $TASK_IDS \
    --n-episodes '$N_EPISODES' \
    --resolution '$RESOLUTION' \
    --preset '$PRESET' \
    --base-seed '$BASE_SEED' \
    --ckpt-path '$CKPT_PATH'$PROMPT_QUOTED
echo 'done; outputs under: $OUT_DIR'
echo 'Next: pair_inputs_by_similarity.sh IN_DIR=$OUT_DIR'
"
fi
