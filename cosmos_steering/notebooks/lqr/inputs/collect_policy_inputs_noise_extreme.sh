#!/bin/bash
# Submit collect_policy_inputs_noise_extreme.py on the ghx4 partition. The .py
# runs N_EPISODES clean libero_10 task 0 rollouts under the original task
# prompt, captures every policy input, then post-hoc adds Gaussian pixel
# noise (sigma=NOISE_SIGMA, default 90 to match `noise_extreme`) to build
# the negatives.
#
# Parallelism
# -----------
# PARALLEL=1 (default) fans the episodes out across WORLD_SIZE sbatch array
# ranks (one GPU each). Rank R handles episodes range(R, N_EPISODES, WORLD_SIZE).
# A dependent CPU-light merge job concatenates the per-rank shards into
# <OUT_DIR>/{positive,negative,manifest}.{npz,json}. With stride sharding,
# rank 0 owns episode 0, so row 0 of the merged negative.npz remains the
# first inference of episode 0 — exactly the linearization point the
# downstream jacobians / LQR pipeline expects.
#
# Set PARALLEL=0 to run all episodes serially on a single GPU.
#
# Usage:
#   ./collect_policy_inputs_noise_extreme.sh
#   N_EPISODES=10 WORLD_SIZE=5 ./collect_policy_inputs_noise_extreme.sh
#   PARALLEL=0 ./collect_policy_inputs_noise_extreme.sh
#   OUT_DIR=/custom/path ./collect_policy_inputs_noise_extreme.sh
#
# sbatch stdout/stderr land under this directory as
# logs/nb_<arrayid>_<rank>.{out,err} + logs/merge_<jobid>.* (parallel mode)
# or logs/nb_<jobid>.{out,err} (serial mode).

set -euo pipefail

PY_NAME="${PY_NAME:-collect_policy_inputs_noise_extreme.py}"

TIME="${TIME:-01:30:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY="$SCRIPT_DIR/$PY_NAME"
[[ -f "$PY" ]] || { echo "ERROR: .py not found: $PY" >&2; exit 1; }

NB_BASE="$(basename "$PY" .py)"

SUITE="${SUITE:-libero_10}"
TASK_ID="${TASK_ID:-0}"
N_EPISODES="${N_EPISODES:-10}"
RESOLUTION="${RESOLUTION:-256}"
NOISE_SIGMA="${NOISE_SIGMA:-90.0}"   # noise_extreme = 90
NOISE_PER_EPISODE_SEED="${NOISE_PER_EPISODE_SEED:-1}"
CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"

# WORLD_SIZE is capped at N_EPISODES below (no point spawning ranks with no
# work). Default 5 = two episodes per rank for the default 10-episode run.
WORLD_SIZE="${WORLD_SIZE:-5}"

# PARALLEL=1 -> sbatch array (one rank per shard) + CPU merge.
# PARALLEL=0 -> single sbatch job runs all episodes.
PARALLEL="${PARALLEL:-1}"
MERGE_TIME="${MERGE_TIME:-00:20:00}"

OUT_DIR="${OUT_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_extreme_pos_neg}"

mkdir -p "$SCRIPT_DIR/logs"

# Cap WORLD_SIZE at N_EPISODES.
if (( WORLD_SIZE > N_EPISODES )); then
    echo "(capping WORLD_SIZE=$WORLD_SIZE to N_EPISODES=$N_EPISODES)"
    WORLD_SIZE=$N_EPISODES
fi

PER_EP_SEED_FLAG="--noise-per-episode-seed"
if [[ "$NOISE_PER_EPISODE_SEED" != "1" && "$NOISE_PER_EPISODE_SEED" != "true" ]]; then
    PER_EP_SEED_FLAG="--no-noise-per-episode-seed"
fi

echo "=== config ==="
echo "  out_dir:                $OUT_DIR"
echo "  suite:                  $SUITE"
echo "  task_id:                $TASK_ID"
echo "  n_episodes:             $N_EPISODES"
echo "  resolution:             $RESOLUTION"
echo "  prompt:                 $PROMPT"
echo "  noise_sigma:            $NOISE_SIGMA"
echo "  noise_per_episode_seed: $NOISE_PER_EPISODE_SEED"
echo "  ckpt_path:              $CKPT_PATH"
echo "  parallel:               $PARALLEL"
echo "  world_size:             $WORLD_SIZE"
echo "  time:                   $TIME"
echo "  partition:              $PARTITION"
echo

COMMON_FLAGS=(
    --suite "$SUITE"
    --task-id "$TASK_ID"
    --n-episodes "$N_EPISODES"
    --resolution "$RESOLUTION"
    --prompt "$PROMPT"
    --noise-sigma "$NOISE_SIGMA"
    "$PER_EP_SEED_FLAG"
    --ckpt-path "$CKPT_PATH"
)

if [[ "$PARALLEL" == "1" || "$PARALLEL" == "true" ]]; then
    # ------------------------------------------------------------------
    # Multi-GPU mode: sbatch array (1 rank per shard) + dependent merge
    # ------------------------------------------------------------------
    MERGE_PY="$SCRIPT_DIR/merge_per_task_inputs.py"
    [[ -f "$MERGE_PY" ]] || {
        echo "ERROR: missing $MERGE_PY (required for PARALLEL=1)" >&2
        exit 1
    }

    # Pre-compute per-rank episode lists (stride sharding) and stash them
    # as space-separated literals embedded in the inner wrap.
    declare -a RANK_EP_LISTS=()
    for ((r = 0; r < WORLD_SIZE; r++)); do
        eps=""
        for ((e = r; e < N_EPISODES; e += WORLD_SIZE)); do
            eps+=" $e"
        done
        RANK_EP_LISTS+=("${eps# }")
    done

    # Per-shard out dirs sit under $OUT_DIR/per_shard/shard<R>.
    PER_SHARD_DIRS_QUOTED=""
    for ((r = 0; r < WORLD_SIZE; r++)); do
        PER_SHARD_DIRS_QUOTED+=" $(printf '%q' "$OUT_DIR/per_shard/shard$r")"
    done

    # Build a bash-array literal that the inner array job indexes by rank.
    # Each element is the space-separated episode list for that rank,
    # itself in quoted form so it survives the heredoc -> shell parse.
    RANK_EP_LITERAL=""
    for s in "${RANK_EP_LISTS[@]}"; do
        RANK_EP_LITERAL+=" $(printf '%q' "$s")"
    done

    FLAGS_QUOTED=""
    for a in "${COMMON_FLAGS[@]}"; do
        FLAGS_QUOTED+=" $(printf '%q' "$a")"
    done

    LAST_IDX=$((WORLD_SIZE - 1))
    echo "=== submitting array of $WORLD_SIZE shards (0-$LAST_IDX) ==="
    for ((r = 0; r < WORLD_SIZE; r++)); do
        echo "  shard $r: episodes=[${RANK_EP_LISTS[$r]}]"
    done

    ARR_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --array=0-"$LAST_IDX" \
        --gpus-per-task=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time="$TIME" \
        --job-name="py_${NB_BASE}_arr" \
        --output="$SCRIPT_DIR/logs/nb_%A_%a.out" \
        --error="$SCRIPT_DIR/logs/nb_%A_%a.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
RANK_EP_LISTS=($RANK_EP_LITERAL)
MY_EPS=\${RANK_EP_LISTS[\$SLURM_ARRAY_TASK_ID]}
PER_SHARD_OUT='$OUT_DIR'/per_shard/shard\$SLURM_ARRAY_TASK_ID
echo \"rank=\$SLURM_ARRAY_TASK_ID  episodes=[\$MY_EPS]  out=\$PER_SHARD_OUT\"
python -u '$PY' \
    --out-dir \"\$PER_SHARD_OUT\" \
    --episode-ids \$MY_EPS \
    $FLAGS_QUOTED
echo \"shard done: \$PER_SHARD_OUT\"
")
    echo "  array job id: $ARR_ID"

    echo "=== submitting merge (afterok:$ARR_ID) ==="
    # NOTE: --gpus-per-task=1 mirrors merge_per_task_inputs.sh; DeltaAI
    # rejects sbatch jobs that don't request a GPU.
    MERGE_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --dependency=afterok:"$ARR_ID" \
        --gpus-per-task=1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time="$MERGE_TIME" \
        --job-name="py_${NB_BASE}_merge" \
        --output="$SCRIPT_DIR/logs/merge_%j.out" \
        --error="$SCRIPT_DIR/logs/merge_%j.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
python -u '$MERGE_PY' \
    --in-dirs$PER_SHARD_DIRS_QUOTED \
    --out-dir '$OUT_DIR'
echo 'merged; outputs under: $OUT_DIR'
")
    echo "  merge job id: $MERGE_ID"
    echo
    echo "=== submitted ==="
    echo "  array:  sbatch $ARR_ID   (0-$LAST_IDX, one shard per rank)"
    echo "  merge:  sbatch $MERGE_ID (afterok:$ARR_ID)"
    echo "  out:    $OUT_DIR"
    echo "  shards: $OUT_DIR/per_shard/shard<R>/"
    echo
    echo "Monitor:  squeue -u \$USER"
    echo "Tail:     tail -f $SCRIPT_DIR/logs/nb_${ARR_ID}_0.err"
else
    # ------------------------------------------------------------------
    # Serial mode (single sbatch job, all episodes)
    # ------------------------------------------------------------------
    FLAGS_QUOTED=""
    for a in "${COMMON_FLAGS[@]}"; do
        FLAGS_QUOTED+=" $(printf '%q' "$a")"
    done

    sbatch \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --gpus-per-node=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time="$TIME" \
        --job-name="py_${NB_BASE}" \
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
    $FLAGS_QUOTED
echo 'done; outputs under: $OUT_DIR'
"
fi
