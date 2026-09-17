#!/usr/bin/env bash
# Submit the complete task-6 Gaussian ActAdd sensitivity pipeline to IDEaS
# L40S. The script prints all dependency-linked Slurm job IDs.

set -euo pipefail

ROOT=/storage/scratch1/9/qdai41/steering_robustness_WAMs
STAGE_SCRIPT="$ROOT/run_cosmos_task6_actadd_stage.sbatch"
RUN_ROOT="${RUN_ROOT:-$ROOT/cosmos_steering/runs/task6_gaussian_actadd_sensitivity_seed99}"
ACCOUNT="${ACCOUNT:-gts-gchou3-ideas_l40s}"
PARTITION="${PARTITION:-gpu-l40s}"
CKPT_PATH="${CKPT_PATH:-/storage/scratch1/9/qdai41/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8}"
TRAIN_EPISODES="${TRAIN_EPISODES:-0 1 2 3 4 5 6 7 8 9}"
EVAL_EPISODES="${EVAL_EPISODES:-20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39}"

read -r -a TRAIN_ARRAY <<< "$TRAIN_EPISODES"
read -r -a EVAL_ARRAY <<< "$EVAL_EPISODES"
TRAIN_LAST=$((${#TRAIN_ARRAY[@]} - 1))
EVAL_LAST=$((${#EVAL_ARRAY[@]} - 1))

mkdir -p "$RUN_ROOT/logs"
COMMON_EXPORT="ALL,RUN_ROOT=$RUN_ROOT,CKPT_PATH=$CKPT_PATH,TRAIN_EPISODES=$TRAIN_EPISODES,EVAL_EPISODES=$EVAL_EPISODES,NOISE_SIGMA=${NOISE_SIGMA:-90.0},NOISE_SEED_BASE=${NOISE_SEED_BASE:-99},ROWS_PER_EPISODE=${ROWS_PER_EPISODE:-4},SAMPLING_STEPS=${SAMPLING_STEPS:-10},ANCHOR_CHUNK=${ANCHOR_CHUNK:-8},BRANCH_HORIZON_CHUNKS=${BRANCH_HORIZON_CHUNKS:-10},RAW_REPEAT_CHUNKS=${RAW_REPEAT_CHUNKS:-3},ROLLOUT_ALPHA=${ROLLOUT_ALPHA:-0.1},POLICY_SEED=${POLICY_SEED:-1},RANDOM_DIRECTION_SEED=${RANDOM_DIRECTION_SEED:-20260917}"
COLLECT_DEPENDENCY=()
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  COLLECT_DEPENDENCY=(--dependency="afterok:$AFTEROK_JOB_ID")
fi

COLLECT_ID=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  "${COLLECT_DEPENDENCY[@]}" \
  --array="0-$TRAIN_LAST" --time=03:00:00 --job-name=cosmos-t6-pairs \
  --output="$RUN_ROOT/logs/pairs_%A_%a.out" \
  --error="$RUN_ROOT/logs/pairs_%A_%a.err" \
  --export="$COMMON_EXPORT,STAGE=collect_pairs" "$STAGE_SCRIPT")

MERGE_ID=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$COLLECT_ID" --time=00:30:00 --job-name=cosmos-t6-pairmerge \
  --output="$RUN_ROOT/logs/pairmerge_%j.out" \
  --error="$RUN_ROOT/logs/pairmerge_%j.err" \
  --export="$COMMON_EXPORT,STAGE=merge_select" "$STAGE_SCRIPT")

ACT_ID=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$MERGE_ID" --time=04:00:00 --job-name=cosmos-t6-acts \
  --output="$RUN_ROOT/logs/acts_%j.out" \
  --error="$RUN_ROOT/logs/acts_%j.err" \
  --export="$COMMON_EXPORT,STAGE=collect_activations" "$STAGE_SCRIPT")

VEC_ID=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$ACT_ID" --time=00:20:00 --job-name=cosmos-t6-vec \
  --output="$RUN_ROOT/logs/vector_%j.out" \
  --error="$RUN_ROOT/logs/vector_%j.err" \
  --export="$COMMON_EXPORT,STAGE=make_vector" "$STAGE_SCRIPT")

EVAL_ID=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$VEC_ID" --array="0-$EVAL_LAST" --time=06:00:00 \
  --job-name=cosmos-t6-sense \
  --output="$RUN_ROOT/logs/eval_%A_%a.out" \
  --error="$RUN_ROOT/logs/eval_%A_%a.err" \
  --export="$COMMON_EXPORT,STAGE=eval" "$STAGE_SCRIPT")

ANALYZE_ID=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --dependency="afterok:$EVAL_ID" --time=00:30:00 --job-name=cosmos-t6-analyze \
  --output="$RUN_ROOT/logs/analyze_%j.out" \
  --error="$RUN_ROOT/logs/analyze_%j.err" \
  --export="$COMMON_EXPORT,STAGE=analyze" "$STAGE_SCRIPT")

cat <<EOF
Cosmos task-6 Gaussian ActAdd sensitivity pipeline submitted on $ACCOUNT / $PARTITION
  collect pairs : $COLLECT_ID  (array 0-$TRAIN_LAST; episodes $TRAIN_EPISODES)
  merge/select  : $MERGE_ID
  activations   : $ACT_ID
  vector        : $VEC_ID
  paired eval   : $EVAL_ID  (array 0-$EVAL_LAST; held-out episodes $EVAL_EPISODES)
  analysis      : $ANALYZE_ID
  output        : $RUN_ROOT
EOF
