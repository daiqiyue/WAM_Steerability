#!/bin/bash
# Slurm launcher for run_lqr_cosmos_policy_noised.py -- closed-loop A-LQR
# rollout where:
#   - ep0/chunk0 takes the negative-npz observation from --pair-dir (the
#     same input the jacobians under JAC_DIR_ACT were computed against)
#     when NO_OVERRIDE_FIRST_CHUNK=0; with the default (=1) every chunk
#     gets its obs from the env and is then noised.
#   - every subsequent inference reads obs from the env and applies σ=NOISE_SIGMA
#     Gaussian noise on the wrist + 3rd-person images before feeding to the policy
#   - R_SCALE follows an exponentially-decaying steering schedule (matches
#     run_lqr_decay_cosmos_policy.ipynb): early chunks see strong steering
#     (R_SCALE=R_SCALE), later chunks see negligible steering (R_SCALE saturates
#     at R_SCALE_FINAL) with time constant R_SCALE_TAU (in chunks):
#         R_SCALE(c) = min(R_SCALE_FINAL, R_SCALE * exp(c / R_SCALE_TAU))
#     Per-chunk K matrices are precomputed up front for c in [0, MAX_CHUNKS).
#
# Parallelism
# -----------
# Submits a sbatch array of WORLD_SIZE workers (one GPU each), each handling
# its slice of N_EPISODES via --rank R --world-size W (episodes are striped
# across ranks: rank R handles range(R, N_EPISODES, WORLD_SIZE)). After they
# all finish, a dependent merge job aggregates results_rank*.json into
# results.json and patches the top-level manifest.json.
#
# Noise seed vs contrastive-vector seed
# -------------------------------------
# The runtime per-episode noise RNG is seeded with
#   seed = NOISE_SEED_BASE + episode_idx.
# With NOISE_SEED_BASE=0 (the default) this matches the
# `np.random.default_rng(seed=int(episode_idx))` used in
# `collect_policy_inputs_noise.ipynb`, so the noise drawn for episode E here
# is IDENTICAL to the noise that built the contrastive negatives for that
# episode -- which is intentional (the LQR linearization point lines up with
# the noise the policy actually sees). To decorrelate the rollout noise from
# the contrastive seed, set NOISE_SEED_BASE to a non-zero offset:
#     NOISE_SEED_BASE=1000 ./run_lqr_cosmos_policy_noised.sh
# Note: when NO_OVERRIDE_FIRST_CHUNK=0, ep0/chunk0 is replaced by the
# baked-in negative.npz observation regardless of NOISE_SEED_BASE -- that
# noise was frozen at collection time. With the default NO_OVERRIDE_FIRST_CHUNK=1
# every chunk's noise is freshly drawn at runtime.
#
# Usage:
#   ./run_lqr_cosmos_policy_noised.sh
#   WORLD_SIZE=4 N_EPISODES=50 ./run_lqr_cosmos_policy_noised.sh
#   LAMBDA=10.0 Q_SCALE=1.0 R_SCALE=10.0 ./run_lqr_cosmos_policy_noised.sh
#   R_SCALE=10.0 R_SCALE_TAU=3.0 R_SCALE_FINAL=1e9 ./run_lqr_cosmos_policy_noised.sh
#   N_EPISODES=3 SEED=42 TIME=00:30:00 ./run_lqr_cosmos_policy_noised.sh
#   NOISE_SEED_BASE=1000 ./run_lqr_cosmos_policy_noised.sh   # decorrelate from contrastive seed
#   MERGE_ONLY=1 ./run_lqr_cosmos_policy_noised.sh           # only run merge
#
# Override SVD_DIR / JAC_DIR_ACT / PAIR_DIR if you want to point at different
# runs from the noise-pair pipeline.

set -euo pipefail

# --- Parallelization --------------------------------------------------------
WORLD_SIZE="${WORLD_SIZE:-4}"

# --- LQR cost hyperparameters ----------------------------------------------
LAMBDA="${LAMBDA:-10.0}"
Q_SCALE="${Q_SCALE:-1.0}"
# R_SCALE is the *initial* control cost at chunk 0; it grows exponentially
# with chunk index toward R_SCALE_FINAL with time constant R_SCALE_TAU
# (in chunks). With R_SCALE_FINAL=1e9 (the default) the steering is
# effectively off once c grows past ~R_SCALE_TAU * log(R_SCALE_FINAL/R_SCALE).
R_SCALE="${R_SCALE:-5.0}"
R_SCALE_TAU="${R_SCALE_TAU:-3.0}"
R_SCALE_FINAL="${R_SCALE_FINAL:-1e9}"
# Upper bound on the number of distinct per-chunk K matrices the python
# script precomputes. Chunks beyond this clamp to MAX_CHUNKS-1 (which
# already has R_SCALE saturated at R_SCALE_FINAL given the default tau).
MAX_CHUNKS="${MAX_CHUNKS:-50}"
QF_SCALE="${QF_SCALE:-1.0}"

# --- Rollout knobs ----------------------------------------------------------
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket}"
# Default is a single rollout because only ep0/chunk0 has the exact jacobian
# linearization point. Bumping N_EPISODES still runs the noise pipeline for
# additional episodes but they start from their own init_states (no override).
N_EPISODES="${N_EPISODES:-50}"
TASK_ID="${TASK_ID:-0}"
SUITE="${SUITE:-libero_10}"
RESOLUTION="${RESOLUTION:-256}"
VIDEO_FPS="${VIDEO_FPS:-30}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
# Cap on env steps per rollout. Overrides the suite's TASK_MAX_STEPS
# (libero_10=520). Default 1000 leaves headroom for steered rollouts that
# are slower than the baseline trajectory distribution.
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1000}"
# 1 = also run an unsteered baseline rollout per episode from the same
# init_state + noise seed (videos suffixed `__baseline`). Set 0 to skip.
RUN_BASELINE="${RUN_BASELINE:-0}"
SEED="${SEED:-1}"  # empty -> use config.json's seed (or 42)

# --- Override + noise -------------------------------------------------------
# PAIR_DIR points at the directory produced by
# notebooks/lqr/inputs/collect_policy_inputs_noise.ipynb (contains
# positive.npz / negative.npz). OBS_INDEX selects which row of negative.npz
# is fed to ep0/chunk0; it MUST match the --obs-index used when the jacobians
# under JAC_DIR_ACT were computed (default 0 = first inference of first
# rollout, which is what run_jacobians_full.sh / run_jacobians_text.sh
# default to).
# PAIR_DIR="${PAIR_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_extreme_pos_neg}"
PAIR_DIR="${PAIR_DIR:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/policy_inputs/libero_10__task00__noise_high_4_pos_neg}"
OBS_INDEX="${OBS_INDEX:-0}"
NOISE_SIGMA="${NOISE_SIGMA:-75.0}"             # noise_extreme: 90, noise_high_4: 75
NOISE_PER_EPISODE_SEED="${NOISE_PER_EPISODE_SEED:-1}"   # 1 = per-episode seed
# Per-episode RNG seed is NOISE_SEED_BASE + ep_idx. Default 0 matches the
# seed used by collect_policy_inputs_noise.ipynb; set to any non-zero offset
# (e.g. 1000) to decorrelate runtime noise from the contrastive negatives.
NOISE_SEED_BASE="${NOISE_SEED_BASE:-0}"
NO_OVERRIDE_FIRST_CHUNK="${NO_OVERRIDE_FIRST_CHUNK:-1}" # 1 = also noise chunk 0

# --- SVD / jacobian inputs --------------------------------------------------
# SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_noise_extreme_pairs_dsall_N-1_k64_p10_ws8_tsall}"
# JAC_DIR_ACT="${JAC_DIR_ACT:-A_tilde_full__put_both_the_alphabet_soup_and_the_tomato_sauce_in__vjp_no_retain__vbf16}"
SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_noise_high_4_pairs_no_action_dsall_N-1_k64_p10_ws8_tsall}"
JAC_DIR_ACT="${JAC_DIR_ACT:-A_tilde_full__put_both_the_alphabet_soup_and_the_tomato_sauce_in__vjp_no_retain__vbf16}"


# --- Slurm resources --------------------------------------------------------
WORKER_TIME="${WORKER_TIME:-${TIME:-00:40:00}}"
MERGE_TIME="${MERGE_TIME:-00:10:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
# Comma-separated nodes to avoid. Defaults match sibling launchers; set
# EXCLUDE_NODES="" to opt out.
# shellcheck disable=SC1091
. "$(dirname -- "${BASH_SOURCE[0]}")/../exclude_nodes.sh"
EXCLUDE_NODES="${EXCLUDE_NODES-$EXCLUDE_NODES_DEFAULT}"
# --- Output naming ----------------------------------------------------------
_slug() { local s="$1"; local n="${2:-24}"; echo "${s:0:$n}" | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_-'; }
PROMPT_SLUG="$(_slug "$PROMPT" 24)"
LQR_TAG="lam${LAMBDA}_q${Q_SCALE}_rinit${R_SCALE}_rfin${R_SCALE_FINAL}_tau${R_SCALE_TAU}_qf${QF_SCALE}"
NOISE_TAG="s${NOISE_SIGMA}_obs${OBS_INDEX}_sb${NOISE_SEED_BASE}"
[[ "$NO_OVERRIDE_FIRST_CHUNK" == "1" ]] && NOISE_TAG="${NOISE_TAG}_noov"

DEFAULT_TAG="${SUITE}__task$(printf '%02d' "$TASK_ID")__lqr_noised_decay__${NOISE_TAG}__${LQR_TAG}__${PROMPT_SLUG}"
RUN_TAG="${RUN_TAG:-}"
CONFIG_TAG="${CONFIG_TAG:-$DEFAULT_TAG}"
if [[ -n "$RUN_TAG" ]]; then
    CONFIG_TAG="${CONFIG_TAG}__${RUN_TAG}"
fi

TAG="${TAG:-seed1}"
OUT_BASE="${OUT_BASE:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/rollouts}"
OUT_DIR="$OUT_BASE${TAG:+/$TAG}/$CONFIG_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_lqr_cosmos_policy_noised.py"

MERGE_ONLY="${MERGE_ONLY:-0}"

[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
if [[ "$MERGE_ONLY" != "1" ]]; then
    [[ -d "$SVD_DIR"  ]] || { echo "ERROR: SVD_DIR not found: $SVD_DIR" >&2; exit 1; }
    [[ -f "$SVD_DIR/config.json" ]] || {
        echo "ERROR: $SVD_DIR has no config.json -- finalize the SVD first" >&2
        exit 1
    }
    [[ -f "$SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" ]] || {
        echo "ERROR: A_tilde missing at $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" >&2
        exit 1
    }
    [[ -f "$PAIR_DIR/negative.npz" ]] || {
        echo "ERROR: $PAIR_DIR/negative.npz not found" >&2
        exit 1
    }
fi

# --- Echo config ------------------------------------------------------------
echo "=== run_lqr_cosmos_policy_noised ==="
echo "  WORLD_SIZE     = $WORLD_SIZE"
echo "  N_EPISODES     = $N_EPISODES"
echo "  SVD_DIR        = $SVD_DIR"
echo "  JAC_DIR_ACT    = $JAC_DIR_ACT"
echo "  PAIR_DIR       = $PAIR_DIR  (obs_index=$OBS_INDEX)"
echo "  NOISE          = sigma=$NOISE_SIGMA  per_episode_seed=$NOISE_PER_EPISODE_SEED  seed_base=$NOISE_SEED_BASE"
echo "                   (seed_base=0 matches collect_policy_inputs_noise; non-zero decorrelates)"
echo "  override       = $([[ "$NO_OVERRIDE_FIRST_CHUNK" == "1" ]] && echo disabled || echo "ep0/chunk0 from negative.npz")"
echo "  PROMPT         = $PROMPT"
echo "  LAMBDA         = $LAMBDA  Q_SCALE = $Q_SCALE  QF_SCALE = $QF_SCALE"
echo "  R_SCALE (init) = $R_SCALE  R_SCALE_TAU = $R_SCALE_TAU  R_SCALE_FINAL = $R_SCALE_FINAL"
echo "  MAX_CHUNKS     = $MAX_CHUNKS"
echo "  TASK_ID        = $TASK_ID  SUITE = $SUITE"
echo "  MAX_ENV_STEPS  = $MAX_ENV_STEPS  RUN_BASELINE = $RUN_BASELINE"
echo "  RESOLUTION     = $RESOLUTION  VIDEO_FPS = $VIDEO_FPS  SEED = ${SEED:-<cfg>}"
echo "  OUT_DIR        = $OUT_DIR"
echo "  account=$ACCOUNT  partition=$PARTITION_SLURM  worker_time=$WORKER_TIME  mem=$MEM"
echo "  merge_only=$MERGE_ONLY  exclude_nodes=${EXCLUDE_NODES:-<none>}"
echo

# --- Build common CLI args for the python script ---------------------------
COMMON_ARGS=(
    --svd-dir "$SVD_DIR"
    --jac-dir-act "$JAC_DIR_ACT"
    --pair-dir "$PAIR_DIR"
    --obs-index "$OBS_INDEX"
    --noise-sigma "$NOISE_SIGMA"
    --noise-seed-base "$NOISE_SEED_BASE"
    --prompt "$PROMPT"
    --lambda-scale "$LAMBDA"
    --q-scale "$Q_SCALE"
    --r-scale "$R_SCALE"
    --r-scale-tau "$R_SCALE_TAU"
    --r-scale-final "$R_SCALE_FINAL"
    --max-chunks "$MAX_CHUNKS"
    --qf-scale "$QF_SCALE"
    --n-episodes "$N_EPISODES"
    --task-id "$TASK_ID"
    --suite "$SUITE"
    --resolution "$RESOLUTION"
    --video-fps "$VIDEO_FPS"
    --num-steps-wait "$NUM_STEPS_WAIT"
    --max-env-steps "$MAX_ENV_STEPS"
    --world-size "$WORLD_SIZE"
    --out-dir "$OUT_DIR"
)
if [[ "$NOISE_PER_EPISODE_SEED" == "1" || "$NOISE_PER_EPISODE_SEED" == "true" ]]; then
    COMMON_ARGS+=( --noise-per-episode-seed )
else
    COMMON_ARGS+=( --no-noise-per-episode-seed )
fi
if [[ "$NO_OVERRIDE_FIRST_CHUNK" == "1" || "$NO_OVERRIDE_FIRST_CHUNK" == "true" ]]; then
    COMMON_ARGS+=( --no-override-first-chunk )
fi
if [[ "$RUN_BASELINE" == "1" || "$RUN_BASELINE" == "true" ]]; then
    COMMON_ARGS+=( --run-baseline )
else
    COMMON_ARGS+=( --no-baseline )
fi
if [[ -n "$SEED" ]]; then
    COMMON_ARGS+=( --seed "$SEED" )
fi
if [[ -n "$TAG" ]]; then
    COMMON_ARGS+=( --tag "$TAG" )
fi

ARGS_QUOTED=""
for a in "${COMMON_ARGS[@]}"; do
    ARGS_QUOTED+=" $(printf '%q' "$a")"
done

# T5 cache wipe (matches sibling launchers).
T5_CLEAR='
python -u -c "
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
"'

# --- Stage 1: rollout array (one rank per task) -----------------------------
if [[ "$MERGE_ONLY" == "1" ]]; then
    echo "=== MERGE_ONLY=1; submitting merge-only job (no rollout array) ==="
    MERGE_ONLY_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION_SLURM" \
        --job-name="lqrnoise_merge_only_${CONFIG_TAG:0:48}" \
        --gpus-per-task=1 \
        --ntasks=1 \
        --cpus-per-task=2 \
        --mem=16G \
        --time="$MERGE_TIME" \
        --output="$LOG_DIR/merge_only_%j.out" \
        --error="$LOG_DIR/merge_only_%j.err" \
        --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
python -u '$PY_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --svd-dir '$SVD_DIR' --jac-dir-act '$JAC_DIR_ACT' \
    --pair-dir '$PAIR_DIR' --prompt '$PROMPT'
")
    echo "  merge-only job id: $MERGE_ONLY_ID"
    echo "  out_dir: $OUT_DIR"
    echo "  Tail log: tail -f $LOG_DIR/merge_only_${MERGE_ONLY_ID}.out"
    exit 0
fi

ROLLOUT_JOB_NAME="lqrnoise_${CONFIG_TAG:0:60}"
LAST_RANK=$((WORLD_SIZE - 1))
echo "=== submitting rollout array (0-$LAST_RANK) ==="

ROLLOUT_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$ROLLOUT_JOB_NAME" \
    --array=0-"$LAST_RANK" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$WORKER_TIME" \
    --output="$LOG_DIR/rollout_rank%a_%A.out" \
    --error="$LOG_DIR/rollout_rank%a_%A.err" \
    --export=ALL \
    ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE\"
$T5_CLEAR
python -u '$PY_SCRIPT' --phase rollout --rank \"\$SLURM_ARRAY_TASK_ID\"$ARGS_QUOTED
")
echo "  rollout job id: $ROLLOUT_ID  (array 0-$LAST_RANK)"

# --- Stage 2: merge (depends on all rollout ranks) -------------------------
MERGE_JOB_NAME="lqrnoise_merge_${CONFIG_TAG:0:48}"
echo "=== submitting merge (depends on $ROLLOUT_ID) ==="

MERGE_ID=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --job-name="$MERGE_JOB_NAME" \
    --dependency=afterok:"$ROLLOUT_ID" \
    --gpus-per-task=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --time="$MERGE_TIME" \
    --output="$LOG_DIR/merge_%j.out" \
    --error="$LOG_DIR/merge_%j.err" \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$SCRIPT_DIR'
python -u '$PY_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --svd-dir '$SVD_DIR' --jac-dir-act '$JAC_DIR_ACT' \
    --pair-dir '$PAIR_DIR' --prompt '$PROMPT'
")
echo "  merge job id:   $MERGE_ID  (afterok:$ROLLOUT_ID)"

echo
echo "=== submitted ==="
echo "  rollout : sbatch $ROLLOUT_ID  (array 0-$LAST_RANK)"
echo "  merge   : sbatch $MERGE_ID    (afterok:$ROLLOUT_ID)"
echo "  out_dir : $OUT_DIR"
echo "  logs    : $LOG_DIR"
echo
echo "Monitor:        squeue -u \$USER --start"
echo "Tail rank 0:    tail -f $LOG_DIR/rollout_rank0_${ROLLOUT_ID}.out"
echo "Tail merge:     tail -f $LOG_DIR/merge_${MERGE_ID}.out"
