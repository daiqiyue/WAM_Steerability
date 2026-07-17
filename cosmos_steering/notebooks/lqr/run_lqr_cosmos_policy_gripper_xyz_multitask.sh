#!/bin/bash
# Slurm launcher for run_lqr_cosmos_policy_gripper_xyz_multitask.py.
#
# Submits an sbatch array of WORLD_SIZE workers (one GPU each), each handling
# its slice of the flat (task_id, episode_idx) job list via --rank R
# --world-size W. After they all finish, a dependent merge job aggregates
# per-rank results into results.json and patches the top-level manifest.json.
#
# With the defaults (TASK_IDS="0 1 2 3 4 5 6 7 8 9", N_EPISODES_PER_TASK=10,
# WORLD_SIZE=10) each rank handles exactly one task's 10 episodes — env built
# once per rank, no reshuffling.

set -euo pipefail

# --- Parallelization --------------------------------------------------------
WORLD_SIZE="${WORLD_SIZE:-10}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
N_EPISODES_PER_TASK="${N_EPISODES_PER_TASK:-10}"

# --- Gripper perturbation ---------------------------------------------------
# Default preset matches the contrastive inputs that fed
# /projects/bhde/jhong7/.../libero10_tasks0-2-4-5_xyz_random_xlarge_2_seed42_paired_multitask_...
# Use BASE_SEED=99 to draw a perturbation distribution distinct from the
# seed=42 used for those inputs (so the LQR is evaluated on out-of-sample
# perturbations).
PRESET="${PRESET:-xyz_random_xlarge_2}"
BASE_SEED="${BASE_SEED:-99}"
GRIPPER_ACTION="${GRIPPER_ACTION:--1.0}"

# --- LQR cost hyperparameters ----------------------------------------------
LAMBDA="${LAMBDA:-5.0}"
Q_SCALE="${Q_SCALE:-10.0}"
R_SCALE="${R_SCALE:-10.0}"
R_SCALE_TAU="${R_SCALE_TAU:-5.0}"
R_SCALE_FINAL="${R_SCALE_FINAL:-1e9}"
MAX_CHUNKS="${MAX_CHUNKS:-50}"
QF_SCALE="${QF_SCALE:-1.0}"

# --- Rollout knobs ----------------------------------------------------------
# Default prompt is empty -> per-task libero default (matches the multitask
# input-collection script). Set PROMPT="<some text>" to override for ALL tasks.
PROMPT="${PROMPT:-}"
SUITE="${SUITE:-libero_10}"
RESOLUTION="${RESOLUTION:-256}"
VIDEO_FPS="${VIDEO_FPS:-30}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1000}"
RUN_BASELINE="${RUN_BASELINE:-0}"  # 1 = also run unsteered baseline per episode
# BASELINE_ONLY=1 skips the LQR-steered pass entirely. Used to produce a
# reference rate for the sweep summary's Δ-vs-base column. Forces CONFIG_TAG
# to "__baseline__" so the summarizer can pick it up via --baseline-dir.
BASELINE_ONLY="${BASELINE_ONLY:-0}"
SEED="${SEED:-1}"

# --- SVD / jacobian inputs --------------------------------------------------
# Default to the multitask paired SVD + the jacobian computed for task 0's
# prompt. Override SVD_DIR / JAC_DIR_ACT to point at any compatible artifacts.
#
# Per-task mode: set JAC_DIR_ACT_PER_TASK to a JSON file mapping task_id ->
# jacobian subdir name under SVD_DIR. When set, JAC_DIR_ACT is ignored and
# each task uses its own K matrices. Example map:
#   {"0":"A_tilde_full__task00__...","1":"A_tilde_full__task01__...",...}
SVD_DIR="${SVD_DIR:-/projects/bhde/jhong7/cosmos-policy/directions/svd/libero10_tasks0-2-4-5_xyz_random_xlarge_2_seed42_paired_multitask_dsall_N-1_k64_p10_ws8_tsall}"
JAC_DIR_ACT="${JAC_DIR_ACT:-A_tilde_full__put_both_the_alphabet_soup_and_the_tomato_sauce_in__vjp_no_retain__vbf16}"
JAC_DIR_ACT_PER_TASK="${JAC_DIR_ACT_PER_TASK:-}"

# --- Slurm resources --------------------------------------------------------
WORKER_TIME="${WORKER_TIME:-02:00:00}"
MERGE_TIME="${MERGE_TIME:-00:10:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
# Use ghx4 (non-interactive) for sweep workers. Override to ghx4-interactive
# only for quick smoke tests.
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
# shellcheck disable=SC1091
. "$(dirname -- "${BASH_SOURCE[0]}")/../exclude_nodes.sh"
EXCLUDE_NODES="${EXCLUDE_NODES-$EXCLUDE_NODES_DEFAULT}"

# --- Output naming ----------------------------------------------------------
_slug() { local s="$1"; local n="${2:-24}"; echo "${s:0:$n}" | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_-'; }
if [[ -n "$PROMPT" ]]; then
    PROMPT_SLUG="$(_slug "$PROMPT" 24)"
else
    PROMPT_SLUG="libero_default"
fi
LQR_TAG="lam${LAMBDA}_q${Q_SCALE}_rinit${R_SCALE}_rfin${R_SCALE_FINAL}_tau${R_SCALE_TAU}_qf${QF_SCALE}"
PERT_TAG="${PRESET}_seed${BASE_SEED}"
N_TASKS_SLUG="$(echo "$TASK_IDS" | wc -w | tr -d ' ')tasks_x${N_EPISODES_PER_TASK}ep"

if [[ "$BASELINE_ONLY" == "1" || "$BASELINE_ONLY" == "true" ]]; then
    # LQR hyperparams don't affect behavior in baseline-only mode (hooks gate
    # off when steering_enabled=False), so fix the tag so the summarizer can
    # find it via --baseline-dir.
    DEFAULT_TAG="__baseline__"
else
    DEFAULT_TAG="${SUITE}__multitask_${N_TASKS_SLUG}__lqr_grxyz__${PERT_TAG}__${LQR_TAG}__${PROMPT_SLUG}"
fi
# When using per-task jacobians, the output config dir lives alongside the
# single-jacobian variants, so disambiguate by injecting "pertaskjac" into
# the prompt slug segment of CONFIG_TAG.
if [[ -n "$JAC_DIR_ACT_PER_TASK" && "$BASELINE_ONLY" != "1" \
        && "$BASELINE_ONLY" != "true" ]]; then
    DEFAULT_TAG="${DEFAULT_TAG}__pertaskjac"
fi
RUN_TAG="${RUN_TAG:-}"
CONFIG_TAG="${CONFIG_TAG:-$DEFAULT_TAG}"
if [[ -n "$RUN_TAG" ]]; then
    CONFIG_TAG="${CONFIG_TAG}__${RUN_TAG}"
fi

TAG="${TAG:-grxyz_multitask_seed99}"
OUT_BASE="${OUT_BASE:-/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/rollouts}"
OUT_DIR="$OUT_BASE${TAG:+/$TAG}/$CONFIG_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_lqr_cosmos_policy_gripper_xyz_multitask.py"

MERGE_ONLY="${MERGE_ONLY:-0}"

[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }
if [[ "$MERGE_ONLY" != "1" ]]; then
    [[ -d "$SVD_DIR" ]] || { echo "ERROR: SVD_DIR not found: $SVD_DIR" >&2; exit 1; }
    [[ -f "$SVD_DIR/config.json" ]] || {
        echo "ERROR: $SVD_DIR has no config.json -- finalize the SVD first" >&2
        exit 1
    }
    if [[ -n "$JAC_DIR_ACT_PER_TASK" ]]; then
        [[ -f "$JAC_DIR_ACT_PER_TASK" ]] || {
            echo "ERROR: --jac-dir-act-per-task file not found: $JAC_DIR_ACT_PER_TASK" >&2
            exit 1
        }
        # Validate each referenced jacobian subdir exists.
        python3 -c "
import json, sys
from pathlib import Path
svd_dir = Path('$SVD_DIR')
m = json.loads(Path('$JAC_DIR_ACT_PER_TASK').read_text())
missing = []
for tid, sub in m.items():
    p = svd_dir / sub / 'A_tilde__full.pt'
    if not p.exists():
        missing.append((tid, str(p)))
if missing:
    for t, p in missing:
        print(f'  task {t}: missing {p}', file=sys.stderr)
    sys.exit(1)
" || { echo "ERROR: per-task jacobian map references missing A_tilde__full.pt files" >&2; exit 1; }
    else
        [[ -f "$SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" ]] || {
            echo "ERROR: A_tilde missing at $SVD_DIR/$JAC_DIR_ACT/A_tilde__full.pt" >&2
            exit 1
        }
    fi
fi

# --- Echo config ------------------------------------------------------------
echo "=== run_lqr_cosmos_policy_gripper_xyz_multitask ==="
echo "  WORLD_SIZE              = $WORLD_SIZE"
echo "  TASK_IDS                = $TASK_IDS"
echo "  N_EPISODES_PER_TASK     = $N_EPISODES_PER_TASK"
echo "  PRESET                  = $PRESET  (base_seed=$BASE_SEED  gripper_action=$GRIPPER_ACTION)"
echo "  SVD_DIR                 = $SVD_DIR"
if [[ -n "$JAC_DIR_ACT_PER_TASK" ]]; then
    echo "  JAC_DIR_ACT_PER_TASK    = $JAC_DIR_ACT_PER_TASK  (per-task jacobians)"
else
    echo "  JAC_DIR_ACT             = $JAC_DIR_ACT"
fi
echo "  PROMPT (override)       = ${PROMPT:-<libero per-task default>}"
echo "  LAMBDA                  = $LAMBDA  Q_SCALE = $Q_SCALE  QF_SCALE = $QF_SCALE"
echo "  R_SCALE (init)          = $R_SCALE  R_SCALE_TAU = $R_SCALE_TAU  R_SCALE_FINAL = $R_SCALE_FINAL"
echo "  MAX_CHUNKS              = $MAX_CHUNKS"
echo "  SUITE = $SUITE  RESOLUTION = $RESOLUTION  MAX_ENV_STEPS = $MAX_ENV_STEPS"
echo "  RUN_BASELINE = $RUN_BASELINE  BASELINE_ONLY = $BASELINE_ONLY  SEED = $SEED"
echo "  OUT_DIR                 = $OUT_DIR"
echo "  account=$ACCOUNT  partition=$PARTITION_SLURM  worker_time=$WORKER_TIME"
echo "  merge_only=$MERGE_ONLY  exclude_nodes=${EXCLUDE_NODES:-<none>}"
echo

# --- Build common CLI args for the python script ---------------------------
# shellcheck disable=SC2206
TASK_IDS_ARR=( $TASK_IDS )

COMMON_ARGS=(
    --svd-dir "$SVD_DIR"
    --preset "$PRESET"
    --base-seed "$BASE_SEED"
    --gripper-action "$GRIPPER_ACTION"
    --lambda-scale "$LAMBDA"
    --q-scale "$Q_SCALE"
    --r-scale "$R_SCALE"
    --r-scale-tau "$R_SCALE_TAU"
    --r-scale-final "$R_SCALE_FINAL"
    --max-chunks "$MAX_CHUNKS"
    --qf-scale "$QF_SCALE"
    --n-episodes-per-task "$N_EPISODES_PER_TASK"
    --task-ids "${TASK_IDS_ARR[@]}"
    --suite "$SUITE"
    --resolution "$RESOLUTION"
    --video-fps "$VIDEO_FPS"
    --num-steps-wait "$NUM_STEPS_WAIT"
    --max-env-steps "$MAX_ENV_STEPS"
    --world-size "$WORLD_SIZE"
    --out-dir "$OUT_DIR"
)
if [[ -n "$JAC_DIR_ACT_PER_TASK" ]]; then
    COMMON_ARGS+=( --jac-dir-act-per-task "$JAC_DIR_ACT_PER_TASK" )
else
    COMMON_ARGS+=( --jac-dir-act "$JAC_DIR_ACT" )
fi
if [[ -n "$PROMPT" ]]; then
    COMMON_ARGS+=( --prompt "$PROMPT" )
fi
if [[ "$BASELINE_ONLY" == "1" || "$BASELINE_ONLY" == "true" ]]; then
    COMMON_ARGS+=( --baseline-only )
elif [[ "$RUN_BASELINE" == "1" || "$RUN_BASELINE" == "true" ]]; then
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

# Merge-phase args (no --task-ids needed — merge just aggregates rank JSONs).
MERGE_ARGS=(
    --phase merge
    --out-dir "$OUT_DIR"
    --world-size "$WORLD_SIZE"
    --svd-dir "$SVD_DIR"
    --preset "$PRESET"
    --task-ids "${TASK_IDS_ARR[@]}"
)
if [[ -n "$JAC_DIR_ACT_PER_TASK" ]]; then
    MERGE_ARGS+=( --jac-dir-act-per-task "$JAC_DIR_ACT_PER_TASK" )
else
    MERGE_ARGS+=( --jac-dir-act "$JAC_DIR_ACT" )
fi
MERGE_ARGS_QUOTED=""
for a in "${MERGE_ARGS[@]}"; do
    MERGE_ARGS_QUOTED+=" $(printf '%q' "$a")"
done

# --- Stage 1: rollout array (one rank per slice) ----------------------------
if [[ "$MERGE_ONLY" == "1" ]]; then
    echo "=== MERGE_ONLY=1; submitting merge-only job (no rollout array) ==="
    MERGE_ONLY_ID=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --partition="$PARTITION_SLURM" \
        --job-name="lqrgxyzm_merge_only_${CONFIG_TAG:0:40}" \
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
python -u '$PY_SCRIPT'$MERGE_ARGS_QUOTED
")
    echo "  merge-only job id: $MERGE_ONLY_ID"
    echo "  out_dir: $OUT_DIR"
    echo "  Tail log: tail -f $LOG_DIR/merge_only_${MERGE_ONLY_ID}.out"
    exit 0
fi

ROLLOUT_JOB_NAME="lqrgxyzm_${CONFIG_TAG:0:40}"
LAST_RANK=$((WORLD_SIZE - 1))
echo "=== submitting rollout array (0-$LAST_RANK) ==="

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
MERGE_JOB_NAME="lqrgxyzm_merge_${CONFIG_TAG:0:40}"
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
python -u '$PY_SCRIPT'$MERGE_ARGS_QUOTED
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
