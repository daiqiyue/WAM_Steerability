#!/bin/bash
# Server-client launcher for LQR-steered noise rollouts.
#
# Each SLURM array task (one GPU) runs:
#   1. server_lqr_noised.py  (dit4dit env) — loads model + LQR, serves on unique port
#   2. client_lqr_noised.py  (libero env)  — runs LIBERO sim, connects to server
#   3. Merge job (dit4dit env) — aggregates results_rank*.json → results.json
#
# Usage:
#   ./run_lqr_dit4dit_noised.sh
#   WORLD_SIZE=4 N_EPISODES=50 NOISE_SIGMA=75 ./run_lqr_dit4dit_noised.sh
#   MERGE_ONLY=1 ./run_lqr_dit4dit_noised.sh

set -euo pipefail

DIT4DIT_ROOT=/work/hdd/bhde/jhong7/DiT4DiT
LQR_ROOT=$DIT4DIT_ROOT/notebooks/lqr

MODEL_PYTHON=/projects/bhde/jhong7/dit4dit-env/dit4dit/bin/python
LIBERO_PYTHON=/work/nvme/bhde/jhong7/LIBERO_pkg/.conda/envs/libero/bin/python

WORLD_SIZE="${WORLD_SIZE:-4}"
N_EPISODES="${N_EPISODES:-50}"
PORT_BASE="${PORT_BASE:-5700}"   # port = PORT_BASE + SLURM_ARRAY_TASK_ID
SERVER_WAIT="${SERVER_WAIT:-60}" # seconds for model + LQR to load

# Noise knobs
NOISE_SIGMA="${NOISE_SIGMA:-75.0}"
NOISE_SEED_BASE="${NOISE_SEED_BASE:-0}"

# LQR cost (exponential R decay: R(c) = min(r_scale * exp(c/tau), r_scale_final))
LAMBDA="${LAMBDA:-1.0}"
Q_SCALE="${Q_SCALE:-10000.0}"
R_SCALE="${R_SCALE:-75000.0}"
R_SCALE_TAU="${R_SCALE_TAU:-3.0}"
R_SCALE_FINAL="${R_SCALE_FINAL:-1e9}"
MAX_CHUNKS="${MAX_CHUNKS:-50}"
QF_SCALE="${QF_SCALE:-1.0}"

# Rollout
PROMPT="${PROMPT:-put both the cream cheese box and the butter in the basket}"
TASK_ID="${TASK_ID:-1}"
SUITE="${SUITE:-libero_10}"
RESOLUTION="${RESOLUTION:-256}"
VIDEO_FPS="${VIDEO_FPS:-30}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1000}"
RUN_BASELINE="${RUN_BASELINE:-1}"
SEED="${SEED:-1}"

SVD_DIR="${SVD_DIR:-}"
JAC_DIR_ACT="${JAC_DIR_ACT:-}"
if [[ -z "$SVD_DIR" || -z "$JAC_DIR_ACT" ]]; then
    echo "ERROR: SVD_DIR and JAC_DIR_ACT must be set." >&2; exit 1
fi

WORKER_TIME="${WORKER_TIME:-02:00:00}"
MERGE_TIME="${MERGE_TIME:-00:10:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-128G}"

CKPT_PATH="${CKPT_PATH:-$DIT4DIT_ROOT/checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt}"

_slug() { local s="$1"; local n="${2:-24}"; echo "${s:0:$n}" | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_-'; }
PROMPT_SLUG="$(_slug "$PROMPT" 24)"
LQR_TAG="lam${LAMBDA}_q${Q_SCALE}_r${R_SCALE}_qf${QF_SCALE}"
NOISE_TAG="noise_sigma${NOISE_SIGMA}"
DEFAULT_TAG="${SUITE}__task$(printf '%02d' "$TASK_ID")__lqr_noised__${NOISE_TAG}__${LQR_TAG}__${PROMPT_SLUG}"
RUN_TAG="${RUN_TAG:-}"
CONFIG_TAG="${CONFIG_TAG:-$DEFAULT_TAG}"
[[ -n "$RUN_TAG" ]] && CONFIG_TAG="${CONFIG_TAG}__${RUN_TAG}"

TAG="${TAG:-seed1}"
OUT_BASE="${OUT_BASE:-$LQR_ROOT/rollouts}"
OUT_DIR="$OUT_BASE${TAG:+/$TAG}/$CONFIG_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SERVER_SCRIPT="$DIT4DIT_ROOT/deployment/model_server/server_lqr.py"
CLIENT_SCRIPT="$SCRIPT_DIR/client_lqr_noised.py"
MERGE_SCRIPT="$SCRIPT_DIR/run_lqr_dit4dit_noised.py"
MERGE_ONLY="${MERGE_ONLY:-0}"

[[ -f "$SERVER_SCRIPT" ]] || { echo "ERROR: missing $SERVER_SCRIPT" >&2; exit 1; }
[[ -f "$CLIENT_SCRIPT" ]] || { echo "ERROR: missing $CLIENT_SCRIPT" >&2; exit 1; }

echo "=== run_lqr_dit4dit_noised (server-client) ==="
echo "  WORLD_SIZE   = $WORLD_SIZE"
echo "  N_EPISODES   = $N_EPISODES"
echo "  NOISE_SIGMA  = $NOISE_SIGMA"
echo "  SVD_DIR      = $SVD_DIR"
echo "  JAC_DIR_ACT  = $JAC_DIR_ACT"
echo "  PROMPT       = $PROMPT"
echo "  LAMBDA=$LAMBDA  Q=$Q_SCALE  R=$R_SCALE  QF=$QF_SCALE"
echo "  TASK_ID=$TASK_ID  SUITE=$SUITE"
echo "  OUT_DIR      = $OUT_DIR"
echo "  PORT_BASE    = $PORT_BASE  (port = PORT_BASE + rank)"
echo

# ── Server args ───────────────────────────────────────────────────────────────
SERVER_ARGS="--svd-dir '$SVD_DIR' --jac-dir-act '$JAC_DIR_ACT' \
    --lambda-scale '$LAMBDA' --q-scale '$Q_SCALE' --r-scale '$R_SCALE' \
    --r-scale-tau '$R_SCALE_TAU' --r-scale-final '$R_SCALE_FINAL' \
    --max-chunks '$MAX_CHUNKS' --qf-scale '$QF_SCALE' \
    --ckpt-path '$CKPT_PATH'"

# ── Client args ───────────────────────────────────────────────────────────────
CLIENT_COMMON=(
    --n-episodes "$N_EPISODES"
    --task-id "$TASK_ID"
    --suite "$SUITE"
    --resolution "$RESOLUTION"
    --video-fps "$VIDEO_FPS"
    --num-steps-wait "$NUM_STEPS_WAIT"
    --max-env-steps "$MAX_ENV_STEPS"
    --noise-sigma "$NOISE_SIGMA"
    --noise-seed-base "$NOISE_SEED_BASE"
    --world-size "$WORLD_SIZE"
    --out-dir "$OUT_DIR"
    --seed "$SEED"
    --tag "$TAG"
)
[[ "$RUN_BASELINE" == "1" ]] && CLIENT_COMMON+=(--run-baseline) || CLIENT_COMMON+=(--no-baseline)
CLIENT_COMMON+=(--save-video)
CLIENT_ARGS_QUOTED=""
for a in "${CLIENT_COMMON[@]}"; do CLIENT_ARGS_QUOTED+=" $(printf '%q' "$a")"; done

# ── dit4dit env activation for server / merge ─────────────────────────────────
DIT4DIT_ACTIVATE="
set -euo pipefail
unset WORLD_SIZE LOCAL_RANK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH='$DIT4DIT_ROOT':\${PYTHONPATH:-}
export LIBERO_HOME=/work/nvme/bhde/jhong7/LIBERO_pkg
export LIBERO_CONFIG_PATH=\$LIBERO_HOME/libero
export PYTHONUTF8=1; export PYTHONIOENCODING=utf-8
"

# ── libero env activation for client ─────────────────────────────────────────
LIBERO_ACTIVATE="
export PYTHONPATH='$DIT4DIT_ROOT':/work/nvme/bhde/jhong7/LIBERO_pkg:\${PYTHONPATH:-}
export LIBERO_HOME=/work/nvme/bhde/jhong7/LIBERO_pkg
export LIBERO_CONFIG_PATH=\$LIBERO_HOME/libero
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl
export PYTHONUTF8=1; export PYTHONIOENCODING=utf-8
"

# ── Merge-only path ───────────────────────────────────────────────────────────
if [[ "$MERGE_ONLY" == "1" ]]; then
    MERGE_ONLY_ID=$(sbatch --parsable \
        --account="$ACCOUNT" --partition="$PARTITION_SLURM" \
        --job-name="dit4dit_noised_merge_${CONFIG_TAG:0:44}" \
        --gpus-per-task=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time="$MERGE_TIME" \
        --output="$LOG_DIR/merge_only_%j.out" --error="$LOG_DIR/merge_only_%j.err" \
        --wrap "${DIT4DIT_ACTIVATE}
'$MODEL_PYTHON' -u '$MERGE_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --svd-dir '$SVD_DIR' --jac-dir-act '$JAC_DIR_ACT' --prompt '$PROMPT' --ckpt-path '$CKPT_PATH'
")
    echo "  merge-only job id: $MERGE_ONLY_ID"
    exit 0
fi

# ── Rollout array: each task = server (dit4dit) + client (libero) ─────────────
LAST_RANK=$((WORLD_SIZE - 1))
ROLLOUT_ID=$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$PARTITION_SLURM" \
    --job-name="dit4dit_noised_${CONFIG_TAG:0:48}" \
    --array=0-"$LAST_RANK" --gpus-per-task=1 --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM" --time="$WORKER_TIME" \
    --output="$LOG_DIR/rollout_rank%a_%A.out" --error="$LOG_DIR/rollout_rank%a_%A.err" \
    --wrap "
set -euo pipefail
RANK=\$SLURM_ARRAY_TASK_ID
PORT=\$(( $PORT_BASE + RANK ))
echo \"=== rank=\$RANK  port=\$PORT ===\"
nvidia-smi -L

# ── Start LQR server (dit4dit env) in background ──────────────────────────────
${DIT4DIT_ACTIVATE}
'$MODEL_PYTHON' -u '$SERVER_SCRIPT' $SERVER_ARGS --port \"\$PORT\" \
    > '$LOG_DIR/server_rank'\"\$RANK\"'_%j.log' 2>&1 &
SERVER_PID=\$!
echo \"Server PID=\$SERVER_PID on port \$PORT — waiting ${SERVER_WAIT}s...\"
sleep $SERVER_WAIT

# ── Run client (libero env) ───────────────────────────────────────────────────
${LIBERO_ACTIVATE}
'$LIBERO_PYTHON' -u '$CLIENT_SCRIPT' \
    --host 127.0.0.1 --port \"\$PORT\" --rank \"\$RANK\"$CLIENT_ARGS_QUOTED

# ── Shutdown server ───────────────────────────────────────────────────────────
kill \$SERVER_PID 2>/dev/null && echo \"Killed server PID \$SERVER_PID\" || true
")
echo "  rollout job id: $ROLLOUT_ID  (array 0-$LAST_RANK)"

# ── Merge job (depends on all rollout tasks) ──────────────────────────────────
MERGE_ID=$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$PARTITION_SLURM" \
    --job-name="dit4dit_noised_merge_${CONFIG_TAG:0:40}" \
    --dependency=afterany:"$ROLLOUT_ID" \
    --gpus-per-task=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time="$MERGE_TIME" \
    --output="$LOG_DIR/merge_%j.out" --error="$LOG_DIR/merge_%j.err" \
    --wrap "${DIT4DIT_ACTIVATE}
'$MODEL_PYTHON' -u '$MERGE_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --svd-dir '$SVD_DIR' --jac-dir-act '$JAC_DIR_ACT' --prompt '$PROMPT' --ckpt-path '$CKPT_PATH'
")
echo "  merge job id:   $MERGE_ID  (afterany:$ROLLOUT_ID)"
echo
echo "  out_dir: $OUT_DIR"
echo "  Tail:    tail -f $LOG_DIR/rollout_rank0_${ROLLOUT_ID}.out"
echo "  Server:  tail -f $LOG_DIR/server_rank0_<job_id>.log"
