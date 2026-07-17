#!/bin/bash
# Slurm launcher to execute run_lqr_cosmos_policy.ipynb headlessly on a single
# GH200 node. Mirrors collect_policy_inputs.sh.
#
# The notebook reads SVD_DIR / JAC_DIR_ACT and the LQR / rollout knobs from
# the environment (see cells 8, 15, 20, 22 of the .ipynb). Override via env
# vars to point at a different SVD run / tune the LQR / change the task.
#
# Usage:
#   ./run_lqr_cosmos_policy.sh
#   PROMPT="open the drawer" LAMBDA=1.5 Q_SCALE=10000 R_SCALE=50000 ./run_lqr_cosmos_policy.sh
#   SVD_DIR=/path/to/svd_run TIME=04:00:00 N_EPISODES=5 ./run_lqr_cosmos_policy.sh

set -euo pipefail

NB_NAME="${NB_NAME:-run_lqr_cosmos_policy.ipynb}"
TIME="${TIME:-00:20:00}"
ACCOUNT="${ACCOUNT:-bhde-dtai-gh}"
PARTITION_SLURM="${PARTITION_SLURM:-ghx4-interactive}"
# Comma-separated list of nodes to avoid. Defaults to nodes we've seen hang
# the bash prologue with 0 CPU / 0 bytes of output. Note the missing ':' --
# the `${VAR-default}` form means "unset -> default, but an explicit empty
# string is honored as empty," so `EXCLUDE_NODES="" ./run_lqr_*.sh` lets you
# opt out of the exclusion when those nodes have been fixed.
# shellcheck disable=SC1091
. "$(dirname -- "${BASH_SOURCE[0]}")/../exclude_nodes.sh"
EXCLUDE_NODES="${EXCLUDE_NODES-$EXCLUDE_NODES_DEFAULT}"
# --- SVD / jacobian inputs (cell 4 of the notebook). ------------------------
SVD_DIR="${SVD_DIR:-/u/jhong7/cosmos-policy/directions/svd/libero10_task00_milk_pairs_dsall_N-1_k64_p10_ws8_tsall}"
JAC_DIR_ACT="${JAC_DIR_ACT:-A_tilde_full__put_both_the_alphabet_soup_and_the_tomato_sauce_in__vjp_no_retain__vbf16}"

# --- LQR / rollout knobs (cells 8, 15, 20, 22). -----------------------------
PROMPT="${PROMPT:-put both the alphabet soup and the tomato sauce in the basket. the cream cheese, ketchup, orange juice, milk, and butter are also on the table.}"
LAMBDA="${LAMBDA:-10.0}"
Q_SCALE="${Q_SCALE:-1.0}"
R_SCALE="${R_SCALE:-10.0}"
QF_SCALE="${QF_SCALE:-1.0}"
N_EPISODES="${N_EPISODES:-1}"

# Disambiguator appended to the save dir when a prior run with the same
# prefix already exists. The notebook's CONFIG_TAG only keeps the first 24
# chars of PROMPT, so two prompts sharing that prefix collide. Set RUN_TAG
# (e.g. "no_milk", "v2") before launching when you're about to run a prompt
# that overlaps an existing rollouts/ subdir.
RUN_TAG="${RUN_TAG:-nonzero_obs}"
TASK_ID="${TASK_ID:-0}"
SUITE_NAME="${SUITE_NAME:-libero_10}"
RESOLUTION="${RESOLUTION:-256}"
VIDEO_FPS="${VIDEO_FPS:-30}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
NB="$SCRIPT_DIR/$NB_NAME"
[[ -f "$NB" ]] || { echo "ERROR: notebook not found: $NB" >&2; exit 1; }

NB_DIR="$(dirname "$NB")"
NB_BASE="$(basename "$NB" .ipynb)"
OUT_NB="${NB_BASE}.executed.ipynb"

mkdir -p "$SCRIPT_DIR/logs"

echo "submitting $NB_NAME"
echo "  SVD_DIR     = $SVD_DIR"
echo "  JAC_DIR_ACT = $JAC_DIR_ACT"
echo "  PROMPT      = $PROMPT"
echo "  LAMBDA      = $LAMBDA"
echo "  Q_SCALE     = $Q_SCALE"
echo "  R_SCALE     = $R_SCALE"
echo "  QF_SCALE    = $QF_SCALE"
echo "  N_EPISODES  = $N_EPISODES"
echo "  TASK_ID     = $TASK_ID"
echo "  SUITE_NAME  = $SUITE_NAME"
echo "  RESOLUTION  = $RESOLUTION"
echo "  VIDEO_FPS   = $VIDEO_FPS"
echo "  RUN_TAG     = ${RUN_TAG:-<none>}"
echo "  account=$ACCOUNT  partition=$PARTITION_SLURM  time=$TIME"
echo "  exclude_nodes=${EXCLUDE_NODES:-<none>}"

# sbatch's --export list is comma-separated, so a value containing a comma
# (e.g. PROMPT) gets silently truncated at the first comma and the rest is
# parsed as bogus key=value entries. Export the vars into this shell's env
# and use --export=ALL so Slurm inherits them through the environment rather
# than parsing them out of a CSV.
export SVD_DIR JAC_DIR_ACT PROMPT LAMBDA Q_SCALE R_SCALE QF_SCALE
export N_EPISODES TASK_ID SUITE_NAME RESOLUTION VIDEO_FPS RUN_TAG

sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION_SLURM" \
    --gpus-per-node=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --time="$TIME" \
    --job-name="nb_${NB_BASE}" \
    --output="$SCRIPT_DIR/logs/nb_%j.out" \
    --error="$SCRIPT_DIR/logs/nb_%j.err" \
    --export=ALL \
    ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
    --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$NB_DIR'
nvidia-smi -L

# Wipe the T5 prompt-embedding cache at the start of every job so corruption
# left behind by a crashed run can't propagate. The notebook's on-demand
# path (cosmos_utils.init_t5_text_embeddings_cache then
# get_t5_embedding_from_cache) will recompute embeddings for the prompts
# this run actually uses and persist them back to this file. We write an
# empty pickle (not an unlink) so init still sets the global path; otherwise
# subsequent on-demand writes would be silently skipped with a warning.
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

echo 'executing: $NB'
echo '  SVD_DIR=$SVD_DIR'
echo '  JAC_DIR_ACT=$JAC_DIR_ACT'
echo '  PROMPT=$PROMPT'
echo '  LAMBDA=$LAMBDA  Q_SCALE=$Q_SCALE  R_SCALE=$R_SCALE  QF_SCALE=$QF_SCALE'
echo '  N_EPISODES=$N_EPISODES  TASK_ID=$TASK_ID  SUITE_NAME=$SUITE_NAME'
jupyter nbconvert --to notebook --execute '$NB' \
    --ExecutePreprocessor.kernel_name=cosmos-policy \
    --ExecutePreprocessor.timeout=14400 \
    --output '$OUT_NB'
echo 'wrote: $NB_DIR/$OUT_NB'
"
