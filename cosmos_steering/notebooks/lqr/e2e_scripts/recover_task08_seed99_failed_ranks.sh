#!/bin/bash
# One-shot recovery for the 8 rollout-array tasks that aborted (SIGABRT,
# exit 134) on flaky nodes gh044 / gh049 during the task08 seed=99 sweep
# that ./run_task08_gripper_xyz_pipeline.sh launched on 2026-05-26.
#
# For each (failed_rank, OUT_DIR) pair:
#   1. scancel the original merge job (parked at DependencyNeverSatisfied)
#   2. submit a fresh single-rank rollout sbatch (with the bad nodes excluded)
#   3. submit a fresh merge sbatch depending on the new rollout
#
# Successful ranks (0,1,2 for some combos; the other three for others) and
# their per-rank results_rank{R}.json shards are left in place — the merge
# globs results_rank*.json under OUT_DIR, so re-running only the missing
# rank reconstructs the full per-combo merge.
#
# Run from anywhere; paths are absolute.
#
# Usage:
#   DRY=1 ./recover_task08_seed99_failed_ranks.sh       # print sbatch lines only
#   ./recover_task08_seed99_failed_ranks.sh             # actually submit

set -euo pipefail

DRY="${DRY:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LQR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$LQR_ROOT/run_lqr_cosmos_policy_gripper_xyz.py"
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: missing $PY_SCRIPT" >&2; exit 1; }

# --- shared (must match the original sweep submission) ----------------------
SVD_DIR="/work/hdd/bhde/jhong7/cosmos-policy/directions/svd/libero10_task08_gripper_xyz_xyz_random_xlarge_3_seed42_paired_N-1_k64_p10_ws8_tsall"
JAC_DIR_ACT="A_tilde_full__put_both_moka_pots_on_the_stove___vjp_no_retain__vbf16"
PROMPT="put both moka pots on the stove"
PRESET="xyz_random_xlarge_3"
BASE_SEED=99
TASK_ID=8
SUITE=libero_10
RESOLUTION=256
VIDEO_FPS=30
NUM_STEPS_WAIT=10
MAX_ENV_STEPS=1000
GRIPPER_ACTION=-1.0
R_SCALE_FINAL=1e9
MAX_CHUNKS=50
QF_SCALE=1.0
N_EPISODES=50
WORLD_SIZE=4
SEED=1
SWEEP_TAG=task08_seed99

# --- slurm resources (match the original sweep) -----------------------------
ACCOUNT="bhde-dtai-gh"
PARTITION_SLURM="ghx4"
WORKER_TIME="01:00:00"
MERGE_TIME="00:10:00"
CPUS_PER_TASK=8
MEM="64G"
# shellcheck disable=SC1091
. "$LQR_ROOT/../exclude_nodes.sh"
EXCLUDE_NODES="${EXCLUDE_NODES-$EXCLUDE_NODES_DEFAULT}"

# --- recovery list (RANK | OLD_PARENT | OLD_MERGE | OUT_DIR_BASENAME) ------
# Generated 2026-05-26 from:
#   sacct -u $USER --starttime=2026-05-26 ... | awk '$3=="FAILED" && /lqrgxyz_libero_10__task08/'
recoveries=(
"3|2347151|2347152|libero_10__task08__lqr_grxyz__xyz_random_xlarge_3_seed99__lam0.5_q1.0_rinit5.0_rfin1e9_tau5.0_qf1.0__put_both_moka_pots_on_th"
"0|2347169|2347170|libero_10__task08__lqr_grxyz__xyz_random_xlarge_3_seed99__lam0.5_q10.0_rinit5.0_rfin1e9_tau5.0_qf1.0__put_both_moka_pots_on_th"
"2|2347183|2347184|libero_10__task08__lqr_grxyz__xyz_random_xlarge_3_seed99__lam0.5_q10.0_rinit20.0_rfin1e9_tau7.0_qf1.0__put_both_moka_pots_on_th"
"3|2347201|2347202|libero_10__task08__lqr_grxyz__xyz_random_xlarge_3_seed99__lam5.0_q1.0_rinit20.0_rfin1e9_tau7.0_qf1.0__put_both_moka_pots_on_th"
"2|2347217|2347218|libero_10__task08__lqr_grxyz__xyz_random_xlarge_3_seed99__lam5.0_q10.0_rinit20.0_rfin1e9_tau5.0_qf1.0__put_both_moka_pots_on_th"
"0|2347219|2347220|libero_10__task08__lqr_grxyz__xyz_random_xlarge_3_seed99__lam5.0_q10.0_rinit20.0_rfin1e9_tau7.0_qf1.0__put_both_moka_pots_on_th"
"0|2347253|2347254|libero_10__task08__lqr_grxyz__xyz_random_xlarge_3_seed99__lam10.0_q10.0_rinit20.0_rfin1e9_tau5.0_qf1.0__put_both_moka_pots_on_th"
"2|2347289|2347290|libero_10__task08__lqr_grxyz__xyz_random_xlarge_3_seed99__lam15.0_q10.0_rinit20.0_rfin1e9_tau5.0_qf1.0__put_both_moka_pots_on_th"
)

OUT_BASE="$LQR_ROOT/rollouts/$SWEEP_TAG"

echo "=== plan ==="
echo "  DRY            : $DRY  (1 = print, 0 = submit)"
echo "  recoveries     : ${#recoveries[@]}"
echo "  exclude_nodes  : $EXCLUDE_NODES"
echo "  svd_dir        : $SVD_DIR"
echo "  jac_dir_act    : $JAC_DIR_ACT"
echo

# T5 cache wipe — matches the original launcher.
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

# ---------- 1) cancel the dead merges ----------
DEAD_MERGES=()
for entry in "${recoveries[@]}"; do
    IFS='|' read -r _RANK _OLD_PARENT OLD_MERGE _BASENAME <<< "$entry"
    DEAD_MERGES+=("$OLD_MERGE")
done
echo "=== scancel dead merges ==="
echo "  IDs: ${DEAD_MERGES[*]}"
if [[ "$DRY" == "0" ]]; then
    scancel "${DEAD_MERGES[@]}" || true
fi
echo

# ---------- 2) resubmit each failed rank + new merge ----------
SUBMITTED=()
for entry in "${recoveries[@]}"; do
    IFS='|' read -r RANK OLD_PARENT OLD_MERGE BASENAME <<< "$entry"
    OUT_DIR="$OUT_BASE/$BASENAME"
    LOG_DIR="$OUT_DIR/logs"
    mkdir -p "$LOG_DIR"

    # Pull LQR cost knobs out of the dir name. They follow the fixed format
    #   ..._lam<L>_q<Q>_rinit<RI>_rfin<RF>_tau<T>_qf<QF>_...
    LAMBDA=$(   echo "$BASENAME" | grep -oP 'lam\K[0-9.]+')
    Q_SCALE=$(  echo "$BASENAME" | grep -oP '_q\K[0-9.]+')
    R_SCALE=$(  echo "$BASENAME" | grep -oP '_rinit\K[0-9.]+')
    R_SCALE_TAU=$(echo "$BASENAME" | grep -oP '_tau\K[0-9.]+')
    [[ -n "$LAMBDA" && -n "$Q_SCALE" && -n "$R_SCALE" && -n "$R_SCALE_TAU" ]] || {
        echo "ERROR: could not parse LQR knobs from $BASENAME" >&2; exit 1
    }

    echo "=== combo $BASENAME ==="
    printf '  rank=%s  lam=%s q=%s rinit=%s tau=%s\n' \
        "$RANK" "$LAMBDA" "$Q_SCALE" "$R_SCALE" "$R_SCALE_TAU"
    echo "  out_dir: $OUT_DIR"

    JOB_NAME="lqrgxyz_resub_r${RANK}_${BASENAME:0:36}"
    if [[ "$DRY" == "1" ]]; then
        ROLLOUT_ID="<dry-run>"
    else
        ROLLOUT_ID=$(sbatch --parsable \
            --account="$ACCOUNT" \
            --partition="$PARTITION_SLURM" \
            --job-name="$JOB_NAME" \
            --array="$RANK" \
            --gpus-per-task=1 \
            --ntasks=1 \
            --cpus-per-task="$CPUS_PER_TASK" \
            --mem="$MEM" \
            --time="$WORKER_TIME" \
            --output="$LOG_DIR/rollout_resub_rank%a_%A.out" \
            --error="$LOG_DIR/rollout_resub_rank%a_%A.err" \
            --export=ALL \
            ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
            --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$LQR_ROOT'
nvidia-smi -L
echo \"rank=\$SLURM_ARRAY_TASK_ID world_size=$WORLD_SIZE  (resubmitted from $OLD_PARENT)\"
$T5_CLEAR
python -u '$PY_SCRIPT' --phase rollout --rank \"\$SLURM_ARRAY_TASK_ID\" \
    --svd-dir '$SVD_DIR' \
    --jac-dir-act '$JAC_DIR_ACT' \
    --preset '$PRESET' \
    --base-seed '$BASE_SEED' \
    --gripper-action '$GRIPPER_ACTION' \
    --prompt '$PROMPT' \
    --lambda-scale '$LAMBDA' \
    --q-scale '$Q_SCALE' \
    --r-scale '$R_SCALE' \
    --r-scale-tau '$R_SCALE_TAU' \
    --r-scale-final '$R_SCALE_FINAL' \
    --max-chunks '$MAX_CHUNKS' \
    --qf-scale '$QF_SCALE' \
    --n-episodes '$N_EPISODES' \
    --task-id '$TASK_ID' \
    --suite '$SUITE' \
    --resolution '$RESOLUTION' \
    --video-fps '$VIDEO_FPS' \
    --num-steps-wait '$NUM_STEPS_WAIT' \
    --max-env-steps '$MAX_ENV_STEPS' \
    --world-size '$WORLD_SIZE' \
    --out-dir '$OUT_DIR' \
    --no-baseline \
    --seed '$SEED' \
    --tag '$SWEEP_TAG'
")
    fi
    echo "  resub rollout: $ROLLOUT_ID"

    MERGE_NAME="lqrgxyz_resub_merge_${BASENAME:0:30}"
    if [[ "$DRY" == "1" ]]; then
        MERGE_ID="<dry-run>"
    else
        MERGE_ID=$(sbatch --parsable \
            --account="$ACCOUNT" \
            --partition="$PARTITION_SLURM" \
            --dependency=afterok:"$ROLLOUT_ID" \
            --job-name="$MERGE_NAME" \
            --gpus-per-task=1 \
            --ntasks=1 \
            --cpus-per-task=2 \
            --mem=16G \
            --time="$MERGE_TIME" \
            --output="$LOG_DIR/merge_resub_%j.out" \
            --error="$LOG_DIR/merge_resub_%j.err" \
            --wrap "
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-policy
cd '$LQR_ROOT'
python -u '$PY_SCRIPT' --phase merge --out-dir '$OUT_DIR' --world-size '$WORLD_SIZE' \
    --svd-dir '$SVD_DIR' --jac-dir-act '$JAC_DIR_ACT' \
    --prompt '$PROMPT' --preset '$PRESET'
")
    fi
    echo "  resub merge:   $MERGE_ID (afterok:$ROLLOUT_ID)"
    SUBMITTED+=("$BASENAME : rollout=$ROLLOUT_ID merge=$MERGE_ID")
    echo
done

echo "=== summary ==="
for s in "${SUBMITTED[@]}"; do echo "  $s"; done
echo
if [[ "$DRY" == "1" ]]; then
    echo "DRY=1; nothing was actually submitted. Re-run with DRY=0 to launch."
else
    echo "Monitor with:  squeue -u \$USER -n lqrgxyz_resub_merge_$(echo "${recoveries[0]}" | cut -d'|' -f4 | cut -c1-30)"
    echo "Or just:       squeue -u \$USER | grep resub"
fi
