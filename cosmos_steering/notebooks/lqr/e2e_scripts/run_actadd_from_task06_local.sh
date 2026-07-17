#!/bin/bash
# End-to-end ActAdd pipeline driver (local, no-slurm) — applies a contrastive
# direction derived from task-06's noise_extreme positive/negative pairs to
# libero_10 tasks {6, 0, 1, 4, 7}, sweeping ALPHA per task.
#
# Pipeline (each stage blocks on the previous):
#   1. violet/collect_actadd_acts.sh — collect per-DiT-block activations from
#      task-06's positive.npz and negative.npz (single GPU; ~hundreds of rows).
#   2. violet/make_actadd_vec.sh     — mean(positive) - mean(negative) per
#      block, saved as a per-layer dict {layer: Tensor(D,)} (CPU).
#   3. For each TARGET_TASK in {6, 0, 1, 4, 7}:
#      violet/actadd_sweep.sh        — sweep ALPHAS via violet/run_actadd_combo.sh
#                                       (gpu_pool fan-out + merge per alpha).
#   4. Append a "## ActAdd sweep (using task-06 contrastive direction)" section
#      to the existing task-06 sweep md
#      ($LQR_ROOT/noise_extreme_seed${NOISE_SEED_BASE}_task06_sweep.md), with
#      per-task subsections + a summary table whose deltas are computed against
#      the pre-existing noise_extreme baselines:
#        - task 06:   rollouts/noise_extreme_seed${SB}_task06/__baseline__/results.json
#        - other:     rollouts/noise_extreme_seed${SB}_task${NN}_from_task06/__baseline__/results.json
#
# Source artifacts (task-06 noise_extreme pair_dir) must already exist; produced
# by run_noise_extreme_pipeline_local.sh on task 06.
#
# Re-run safe: each stage skips if its primary artifact exists. Force per-step
# with FORCE_STEP{1,2,3,4}=1 (or FORCE=1 globally).
#
# Usage:
#   ./run_actadd_from_task06_local.sh
#   START_AT=3 ./run_actadd_from_task06_local.sh                 # resume at sweeps
#   ONLY_TASKS="6"     ./run_actadd_from_task06_local.sh
#   ALPHAS="0.0 1.0 5.0" ./run_actadd_from_task06_local.sh
#   N_EPISODES=10      ./run_actadd_from_task06_local.sh         # smoke test
#   FORCE_STEP4=1      ./run_actadd_from_task06_local.sh         # only rebuild md
#
# Artifacts:
#   activations : $LQR_ROOT/artifacts/actadd/activations__libero_10__task06.npz
#   contrast vec: $LQR_ROOT/artifacts/actadd/v__libero_10__task06.pt
#   rollouts    : $LQR_ROOT/rollouts/actadd_from_task06_seed${SB}_task${NN}/...
#   summary md  : appended-to $LQR_ROOT/noise_extreme_seed${SB}_task06_sweep.md

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." &>/dev/null && pwd)}"
LQR_ROOT="${LQR_ROOT:-$REPO_ROOT/notebooks/lqr}"
VIOLET_DIR="$LQR_ROOT/violet"

# Source the common helpers (paths, env activation, gpu pool).
. "$VIOLET_DIR/_common.sh"

# --- Source-task (task-06) pair_dir + prompt --------------------------------
SUITE="${SUITE:-libero_10}"
SOURCE_TASK_ID="${SOURCE_TASK_ID:-6}"
SOURCE_TASK_STR="$(printf 'task%02d' "$SOURCE_TASK_ID")"
if [[ -z "${SOURCE_PROMPT:-}" ]]; then
    SOURCE_PROMPT="$(get_libero_prompt "$SUITE" "$SOURCE_TASK_ID")"
fi
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-$LQR_ROOT/inputs/policy_inputs/${SUITE}__${SOURCE_TASK_STR}__noise_extreme_pos_neg}"

# --- Per-step config --------------------------------------------------------
N_EPISODES="${N_EPISODES:-30}"
WORLD_SIZE="${WORLD_SIZE:-$NUM_GPUS_AVAIL}"
NOISE_SIGMA="${NOISE_SIGMA:-90.0}"
NOISE_SEED_BASE="${NOISE_SEED_BASE:-99}"
NOISE_PER_EPISODE_SEED="${NOISE_PER_EPISODE_SEED:-1}"
SAMPLING_STEPS="${SAMPLING_STEPS:-10}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-520}"
SEED="${SEED:-1}"
RESOLUTION="${RESOLUTION:-256}"

# GPU V-cache cap inherited from run_noise_extreme_pipeline_local.sh — actadd
# doesn't actually use the V cache but exporting it is harmless and keeps the
# CUDA allocator config consistent across stages.
VCACHE_MAX_GPU_TILES="${VCACHE_MAX_GPU_TILES-100}"
export VCACHE_MAX_GPU_TILES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAX_ENV_STEPS

# Alpha sweep grid
# read -r -a ALPHAS <<< "${ALPHAS:-0.0 0.25 0.5 1.0 2.0 5.0 10.0 20.0}"
read -r -a ALPHAS <<< "${ALPHAS:-0.1}"

# Target tasks
read -r -a TARGET_TASKS <<< "${ONLY_TASKS:-6 0 1 4 7}"

# Per-timestep mode: when set, the collector keeps the latent-T axis and the
# steering vector becomes (T, D) per layer (broadcast as (1, T, 1, 1, D) in
# the actadd hook). Artifacts, rollout TAG, and the md section header get a
# `_perT` / `(per-timestep, ...)` suffix so per-layer and per-timestep runs
# coexist on disk and in the unified md.
PER_TIMESTEP="${PER_TIMESTEP:-0}"
PERT_FILE_SUFFIX=""
PERT_TAG_SUFFIX=""
PERT_MODE_LABEL="per-layer"
if [[ "$PER_TIMESTEP" == "1" || "$PER_TIMESTEP" == "true" ]]; then
    PERT_FILE_SUFFIX="__perT"
    PERT_TAG_SUFFIX="_perT"
    PERT_MODE_LABEL="per-timestep"
fi

# Artifact paths (shared with violet/collect_actadd_acts.sh and make_actadd_vec.sh)
ACTADD_ARTIFACTS_BASE="${ACTADD_ARTIFACTS_BASE:-$ARTIFACTS_BASE/actadd}"
ACTIVATIONS_PATH="${ACTIVATIONS_PATH:-$ACTADD_ARTIFACTS_BASE/activations__${SUITE}__${SOURCE_TASK_STR}${PERT_FILE_SUFFIX}.npz}"
V_PATH="${V_PATH:-$ACTADD_ARTIFACTS_BASE/v__${SUITE}__${SOURCE_TASK_STR}${PERT_FILE_SUFFIX}.pt}"

# Existing noise_extreme baselines (from run_noise_extreme_pipeline_local.sh
# + run_top10_from_task06_on_tasks_0_1_4_7.sh).
baseline_dir_for_task() {
    local tid="$1"
    local ts; ts="$(printf 'task%02d' "$tid")"
    if [[ "$tid" == "$SOURCE_TASK_ID" ]]; then
        echo "$LQR_ROOT/rollouts/noise_extreme_seed${NOISE_SEED_BASE}_${ts}/__baseline__"
    else
        echo "$LQR_ROOT/rollouts/noise_extreme_seed${NOISE_SEED_BASE}_${ts}_from_task06/__baseline__"
    fi
}

# Existing task-06 sweep md to append the actadd section to.
UNIFIED_MD="${UNIFIED_MD:-$LQR_ROOT/noise_extreme_seed${NOISE_SEED_BASE}_task06_sweep.md}"

START_AT="${START_AT:-1}"
STOP_AFTER="${STOP_AFTER:-4}"

# Forward FORCE globally / per-step.
[[ -n "${FORCE:-}" ]] && export FORCE_STEP1="${FORCE_STEP1:-$FORCE}" \
                                FORCE_STEP2="${FORCE_STEP2:-$FORCE}" \
                                FORCE_STEP3="${FORCE_STEP3:-$FORCE}" \
                                FORCE_STEP4="${FORCE_STEP4:-$FORCE}"

banner "actadd local pipeline: source=${SOURCE_TASK_STR}, targets={${TARGET_TASKS[*]}}, mode=${PERT_MODE_LABEL}"
echo "  REPO_ROOT       : $REPO_ROOT"
echo "  source prompt   : $SOURCE_PROMPT"
echo "  source pair_dir : $SOURCE_PAIR_DIR"
echo "  GPUs            : $GPUS  (NUM_GPUS=$NUM_GPUS_AVAIL)"
echo "  WORLD_SIZE      : $WORLD_SIZE"
echo "  N_EPISODES      : $N_EPISODES"
echo "  NOISE_SIGMA     : $NOISE_SIGMA  seed_base=$NOISE_SEED_BASE per_ep=$NOISE_PER_EPISODE_SEED"
echo "  ALPHAS          : [${ALPHAS[*]}]"
echo "  PER_TIMESTEP    : $PER_TIMESTEP (mode=$PERT_MODE_LABEL)"
echo "  activations npz : $ACTIVATIONS_PATH"
echo "  contrast vec pt : $V_PATH"
echo "  unified md      : $UNIFIED_MD"
echo "  START_AT / STOP_AFTER : $START_AT / $STOP_AFTER"

should_run() {
    local n="$1"
    if (( n < START_AT )); then echo "[skip] step $n below START_AT=$START_AT"; return 1; fi
    if (( n > STOP_AFTER )); then echo "[skip] step $n above STOP_AFTER=$STOP_AFTER"; return 1; fi
    return 0
}

# =========================================================================
# Step 1 — collect activations from task-06 positive/negative
# =========================================================================
if should_run 1; then
    banner "Step 1: collect actadd activations  ($SOURCE_TASK_STR, $PERT_MODE_LABEL)"
    FORCE="${FORCE_STEP1:-}" \
    SUITE="$SUITE" TASK_ID="$SOURCE_TASK_ID" PROMPT="$SOURCE_PROMPT" \
    PAIR_DIR="$SOURCE_PAIR_DIR" \
    PER_TIMESTEP="$PER_TIMESTEP" \
    OUT_PATH="$ACTIVATIONS_PATH" GPUS="$GPUS" \
        bash "$VIOLET_DIR/collect_actadd_acts.sh"
fi

# =========================================================================
# Step 2 — build per-layer contrastive vector
# =========================================================================
if should_run 2; then
    banner "Step 2: make actadd contrastive vec  ($SOURCE_TASK_STR)"
    FORCE="${FORCE_STEP2:-}" \
    SUITE="$SUITE" TASK_ID="$SOURCE_TASK_ID" \
    ACTIVATIONS_PATH="$ACTIVATIONS_PATH" OUT_PATH="$V_PATH" \
        bash "$VIOLET_DIR/make_actadd_vec.sh"
fi

# =========================================================================
# Step 3 — per-target-task alpha sweeps
# =========================================================================
if should_run 3; then
    [[ -f "$V_PATH" ]] || { echo "ERROR: $V_PATH missing (step 2 produced it)" >&2; exit 1; }
    for tid in "${TARGET_TASKS[@]}"; do
        TASK_STR=$(printf "task%02d" "$tid")
        TAG="actadd_from_task06${PERT_TAG_SUFFIX}_seed${NOISE_SEED_BASE}_${TASK_STR}"
        ROLLOUTS_ROOT="$LQR_ROOT/rollouts/$TAG"
        banner "Step 3: alpha sweep  ${TASK_STR}  ->  $TAG  (mode=$PERT_MODE_LABEL)"
        FORCE="${FORCE_STEP3:-}" \
        V_PATH="$V_PATH" \
        ALPHAS="${ALPHAS[*]}" \
        SUITE="$SUITE" TASK_ID="$tid" \
        N_EPISODES="$N_EPISODES" WORLD_SIZE="$WORLD_SIZE" \
        NOISE_SIGMA="$NOISE_SIGMA" NOISE_SEED_BASE="$NOISE_SEED_BASE" \
        NOISE_PER_EPISODE_SEED="$NOISE_PER_EPISODE_SEED" \
        RESOLUTION="$RESOLUTION" SAMPLING_STEPS="$SAMPLING_STEPS" \
        MAX_ENV_STEPS="$MAX_ENV_STEPS" SEED="$SEED" \
        TAG="$TAG" OUT_BASE="$LQR_ROOT/rollouts" GPUS="$GPUS" \
            bash "$VIOLET_DIR/actadd_sweep.sh"
    done
fi

# =========================================================================
# Step 4 — build actadd section and append it to UNIFIED_MD
# =========================================================================
if should_run 4; then
    banner "Step 4: build actadd section and append to $UNIFIED_MD (mode=$PERT_MODE_LABEL)"

    # Mode-aware header lets per-layer and per-timestep sections coexist in
    # the same md. Strip-on-re-run targets the EXACT current-mode header, so
    # other-mode sections are untouched.
    ACTADD_HEADER="## ActAdd sweep (${PERT_MODE_LABEL}, using task-06 contrastive direction)"

    # Compose the actadd section in a temp file via inline python (mirrors the
    # cross-task summary table builder in run_top10_from_task06_on_tasks_0_1_4_7.sh).
    ACTADD_MD="$(mktemp -t actadd_section.XXXXXX.md)"
    trap 'rm -f "$ACTADD_MD"' EXIT

    _ALPHAS_CSV=$(IFS=,; echo "${ALPHAS[*]}")
    _TARGETS_CSV=$(IFS=,; echo "${TARGET_TASKS[*]}")

    activate_env
    LQR_ROOT="$LQR_ROOT" \
    SUITE="$SUITE" \
    SOURCE_TASK_ID="$SOURCE_TASK_ID" \
    NOISE_SEED_BASE="$NOISE_SEED_BASE" \
    NOISE_SIGMA="$NOISE_SIGMA" \
    SOURCE_PROMPT="$SOURCE_PROMPT" \
    V_PATH="$V_PATH" \
    ALPHAS_CSV="$_ALPHAS_CSV" \
    CROSS_TASKS="$_TARGETS_CSV" \
    ACTADD_HEADER="$ACTADD_HEADER" \
    PERT_TAG_SUFFIX="$PERT_TAG_SUFFIX" \
    N_EPISODES="$N_EPISODES" \
        python -u - > "$ACTADD_MD" <<'PY'
import json, os, sys
from pathlib import Path

LQR_ROOT  = Path(os.environ["LQR_ROOT"])
SUITE     = os.environ["SUITE"]
SRC_TID   = int(os.environ["SOURCE_TASK_ID"])
NSB       = int(os.environ["NOISE_SEED_BASE"])
SIGMA     = os.environ.get("NOISE_SIGMA", "90.0")
SRC_PR    = os.environ["SOURCE_PROMPT"]
V_PATH    = os.environ["V_PATH"]
ALPHAS    = [a for a in os.environ["ALPHAS_CSV"].split(",") if a]
TASKS     = [int(x) for x in os.environ["CROSS_TASKS"].split(",") if x]
HEADER    = os.environ["ACTADD_HEADER"]
PERT_TAG  = os.environ.get("PERT_TAG_SUFFIX", "")
N_EP_EXPECTED = int(os.environ.get("N_EPISODES", "30"))

def prompt_slug_rollout(s):
    # mirror violet/_common.sh prompt_slug_rollout()
    out = s[:24].lower().replace(" ", "_")
    keep = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    return "".join(c for c in out if c in keep)

def task_root(tid):
    ts = f"task{tid:02d}"
    return LQR_ROOT / "rollouts" / f"actadd_from_task06{PERT_TAG}_seed{NSB}_{ts}"

def baseline_path(tid):
    ts = f"task{tid:02d}"
    if tid == SRC_TID:
        return LQR_ROOT / "rollouts" / f"noise_extreme_seed{NSB}_{ts}" \
               / "__baseline__" / "results.json"
    return LQR_ROOT / "rollouts" / f"noise_extreme_seed{NSB}_{ts}_from_task06" \
           / "__baseline__" / "results.json"

def read_succ(p):
    if not p.exists():
        return None
    try:
        r = json.load(open(p))
    except Exception:
        return None
    if isinstance(r, list) and r:
        s = sum(1 for e in r if e.get("success"))
        return (s, len(r))
    return None

def find_combo_results(tid, alpha):
    root = task_root(tid)
    if not root.exists():
        return None
    ts = f"task{tid:02d}"
    noise_tag = f"s{SIGMA}_sb{NSB}"
    # We need each task's actual prompt slug. Glob lets us tolerate slight
    # differences (e.g. trailing punctuation in the libero language).
    import glob
    pattern = (
        f"{SUITE}__{ts}__actadd_noised__{noise_tag}__alpha{alpha}__*"
    )
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    return matches[0] / "results.json"

# Pull baselines once per task (from the existing noise_extreme pipeline).
baselines = {tid: read_succ(baseline_path(tid)) for tid in TASKS}

def fmt_baseline(b):
    if b is None:
        return "no baseline"
    s, n = b
    return f"base {100*s/n:.0f}%, {s}/{n}"

def delta_pp(info, base):
    if info is None or base is None:
        return None
    s, n   = info
    bs, bn = base
    return 100 * s / n - 100 * bs / bn

def fmt_cell(info, base):
    if info is None:
        return "—"
    s, n = info
    rate = 100 * s / n
    if base is None:
        return f"{rate:.0f}% ({s}/{n})"
    delta = rate - 100 * (base[0] / base[1])
    sign  = "+" if delta >= 0 else ""
    return f"{rate:.0f}% ({sign}{delta:.0f})"

# First pass: gather results for every (alpha, tid) so we can render a table.
# data[alpha][tid] = (info, baseline)
data = {}
for a in ALPHAS:
    per_task = {}
    for tid in TASKS:
        rp = find_combo_results(tid, a)
        per_task[tid] = (read_succ(rp) if rp else None, baselines[tid])
    data[a] = per_task

# ---- Markdown -----------------------------------------------------------
out = []
out.append(HEADER)
out.append("")
out.append("Each cell shows the steered success rate (over "
           f"{N_EP_EXPECTED} episodes) and the Δ vs that task's "
           "noise_extreme baseline in percentage points (the same baseline as "
           "the LQR cross-task section above). The steering vector is the per-"
           "block mean(positive_activations) − mean(negative_activations) "
           f"computed from task-{SRC_TID:02d}'s noise_extreme pos/neg pair, "
           "hooked at every DiT block as `output += alpha * v[block]`.")
out.append("")
out.append(f"- Source task: **task{SRC_TID:02d}**  "
           f"_(prompt: \"{SRC_PR}\")_")
out.append(f"- Steering vector: `{V_PATH}`")
out.append(f"- Image noise: σ={SIGMA}, "
           f"per-episode seed (base={NSB}).")
out.append("")

# ---- Per-task subsections (one short line per alpha) ----
for tid in TASKS:
    ts = f"task{tid:02d}"
    bp = baseline_path(tid)
    out.append(f"### {ts}  ({fmt_baseline(baselines[tid])})")
    out.append("")
    out.append(f"- Rollouts root: `{task_root(tid)}/`")
    out.append(f"- Baseline source: `{bp}`")
    out.append("")
    out.append("| alpha | success rate | Δ vs baseline (pp) |")
    out.append("|---|---|---|")
    for a in ALPHAS:
        info, base = data[a][tid]
        cell = fmt_cell(info, base)
        if info is None:
            rate_str = "—"
            d_str    = "—"
        else:
            s, n = info
            rate_str = f"{100*s/n:.0f}% ({s}/{n})"
            d = delta_pp(info, base)
            d_str = "—" if d is None else (f"{'+' if d >= 0 else ''}{d:.0f}")
        out.append(f"| {a} | {rate_str} | {d_str} |")
    out.append("")

# ---- Summary table (rows = alpha, cols = tasks) ----
out.append("### Summary table (rows = alpha, cols = tasks)")
out.append("")
hdr = ["alpha", "**mean Δ**"]
for tid in TASKS:
    hdr.append(f"task{tid:02d} ({fmt_baseline(baselines[tid])})")
out.append("| " + " | ".join(hdr) + " |")
out.append("|" + "|".join(["---"] * len(hdr)) + "|")

def mean_delta_for_alpha(per_task):
    deltas = [delta_pp(info, base) for (info, base) in per_task.values()
              if delta_pp(info, base) is not None]
    if not deltas:
        return (None, 0)
    return (sum(deltas) / len(deltas), len(deltas))

for a in ALPHAS:
    per_task = data[a]
    md, n_avg = mean_delta_for_alpha(per_task)
    if md is None:
        mean_cell = "—"
    else:
        sign = "+" if md >= 0 else ""
        mean_cell = (f"**{sign}{md:.1f}**" if n_avg == len(TASKS)
                     else f"**{sign}{md:.1f}** ({n_avg}/{len(TASKS)})")
    row = [a, mean_cell]
    for tid in TASKS:
        info, base = per_task[tid]
        row.append(fmt_cell(info, base))
    out.append("| " + " | ".join(row) + " |")

out.append("")
out.append(f"Legend: `XX% (+Δ)` = steered success rate with Δ-vs-baseline in "
           "percentage points. `—` = no `results.json` for that (task, alpha) "
           "pair yet. `mean Δ` is the unweighted mean of available per-task Δs.")
print("\n".join(out))
PY

    echo "  wrote $(wc -l <"$ACTADD_MD") lines to $ACTADD_MD"

    # Append (or replace) the section in UNIFIED_MD. If a previous actadd
    # section with THIS mode's header is present, strip from that header until
    # the next "## " heading (or EOF). Other-mode sections (or other top-level
    # sections that happen to come after) are preserved.
    if [[ -f "$UNIFIED_MD" ]]; then
        if grep -Fq "$ACTADD_HEADER" "$UNIFIED_MD"; then
            echo "  $UNIFIED_MD already has a '${PERT_MODE_LABEL}' actadd section; replacing it in place"
            TMP_MD="$(mktemp -t unified_md.XXXXXX.md)"
            awk -v hdr="$ACTADD_HEADER" '
                $0 == hdr               { skip=1; next }
                skip && /^## [^#]/      { skip=0 }
                !skip                   { print }
            ' "$UNIFIED_MD" > "$TMP_MD"
            mv "$TMP_MD" "$UNIFIED_MD"
        fi
        {
            printf '\n---\n\n'
            cat "$ACTADD_MD"
        } >> "$UNIFIED_MD"
    else
        echo "  WARN: $UNIFIED_MD doesn't exist yet; writing actadd section as a new file"
        cat "$ACTADD_MD" > "$UNIFIED_MD"
    fi

    echo "  appended actadd section -> $UNIFIED_MD"
    echo "  total md size: $(wc -l <"$UNIFIED_MD") lines, $(wc -c <"$UNIFIED_MD") bytes"
fi

banner "PIPELINE COMPLETE (mode=$PERT_MODE_LABEL)"
echo "  activations : $ACTIVATIONS_PATH"
echo "  contrast vec: $V_PATH"
echo "  rollouts    :"
for tid in "${TARGET_TASKS[@]}"; do
    TASK_STR=$(printf "task%02d" "$tid")
    printf '    %-7s %s\n' "${TASK_STR}:" \
        "$LQR_ROOT/rollouts/actadd_from_task06${PERT_TAG_SUFFIX}_seed${NOISE_SEED_BASE}_${TASK_STR}/"
done
echo "  unified summary : $UNIFIED_MD"
