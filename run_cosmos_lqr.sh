#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
COSMOS_DIR="${COSMOS_DIR:-$ROOT/cosmos_steering}"
export REPO_ROOT="${REPO_ROOT:-$COSMOS_DIR}"
export LQR_ROOT="${COSMOS_LQR_ROOT:-${LQR_ROOT:-$COSMOS_DIR/notebooks/lqr}}"

PERTURBATION="${PERTURBATION:-noise}"        # noise | gaussian | gripper | camera
case "$PERTURBATION" in
  camera|cam)
    DEFAULT_COLLECT_TASK_ID=0
    DEFAULT_EVAL_TASK_IDS="0 2 4 5 9"
    ;;
  gripper|gripper_xyz)
    DEFAULT_COLLECT_TASK_ID=1
    DEFAULT_EVAL_TASK_IDS="1 2 3 7 9"
    ;;
  noise|gaussian)
    DEFAULT_COLLECT_TASK_ID=6
    DEFAULT_EVAL_TASK_IDS="0 1 4 6 7"
    ;;
  *)
    DEFAULT_COLLECT_TASK_ID=0
    DEFAULT_EVAL_TASK_IDS="0"
    ;;
esac
COLLECT_TASK_ID="${COLLECT_TASK_ID:-${TASK_ID:-$DEFAULT_COLLECT_TASK_ID}}"
EVAL_TASK_IDS="${EVAL_TASK_IDS:-${TASK_IDS:-$DEFAULT_EVAL_TASK_IDS}}"

export TASK_ID="$COLLECT_TASK_ID"
export TASK_IDS="$EVAL_TASK_IDS"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export START_AT="${START_AT:-1}"
export N_EPISODES="${N_EPISODES:-${NUM_EPISODES:-10}}"
export OUT_BASE="${OUT_BASE:-$LQR_ROOT/rollouts}"

check_cosmos_env() {
  "${PYTHON:-python}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys

def get_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None

def major_minor(ver):
    parts = []
    for chunk in ver.replace("-", ".").split("."):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            digits = ""
            for ch in chunk:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                parts.append(int(digits))
            break
    return tuple(parts + [0] * (2 - len(parts)))

robosuite_v = get_version("robosuite")
mujoco_v = get_version("mujoco")
if robosuite_v is None or mujoco_v is None:
    missing = []
    if robosuite_v is None:
        missing.append("robosuite")
    if mujoco_v is None:
        missing.append("mujoco")
    sys.exit(
        "Missing Cosmos/LIBERO simulation dependency: "
        + ", ".join(missing)
        + "\nInstall robosuite==1.4.1 and mujoco==2.3.7 in the active env."
    )

if major_minor(robosuite_v) < (1, 5) and major_minor(mujoco_v) >= (3, 0):
    sys.exit(
        f"Incompatible robosuite/mujoco versions: robosuite=={robosuite_v}, "
        f"mujoco=={mujoco_v}.\n"
        "LIBERO with robosuite==1.4.1 uses the MuJoCo 2.x mj_fullM API. "
        "Run: python -m pip install 'mujoco==2.3.7'"
    )
PY
}

if [[ "${SKIP_ENV_CHECK:-0}" != "1" ]]; then
  check_cosmos_env
fi
if [[ -z "${SUBMIT_SWEEP:-}" ]]; then
  if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
    export SUBMIT_SWEEP=0
  else
    export SUBMIT_SWEEP=1
  fi
fi

case "$PERTURBATION" in
  noise|gaussian)
    export SVD_TASK="${SVD_TASK:-$COLLECT_TASK_ID}"
    exec bash "$LQR_ROOT/e2e_scripts/run_noise_pipeline.sh"
    ;;
  gripper|gripper_xyz)
    exec bash "$LQR_ROOT/e2e_scripts/run_gripper_xyz_pipeline.sh"
    ;;
  camera|cam)
    exec bash "$LQR_ROOT/e2e_scripts/run_cam_random_large_pipeline.sh"
    ;;
  *)
    echo "Unknown PERTURBATION=$PERTURBATION (use noise, gripper, or camera)" >&2
    exit 2
    ;;
esac
