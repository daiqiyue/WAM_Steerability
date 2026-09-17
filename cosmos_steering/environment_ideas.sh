#!/usr/bin/env bash
# IDEaS L40S/DGX runtime for the Cosmos-Policy experiments in this checkout.

set -eo pipefail

COSMOS_WAMS_ROOT="${COSMOS_WAMS_ROOT:-/storage/scratch1/9/qdai41/steering_robustness_WAMs/cosmos_steering}"
COSMOS_CONDA_ENV="${COSMOS_CONDA_ENV:-/storage/scratch1/9/qdai41/.conda/envs/cosmos-policy}"
LIBERO_HOME="${LIBERO_HOME:-/storage/scratch1/9/qdai41/cosmos/LIBERO}"

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate "$COSMOS_CONDA_ENV"
set -u

export COSMOS_WAMS_ROOT COSMOS_CONDA_ENV LIBERO_HOME
export PYTHONPATH="$COSMOS_WAMS_ROOT:$LIBERO_HOME${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME="${HF_HOME:-/storage/scratch1/9/qdai41/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/storage/scratch1/9/qdai41/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/storage/scratch1/9/qdai41/.config/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

python - <<'PY'
from importlib.metadata import version

expected = {
    "robosuite": "1.4.1",
    "mujoco": "2.3.7",
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
}
bad = []
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        bad.append(f"{package}=={actual} (need {wanted})")
if bad:
    raise SystemExit("Incompatible Cosmos/LIBERO environment: " + ", ".join(bad))
PY
