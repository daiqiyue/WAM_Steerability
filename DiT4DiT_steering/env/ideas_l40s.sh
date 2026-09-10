#!/usr/bin/env bash
# Site configuration for DiT4DiT jobs on the Georgia Tech IDEaS L40S queue.
# Source this file; every value can still be overridden by the caller.

_DIT4DIT_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
_DIT4DIT_PROJECT_ROOT="$(cd -- "${_DIT4DIT_ENV_DIR}/../.." >/dev/null 2>&1 && pwd)"

export ACCOUNT="${ACCOUNT:-gts-gchou3-ideas_l40s}"
export PARTITION="${PARTITION:-gpu-l40s}"
export PARTITION_SLURM="${PARTITION_SLURM:-$PARTITION}"

# This repository contains the steering/LQR code.  The original model source is
# kept in a separate checkout and is deliberately configured independently.
export DIT4DIT_ROOT="${DIT4DIT_ROOT:-${_DIT4DIT_PROJECT_ROOT}/DiT4DiT_steering}"
export DIT4DIT_LQR_ROOT="${DIT4DIT_LQR_ROOT:-${DIT4DIT_ROOT}/lqr}"
export DIT4DIT_CODE_ROOT="${DIT4DIT_CODE_ROOT:-${DIT4DIT_ROOT}/upstream/DiT4DiT_source}"
export DIT4DIT_PYTHON="${DIT4DIT_PYTHON:-/storage/scratch1/9/qdai41/.conda/envs/dit4dit/bin/python}"
export PYTHON="${PYTHON:-$DIT4DIT_PYTHON}"
export MODEL_PYTHON="${MODEL_PYTHON:-$DIT4DIT_PYTHON}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-$DIT4DIT_PYTHON}"

export COSMOS25_MODEL_DIR="${COSMOS25_MODEL_DIR:-/storage/scratch1/9/qdai41/cosmos/DiT4DiT/models/Cosmos-Predict2.5-2B}"
export CKPT_PATH="${CKPT_PATH:-${DIT4DIT_ROOT}/env/checkpoint_runtime/dit4dit_libero/final_model/pytorch_model.pt}"
export LIBERO_HOME="${LIBERO_HOME:-/storage/home/hcoda1/9/qdai41/workspace/ctrlwam/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${_DIT4DIT_PROJECT_ROOT}/DiT4DiT_steering/env/libero}"
export PYTHONPATH="${DIT4DIT_CODE_ROOT}:${DIT4DIT_ROOT}:${LIBERO_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/dit4dit-matplotlib-${USER}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/dit4dit-xdg-cache-${USER}}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONNOUSERSITE=1

unset _DIT4DIT_ENV_DIR _DIT4DIT_PROJECT_ROOT
