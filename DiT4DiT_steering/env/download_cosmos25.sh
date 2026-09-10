#!/usr/bin/env bash
set -euo pipefail

PY_ENV="${PY_ENV:-/storage/scratch1/9/qdai41/.conda/envs/dit4dit}"
MODEL_DIR="${COSMOS25_MODEL_DIR:-/storage/scratch1/9/qdai41/cosmos/DiT4DiT/models/Cosmos-Predict2.5-2B}"

exec "${PY_ENV}/bin/huggingface-cli" download \
  nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained \
  --local-dir "${MODEL_DIR}" \
  --max-workers "${HF_DOWNLOAD_WORKERS:-2}"
