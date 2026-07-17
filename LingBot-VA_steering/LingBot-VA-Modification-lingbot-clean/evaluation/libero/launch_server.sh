#!/usr/bin/env bash
set -euo pipefail

save_root="${SAVE_ROOT:-}"
if [[ -n "${save_root}" ]]; then
  mkdir -p "${save_root}"
fi

# Match sbatch: websocket port and DDP master port per job when SLURM_JOB_ID is set.
PORT="${PORT:-$((29056 + (${SLURM_JOB_ID:-0} % 1000)))}"
SERVER_PORT="${SERVER_PORT:-${PORT}}"
MASTER_PORT="${MASTER_PORT:-$((12000 + (${SLURM_JOB_ID:-0} % 20000)))}"

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "${MASTER_PORT}" \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port "${SERVER_PORT}" \
    --save_root "${save_root}"
