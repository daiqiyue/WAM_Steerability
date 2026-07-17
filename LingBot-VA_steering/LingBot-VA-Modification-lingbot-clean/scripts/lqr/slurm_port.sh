#!/usr/bin/env bash
# Resolve WebSocket port: PORT_BASE + (SLURM_JOB_ID % 1000).
# Override with PORT=... or PORT_BASE=... (default 29056).
: "${PORT_BASE:=29056}"
if [[ -z "${PORT:-}" ]]; then
  PORT="$((PORT_BASE + (${SLURM_JOB_ID:-0} % 1000)))"
fi
export PORT
