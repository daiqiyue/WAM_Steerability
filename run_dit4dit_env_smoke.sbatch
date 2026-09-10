#!/usr/bin/env bash
#SBATCH --job-name=dit4dit-env-smoke
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --chdir=/storage/scratch1/9/qdai41/steering_robustness_WAMs
#SBATCH --output=DiT4DiT_steering/lqr/logs/env_smoke_%j.out
#SBATCH --error=DiT4DiT_steering/lqr/logs/env_smoke_%j.err

set -euo pipefail
source DiT4DiT_steering/env/ideas_l40s.sh
nvidia-smi -L
"$DIT4DIT_PYTHON" DiT4DiT_steering/env/verify_environment.py --load-model
