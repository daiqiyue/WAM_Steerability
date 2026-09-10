#!/usr/bin/env bash
#SBATCH --job-name=cosmos25-download
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/scratch1/9/qdai41/steering_robustness_WAMs/DiT4DiT_steering/lqr/logs/cosmos25_download_%j.out
#SBATCH --error=/storage/scratch1/9/qdai41/steering_robustness_WAMs/DiT4DiT_steering/lqr/logs/cosmos25_download_%j.err
#SBATCH --chdir=/storage/scratch1/9/qdai41/steering_robustness_WAMs

set -euo pipefail
export PYTHONNOUSERSITE=1
export HF_DOWNLOAD_WORKERS=2

DiT4DiT_steering/env/download_cosmos25.sh

test -s /storage/scratch1/9/qdai41/cosmos/DiT4DiT/models/Cosmos-Predict2.5-2B/model_index.json
test -s /storage/scratch1/9/qdai41/cosmos/DiT4DiT/models/Cosmos-Predict2.5-2B/transformer/diffusion_pytorch_model.safetensors
test -s /storage/scratch1/9/qdai41/cosmos/DiT4DiT/models/Cosmos-Predict2.5-2B/vae/diffusion_pytorch_model.safetensors
