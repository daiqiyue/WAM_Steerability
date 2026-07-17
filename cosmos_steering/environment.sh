# Source this file to activate the cosmos-policy env: `source environment.sh`
#
# Env lives at /nethome/jhong392/home/envs/cosmos-policy
# (symlink target: /usr/scratch/jhong392/envs/cosmos-policy)

source ~/home/src_conda.sh
conda activate /nethome/jhong392/home/envs/cosmos-policy

# Conda ships ffmpeg 4 in the env so decord 0.6.0 can find libavformat.so.58.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Caches and configs live under /usr/scratch (home is space-limited).
export HF_HOME="${HF_HOME:-/usr/scratch/jhong392/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/usr/scratch/jhong392/.libero}"

# Pick up HF token if present (needed for gated nvidia/Cosmos-Predict2-* repos).
# Default: dedicated token file for this repo; falls back to standard HF locations.
for _hf_token_path in \
    /nethome/jhong392/home/hf_token/cosmos_policy.txt \
    "${HOME}/.huggingface/token" \
    "${HOME}/.cache/huggingface/token"; do
    if [ -z "${HF_TOKEN:-}" ] && [ -f "${_hf_token_path}" ]; then
        export HF_TOKEN="$(cat "${_hf_token_path}")"
    fi
done
unset _hf_token_path
