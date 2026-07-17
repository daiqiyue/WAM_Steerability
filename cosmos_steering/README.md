# Cosmos-Policy Steering
This directory is directly built off of the [original Cosmos-Policy repo](https://github.com/nvlabs/cosmos-policy) --- see their README.md for information related to system requirements.

# Setup Guide

This guide describes the working setup for Cosmos Policy on **NCSA Delta AI**
(Grace-Hopper GH200, aarch64). The original NVIDIA setup expected x86 Docker;
that flow is not available on Delta AI, so we use a conda env directly on the
shared filesystem instead.


## Installation

This package expects a conda environment named **`cosmos-policy`**.

```bash
conda create -y -n cosmos-policy python=3.10 pip
conda activate cosmos-policy
pip install uv

# ffmpeg 4 supplies libavformat.so.58 needed by decord 0.6.0.
conda install -y -c conda-forge "ffmpeg=4.*"

cd <path-to-cosmos-policy>
uv pip install --python "$(which python)" -e ".[cu128]"

# LIBERO: top-level libero/__init__.py is missing in upstream, so install
# from a local checkout that adds it.
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    <path-to-libero>
touch <path-to-libero>/libero/__init__.py
uv pip install --python "$(which python)" --no-deps -e <path-to-libero>

# LIBERO needs robosuite 1.4.x (1.5.x removed the manipulation/single_arm_env path).
uv pip install --python "$(which python)" "robosuite==1.4.1"
uv pip install --python "$(which python)" "mujoco==2.3.7"

# Misc LIBERO sim deps.
uv pip install --python "$(which python)" draccus easydict gym bddl cloudpickle
```

Keep `robosuite==1.4.1` with `mujoco==2.3.7`. MuJoCo 3.x changed the
`mj_fullM` Python API and is not compatible with robosuite 1.4.x.

### Setting environment variables for the conda env

Place environment variables in the conda activate hook so they are set
automatically whenever the env is activated:

```bash
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
# create or edit the hook file:
nano "${CONDA_PREFIX}/etc/conda/activate.d/cosmos_policy_env.sh"
```

Inside that file, export the variables pointing to your local paths, for
example:

```bash
export HF_HOME=<path-to-huggingface-cache>
export HF_HUB_CACHE="${HF_HOME}/hub"
export LIBERO_CONFIG_PATH=<path-to-libero-config>
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
```

The hook runs every time you `conda activate cosmos-policy`.

## Environment variables

The following variables must be set in
`${CONDA_PREFIX}/etc/conda/activate.d/cosmos_policy_env.sh` (see Installation
above). Adjust paths to match your system:

| Var                  | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| `HF_HOME`            | Root directory for Hugging Face cache                |
| `HF_HUB_CACHE`       | Hub model cache; typically `${HF_HOME}/hub`          |
| `LIBERO_CONFIG_PATH` | Directory where LIBERO writes config/task files      |
| `LD_LIBRARY_PATH`    | Prepend `${CONDA_PREFIX}/lib` so decord finds ffmpeg 4 |
| `HF_TOKEN`           | (Optional) read from `~/.huggingface/token` or `~/.cache/huggingface/token` if present |

NVIDIA-released Cosmos *base* models (`nvidia/Cosmos-Predict2-*`) are gated.
Accept the license on Hugging Face and run:

```bash
huggingface-cli login
```

The hook will pick up the token next time you `conda activate cosmos-policy`.

## Running

Run the desired pipeline or evaluation script e.g., 

```bash
./notebooks/lqr/e2e_scripts/run_noise_pipeline.sh
./notebooks/lqr/e2e_scripts/run_gripper_xyz_pipeline.sh
```
Each script prints a usage block at the top and supports `START_AT=<step>` and `FORCE_STEP<N>=1` to resume or re-run individual steps.
