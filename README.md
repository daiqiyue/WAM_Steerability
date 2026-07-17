# Steering Robustness into World Action Models via Mechanistic Interpretability and Optimal Control

This repo contains the implementation for WA-LQR, and ActAdd (for WAMs).

[![Paper](https://img.shields.io/badge/arXiv-2604.19018-b31b1b.svg)](https://arxiv.org/abs/2607.14943)
[![Website](https://img.shields.io/badge/Website-Visit-blue.svg)](https://trustworthyrobotics.github.io/steering_robust_wam_site/)



# Usage Guide

This directory collects LQR steering demos for three policy models:

- `cosmos_steering/`: Cosmos-Policy + LQR.
- `DiT4DiT_steering/`: DiT4DiT + LQR.
- `LingBot-VA_steering/`: LingBot-VA + LQR.

The project-level Slurm entrypoints are:

```text
run_cosmos_lqr.sh
run_dit4dit_lqr.sh
run_lingbot_lqr.sh
```

The project-level local, no-Slurm entrypoints are:

```text
run_cosmos_lqr_local.sh
run_dit4dit_lqr_local.sh
run_lingbot_lqr_local.sh
```

These top-level scripts only set variables and dispatch to the model-specific
pipelines. The actual collection, SVD, Jacobian, rollout, and evaluation code
lives in each model subdirectory.

## 0. Quick Start

Run all commands from the project root:

```bash
export REPO_ROOT=/path/to/repo
cd "${REPO_ROOT}"
```

Keep `REPO_ROOT` exported while following the setup and run commands below.

Make the entrypoints executable:

```bash
chmod +x run_cosmos_lqr.sh run_dit4dit_lqr.sh run_lingbot_lqr.sh
chmod +x run_cosmos_lqr_local.sh run_dit4dit_lqr_local.sh run_lingbot_lqr_local.sh
```

Optional syntax checks:

```bash
bash -n run_cosmos_lqr.sh run_dit4dit_lqr.sh run_lingbot_lqr.sh 
bash -n run_cosmos_lqr_local.sh run_dit4dit_lqr_local.sh run_lingbot_lqr_local.sh 
```

## 1. Common Runtime Variables

The top-level LQR scripts use environment variables. If a variable is unset, the
script falls back to its default. The local scripts and Slurm scripts share the
same task-split variables.

```bash
PERTURBATION=noise        # Perturbation family.
COLLECT_TASK_ID=0         # Task id used to collect LQR artifacts.
EVAL_TASK_IDS="0 1 2"     # Evaluation tasks, space-separated; Cosmos/LingBot.
EVAL_TASK_ID=3            # DiT4DiT generalization target task.
N_EPISODES=10             # Collection episodes.
EVAL_NUM_EPISODES=20      # Evaluation episodes, when supported downstream.
START_AT=1                # Resume from this e2e pipeline step, when supported.
SKIP_EVAL=0               # Set 1 to build artifacts without rollout/eval.
FORCE_STEP1=1             # Force rerun of a step; step numbers are pipeline-specific.
```

Common artifact and controller variables:

```bash
export REPO_ROOT=/path/to/repo
COSMOS_DIR="${REPO_ROOT}/cosmos_steering"
COSMOS_LQR_ROOT="${REPO_ROOT}/cosmos_steering/notebooks/lqr"
DIT4DIT_ROOT="${REPO_ROOT}/DiT4DiT_steering"
DIT4DIT_LQR_ROOT="${REPO_ROOT}/DiT4DiT_steering/lqr"
SVD_DIR=/path/to/existing/svd
EXISTING_COLLECT_DIR=/path/to/pairs_dir
EXISTING_PAIRS_ALL_DIR=/path/to/paired_dir
K_TARGET=64
NUM_SAMPLES=-1
P_OVER=10
PARTITIONS="0-9,10-19,20-29"
PORT=29500
ACCOUNT=your-slurm-account
PARTITION=your-slurm-partition
PARTITION_SLURM=your-slurm-partition
```

Local-only execution variables:

```bash
PYTHON=python             # Python executable used by local wrappers.
WORLD_SIZE=1              # Local sequential rank count for small runs.
SVD_WORLD_SIZE=1          # Override SVD rank count.
JAC_WORLD_SIZE=1          # Override Jacobian rank count.
EVAL_WORLD_SIZE=1         # Override rollout rank count.
OUT_BASE=/path/to/outputs # Local output root.
```

Default task splits:

```text
camera position    collect task 0    eval tasks 0 2 4 5 9
gripper position   collect task 1    eval tasks 1 2 3 7 9
gaussian/noise     collect task 6    eval tasks 0 1 4 6 7
```

LIBERO `libero_10` task ids:

```text
0  put both the alphabet soup and the tomato sauce in the basket
1  put both the cream cheese box and the butter in the basket
2  turn on the stove and put the moka pot on it
3  put the black bowl in the bottom drawer of the cabinet and close it
4  put the white mug on the left plate and put the yellow and white mug on the right plate
5  pick up the book and place it in the back compartment of the caddy
6  put the white mug on the plate and put the chocolate pudding to the right of the plate
7  put both the alphabet soup and the cream cheese box in the basket
8  put both moka pots on the stove
9  put the yellow and white mug in the microwave and close it
```

## 2. Cosmos-Policy Environment Setup

Cosmos scripts assume a conda environment named `cosmos-policy`.

```bash
conda create -y -n cosmos-policy python=3.10 pip
conda activate cosmos-policy
python -m pip install uv

# decord 0.6.0 needs libavformat.so.58.
conda install -y -c conda-forge "ffmpeg=4.*"

cd "${REPO_ROOT}/cosmos_steering"
uv pip install --python "$(which python)" -e ".[cu128]"
```

Install LIBERO:

```bash
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ./LIBERO

# Some LIBERO checkouts miss this package marker. Create it before editable
# install so setuptools can discover the `libero` package.
touch ./LIBERO/libero/__init__.py

uv pip install --python "$(which python)" --no-deps -e ./LIBERO
uv pip install --python "$(which python)" "robosuite==1.4.1"
uv pip install --python "$(which python)" "mujoco==2.3.7"
uv pip install --python "$(which python)" draccus easydict gym bddl cloudpickle
```

Keep `robosuite==1.4.1` with `mujoco==2.3.7`. MuJoCo 3.x changed the
`mj_fullM` Python API and will fail when robosuite 1.4.x builds LIBERO envs.

Verify the import from the same conda environment:

```bash
python - <<'PY'
from libero.libero import benchmark
print("LIBERO import OK")
PY
```

Set paths for your machine before running Cosmos scripts. These variables mean:

- `HF_HOME`: root directory for Hugging Face caches. Put this on a filesystem
  with enough space for model downloads.
- `HF_HUB_CACHE`: Hugging Face Hub cache. This is usually `${HF_HOME}/hub`.
- `LIBERO_CONFIG_PATH`: LIBERO user config directory. It should contain or
  create `config.yaml`; it is not the cloned `LIBERO/libero` source directory.
  LIBERO defaults to `${HOME}/.libero` if this variable is unset.
- `LD_LIBRARY_PATH`: Linux dynamic-library search path for `.so` files. Prepending
  `${CONDA_PREFIX}/lib` helps Python packages find conda-provided ffmpeg,
  MuJoCo, OpenGL, and CUDA-adjacent shared libraries before system copies.
- `MUJOCO_GL` and `PYOPENGL_PLATFORM`: select EGL headless rendering for LIBERO /
  MuJoCo on servers without a display.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`: compatibility switch for PyTorch 2.6+
  when loading official LIBERO init-state files. Leave it at `1` only for
  trusted LIBERO assets/checkpoints.

```bash
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export LIBERO_HOME="${REPO_ROOT}/LIBERO"
export LIBERO_CONFIG_PATH="${HOME}/.libero"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

If you want LIBERO assets and datasets outside your home directory, edit
`${LIBERO_CONFIG_PATH}/config.yaml` after LIBERO creates it. The file contains
keys such as `benchmark_root`, `bddl_files`, `init_states`, `datasets`, and
`assets`.

If the run prints warnings that `assets` or `datasets` point to an old conda
environment or another project, set `LIBERO_CONFIG_PATH` to a fresh directory
or edit `${LIBERO_CONFIG_PATH}/config.yaml` so those paths point to your current
LIBERO assets, init states, and datasets.

Optional: put the same exports in a conda activation hook so they are applied
automatically whenever you run `conda activate cosmos-policy`:

```bash
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
nano "${CONDA_PREFIX}/etc/conda/activate.d/cosmos_policy_env.sh"
```

Cosmos base models are gated Hugging Face models. Accept the license and log in
before the first run:

```bash
huggingface-cli login
```

Pass the Slurm account and partition through environment variables:

```bash
ACCOUNT=your-slurm-account PARTITION=your-slurm-partition bash run_cosmos_lqr.sh
```

## 3. DiT4DiT Environment Setup

The top-level DiT4DiT script treats `DiT4DiT_steering` as `DIT4DIT_ROOT` by
default, and uses `DiT4DiT_steering/lqr` as `DIT4DIT_LQR_ROOT`. The launcher
exports this value as `LQR_ROOT` only for the downstream DiT4DiT wrappers.

The DiT4DiT package list is pinned in `DiT4DiT_steering/requirements.txt`. It
includes the model stack (`torch`, `torchvision`, `transformers`, `diffusers`,
`accelerate`, `deepspeed`, `peft`, `timm`, `einops`), CUDA 12.8 wheels, and
common vision/data packages (`opencv-python`, `imageio`, `pandas`, `scipy`,
`safetensors`, `huggingface-hub`, `wandb`). LIBERO must be installed separately
for simulation.

```bash
conda create -y -n dit4dit python=3.10 pip
conda activate dit4dit

cd "${REPO_ROOT}/DiT4DiT_steering"
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt
```

Install LIBERO in the same environment:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO

touch /path/to/LIBERO/libero/__init__.py

pip install --no-deps -e /path/to/LIBERO
pip install "robosuite==1.4.1" "mujoco==2.3.7" \
  bddl cloudpickle draccus easydict gym \
  "imageio[ffmpeg]"
```

Keep `robosuite==1.4.1` with `mujoco==2.3.7`; MuJoCo 3.x is not compatible
with robosuite 1.4.x's `mj_fullM` call path.

Verify the import from the same conda environment:

```bash
python - <<'PY'
from libero.libero import benchmark
print("LIBERO import OK")
PY
```

Prepare the following paths:

```bash
export DIT4DIT_ROOT="${REPO_ROOT}/DiT4DiT_steering"
export DIT4DIT_LQR_ROOT="${DIT4DIT_ROOT}/lqr"
export DIT4DIT_CKPT_DIR=/path/to/dit4dit-model
export CKPT_PATH="${DIT4DIT_CKPT_DIR}/dit4dit_libero/final_model/pytorch_model.pt"

export LIBERO_HOME=/path/to/LIBERO
export LIBERO_CONFIG_PATH="${HOME}/.libero"
export PYTHONPATH="${DIT4DIT_ROOT}:${LIBERO_HOME}:${PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Server-client rollout scripts can use one environment or two separate ones.
export MODEL_PYTHON="$(which python)"
export LIBERO_PYTHON="$(which python)"
```

## 4. LingBot-VA Environment Setup

LingBot scripts assume a conda environment named `lingbot`.

```bash
conda create -y -n lingbot python=3.10
conda activate lingbot
```

Install PyTorch. Example for CUDA 12.6:

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

Install LingBot-VA dependencies:

```bash
cd "${REPO_ROOT}/LingBot-VA_steering/LingBot-VA-Modification-lingbot-clean"
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

Configure the checkpoint:

```bash
export LINGBOT_LIBERO_CKPT_PATH=/path/to/checkpoints/lingbot-libero
export LINGBOT_REPO_DIR="${REPO_ROOT}/LingBot-VA_steering/LingBot-VA-Modification-lingbot-clean"
```

Check the checkpoint attention mode. Inference and evaluation should use an
inference-compatible mode such as `torch` or `flashattn`:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

ckpt = Path(os.environ["LINGBOT_LIBERO_CKPT_PATH"])
cfg = ckpt / "transformer" / "config.json"
print("attn_mode =", json.loads(cfg.read_text()).get("attn_mode"))
PY
```

Prepare LIBERO assets and robosuite macros. First verify that LIBERO imports:

```bash
python - <<'PY'
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
print("LIBERO import OK")
PY
```

If robosuite asks you to create private macros, run the `setup_macros.py`
command it prints. It usually looks like:

```bash
python /path/to/conda/envs/lingbot/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
```

## 5. Download Model Weights

Set the Hugging Face cache first if you have not already done so:

```bash
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
python -m pip install -U huggingface_hub
huggingface-cli login
```

### 5.1 Cosmos-Policy

The Cosmos LQR scripts default to the Hugging Face repo id
`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`, so the first run can download the
checkpoint automatically into `HF_HUB_CACHE`. You can also pre-download it:

```bash
huggingface-cli download nvidia/Cosmos-Policy-LIBERO-Predict2-2B
```

The model is gated under NVIDIA's license. Accept the model license on the
Hugging Face page before downloading:
`https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B`.
You normally do not need to set `CKPT_PATH` for Cosmos; leave the default repo
id unless you downloaded the checkpoint to a custom local directory.

### 5.2 DiT4DiT

The DiT4DiT checkpoint is downloaded as a directory. The LQR scripts receive
the actual `.pt` file inside that directory through `CKPT_PATH`.

By default, the scripts look for this file:

```text
${DIT4DIT_ROOT}/checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt
```

Download the official LIBERO checkpoint from Hugging Face into that layout:
`https://huggingface.co/mondo-robotics/dit4dit-model/tree/main/dit4dit_libero`.

```bash
cd "${REPO_ROOT}/DiT4DiT_steering"
export DIT4DIT_ROOT="${REPO_ROOT}/DiT4DiT_steering"

huggingface-cli download mondo-robotics/dit4dit-model \
  --include "dit4dit_libero/**" \
  --local-dir "${DIT4DIT_ROOT}/checkpoint/dit4dit-model"
```

If you store the checkpoint elsewhere, set:

```bash
export DIT4DIT_CKPT_DIR=/path/to/dit4dit-model
export CKPT_PATH="${DIT4DIT_CKPT_DIR}/dit4dit_libero/final_model/pytorch_model.pt"
```

For example, if the downloaded directory is
`/storage/home/hcoda1/9/qdai41/scratch/models/dit4dit-model`, use:

```bash
export DIT4DIT_CKPT_DIR=/storage/home/hcoda1/9/qdai41/scratch/models/dit4dit-model
export CKPT_PATH="${DIT4DIT_CKPT_DIR}/dit4dit_libero/final_model/pytorch_model.pt"
```

You can verify the file exists before running:

```bash
test -f "${CKPT_PATH}" && echo "DiT4DiT checkpoint found: ${CKPT_PATH}"
```

### 5.3 LingBot-VA

LingBot-VA does not have a hard-coded download command in this repo. Download or
copy the LingBot-VA LIBERO checkpoint to a local directory, then point
`LINGBOT_LIBERO_CKPT_PATH` at it:

```bash
export LINGBOT_LIBERO_CKPT_PATH=/path/to/checkpoints/lingbot-libero
```

The directory should contain the subdirectories used by
`wan_va/configs/va_libero_cfg.py` and `wan_va/wan_va_server.py`:

```text
${LINGBOT_LIBERO_CKPT_PATH}/vae/
${LINGBOT_LIBERO_CKPT_PATH}/tokenizer/
${LINGBOT_LIBERO_CKPT_PATH}/text_encoder/
${LINGBOT_LIBERO_CKPT_PATH}/transformer/
```

The attention-mode check in the LingBot setup section verifies
`${LINGBOT_LIBERO_CKPT_PATH}/transformer/config.json` before evaluation.

## 6. Quick Start Eval Runs

The LQR scripts run this pipeline:

```text
collect positive/negative inputs -> SVD -> projected Jacobians -> rollout/eval
```

Use `SKIP_EVAL=0` to run rollout/eval. If steps 1-3 already finished, resume
only the eval stage with `START_AT=4` for Cosmos and LingBot, or `START_AT=5`
for DiT4DiT.

For local runs, eval can be resumed directly from the local wrapper. Cosmos
camera-position local eval runs one LQR hyperparameter setting; the Slurm camera
path runs the full sweep.

```bash
START_AT=4 SKIP_EVAL=0 PERTURBATION=gripper \
  bash run_cosmos_lqr_local.sh
```

## 7. Run Cosmos LQR

Supported perturbations:

```text
noise
gripper / gripper_xyz
camera / cam
```

Examples:

Camera examples use the original pipeline default, `N_POS=10` and `N_NEG=10`.
Gripper/noise examples use 30 collection episodes. Eval examples use 20 rollout
episodes where the downstream script exposes an eval episode variable.

Slurm examples:

```bash
# Camera-position LQR with Slurm eval.
ACCOUNT=your-slurm-account PARTITION=your-slurm-partition \
  PERTURBATION=camera N_POS=10 N_NEG=10 SWEEP_N_EPISODES=20 \
  SKIP_EVAL=0 SUBMIT_SWEEP=1 \
  bash run_cosmos_lqr.sh

# Gripper-position LQR with Slurm eval.
ACCOUNT=your-slurm-account PARTITION=your-slurm-partition \
  PERTURBATION=gripper N_EPISODES=30 SKIP_EVAL=0 \
  bash run_cosmos_lqr.sh

# Gaussian/noise LQR with Slurm eval.
ACCOUNT=your-slurm-account PARTITION=your-slurm-partition \
  PERTURBATION=gaussian N_EPISODES=30 SKIP_EVAL=0 \
  bash run_cosmos_lqr.sh
```

Local examples:

```bash
# Camera-position LQR with local eval.
# Uses the default LQR coefficients; override LAMBDA/Q_SCALE/R_SCALE/R_SCALE_TAU
# to try a different controller setting.
PERTURBATION=camera N_POS=10 N_NEG=10 EVAL_NUM_EPISODES=20 \
  WORLD_SIZE=1 EVAL_WORLD_SIZE=1 SKIP_EVAL=0 \
  bash run_cosmos_lqr_local.sh

# Gripper-position LQR with local eval.
PERTURBATION=gripper N_EPISODES=30 EVAL_NUM_EPISODES=20 \
  WORLD_SIZE=1 EVAL_WORLD_SIZE=1 SKIP_EVAL=0 \
  bash run_cosmos_lqr_local.sh

# Gaussian/noise LQR with local eval.
PERTURBATION=gaussian N_EPISODES=30 EVAL_NUM_EPISODES=20 \
  WORLD_SIZE=1 EVAL_WORLD_SIZE=1 SKIP_EVAL=0 \
  bash run_cosmos_lqr_local.sh
```

Useful Cosmos variables:

```bash
SVD_BASE=/path/to/svd_root
SVD_DIR=/path/to/existing/svd_run
SVD_TASK=0
SUBMIT_SWEEP=1        # Submit the Cosmos camera eval sweep.
N_POS=10
N_NEG=10
CAM_BASE_SEED=42
PRESET=xyz_random_xlarge_3
```

Default outputs:

```text
cosmos_steering/notebooks/lqr/inputs/policy_inputs/
cosmos_steering/directions/svd/
cosmos_steering/notebooks/lqr/rollouts/
```

## 8. Run DiT4DiT LQR

Supported perturbations:

```text
gaussian / noise
gripper / gripper_xyz
```

This checkout does not include a DiT4DiT camera-view LQR pipeline.

Examples:

All examples below use 30 collection episodes and 20 local eval episodes.
DiT4DiT does not currently include a camera-view LQR pipeline in this checkout.

Slurm examples:

```bash
# Gaussian/noise LQR with Slurm eval.
ACCOUNT=your-slurm-account PARTITION=your-slurm-partition \
  PERTURBATION=gaussian N_EPISODES=30 \
  bash run_dit4dit_lqr.sh

# Gripper-position LQR with Slurm eval.
ACCOUNT=your-slurm-account PARTITION=your-slurm-partition \
  PERTURBATION=gripper N_EPISODES=30 \
  bash run_dit4dit_lqr.sh
```

Local examples:

```bash
# Gaussian/noise LQR with local eval.
PERTURBATION=gaussian N_EPISODES=30 EVAL_NUM_EPISODES=20 \
  WORLD_SIZE=1 EVAL_WORLD_SIZE=1 SKIP_EVAL=0 \
  bash run_dit4dit_lqr_local.sh

# Gripper-position LQR with local eval.
PERTURBATION=gripper N_EPISODES=30 EVAL_NUM_EPISODES=20 \
  WORLD_SIZE=1 EVAL_WORLD_SIZE=1 SKIP_EVAL=0 \
  bash run_dit4dit_lqr_local.sh
```

Useful DiT4DiT variables:

```bash
SVD_NOISE_SIGMA=75.0
EVAL_NOISE_SIGMA=22.0
PRESET=xyz_random_xlarge_3
BASE_SEED=42
SWEEP_BASE_SEED=99
DIT4DIT_CKPT_DIR=/path/to/dit4dit-model
CKPT_PATH="${DIT4DIT_CKPT_DIR}/dit4dit_libero/final_model/pytorch_model.pt"
```

Default outputs:

```text
DiT4DiT_steering/lqr/inputs/policy_inputs/
DiT4DiT_steering/lqr/directions/svd/
DiT4DiT_steering/lqr/rollouts/
```

## 9. Run LingBot-VA LQR

Supported perturbations:

```text
init_pos / init / position
gaussian / noise
camera / cam
```

Examples:

```bash
export LINGBOT_LIBERO_CKPT_PATH=/path/to/checkpoints/lingbot-libero
```

The examples below use 30 collection episodes and 20 eval episodes.

Slurm examples:

```bash
# Camera-position LQR with Slurm eval.
ACCOUNT=your-slurm-account PARTITION=your-slurm-partition \
  PERTURBATION=camera NUM_EPISODES=30 EVAL_NUM_EPISODES=20 SKIP_EVAL=0 \
  bash run_lingbot_lqr.sh

# Gripper/init-position LQR with Slurm eval.
ACCOUNT=your-slurm-account PARTITION=your-slurm-partition \
  PERTURBATION=init_pos NUM_EPISODES=30 EVAL_NUM_EPISODES=20 SKIP_EVAL=0 \
  bash run_lingbot_lqr.sh

# Gaussian/noise LQR with Slurm eval.
ACCOUNT=your-slurm-account PARTITION=your-slurm-partition \
  PERTURBATION=gaussian NUM_EPISODES=30 EVAL_NUM_EPISODES=20 SKIP_EVAL=0 \
  bash run_lingbot_lqr.sh
```

Local examples:

```bash
export LINGBOT_LIBERO_CKPT_PATH=/path/to/checkpoints/lingbot-libero

# Camera-position LQR with local eval.
PERTURBATION=camera NUM_EPISODES=30 EVAL_NUM_EPISODES=20 SKIP_EVAL=0 \
  bash run_lingbot_lqr_local.sh

# Gripper/init-position LQR with local eval.
PERTURBATION=init_pos NUM_EPISODES=30 EVAL_NUM_EPISODES=20 SKIP_EVAL=0 \
  bash run_lingbot_lqr_local.sh

# Gaussian/noise LQR with local eval.
PERTURBATION=gaussian NUM_EPISODES=30 EVAL_NUM_EPISODES=20 SKIP_EVAL=0 \
  bash run_lingbot_lqr_local.sh
```

Useful LingBot variables:

```bash
CONFIG_NAME=libero
LIBERO_BENCHMARK=libero_10
COLLECT_MODE=video       # Common for init_pos/camera.
COLLECT_MODE=action      # Common for gaussian.
SELECTED_TIMESTEPS=0,4,9,14,19
PERTURB_SPEC=scripts/lqr/configs/perturb_spec_camera.yaml
LQR_CONFIG=scripts/lqr/configs/lqr_config_video.yaml
EVAL_STARTUP_WAIT_SEC=1200
PORT=29500
```

Resume eval from existing LingBot artifacts:

```bash
EXISTING_COLLECT_DIR=outputs/lqr/pairs_init_pos_xxx \
START_AT=4 SKIP_EVAL=0 \
PERTURBATION=init_pos bash run_lingbot_lqr.sh

EXISTING_PAIRS_ALL_DIR=outputs/lqr/pairs_all_init_pos_xxx \
START_AT=4 SKIP_EVAL=0 \
PERTURBATION=init_pos bash run_lingbot_lqr.sh

SVD_DIR=outputs/lqr/svd_all_perturb_init_pos_xxx \
JAC_SUBDIR=A_tilde_lingbot \
START_AT=4 SKIP_EVAL=0 \
PERTURBATION=init_pos bash run_lingbot_lqr.sh
```

Default outputs:

```text
LingBot-VA_steering/LingBot-VA-Modification-lingbot-clean/outputs/lqr/
LingBot-VA_steering/LingBot-VA-Modification-lingbot-clean/outputs/lqr_eval_*/
```

## 10. Resume and Debug

Common resume patterns:

```bash
# Resume from step 3.
START_AT=3 bash run_cosmos_lqr.sh

# Force rerun of one step.
FORCE_STEP1=1 bash run_dit4dit_lqr.sh

# Resume eval after artifacts are ready.
START_AT=4 SKIP_EVAL=0 bash run_lingbot_lqr.sh

# Pin the server/client port.
PORT=29500 bash run_lingbot_lqr.sh
```

Check Slurm:

```bash
squeue -u "$USER"
```

Check generated files:

```bash
find . -maxdepth 4 -type f \( -name "positive.npz" -o -name "svd_summary.pt" -o -name "A_tilde__full.pt" -o -name "results.json" \)
```

## 12. Directory and Output Reference

```text
run_cosmos_lqr.sh
  -> cosmos_steering/notebooks/lqr/e2e_scripts/*

run_cosmos_lqr_local.sh
  -> cosmos_steering/notebooks/lqr/local_runs/
     Local input/SVD/Jacobian artifacts. Override with OUT_BASE=/path/to/output.
  -> cosmos_steering/notebooks/lqr/rollouts/local_runs/
     Local rollout/eval results and videos.

run_dit4dit_lqr.sh
  -> DiT4DiT_steering/lqr/e2e_scripts/*

run_lingbot_lqr.sh
  -> LingBot-VA_steering/LingBot-VA-Modification-lingbot-clean/run_lqr_*.sh
