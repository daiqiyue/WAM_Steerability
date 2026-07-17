# LingBot-VA LIBERO Baseline and LQR Workflow

This repository contains a cleaned LIBERO evaluation workflow for two experiment
families:

- Vanilla policy baseline under perturbations.
- LQR activation steering under the same perturbations.

The code is intentionally free of machine-specific paths. Any path that depends
on your server, user name, storage mount, conda installation, checkpoint
location, or local cache should be passed through an environment variable. Dummy
paths in this README use `/path/to/...`; replace them with real paths on your
machine before running.

## Repository Layout

Important files and directories:

- `wan_va/`: LingBot-VA model, server, training, and config code.
- `evaluation/libero/client.py`: LIBERO rollout client. Both baseline and LQR
  eval use this client.
- `scripts/lqr/`: shared baseline/LQR experiment code.
- `scripts/lqr/run_libero_policy_eval.py`: vanilla baseline evaluator.
- `scripts/lqr/run_libero_lqr_eval.py`: LQR evaluator.
- `scripts/lqr/run_lqr_pipeline.sh`: end-to-end LQR artifact and eval pipeline.
- `scripts/lqr/configs/`: perturbation and controller configs.
- `run_baseline_perturbations.sh`: runs baseline eval for init-position,
  Gaussian, and camera perturbations.
- `run_lqr_init_pos.sh`: runs the init-position LQR pipeline.
- `run_lqr_gaussian.sh`: runs the Gaussian-noise LQR pipeline.
- `run_lqr_camera.sh`: runs the camera-perturbation LQR pipeline.

The old `scripts/activation_steering/` and single `script/` workflows are not
needed for the current baseline/LQR LIBERO flow.

## Dummy Paths and Local Configuration

The repo uses these environment variables for local paths and machine-specific
settings:

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `LINGBOT_REPO_DIR` | No | directory containing the launcher script | Override the repo root if launching from another location. |
| `CONDA_ENV` | No | `lingbot` | Conda environment to activate. |
| `LINGBOT_LIBERO_CKPT_PATH` | Yes for eval | `/path/to/lingbot-libero` | Local LingBot-VA LIBERO checkpoint directory. |
| `PORT` | No | computed from `PORT_BASE` and optional `SLURM_JOB_ID` | WebSocket port used by server/client. |
| `PORT_BASE` | No | `29056` | Base port used when `PORT` is not set. |
| `OUT_BASE` | No | depends on script | Output root for generated results. |
| `LINGBOT_DEBUG_LOG_PATH` | No | `outputs/debug/build_all_pairs_debug.jsonl` | Debug log path for `build_all_pairs.py`; set empty to disable. |
| `WANDB_API_KEY` | Only for training with wandb | unset | W&B API key. Not needed for baseline or LQR eval. |
| `WANDB_BASE_URL` | Only for training with wandb | unset | W&B server URL. |
| `WANDB_TEAM_NAME` | Only for training with wandb | unset | W&B entity/team name. |
| `WANDB_PROJECT` | Only for training with wandb | `va_robotwin` | W&B project name. |

Minimum path replacement before evaluation:

```bash
export LINGBOT_LIBERO_CKPT_PATH=/path/to/checkpoints/lingbot-libero
```

Optional repo override:

```bash
export LINGBOT_REPO_DIR=/path/to/LingBot-VA-Modification
```

The launcher scripts now compute the repo root from their own location, so in
normal use you can run them directly from the repo directory without setting
`LINGBOT_REPO_DIR`.

## Environment Setup

These steps assume a Linux machine with CUDA-capable GPU access and conda.

1. Create and activate a Python 3.10 environment:

```bash
conda create -n lingbot python=3.10 -y
conda activate lingbot
```

2. Install PyTorch. Pick the wheel that matches your CUDA runtime. For CUDA
12.6, the pinned environment used by this project is:

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

3. Install repository dependencies:

```bash
cd /path/to/LingBot-VA-Modification
pip install -r requirements.txt
```

If `flash_attn` fails from source build isolation, install it after PyTorch:

```bash
pip install flash-attn --no-build-isolation
```

4. Install this repo in editable mode if your environment needs package metadata:

```bash
pip install -e .
```

5. Install or prepare LIBERO assets. The exact command depends on your LIBERO
installation. A common pattern is:

```bash
python -c "import libero; print(libero.__file__)"
```

Then follow your LIBERO package's asset setup instructions. Verify that
`evaluation/libero/client.py` can import:

```bash
python - <<'PY'
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
print("LIBERO import OK")
PY
```

6. If robosuite asks you to create private macros, run the setup command printed
by robosuite in your environment. It usually looks like:

```bash
python /path/to/conda/envs/lingbot/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
```

7. Configure the LingBot checkpoint path:

```bash
export LINGBOT_LIBERO_CKPT_PATH=/path/to/checkpoints/lingbot-libero
```

The checkpoint directory should contain the model files expected by
`wan_va/configs/va_libero_cfg.py`. In particular, the transformer config inside
the checkpoint must use an inference-compatible attention mode.

## Attention Mode Check

LingBot-VA reads `attn_mode` from the checkpoint's `transformer/config.json`.
Before inference or evaluation, check:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

ckpt = Path(os.environ["LINGBOT_LIBERO_CKPT_PATH"])
cfg = ckpt / "transformer" / "config.json"
data = json.loads(cfg.read_text())
print("attn_mode =", data.get("attn_mode"))
PY
```

For baseline/LQR inference, use an inference mode such as `torch` or
`flashattn`. If your checkpoint is set to a training-only mode, edit
`transformer/config.json` before running evaluation.

## Perturbation Types

The workflow supports three perturbation families:

| Perturbation | Baseline config | LQR collection config | Default eval tasks |
| --- | --- | --- | --- |
| Init position | `scripts/lqr/configs/eval_init_pos.yaml` | `scripts/lqr/configs/perturb_spec_init_pos.yaml` | `1 2 3 7 9` |
| Gaussian image noise | `scripts/lqr/configs/perturb_spec_gaussian_30.yaml` | `scripts/lqr/configs/perturb_spec_gaussian_30.yaml` | `6 0 1 4 7` |
| Camera randomization | `scripts/lqr/configs/eval_camera.yaml` | `scripts/lqr/configs/perturb_spec_camera.yaml` | `0 2 4 5 9` |

Baseline uses `run_libero_policy_eval.py`, which starts the unmodified
LingBot-VA server and then runs the LIBERO client under perturbations.

LQR uses `run_lqr_pipeline.sh`, which collects positive/negative examples,
builds row-aligned pairs, runs SVD, computes Jacobians, starts an LQR-injected
server, and evaluates.

## Run the Vanilla Baseline

First configure the checkpoint:

```bash
cd /path/to/LingBot-VA-Modification
conda activate lingbot
export LINGBOT_LIBERO_CKPT_PATH=/path/to/checkpoints/lingbot-libero
```

Run all three perturbation families:

```bash
bash run_baseline_perturbations.sh
```

Default outputs:

```text
outputs/baseline/init_pos/
outputs/baseline/gaussian/
outputs/baseline/camera/
```

Each output directory contains per-variant rollout videos and a `summary.json`.

Useful baseline overrides:

```bash
# Use fewer episodes for a smoke test.
EVAL_NUM_EPISODES=2 bash run_baseline_perturbations.sh

# Resume partial outputs.
RESUME=1 bash run_baseline_perturbations.sh

# Change output root.
OUT_BASE=outputs/baseline_debug bash run_baseline_perturbations.sh

# Evaluate different task ids.
INIT_POS_TASK_IDS="1 7" \
GAUSSIAN_TASK_IDS="6" \
CAMERA_TASK_IDS="0 4" \
bash run_baseline_perturbations.sh

# Pin the WebSocket port.
PORT=29500 bash run_baseline_perturbations.sh
```

Run one perturbation manually:

```bash
python scripts/lqr/run_libero_policy_eval.py \
  --config-name libero \
  --libero-benchmark libero_10 \
  --task-ids 1 2 3 7 9 \
  --num-episodes 20 \
  --startup-wait-sec 1200 \
  --perturb-spec scripts/lqr/configs/eval_init_pos.yaml \
  --out-dir outputs/baseline/init_pos
```

## Run LQR Pipelines

The LQR launchers are shell scripts, not sbatch files. They can run inside an
interactive GPU allocation, a normal terminal with GPU access, or any job system
wrapper that calls `bash`.

Always set the checkpoint path first:

```bash
cd /path/to/LingBot-VA-Modification
conda activate lingbot
export LINGBOT_LIBERO_CKPT_PATH=/path/to/checkpoints/lingbot-libero
```

Run init-position LQR:

```bash
bash run_lqr_init_pos.sh
```

Run Gaussian-noise LQR:

```bash
bash run_lqr_gaussian.sh
```

Run camera-perturbation LQR:

```bash
bash run_lqr_camera.sh
```

Useful LQR smoke tests:

```bash
# Build artifacts only; skip rollout eval.
SKIP_EVAL=1 NUM_EPISODES=2 bash run_lqr_init_pos.sh

# Very small Gaussian run.
SKIP_EVAL=1 NUM_EPISODES=2 NUM_SAMPLES=2 K_TARGET=2 bash run_lqr_gaussian.sh

# Very small camera run.
SKIP_EVAL=1 NUM_EPISODES=2 N_POS=2 N_NEG=2 NUM_SAMPLES=2 K_TARGET=2 bash run_lqr_camera.sh
```

Default LQR outputs are written under `outputs/lqr/`, with timestamped
subdirectories. The most important artifacts are:

- `pairs_*`: raw positive and negative rollout records.
- `pairs_all_*`: row-aligned positive/negative pairs.
- `svd_*`: SVD basis, contrastive vectors, and projected diffs.
- `A_tilde_lingbot/`: Jacobian artifacts under the SVD directory.
- `outputs/lqr_eval_*`: LQR rollout videos and success metrics.

## LQR Pipeline Stages

The high-level stages are:

1. `run_collect_pairs.py`: collect positive/negative records under the selected
   perturbation.
2. `pair_inputs_by_similarity.py`: for init-position, match unaligned success
   and failure buckets by similarity.
3. `run_partition_svd.py`: compute contrastive SVD bases from activation deltas.
4. `run_compute_jacobians.py`: fit projected Jacobians with VJP by default.
5. `run_libero_lqr_eval.py`: start the LQR-injected server and run LIBERO eval.

`run_lqr_pipeline.sh` orchestrates these stages.

## Reusing Existing LQR Artifacts

If you already collected pairs:

```bash
EXISTING_COLLECT_DIR=outputs/lqr/pairs_init_pos_xxx \
SKIP_EVAL=1 \
bash run_lqr_init_pos.sh
```

If you already have row-aligned pairs:

```bash
EXISTING_PAIRS_ALL_DIR=outputs/lqr/pairs_all_init_pos_xxx \
SKIP_EVAL=1 \
bash run_lqr_init_pos.sh
```

If you already have an SVD directory and only want eval:

```bash
SVD_DIR=outputs/lqr/svd_all_perturb_init_pos_xxx \
JAC_SUBDIR=A_tilde_lingbot \
bash run_lqr_init_pos.sh
```

## Important Runtime Variables

Core variables:

| Variable | Default | Used by | Description |
| --- | --- | --- | --- |
| `CONFIG_NAME` | `libero` | baseline, LQR | LingBot config name. |
| `LIBERO_BENCHMARK` | `libero_10` | baseline, LQR | LIBERO benchmark suite. |
| `TASK_ID` | script-specific | LQR | Collection task id. |
| `TASK_IDS` | script-specific | baseline, LQR eval | Space-separated eval task ids. |
| `NUM_EPISODES` | script-specific | LQR collect | Collection episodes. |
| `EVAL_NUM_EPISODES` | `20` | baseline, LQR eval | Eval episodes per task/variant. |
| `PERTURB_SPEC` | script-specific | LQR collect/eval | Perturbation config. |
| `OUT_BASE` | script-specific | baseline, LQR | Output root. |
| `PORT` | auto | baseline, LQR | Server/client WebSocket port. |
| `EVAL_STARTUP_WAIT_SEC` | `1200` | baseline, LQR eval | Server startup wait timeout. |

LQR-specific variables:

| Variable | Default | Description |
| --- | --- | --- |
| `COLLECT_MODE` | script-specific | Capture `action` or `video` branch activations. |
| `SELECTED_TIMESTEPS` | script-specific | Denoising timesteps used for SVD/Jacobian. |
| `NUM_SAMPLES` | script-specific | Number of paired rows used for SVD; `-1` means all. |
| `K_TARGET` | `64` | Projection rank. |
| `P_OVER` | `10` | PCA oversampling rank. |
| `PARTITIONS` | script-specific | Layer partitions, e.g. `0-9,10-19,20-29`; empty means auto. |
| `JAC_METHOD` | `vjp` | Jacobian method. |
| `JAC_SUBDIR` | `A_tilde_lingbot` | Jacobian artifact subdirectory under `SVD_DIR`. |
| `LQR_CONFIG` | script-specific | LQR controller config. |
| `INJECT_MODE` | `auto` | Injection branch during eval. |
| `SKIP_EVAL` | `0` | Set `1` to stop after artifact generation. |

## Dummy Path Replacement Checklist

Before sharing or running on a new machine:

1. Search for non-dummy absolute paths:

```bash
PRIVATE_PATTERNS='old_user|old_account|old_storage_mount|old_email_domain'
rg -n "${PRIVATE_PATTERNS}" \
  --glob '!outputs/**' \
  --glob '!logs/**' \
  --glob '!*.pdf' \
  --glob '!*.pyc'
```

2. Replace required local paths through environment variables, not hardcoded
   edits:

```bash
export LINGBOT_LIBERO_CKPT_PATH=/path/to/checkpoints/lingbot-libero
export LINGBOT_REPO_DIR=/path/to/LingBot-VA-Modification
export LINGBOT_DEBUG_LOG_PATH=outputs/debug/build_all_pairs_debug.jsonl
```

3. Keep generated artifacts out of commits. This repo ignores:

```text
logs/
outputs/
*.out
*.err
__pycache__/
```

4. If you need to publish experiment outputs, sanitize their manifests/logs
   separately because generated files can contain absolute paths from the
   machine that produced them.

## Troubleshooting

Server does not start:

- Check `PORT` is free.
- Increase `EVAL_STARTUP_WAIT_SEC`.
- Confirm `LINGBOT_LIBERO_CKPT_PATH` points to a real checkpoint.
- Confirm `transformer/config.json` has an inference-compatible `attn_mode`.

Client cannot create LIBERO env:

- Verify LIBERO imports.
- Verify assets are installed.
- Run robosuite macro setup if prompted.
- On headless machines, confirm EGL/MuJoCo rendering works in your environment.

Shape mismatch during LQR:

- Make sure `COLLECT_MODE`, `INJECT_MODE`, `LQR_CONFIG`, and the SVD/Jacobian
  artifacts all correspond to the same activation branch.
- For video-mode artifacts, use `LQR_CONFIG=scripts/lqr/configs/lqr_config_video.yaml`.
- For action-mode artifacts, use `LQR_CONFIG=scripts/lqr/configs/lqr_config.yaml`.

Partial run or crash:

- For baseline, use `RESUME=1`.
- For LQR, reuse `EXISTING_COLLECT_DIR`, `EXISTING_PAIRS_ALL_DIR`, or `SVD_DIR`.
- Keep `PORT` fixed if you are restarting a run manually and want predictable
  server/client behavior.

## Minimal End-to-End Example

```bash
cd /path/to/LingBot-VA-Modification
conda activate lingbot
export LINGBOT_LIBERO_CKPT_PATH=/path/to/checkpoints/lingbot-libero

# Baseline, all three perturbations, quick smoke test.
EVAL_NUM_EPISODES=2 bash run_baseline_perturbations.sh

# LQR artifact smoke test for init-position perturbation.
SKIP_EVAL=1 NUM_EPISODES=2 K_TARGET=2 NUM_SAMPLES=2 bash run_lqr_init_pos.sh

# Full init-position LQR run.
bash run_lqr_init_pos.sh
```
