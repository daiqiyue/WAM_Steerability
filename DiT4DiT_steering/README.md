# DiT4DiT Steering
This directory is directly built off of the [original DiT4DiT repo](https://dit4dit.github.io/) — see that README for system requirements and installation instructions. Otherwise, the structure is similar to that in [cosmos_steering](../cosmos_steering/).

## Installation

This repo follows the setup guide in the original DiT4DiT repository. The scripts here assume the DiT4DiT repo and conda environment are installed.

## Environment variables

The local runner separates this steering checkout from the upstream model
source. Configure them explicitly (or source the IDEaS configuration below):

```bash
DIT4DIT_ROOT=<path-to-this-repo>/DiT4DiT_steering
DIT4DIT_CODE_ROOT=<path-to-upstream-DiT4DiT-checkout>
DIT4DIT_LQR_ROOT=$DIT4DIT_ROOT/lqr
```

Outputs (inputs, SVD directions, Jacobians, and rollouts) are derived from the
configured LQR/output roots; no author-specific `/projects/...` path is needed.

## Running

Run the desired end-to-end pipeline, e.g.:

```bash
./lqr/e2e_scripts/run_noised_pipeline.sh
./lqr/e2e_scripts/run_noise_pipeline_generalize.sh
./lqr/e2e_scripts/run_gripper_xyz_pipeline_generalize.sh
```

Each script prints a usage block at the top and supports `START_AT=<step>` and `FORCE_STEP<N>=1` to resume or re-run individual steps.

## IDEaS L40S configuration and activation analysis

For the Georgia Tech IDEaS allocation used by this checkout, source the
versioned site configuration before running commands:

```bash
source DiT4DiT_steering/env/ideas_l40s.sh
"$DIT4DIT_PYTHON" DiT4DiT_steering/env/verify_environment.py --load-model
```

This keeps the steering code/output in this repository while selecting the
original model checkout with `DIT4DIT_CODE_ROOT`. It also disables user-site
packages so an older `~/.local` tokenizer cannot shadow the pinned conda
environment.

The DiT4DiT checkpoint configuration must also point to the Diffusers revision
of Cosmos-Predict2.5-2B. The site setup uses a small runtime checkpoint bundle
under `env/checkpoint_runtime/`: its config contains the real base-model path,
while the 21 GB policy weights and normalization statistics are symlinked to the
existing checkpoint. If the Cosmos bundle is absent, download/resume it with:

```bash
DiT4DiT_steering/env/download_cosmos25.sh
```

The end-to-end Slurm entry point is:

```bash
sbatch run_dit4dit_noise_lqr_interpretability.sbatch
```

Use this entry point (or `run_dit4dit_lqr_local.sh`) on IDEaS. The older
server/client launchers under `lqr/e2e_scripts/` still describe the original
authors' cluster layout and refer to server/sweep files that are not present in
this checkout; they are retained as provenance, not used by this run.

It runs paired clean/noisy collection, contrastive SVD, transition Jacobians,
LQR rollouts with equally perturbed baseline and steered arms, activation
capture, the success/failure/steered overlap plot, and the final action-token
directional Jacobian and Jacobian heatmap. The plotting report evaluates
`1 - d(steered, success) / d(failure, success)` in both the displayed PCA plane
and the full mean-pooled activation space; positive values mean the steered
centroid moved closer to the baseline-success centroid.

For the checked-in task-1 job, final artifacts are written below
`lqr/runs/ideas_l40s_noise_task01/eval/noise_task01/`:

- `results.json`: paired perturbed baseline and steered rollout outcomes.
- `activations/`: per-episode post-hook activation traces.
- `interpretability/steering_activation_overlap.{png,pdf}` and the associated
  metrics JSON.
- `interpretability/steer_output_jacobian.{pt,json}` plus
  `steer_output_jacobian_heatmap.png`. The tensor is the directional derivative
  of all eight normalized action-horizon output tokens along the reconstructed
  contrastive activation direction; the JSON also reports continuous physical
  action-scale derivatives (the deployed gripper threshold is nondifferentiable).

See [interpretability/JACOBIAN.md](interpretability/JACOBIAN.md) for the exact
Jacobian definition, tensor dimensions, heatmap interpretation, all-step and
all-block commands, saved-file schema, action-unit caveats, and replotting
instructions.

Output-action success/failure separation can be analyzed in the existing 3D
PCA views with a linear SVM using
`interpretability/analyze_output_action_svm_3d.py`. The script fits planes for
rollout means, complete `8 x 7` inference chunks, and individual 7D tokens. It
reports both in-sample accuracy and grouped leave-one-rollout-out accuracy. In
the grouped evaluation, standardization, PCA, and the SVM are refit without the
held-out rollout, so tokens from one rollout cannot leak across train and test.
The checked-in IDEaS wrapper is:

```bash
sbatch run_dit4dit_output_action_svm_3d.sbatch
```

Gaussian-noise rollout videos display three synchronized panels: the clean
environment agent view, the exact noisy agent view consumed by the model, and
the exact noisy wrist view. The noisy panels remain fixed while the eight
actions from one inference chunk execute, and update at the next labeled
inference. Set `NOISE_OUT_DIR` when submitting the reproducible-rollout Slurm
job to preserve an older output directory.

To branch a saved Gaussian-noise inference into matched raw and ActAdd
continuations, use `interpretability/replay_snapshot_actadd.py`; the checked-in
episode-6/inference-20 experiment is submitted with:

```bash
sbatch run_dit4dit_snapshot_actadd_comparison.sbatch
```

Both branches restore the saved MuJoCo state, exact first observation, Gaussian
noise-generator state, and policy seed. The comparison video labels every
inference and displays the clean view, exact noisy agent/wrist inputs, and
end-effector displacement. The accompanying JSON records per-axis and L2 EEF
displacement, success, executed inference/action counts, and whether the first
model input was reproduced exactly. By default the wrapper uses unit-norm
reconstructed contrastive directions, `alpha=1`, and every block/denoising
step; all three choices are explicit command-line arguments in the Python
entry point.
