# Steering-to-output Jacobian

This directory contains the code used to measure how a hidden-state steering
intervention changes DiT4DiT's final action tokens. The main entry points are:

- `compute_steer_output_jacobian.py`: compute the directional Jacobian for
  selected denoising steps and transformer blocks, then save tensors and plots.
- `replot_steer_output_jacobian.py`: regenerate the per-step heatmaps from a
  saved result without loading the policy model.
- `../../run_dit4dit_all_blocks_steps_jacobian.sbatch`: run the calculation for
  Gaussian-noise and initial-gripper-position perturbations as a two-task Slurm
  array.
- `../../replot_dit4dit_all_jacobians.sbatch`: redraw both saved experiments
  with a colorbar placed outside the block panels.

## Quantity being computed

For one fixed observation, denoising step `t`, and transformer block `l`, let
`h` denote the action-token activation after that block. The SVD pipeline
provides a reconstructed contrastive direction `v`, which this script
normalizes to unit length:

```text
v_hat = v / ||v||_2
h(alpha) = h + alpha * v_hat
a(alpha) = final DiT4DiT action tokens after completing denoising
```

The saved directional Jacobian is

```text
d a(alpha) / d alpha |_(alpha=0)
    = (partial a / partial h) @ v_hat
```

The full activation Jacobian `partial a / partial h` would have one column for
every activation coordinate. It is not materialized. Multiplication by the
contrastive direction reduces it to one derivative per output coordinate. For
the current model, the result is reshaped as:

```text
[8 output-horizon tokens, 7 action dimensions]
```

Thus each heatmap cell answers: if the activation at this one denoising
step/block is moved an infinitesimal amount in the unit contrastive direction,
how does this output-token coordinate initially change?

The primary implementation uses `torch.func.jvp`. If a fused model operation
does not support forward-mode automatic differentiation, the script records a
fallback method and uses the central difference
`(a(+epsilon) - a(-epsilon)) / (2 * epsilon)`.

## Reading the heatmaps

Rows are output action tokens `0..7`. Columns are:

```text
dx, dy, dz, droll, dpitch, dyaw, gripper
```

- Red: the continuous output coordinate increases as `alpha` increases.
- Blue: the continuous output coordinate decreases as `alpha` increases.
- White/light gray: the local derivative is close to zero.
- `|Jv|` or `||Jv||_2`: the Frobenius/L2 norm of the entire `8 x 7`
  directional-derivative array for that block and denoising step. It is a
  summary of sensitivity, not an action displacement by itself.

For a sufficiently small steering-strength change `Delta alpha`, the local
first-order prediction is:

```text
Delta action[token, dim] ~= (d action[token, dim] / d alpha) * Delta alpha
```

This approximation is local to `alpha=0`, the selected observation, the fixed
diffusion-noise seed, and one intervention location. Large steering strengths
can be nonlinear. The rollout LQR controller also intervenes at multiple
blocks/steps with state-dependent amplitudes, so a single-location heatmap does
not by itself equal the action difference observed in a complete rollout.

## Action scale and gripper limitation

Two derivatives are saved:

- `d_action_tokens_d_steer_alpha`: normalized model-output units per unit
  `alpha`; this is the array displayed in the default heatmaps.
- `d_pre_binarization_physical_actions_d_steer_alpha`: the same local
  derivative after applying the checkpoint's dataset min/max unnormalization.

The latter is in the action scale used by the LIBERO policy/controller. It
should not automatically be interpreted as centimeters or radians without
checking the dataset/controller convention.

At deployment, the gripper output is thresholded into `+1` (close) or `-1`
(open). A hard threshold has derivative zero away from the threshold and is
undefined at the threshold. Therefore the saved gripper derivative describes
the continuous **pre-binarization** score. To determine whether a real command
flips, compare the continuous score with the threshold or rerun inference at
the desired finite steering strength.

## Running every denoising step and block

After configuring the IDEaS runtime:

```bash
source DiT4DiT_steering/env/ideas_l40s.sh
sbatch run_dit4dit_all_blocks_steps_jacobian.sbatch
```

Slurm array task `0` processes the Gaussian-noise experiment and task `1`
processes the initial-gripper-position experiment. The checked-in job requests
all four available denoising steps and all 16 transformer blocks.

The equivalent direct command is:

```bash
"$DIT4DIT_PYTHON" \
  DiT4DiT_steering/interpretability/compute_steer_output_jacobian.py \
  --svd-dir /path/to/svd_directory \
  --inputs-npz /path/to/negative.npz \
  --ckpt-path "$CKPT_PATH" \
  --prompt "put both the cream cheese box and the butter in the basket" \
  --obs-index 0 \
  --blocks all \
  --steps all \
  --out-path /path/to/steer_output_jacobian_all_blocks_steps.pt
```

Use comma-separated subsets such as `--blocks 0,8,15 --steps 0,3` for a smaller
diagnostic run. A requested denoising step must have a corresponding SVD
direction in `config.json` and `svd_summary.pt`.

## Saved outputs

Given `--out-path result.pt`, the computation produces:

- `result.pt`: complete metadata and tensors for every `stepN_blockM` record.
- `result.json`: JSON-readable version of each record.
- `result_heatmap_stepN.png`: all requested blocks for denoising step `N`, with
  a shared symmetric color scale across the entire run.
- `result_heatmaps_all.pdf`: all per-step panels in one PDF.
- `result_block_step_overview.{png,pdf}`: block-by-step maps of total
  sensitivity and the largest absolute derivative.

Important fields inside each record are:

| Field | Meaning |
| --- | --- |
| `baseline_normalized_action_tokens` | `a(0)` for the fixed input and seed |
| `d_action_tokens_d_steer_alpha` | normalized `8 x 7` directional Jacobian |
| `d_pre_binarization_physical_actions_d_steer_alpha` | unnormalized continuous derivative |
| `per_token_l2` | derivative norm for each output token |
| `per_action_dim_l2` | derivative norm for each action dimension |
| `total_l2` | `||Jv||_2` across every token and action dimension |
| `max_abs` | largest absolute heatmap entry |
| `method` | forward-mode JVP or central-difference fallback |
| `contrastive_direction_raw_norm` | norm before converting `v` to `v_hat` |

## Replotting without model inference

If the Jacobian tensor is already available, regenerate figures with:

```bash
"$DIT4DIT_PYTHON" \
  DiT4DiT_steering/interpretability/replot_steer_output_jacobian.py \
  /path/to/steer_output_jacobian_all_blocks_steps.pt
```

Use `--steps 0,3` to select steps or `--no-pdf` to write only PNGs. Replotting
does not load DiT4DiT and does not require a GPU, although the provided Slurm
wrapper uses the same site environment for convenience.

## Scope of the result

The Jacobian is observation-specific, seed-specific, and local in steering
strength. To estimate the effect over actual rollouts, compute it for multiple
saved inference observations (see `extract_inference_snapshot.py`) or replay
those observations at finite positive and negative steering strengths. Report
the distribution across observations rather than treating one heatmap as a
global symbolic formula for the policy.
