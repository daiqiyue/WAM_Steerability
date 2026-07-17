# DiT4DiT Steering
This directory is directly built off of the [original DiT4DiT repo](https://dit4dit.github.io/) — see that README for system requirements and installation instructions. Otherwise, the structure is similar to that in [cosmos_steering](../cosmos_steering/).

## Installation

This repo follows the setup guide in the original DiT4DiT repository. The scripts here assume the DiT4DiT repo and conda environment are installed.

## Environment variables

Each pipeline script sets `DIT4DIT_ROOT` near the top of the file. Update this variable to point to your local DiT4DiT checkout before running:

```bash
DIT4DIT_ROOT=<path-to-your-DiT4DiT-checkout>
LQR_ROOT=$DIT4DIT_ROOT/notebooks/lqr
```

This appears in every end-to-end pipeline script under `lqr/e2e_scripts/`. All output paths (inputs, SVD directions, Jacobians, rollouts) are derived from `LQR_ROOT`, so setting `DIT4DIT_ROOT` correctly is the only path configuration required.

## Running

Run the desired end-to-end pipeline, e.g.:

```bash
./lqr/e2e_scripts/run_noised_pipeline.sh
./lqr/e2e_scripts/run_noise_pipeline_generalize.sh
./lqr/e2e_scripts/run_gripper_xyz_pipeline_generalize.sh
```

Each script prints a usage block at the top and supports `START_AT=<step>` and `FORCE_STEP<N>=1` to resume or re-run individual steps.