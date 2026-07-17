#!/usr/bin/env python
"""Collect per-DiT-block activations from a positive/negative .npz pair.

For each row in positive.npz (and each row in negative.npz), run one denoising
step of the Cosmos-Policy model under the supplied prompt with forward hooks
on every model.net.blocks[*] capturing the mean-pooled output (shape D, the
per-token hidden dim).

Output (.npz file with two stacked arrays + small metadata):
    positive     : float32 (N_pos, N_blocks, D)        # default
                   float32 (N_pos, N_blocks, T, D)     # --per-timestep
    negative     : float32 (N_neg, N_blocks, D)        # default
                   float32 (N_neg, N_blocks, T, D)     # --per-timestep
    task_id      : int (scalar)
    prompt       : str
    per_timestep : bool (scalar)

The T axis above is the model's latent-temporal axis (the second dim of the
DiT block output [B, T, H, W, D]). With --per-timestep we mean-pool over
(B, H, W) only, preserving T; default behaviour also pools over T to yield
one D-vec per block.

Single-task; one (pair_dir, prompt) pair per invocation. Fan-out across tasks
is handled by the caller (e.g. violet/collect_actadd_acts.sh).

Usage:
    python collect_acts_passive.py \
        --pair-dir notebooks/lqr/inputs/policy_inputs/libero_10__task06__noise_extreme_pos_neg \
        --prompt "put the white mug on the plate and put the chocolate pudding to the right of the plate" \
        --task-id 6 \
        --out-path notebooks/lqr/artifacts/actadd/activations__libero_10__task06.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch as th


def _setup_env():
    """Set sensible cache defaults if the caller hasn't already (matches the
    pattern used by run_actadd_cosmos_policy.py). environment.sh wins via
    setdefault."""
    hf = os.environ.get("HF_HOME", "/usr/scratch/jhong392/huggingface")
    hub = os.environ.get("HF_HUB_CACHE", str(Path(hf, "hub")))
    Path(hub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf)
    os.environ.setdefault("HF_HUB_CACHE", hub)
    os.environ.setdefault("TRANSFORMERS_CACHE", hub)

    libero_cfg = os.environ.get(
        "LIBERO_CONFIG_PATH", "/usr/scratch/jhong392/.libero"
    )
    Path(libero_cfg).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", libero_cfg)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # notebooks/actadd/collect_acts_passive.py -> repo root is two up
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pair-dir", type=Path, required=True,
                    help="directory containing positive.npz and negative.npz "
                         "(both with primary_images, wrist_images, proprios)")
    ap.add_argument("--prompt", type=str, required=True,
                    help="task description used during the denoising step "
                         "(prompts the policy, then we capture activations)")
    ap.add_argument("--task-id", type=int, required=True,
                    help="task id (recorded in the output for provenance)")
    ap.add_argument("--out-path", type=Path, required=True,
                    help=".npz output path; will overwrite if it exists")
    ap.add_argument("--ckpt-path", type=str,
                    default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--per-timestep", dest="per_timestep",
                    action="store_true", default=False,
                    help="keep the latent T axis: hook reduces over (B,H,W) "
                         "and produces a (T, D) vector per block instead of "
                         "(D,). Use with run_actadd_cosmos_policy.py's "
                         "2-D vector path.")
    return ap.parse_args()


def collect_activations_from_npz(npz_path, prompt, model, n_blocks,
                                 cfg, dataset_stats, get_action,
                                 per_timestep=False):
    """Return float32 array of shape (N_rows, n_blocks, D) (default) or
    (N_rows, n_blocks, T, D) (per_timestep=True).

    For each row, runs one denoising step under `prompt` with forward hooks
    installed on every DiT block; the hook captures the mean-pooled output
    either as a length-D vector (mean over batch/time/H/W) or a (T, D) matrix
    (mean over batch/H/W only).
    """
    d = np.load(npz_path)
    primary_images = d["primary_images"]   # [N, H, W, C] uint8
    wrist_images = d["wrist_images"]       # [N, H, W, C] uint8
    proprios = d["proprios"]               # [N, 9] float32
    n = int(primary_images.shape[0])

    buf = {}
    hooks = []
    for b in range(n_blocks):
        if per_timestep:
            def _hook(_module, _inp, out, _b=b):
                # out: [B, T, H, W, D] -- mean over (B, H, W) -> [T, D]
                buf[_b] = out.detach().float().mean(dim=(0, 2, 3)).cpu().numpy()
        else:
            def _hook(_module, _inp, out, _b=b):
                # out: [B, T, H, W, D] -- mean over all non-channel dims -> [D]
                buf[_b] = out.detach().float().mean(dim=(0, 1, 2, 3)).cpu().numpy()
        hooks.append(model.net.blocks[b].register_forward_hook(_hook))

    rows = []
    try:
        for i in range(n):
            buf.clear()
            obs = {
                "primary_image": primary_images[i],
                "wrist_image":   wrist_images[i],
                "proprio":       proprios[i],
            }
            get_action(cfg, model, dataset_stats, obs, prompt,
                       num_denoising_steps_action=1)
            row = np.stack(
                [buf[b] for b in range(n_blocks)], axis=0
            )  # (n_blocks, D) or (n_blocks, T, D)
            rows.append(row)
            if (i + 1) % 50 == 0 or (i + 1) == n:
                print(f"    [{npz_path.name}] {i + 1}/{n} rows",
                      end="\r", flush=True)
    finally:
        for h in hooks:
            h.remove()

    return np.stack(rows, axis=0).astype(np.float32)  # (N, n_blocks, [T,] D)


def main():
    args = parse_args()
    _setup_env()

    pos_npz = args.pair_dir / "positive.npz"
    neg_npz = args.pair_dir / "negative.npz"
    for p in (pos_npz, neg_npz):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        PolicyEvalConfig,
    )
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        load_dataset_stats,
        init_t5_text_embeddings_cache,
    )

    cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=args.ckpt_path,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{args.ckpt_path}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{args.ckpt_path}/libero_t5_embeddings.pkl",
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        chunk_size=16,
        num_open_loop_steps=16,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=1,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        task_suite_name=args.suite,
        suite="libero",
    )
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    model, _ = get_model(cfg)

    n_blocks = len(model.net.blocks)
    print(f"[model] {n_blocks} DiT blocks loaded  "
          f"(prompt='{args.prompt}', task_id={args.task_id}, "
          f"per_timestep={args.per_timestep})")

    print(f"[collect] positive: {pos_npz}")
    pos = collect_activations_from_npz(
        pos_npz, args.prompt, model, n_blocks, cfg, dataset_stats, get_action,
        per_timestep=args.per_timestep,
    )
    print(f"\n[collect] positive done -> shape {pos.shape}")

    print(f"[collect] negative: {neg_npz}")
    neg = collect_activations_from_npz(
        neg_npz, args.prompt, model, n_blocks, cfg, dataset_stats, get_action,
        per_timestep=args.per_timestep,
    )
    print(f"\n[collect] negative done -> shape {neg.shape}")

    np.savez(
        args.out_path,
        positive=pos,
        negative=neg,
        task_id=np.int64(args.task_id),
        prompt=np.array(args.prompt),
        per_timestep=np.bool_(args.per_timestep),
    )
    print(f"[done] wrote {args.out_path}  "
          f"(positive={pos.shape}, negative={neg.shape}, "
          f"per_timestep={args.per_timestep})")


if __name__ == "__main__":
    main()
