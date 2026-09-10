#!/usr/bin/env python
"""Jacobian of final DiT4DiT action tokens with respect to steering strength.

For each requested (denoising step, block), reconstruct the learned SVD
contrastive direction in the full action-token activation space and insert
``alpha * unit_direction`` after that block.  The reported tensor is

    d final_normalized_action_tokens / d alpha

so it directly measures which output token/dimension changes when that hidden
activation is steered.  This is a directional Jacobian (J @ v), not merely the
finite difference between two rollouts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LQR_ROOT = HERE.parent / "lqr"
if str(LQR_ROOT) not in sys.path:
    sys.path.insert(0, str(LQR_ROOT))
from runtime_paths import configure_runtime  # noqa: E402

CODE_ROOT, _ = configure_runtime(HERE.parent)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--svd-dir", type=Path, required=True)
    p.add_argument("--inputs-npz", type=Path, required=True)
    p.add_argument("--ckpt-path", default=os.environ.get("CKPT_PATH"), required=False)
    p.add_argument("--prompt", required=True)
    p.add_argument("--obs-index", type=int, default=0)
    p.add_argument("--blocks", default="0,8,15",
                   help="comma-separated block ids, or 'all'")
    p.add_argument("--steps", default="0",
                   help="comma-separated denoising steps, or 'all'")
    p.add_argument("--output-dims", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fd-epsilon", type=float, default=1e-2,
                   help="fallback central-difference epsilon if forward-mode AD is unsupported")
    p.add_argument("--out-path", type=Path, required=True)
    return p.parse_args()


def prepare_observation(npz_path: Path, index: int, state_dim: int):
    d = np.load(npz_path)
    primary = cv2.resize(d["primary_images"][index], (224, 224), interpolation=cv2.INTER_AREA)
    wrist = cv2.resize(d["wrist_images"][index], (224, 224), interpolation=cv2.INTER_AREA)
    image = np.concatenate([primary, wrist], axis=1)
    proprio = d["proprios"][index].astype(np.float32)[None]
    state = np.stack([np.sin(proprio), np.cos(proprio)], axis=-1).reshape(1, -1)
    if state.shape[-1] < state_dim:
        state = np.pad(state, ((0, 0), (0, state_dim - state.shape[-1])))
    return image, state.astype(np.float32)


def direction_for(svd_dir: Path, cfg: dict, summary: dict, block: int, step: int):
    selected = [int(x) for x in cfg["selected_timesteps"]]
    t_pos = selected.index(step)
    p = int(summary["layer_to_part"][block])
    l0, l1 = [int(x) for x in cfg["partitions"][p]]
    k = int(cfg["k_target"])
    v_path = svd_dir / f"V_part{p}_layers{l0}-{l1}_t{step}_k{k}.pt"
    V = torch.load(v_path, map_location="cpu", weights_only=False)["V"].float()
    c = summary["c_means"][block, t_pos].float()
    full = V @ c
    norm = full.norm().clamp(min=1e-12)
    return full / norm, float(norm), v_path


def main() -> int:
    args = parse_args()
    if not args.ckpt_path:
        raise ValueError("--ckpt-path or CKPT_PATH is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((args.svd_dir / "config.json").read_text())
    summary = torch.load(args.svd_dir / "svd_summary.pt", map_location="cpu", weights_only=False)
    n_blocks = int(summary["c_means"].shape[0])
    selected_steps = [int(x) for x in cfg["selected_timesteps"]]
    blocks = list(range(n_blocks)) if args.blocks.strip().lower() == "all" else [
        int(x) for x in args.blocks.split(",")
    ]
    steps = selected_steps if args.steps.strip().lower() == "all" else [
        int(x) for x in args.steps.split(",")
    ]
    invalid_blocks = [x for x in blocks if x < 0 or x >= n_blocks]
    invalid_steps = [x for x in steps if x not in selected_steps]
    if invalid_blocks:
        raise ValueError(f"invalid blocks {invalid_blocks}; valid range is 0..{n_blocks - 1}")
    if invalid_steps:
        raise ValueError(
            f"steps {invalid_steps} do not have SVD directions; available={selected_steps}"
        )

    import DiT4DiT.model.framework.DiT4DiT  # noqa: F401
    from DiT4DiT.model.framework.base_framework import baseframework
    from DiT4DiT.model.framework.share_tools import read_mode_config

    print(f"loading {args.ckpt_path} on {device}", flush=True)
    model = baseframework.from_pretrained(args.ckpt_path).to(device).eval()
    action_model = model.action_model
    dit = action_model.model
    _, norm_stats = read_mode_config(args.ckpt_path)
    action_stats = norm_stats[next(iter(norm_stats))]["action"]
    image, state_np = prepare_observation(
        args.inputs_npz, args.obs_index, int(action_model.config.state_dim)
    )

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        bi = model.backbone_interface.build_cosmos_inputs(
            images=[[image]], instructions=[args.prompt]
        )
        bout = model.backbone_interface(
            **bi, output_hidden_states=True, output_attentions=False, return_dict=True
        )
        vl_embs = bout.hidden_states[-1].detach()
    state_t = torch.from_numpy(state_np).unsqueeze(0).to(device=device, dtype=vl_embs.dtype)

    horizon = int(action_model.action_horizon)
    action_dim = int(action_model.config.action_dim)
    output_dims = min(args.output_dims, action_dim, len(action_stats["max"]))
    action_high = np.asarray(action_stats["max"], dtype=np.float32)[:output_dims]
    action_low = np.asarray(action_stats["min"], dtype=np.float32)[:output_dims]
    action_mask = np.asarray(
        action_stats.get("mask", np.ones_like(action_high, dtype=bool)), dtype=bool
    )[:output_dims]
    unnormalize_scale = np.where(action_mask, 0.5 * (action_high - action_low), 1.0)
    n_steps = int(cfg["sampling_steps"])
    state_tokens = 1 if action_model.state_encoder is not None else 0
    gen = torch.Generator(device=device).manual_seed(args.seed)
    initial_actions = torch.randn(
        1, horizon, action_dim, generator=gen, device=device, dtype=vl_embs.dtype
    )

    def final_actions(alpha, target_block: int, target_step: int, direction: torch.Tensor):
        actions = initial_actions.clone()
        state_features = action_model.state_encoder(state_t) if state_tokens else None
        direction = direction.to(device=device, dtype=actions.dtype).reshape(horizon, -1)
        for step in range(n_steps):
            t_disc = int((1.0 - step / float(n_steps)) * action_model.num_timestep_buckets)
            ts = torch.full((1,), t_disc, device=device, dtype=torch.long)
            action_features = action_model.action_encoder(actions, ts)
            if action_model.config.add_pos_embed:
                pos = torch.arange(horizon, device=device, dtype=torch.long)
                action_features = action_features + action_model.position_embedding(pos).unsqueeze(0)
            hidden = torch.cat((state_features, action_features), dim=1) if state_tokens else action_features
            temb = dit.timestep_encoder(ts)
            for block_idx, block in enumerate(dit.transformer_blocks):
                interleaved = block_idx % 2 == 1 and dit.config.interleave_self_attention
                hidden = block(
                    hidden,
                    attention_mask=None,
                    encoder_hidden_states=None if interleaved else vl_embs,
                    encoder_attention_mask=None,
                    temb=temb,
                )
                if step == target_step and block_idx == target_block:
                    if state_tokens:
                        prefix = torch.zeros(
                            hidden.shape[0], state_tokens, hidden.shape[-1],
                            device=hidden.device, dtype=hidden.dtype,
                        )
                        update = torch.cat((prefix, direction.unsqueeze(0)), dim=1)
                    else:
                        update = direction.unsqueeze(0)
                    hidden = hidden + alpha.to(hidden.dtype) * update
            shift, scale = dit.proj_out_1(torch.nn.functional.silu(temb)).chunk(2, dim=1)
            hidden = dit.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
            model_out = dit.proj_out_2(hidden)
            velocity = action_model.action_decoder(model_out)[:, -horizon:]
            actions = actions - (1.0 / n_steps) * velocity
        return actions[0, :, :output_dims].float()

    records = {}
    for step in steps:
        for block in blocks:
            direction, raw_norm, v_path = direction_for(args.svd_dir, cfg, summary, block, step)
            alpha0 = torch.zeros((), device=device, dtype=torch.float32)

            def fn(alpha):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    return final_actions(alpha, block, step, direction)

            method = "torch.func.jvp"
            try:
                output, derivative = torch.func.jvp(
                    fn, (alpha0,), (torch.ones_like(alpha0),), strict=False
                )
            except Exception as exc:  # some fused attention kernels lack forward AD
                method = f"central_difference_fallback ({type(exc).__name__})"
                with torch.no_grad():
                    output = fn(alpha0)
                    derivative = (fn(alpha0 + args.fd_epsilon) - fn(alpha0 - args.fd_epsilon)) / (
                        2.0 * args.fd_epsilon
                    )
            derivative = derivative.detach().cpu()
            output = output.detach().cpu()
            derivative_physical = derivative * torch.from_numpy(unnormalize_scale)
            key = f"step{step}_block{block}"
            records[key] = {
                "step": step,
                "block": block,
                "method": method,
                "contrastive_direction_raw_norm": raw_norm,
                "contrastive_basis_path": str(v_path),
                "baseline_normalized_action_tokens": output,
                "d_action_tokens_d_steer_alpha": derivative,
                "per_token_l2": derivative.norm(dim=-1),
                "per_action_dim_l2": derivative.norm(dim=0),
                "d_pre_binarization_physical_actions_d_steer_alpha": derivative_physical,
                "physical_per_token_l2": derivative_physical.norm(dim=-1),
                "physical_per_action_dim_l2": derivative_physical.norm(dim=0),
                "total_l2": float(derivative.norm()),
                "physical_total_l2": float(derivative_physical.norm()),
                "max_abs": float(derivative.abs().max()),
            }
            print(f"{key}: method={method}, |Jv|={derivative.norm():.6g}, "
                  f"max={derivative.abs().max():.6g}", flush=True)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "definition": "d final normalized action tokens / d alpha, where alpha multiplies a unit-norm reconstructed SVD contrastive activation direction",
        "physical_definition": "continuous action derivative after dataset min-max unnormalization; the deployed gripper command is subsequently thresholded and has no ordinary derivative at the threshold",
        "prompt": args.prompt,
        "inputs_npz": str(args.inputs_npz),
        "obs_index": args.obs_index,
        "seed": args.seed,
        "records": records,
    }, args.out_path)
    json_path = args.out_path.with_suffix(".json")
    serializable = {
        key: {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in row.items()}
        for key, row in records.items()
    }
    json_path.write_text(json.dumps(serializable, indent=2))

    # Causal-effect views: rows are action-horizon output tokens and columns are
    # action dimensions. Use one symmetric color scale across every requested
    # block and timestep, so magnitudes are directly comparable.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"][:output_dims]
    arrays = [row["d_action_tokens_d_steer_alpha"].numpy() for row in records.values()]
    vmax = max(float(np.abs(a).max()) for a in arrays) or 1.0
    from matplotlib.backends.backend_pdf import PdfPages

    detail_paths = []
    pdf_path = args.out_path.with_name(args.out_path.stem + "_heatmaps_all.pdf")
    with PdfPages(pdf_path) as pdf:
        for step in steps:
            step_records = [
                (key, row) for key, row in records.items() if int(row["step"]) == step
            ]
            ncols = min(4, len(step_records))
            nrows = math.ceil(len(step_records) / ncols)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.1 * ncols, 3.7 * nrows), squeeze=False
            )
            image = None
            for ax, (key, row) in zip(axes.ravel(), step_records):
                values = row["d_action_tokens_d_steer_alpha"].numpy()
                image = ax.imshow(
                    values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto"
                )
                ax.set_title(f"block {row['block']}   |Jv|={row['total_l2']:.3g}")
                ax.set_xlabel("action dimension")
                ax.set_ylabel("output token")
                ax.set_xticks(range(output_dims), labels, rotation=40, ha="right")
                ax.set_yticks(range(horizon))
            for ax in axes.ravel()[len(step_records):]:
                ax.set_visible(False)
            fig.suptitle(
                f"Directional output Jacobian — denoising step {step} (all blocks)"
            )
            fig.subplots_adjust(
                left=0.06, right=0.84, bottom=0.08, top=0.92, hspace=0.45, wspace=0.32
            )
            cbar_ax = fig.add_axes([0.875, 0.12, 0.018, 0.74])
            fig.colorbar(image, cax=cbar_ax,
                         label=r"$d\,action / d\,steer\;alpha$")
            step_path = args.out_path.with_name(
                args.out_path.stem + f"_heatmap_step{step}.png"
            )
            fig.savefig(step_path, dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            detail_paths.append(step_path)

    # Layer/time overview. This is the easiest plot for locating sensitive
    # blocks before consulting the detailed 8 x action-dimension panels.
    norm_grid = np.full((len(steps), len(blocks)), np.nan, dtype=np.float32)
    max_grid = np.full_like(norm_grid, np.nan)
    for si, step in enumerate(steps):
        for bi_idx, block in enumerate(blocks):
            row = records[f"step{step}_block{block}"]
            norm_grid[si, bi_idx] = row["total_l2"]
            max_grid[si, bi_idx] = row["max_abs"]
    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.7 * len(blocks)), 6.2), squeeze=False)
    for ax, grid, title, cbar_label in (
        (axes[0, 0], norm_grid, r"Total sensitivity $\|Jv\|_2$", r"$\|Jv\|_2$"),
        (axes[1, 0], max_grid, "Largest individual action derivative", "max absolute derivative"),
    ):
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_ylabel("denoising step")
        ax.set_xlabel("transformer block")
        ax.set_xticks(range(len(blocks)), blocks)
        ax.set_yticks(range(len(steps)), steps)
        for si in range(len(steps)):
            for bi_idx in range(len(blocks)):
                ax.text(bi_idx, si, f"{grid[si, bi_idx]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if grid[si, bi_idx] > np.nanmax(grid) * 0.55 else "black")
        fig.colorbar(im, ax=ax, shrink=0.88, label=cbar_label)
    fig.suptitle("Steering-to-output Jacobian across all denoising steps and blocks")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    overview_path = args.out_path.with_name(args.out_path.stem + "_block_step_overview.png")
    overview_pdf_path = overview_path.with_suffix(".pdf")
    fig.savefig(overview_path, dpi=180, bbox_inches="tight")
    fig.savefig(overview_pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out_path}, {json_path}, {pdf_path}, {overview_path}, "
          f"and {len(detail_paths)} per-step heatmaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
