#!/usr/bin/env python3
"""Convert saved output-token Jacobians to deployed LIBERO action changes.

The policy emits normalized continuous action tokens.  LIBERO receives the
tokens after clipping, dataset min/max unnormalization, and a hard gripper
threshold.  This script applies that exact post-processing to a saved
directional Jacobian and plots both the local continuous derivative and the
finite, linearized action change for a chosen delta-alpha.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper*"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", nargs=2, action="append", required=True,
        metavar=("NAME", "JACOBIAN_RESULT"),
        help="Dataset label and saved .json or .pt Jacobian; repeatable.",
    )
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta-alpha", type=float, default=0.1)
    parser.add_argument("--selected-step", type=int, default=0)
    parser.add_argument("--selected-blocks", default="0,8,15")
    return parser.parse_args()


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".pt":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        records = payload["records"]
        return {
            key: {
                field: value.detach().cpu().tolist() if torch.is_tensor(value) else value
                for field, value in row.items()
            }
            for key, row in records.items()
        }
    raise ValueError(f"Expected .json or .pt result, got {path}")


def load_action_stats(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text())
    if "action" in payload:
        raw = payload["action"]
    else:
        first = next(iter(payload.values()))
        raw = first["action"]
    high = np.asarray(raw["max"], dtype=np.float64)[:7]
    low = np.asarray(raw["min"], dtype=np.float64)[:7]
    mask = np.asarray(raw.get("mask", [True] * 7), dtype=bool)[:7]
    scale = np.where(mask, 0.5 * (high - low), 1.0)
    offset = np.where(mask, 0.5 * (high + low), 0.0)
    return {"high": high, "low": low, "mask": mask, "scale": scale, "offset": offset}


def deployed_libero_action(normalized: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    clipped = np.clip(normalized[:, :7], -1.0, 1.0)
    action = np.where(
        stats["mask"][None, :],
        clipped * stats["scale"][None, :] + stats["offset"][None, :],
        clipped,
    )
    # Exact rollout convention: normalized score < 0.5 is close (+1).
    action[:, 6] = np.where(normalized[:, 6] < 0.5, 1.0, -1.0)
    return action


def convert_record(
    row: dict[str, Any], stats: dict[str, np.ndarray], delta_alpha: float
) -> dict[str, Any]:
    baseline = np.asarray(row["baseline_normalized_action_tokens"], dtype=np.float64)
    normalized_jacobian = np.asarray(row["d_action_tokens_d_steer_alpha"], dtype=np.float64)
    clip_slope = ((baseline > -1.0) & (baseline < 1.0)).astype(np.float64)
    local = normalized_jacobian * stats["scale"][None, :] * clip_slope
    # The deployed gripper command is piecewise constant, not differentiable at 0.5.
    local[:, 6] = 0.0

    baseline_action = deployed_libero_action(baseline, stats)
    linearized_normalized = baseline + delta_alpha * normalized_jacobian
    changed_action = deployed_libero_action(linearized_normalized, stats)
    finite_delta = changed_action - baseline_action

    with np.errstate(divide="ignore", invalid="ignore"):
        gripper_crossing = (0.5 - baseline[:, 6]) / normalized_jacobian[:, 6]
    gripper_crossing[~np.isfinite(gripper_crossing)] = np.nan
    positive = gripper_crossing[gripper_crossing > 0]
    return {
        "step": int(row["step"]),
        "block": int(row["block"]),
        "method": row.get("method", "unknown"),
        "baseline_normalized_action_tokens": baseline,
        "normalized_jacobian": normalized_jacobian,
        "baseline_libero_action": baseline_action,
        "local_libero_action_jacobian": local,
        "linearized_delta_libero_action": finite_delta,
        "local_total_l2": float(np.linalg.norm(local)),
        "local_max_abs": float(np.abs(local).max()),
        "delta_total_l2": float(np.linalg.norm(finite_delta)),
        "delta_max_abs": float(np.abs(finite_delta).max()),
        "gripper_flip_count": int(np.count_nonzero(finite_delta[:, 6])),
        "positive_alpha_to_nearest_gripper_flip": (
            float(positive.min()) if len(positive) else None
        ),
    }


def converted_records(
    records: dict[str, dict[str, Any]], stats: dict[str, np.ndarray], delta_alpha: float
) -> dict[str, dict[str, Any]]:
    return {
        key: convert_record(row, stats, delta_alpha)
        for key, row in records.items()
    }


def heatmap_pages(
    name: str,
    records: dict[str, dict[str, Any]],
    output_dir: Path,
    field: str,
    title_prefix: str,
    colorbar_label: str,
    filename_tag: str,
) -> list[Path]:
    steps = sorted({row["step"] for row in records.values()})
    all_values = [np.asarray(row[field]) for row in records.values()]
    vmax = max(float(np.abs(values).max()) for values in all_values) or 1.0
    paths: list[Path] = []
    pdf_path = output_dir / f"{name}_{filename_tag}_all_steps.pdf"
    with PdfPages(pdf_path) as pdf:
        for step in steps:
            selected = sorted(
                [row for row in records.values() if row["step"] == step],
                key=lambda row: row["block"],
            )
            ncols = min(4, len(selected))
            nrows = math.ceil(len(selected) / ncols)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.15 * ncols, 3.7 * nrows), squeeze=False
            )
            image = None
            for ax, row in zip(axes.ravel(), selected):
                values = np.asarray(row[field])
                image = ax.imshow(
                    values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto"
                )
                ax.axvspan(5.5, 6.5, color="#8c8c8c", alpha=0.18)
                ax.set_title(
                    f"block {row['block']}   L2={np.linalg.norm(values):.3g}", fontsize=10
                )
                ax.set_xlabel("LIBERO action dimension")
                ax.set_ylabel("output token")
                ax.set_xticks(range(7), ACTION_NAMES, rotation=40, ha="right")
                ax.set_yticks(range(values.shape[0]))
            for ax in axes.ravel()[len(selected):]:
                ax.set_visible(False)
            fig.suptitle(f"{title_prefix} — {name}, denoising step {step}")
            fig.subplots_adjust(
                left=0.06, right=0.84, bottom=0.08, top=0.92, hspace=0.46, wspace=0.34
            )
            colorbar_axis = fig.add_axes([0.875, 0.12, 0.018, 0.74])
            fig.colorbar(image, cax=colorbar_axis, label=colorbar_label)
            fig.text(
                0.875, 0.055,
                "*gripper is thresholded; local deployed derivative is 0 away from the threshold",
                fontsize=8, ha="left",
            )
            path = output_dir / f"{name}_{filename_tag}_step{step}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    paths.append(pdf_path)
    return paths


def overview_plot(
    name: str, records: dict[str, dict[str, Any]], output_dir: Path, delta_alpha: float
) -> Path:
    steps = sorted({row["step"] for row in records.values()})
    blocks = sorted({row["block"] for row in records.values()})
    local = np.full((len(steps), len(blocks)), np.nan)
    finite = np.full_like(local, np.nan)
    gripper = np.full_like(local, np.nan)
    for si, step in enumerate(steps):
        for bi, block in enumerate(blocks):
            row = records[f"step{step}_block{block}"]
            local[si, bi] = row["local_total_l2"]
            finite[si, bi] = row["delta_total_l2"]
            crossing = row["positive_alpha_to_nearest_gripper_flip"]
            if crossing is not None:
                gripper[si, bi] = crossing

    fig, axes = plt.subplots(3, 1, figsize=(max(11, 0.68 * len(blocks)), 9.0))
    panels = [
        (local, "viridis", r"Local continuous sensitivity  $||\partial u_{LIBERO}/\partial\alpha||_2$", "action units / alpha"),
        (finite, "magma", rf"Linearized finite change  $||\Delta u_{{LIBERO}}||_2$,  $\Delta\alpha={delta_alpha:g}$", "action units"),
        (gripper, "viridis_r", "Smallest positive alpha predicted to flip a gripper token", "alpha to threshold"),
    ]
    for ax, (grid, cmap, title, label) in zip(axes, panels):
        display_grid = grid.copy()
        is_gripper_threshold = label == "alpha to threshold"
        if is_gripper_threshold:
            # Extremely large linearized crossings are numerically uninformative
            # and otherwise flatten every useful color into the bottom bin.
            display_grid[display_grid > 50.0] = np.nan
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("#b9b9b9")
        image = ax.imshow(display_grid, cmap=cmap_obj, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("transformer block")
        ax.set_ylabel("denoising step")
        ax.set_xticks(range(len(blocks)), blocks)
        ax.set_yticks(range(len(steps)), steps)
        finite_values = display_grid[np.isfinite(display_grid)]
        pivot = float(np.median(finite_values)) if len(finite_values) else 0.0
        for si in range(len(steps)):
            for bi in range(len(blocks)):
                value = grid[si, bi]
                if not np.isfinite(value):
                    text = "—"
                elif is_gripper_threshold and value > 50.0:
                    text = ">50"
                else:
                    text = f"{value:.3f}"
                ax.text(
                    bi, si, text, ha="center", va="center", fontsize=7,
                    color=(
                        "white" if np.isfinite(display_grid[si, bi])
                        and display_grid[si, bi] > pivot else "black"
                    ),
                )
        fig.colorbar(image, ax=ax, shrink=0.88, label=label)
    fig.suptitle(f"Steering effect in deployed LIBERO action space — {name}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = output_dir / f"{name}_libero_action_jacobian_overview.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def comparison_plot(
    datasets: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    step: int,
    blocks: list[int],
    delta_alpha: float,
) -> Path:
    selected: list[tuple[str, dict[str, Any]]] = []
    for name, records in datasets.items():
        for block in blocks:
            key = f"step{step}_block{block}"
            if key in records:
                selected.append((name, records[key]))
    if not selected:
        raise ValueError("No selected step/block records were found")
    vmax = max(
        float(np.abs(row["linearized_delta_libero_action"]).max())
        for _, row in selected
    ) or 1.0
    nrows = len(datasets)
    ncols = len(blocks)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.35 * ncols, 4.8 * nrows), squeeze=False
    )
    image = None
    for row_idx, (name, records) in enumerate(datasets.items()):
        for col_idx, block in enumerate(blocks):
            ax = axes[row_idx, col_idx]
            key = f"step{step}_block{block}"
            if key not in records:
                ax.set_visible(False)
                continue
            row = records[key]
            values = row["linearized_delta_libero_action"]
            image = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.axvspan(5.5, 6.5, color="#8c8c8c", alpha=0.18)
            ax.set_title(f"{name} · block {block}\nL2={row['delta_total_l2']:.4f}")
            ax.set_xlabel("LIBERO action dimension")
            ax.set_ylabel("output token")
            ax.set_xticks(range(7), ACTION_NAMES, rotation=40, ha="right")
            ax.set_yticks(range(8))
    fig.suptitle(
        rf"Predicted deployed LIBERO action change, step {step}, $\Delta\alpha={delta_alpha:g}$"
    )
    fig.subplots_adjust(
        left=0.065, right=0.84, bottom=0.09, top=0.91, hspace=0.78, wspace=0.32
    )
    colorbar_axis = fig.add_axes([0.875, 0.16, 0.022, 0.66])
    fig.colorbar(image, cax=colorbar_axis, label=rf"$\Delta u_{{LIBERO}}$ for $\Delta\alpha={delta_alpha:g}$")
    fig.text(
        0.875, 0.09,
        "*gripper shows finite threshold flips; zero means the command did not flip",
        fontsize=8, ha="left",
    )
    path = output_dir / f"libero_action_delta_alpha_{str(delta_alpha).replace('.', 'p')}_comparison.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = load_action_stats(args.action_stats)
    datasets: dict[str, dict[str, dict[str, Any]]] = {}
    outputs: dict[str, Any] = {}
    for name, raw_path in args.result:
        records = converted_records(load_records(Path(raw_path)), stats, args.delta_alpha)
        datasets[name] = records
        derivative_paths = heatmap_pages(
            name, records, args.output_dir,
            "local_libero_action_jacobian",
            r"Local deployed LIBERO action Jacobian",
            r"$\partial u_{LIBERO}/\partial\alpha$",
            "libero_action_jacobian",
        )
        delta_paths = heatmap_pages(
            name, records, args.output_dir,
            "linearized_delta_libero_action",
            rf"Predicted deployed LIBERO action change ($\Delta\alpha={args.delta_alpha:g}$)",
            rf"$\Delta u_{{LIBERO}}$ for $\Delta\alpha={args.delta_alpha:g}$",
            f"libero_action_delta_alpha_{str(args.delta_alpha).replace('.', 'p')}",
        )
        overview = overview_plot(name, records, args.output_dir, args.delta_alpha)
        outputs[name] = {
            "source": str(Path(raw_path).resolve()),
            "derivative_plots": [str(path.resolve()) for path in derivative_paths],
            "delta_plots": [str(path.resolve()) for path in delta_paths],
            "overview": str(overview.resolve()),
            "records": serializable(records),
        }

    blocks = [int(item) for item in args.selected_blocks.split(",")]
    comparison = comparison_plot(
        datasets, args.output_dir, args.selected_step, blocks, args.delta_alpha
    )
    summary = {
        "definition": "LIBERO action change after clipping, dataset min/max unnormalization, and hard gripper threshold, using a first-order normalized-output approximation",
        "delta_alpha": args.delta_alpha,
        "action_names": ACTION_NAMES,
        "action_scale_d_libero_d_normalized": stats["scale"].tolist(),
        "action_offset": stats["offset"].tolist(),
        "gripper_rule": "+1 close if normalized score < 0.5, else -1 open",
        "comparison_plot": str(comparison.resolve()),
        "datasets": outputs,
    }
    summary_path = args.output_dir / "libero_action_jacobian.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "comparison_plot": str(comparison.resolve()),
        "summary": str(summary_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
