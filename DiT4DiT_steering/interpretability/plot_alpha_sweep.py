#!/usr/bin/env python3
"""Plot finite-alpha steering rates and exact deployed LIBERO action changes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", nargs=2, action="append", required=True,
        metavar=("NAME", "SWEEP_RESULT"),
        help="Dataset label and saved multi-epsilon .pt or .json result; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-step", type=int, default=0)
    parser.add_argument("--selected-blocks", default="0,8,15")
    return parser.parse_args()


def plain(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".json":
        records = json.loads(path.read_text())
    elif path.suffix == ".pt":
        import torch

        records = plain(torch.load(path, map_location="cpu", weights_only=False)["records"])
    else:
        raise ValueError(f"expected .pt or .json, got {path}")
    if not records or not next(iter(records.values())).get("finite_difference_sweep"):
        raise ValueError(f"{path} does not contain a finite_difference_sweep")
    return records


def epsilon_values(records: dict[str, dict[str, Any]]) -> list[float]:
    first = next(iter(records.values()))["finite_difference_sweep"]
    return sorted(float(key) for key in first)


def sweep_row(row: dict[str, Any], epsilon: float) -> dict[str, Any]:
    return row["finite_difference_sweep"][f"{epsilon:g}"]


def continuous_norm(item: dict[str, Any], field: str) -> float:
    values = np.asarray(item[field], dtype=np.float64)
    return float(np.linalg.norm(values[:, :6]))


def relative_continuous_rate_error(item: dict[str, Any], reference: dict[str, Any]) -> float:
    value = np.asarray(item["libero_central_rate"], dtype=np.float64)[:, :6]
    ref = np.asarray(reference["libero_central_rate"], dtype=np.float64)[:, :6]
    denominator = np.linalg.norm(ref)
    return float(np.linalg.norm(value - ref) / denominator) if denominator > 1e-12 else np.nan


def quantile_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanpercentile(values, 25, axis=0),
        np.nanmedian(values, axis=0),
        np.nanpercentile(values, 75, axis=0),
    )


def summary_curves(
    name: str, records: dict[str, dict[str, Any]], output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    epsilons = epsilon_values(records)
    reference_epsilon = min(epsilons)
    rate_rows = []
    error_rows = []
    delta_rows = []
    flip_rows = []
    for row in records.values():
        reference = sweep_row(row, reference_epsilon)
        rate_rows.append([
            continuous_norm(sweep_row(row, epsilon), "libero_central_rate")
            for epsilon in epsilons
        ])
        error_rows.append([
            relative_continuous_rate_error(sweep_row(row, epsilon), reference)
            for epsilon in epsilons
        ])
        delta_rows.append([
            continuous_norm(sweep_row(row, epsilon), "libero_positive_delta")
            for epsilon in epsilons
        ])
        flip_rows.append([
            int(sweep_row(row, epsilon)["positive_gripper_flip_count"])
            for epsilon in epsilons
        ])
    rate_rows = np.asarray(rate_rows)
    error_rows = np.asarray(error_rows)
    delta_rows = np.asarray(delta_rows)
    flip_rows = np.asarray(flip_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), layout="constrained")
    panels = [
        (axes[0, 0], rate_rows, "Symmetric rate magnitude", "continuous LIBERO action units / alpha"),
        (
            axes[0, 1], error_rows,
            rf"Rate disagreement with $\epsilon={reference_epsilon:g}$",
            "relative L2 error",
        ),
        (axes[1, 0], delta_rows, "Actual action change from 0 to +alpha", "continuous LIBERO action L2"),
    ]
    summary: dict[str, Any] = {"epsilons": epsilons, "records": len(records)}
    for ax, values, title, ylabel in panels:
        low, median, high = quantile_band(values)
        for row in values:
            ax.plot(epsilons, row, color="0.72", linewidth=0.55, alpha=0.28)
        ax.fill_between(epsilons, low, high, color="C0", alpha=0.2, label="25–75%")
        ax.plot(epsilons, median, "o-", color="C0", linewidth=2.0, label="median")
        ax.set_xscale("log")
        ax.set_xticks(epsilons, [f"{epsilon:g}" for epsilon in epsilons])
        ax.set_xlabel(r"finite-difference radius / positive $\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        summary[title] = {
            "median": median.tolist(), "q25": low.tolist(), "q75": high.tolist(),
            "max": np.nanmax(values, axis=0).tolist(),
        }

    total_flips = flip_rows.sum(axis=0)
    axes[1, 1].bar(range(len(epsilons)), total_flips, color="C3", width=0.58)
    axes[1, 1].set_xticks(range(len(epsilons)), [f"{epsilon:g}" for epsilon in epsilons])
    axes[1, 1].set_xlabel(r"positive $\alpha$")
    axes[1, 1].set_ylabel("flipped output-token commands")
    axes[1, 1].set_title("Exact deployed gripper flips across all step/block probes")
    axes[1, 1].grid(axis="y", alpha=0.2)
    summary["positive_gripper_flip_total"] = total_flips.tolist()
    fig.suptitle(f"Finite-alpha steering linearity — {name}", fontsize=15)
    path = output_dir / f"{name}_alpha_sweep_summary.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path, summary


def block_step_overview(
    name: str, records: dict[str, dict[str, Any]], output_dir: Path
) -> Path:
    epsilons = epsilon_values(records)
    reference_epsilon = min(epsilons)
    steps = sorted({int(row["step"]) for row in records.values()})
    blocks = sorted({int(row["block"]) for row in records.values()})
    delta_grids = []
    error_grids = []
    for epsilon in epsilons:
        delta_grid = np.full((len(steps), len(blocks)), np.nan)
        error_grid = np.full_like(delta_grid, np.nan)
        for step_index, step in enumerate(steps):
            for block_index, block in enumerate(blocks):
                row = records[f"step{step}_block{block}"]
                item = sweep_row(row, epsilon)
                reference = sweep_row(row, reference_epsilon)
                delta_grid[step_index, block_index] = continuous_norm(
                    item, "libero_positive_delta"
                )
                error_grid[step_index, block_index] = relative_continuous_rate_error(
                    item, reference
                )
        delta_grids.append(delta_grid)
        error_grids.append(error_grid)

    ncols = len(epsilons)
    fig, axes = plt.subplots(
        2, ncols, figsize=(max(13.5, 3.7 * ncols), 6.8), squeeze=False,
        layout="constrained",
    )
    delta_max = max(float(np.nanmax(grid)) for grid in delta_grids) or 1.0
    error_max = max(float(np.nanmax(grid)) for grid in error_grids) or 1.0
    delta_images = []
    error_images = []
    for col, epsilon in enumerate(epsilons):
        delta_image = axes[0, col].imshow(
            delta_grids[col], cmap="magma", vmin=0, vmax=delta_max, aspect="auto"
        )
        error_image = axes[1, col].imshow(
            error_grids[col], cmap="viridis", vmin=0, vmax=error_max, aspect="auto"
        )
        delta_images.append(delta_image)
        error_images.append(error_image)
        axes[0, col].set_title(rf"$\alpha=+{epsilon:g}$")
        axes[1, col].set_title(rf"central radius $\epsilon={epsilon:g}$")
        for row_index in range(len(steps)):
            for block_index in range(len(blocks)):
                axes[0, col].text(
                    block_index, row_index, f"{delta_grids[col][row_index, block_index]:.3f}",
                    ha="center", va="center", fontsize=6.2,
                    color="white" if delta_grids[col][row_index, block_index] < 0.55 * delta_max else "black",
                )
                value = error_grids[col][row_index, block_index]
                axes[1, col].text(
                    block_index, row_index, "—" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center", va="center", fontsize=6.2,
                    color="white" if np.isfinite(value) and value < 0.55 * error_max else "black",
                )
        for row in range(2):
            axes[row, col].set_xlabel("transformer block")
            axes[row, col].set_xticks(range(len(blocks)), blocks, fontsize=7)
            axes[row, col].set_yticks(range(len(steps)), steps)
            if col == 0:
                axes[row, col].set_ylabel("denoising step")
    fig.colorbar(
        delta_images[-1], ax=axes[0, :].tolist(), location="right", pad=0.015,
        label="actual continuous LIBERO action change L2",
    )
    fig.colorbar(
        error_images[-1], ax=axes[1, :].tolist(), location="right", pad=0.015,
        label=rf"relative rate error vs $\epsilon={reference_epsilon:g}$",
    )
    fig.suptitle(f"Finite steering effect across every denoising step and block — {name}")
    path = output_dir / f"{name}_alpha_sweep_block_step_overview.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def exact_delta_heatmaps(
    name: str, records: dict[str, dict[str, Any]], output_dir: Path
) -> list[Path]:
    epsilons = epsilon_values(records)
    steps = sorted({int(row["step"]) for row in records.values()})
    paths: list[Path] = []
    for epsilon in epsilons:
        all_values = [
            np.asarray(sweep_row(row, epsilon)["libero_positive_delta"], dtype=np.float64)
            for row in records.values()
        ]
        vmax = max(float(np.abs(values).max()) for values in all_values) or 1.0
        pdf_path = output_dir / f"{name}_exact_action_delta_alpha_{epsilon:g}_all_steps.pdf"
        with PdfPages(pdf_path) as pdf:
            for step in steps:
                selected = sorted(
                    [row for row in records.values() if int(row["step"]) == step],
                    key=lambda row: int(row["block"]),
                )
                ncols = min(4, len(selected))
                nrows = math.ceil(len(selected) / ncols)
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=(4.0 * ncols + 1.2, 3.6 * nrows),
                    squeeze=False, layout="constrained",
                )
                image = None
                for ax, row in zip(axes.ravel(), selected):
                    item = sweep_row(row, epsilon)
                    values = np.asarray(item["libero_positive_delta"], dtype=np.float64)
                    image = ax.imshow(
                        values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto"
                    )
                    ax.axvspan(5.5, 6.5, color="0.45", alpha=0.15)
                    ax.set_title(
                        f"block {row['block']}  L2={np.linalg.norm(values):.3g}"
                        f"  flips={item['positive_gripper_flip_count']}",
                        fontsize=9,
                    )
                    ax.set_xlabel("LIBERO action dimension")
                    ax.set_ylabel("output token")
                    ax.set_xticks(range(values.shape[1]), ACTION_NAMES[: values.shape[1]], rotation=40, ha="right")
                    ax.set_yticks(range(values.shape[0]))
                for ax in axes.ravel()[len(selected):]:
                    ax.set_visible(False)
                fig.colorbar(
                    image, ax=[ax for ax in axes.ravel() if ax.get_visible()],
                    location="right", pad=0.02,
                    label=rf"exact $u(+{epsilon:g})-u(0)$ in LIBERO action units",
                )
                fig.suptitle(
                    rf"Exact deployed action change — {name}, step {step}, $\alpha=+{epsilon:g}$"
                )
                path = output_dir / f"{name}_exact_action_delta_alpha_{epsilon:g}_step{step}.png"
                fig.savefig(path, dpi=180)
                pdf.savefig(fig)
                plt.close(fig)
                paths.append(path)
        paths.append(pdf_path)
    return paths


def selected_comparison(
    datasets: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    step: int,
    blocks: list[int],
) -> Path:
    fig, axes = plt.subplots(
        len(datasets), len(blocks),
        figsize=(4.1 * len(blocks), 3.7 * len(datasets)),
        squeeze=False, layout="constrained",
    )
    for row_index, (name, records) in enumerate(datasets.items()):
        epsilons = epsilon_values(records)
        reference_epsilon = min(epsilons)
        for col_index, block in enumerate(blocks):
            ax = axes[row_index, col_index]
            key = f"step{step}_block{block}"
            if key not in records:
                ax.set_visible(False)
                continue
            row = records[key]
            reference = sweep_row(row, reference_epsilon)
            rate = [continuous_norm(sweep_row(row, epsilon), "libero_central_rate") for epsilon in epsilons]
            delta = [continuous_norm(sweep_row(row, epsilon), "libero_positive_delta") for epsilon in epsilons]
            error = [relative_continuous_rate_error(sweep_row(row, epsilon), reference) for epsilon in epsilons]
            ax.plot(epsilons, rate, "o-", label="central rate L2")
            ax.plot(epsilons, delta, "s-", label="actual +alpha delta L2")
            ax.plot(epsilons, error, "^-", label=f"rate rel. error vs {reference_epsilon:g}")
            ax.set_xscale("log")
            ax.set_xticks(epsilons, [f"{epsilon:g}" for epsilon in epsilons])
            ax.set_title(f"{name} · block {block}")
            ax.set_xlabel(r"$\epsilon$ / positive $\alpha$")
            ax.set_ylabel("magnitude")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Selected steering probes at denoising step {step}")
    path = output_dir / f"alpha_sweep_selected_step{step}_comparison.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_records(Path(path)) for name, path in args.result}
    epsilon_sets = {tuple(epsilon_values(records)) for records in datasets.values()}
    if len(epsilon_sets) != 1:
        raise ValueError(f"datasets have different epsilon values: {epsilon_sets}")

    outputs: dict[str, Any] = {}
    for name, records in datasets.items():
        summary_path, numeric_summary = summary_curves(name, records, args.output_dir)
        overview_path = block_step_overview(name, records, args.output_dir)
        heatmaps = exact_delta_heatmaps(name, records, args.output_dir)
        outputs[name] = {
            "source": str(dict(args.result)[name]),
            "summary_plot": str(summary_path.resolve()),
            "block_step_overview": str(overview_path.resolve()),
            "exact_delta_heatmaps": [str(path.resolve()) for path in heatmaps],
            "numeric_summary": numeric_summary,
        }

    comparison = selected_comparison(
        datasets,
        args.output_dir,
        args.selected_step,
        [int(item) for item in args.selected_blocks.split(",")],
    )
    payload = {
        "definition": (
            "central rate uses [u(+epsilon)-u(-epsilon)]/(2 epsilon); exact positive "
            "delta uses the independently evaluated u(+epsilon)-u(0) after rollout "
            "clipping, dataset unnormalization, and gripper thresholding"
        ),
        "comparison_plot": str(comparison.resolve()),
        "datasets": outputs,
    }
    summary_json = args.output_dir / "alpha_sweep_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "summary": str(summary_json.resolve()),
        "comparison_plot": str(comparison.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
