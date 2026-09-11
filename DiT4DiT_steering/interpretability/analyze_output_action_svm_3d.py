#!/usr/bin/env python3
"""Fit linear SVM planes to the existing 3D PCA output-action views.

The input representations and PCA preprocessing match
``analyze_output_action_clusters.py``.  In addition to the in-sample plane
shown in each panel, accuracy is evaluated by leaving out one complete rollout
at a time and refitting standardization, PCA, and the SVM from scratch.  This
prevents chunks or tokens from the held-out rollout leaking into training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import minimize

from analyze_output_action_clusters import load_episodes, make_representations


FAILURE_COLOR = "#D55E00"
SUCCESS_COLOR = "#0072B2"
PLANE_COLOR = "#6A51A3"
REPRESENTATIONS = (
    ("rollout_common_prefix", "Rollout means", True),
    ("chunk_common_prefix", "Inference chunks (8×7)", False),
    ("token_common_prefix", "Individual action tokens (7D)", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", nargs=2, action="append", required=True,
        metavar=("NAME", "SNAPSHOT_DIR"),
        help="Dataset label and directory containing ep*.pt snapshots; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--c", type=float, default=1.0, help="squared-hinge SVM C")
    parser.add_argument("--max-scatter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fit_preprocessor(x: np.ndarray) -> dict[str, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    components = vt[:3]
    coords = standardized @ components.T
    variance = singular**2 / max(len(x) - 1, 1)
    ratio = variance[:3] / max(variance.sum(), 1e-12)
    if components.shape[0] < 3:
        components = np.pad(components, ((0, 3 - components.shape[0]), (0, 0)))
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
        ratio = np.pad(ratio, (0, 3 - len(ratio)))
    return {
        "mean": mean,
        "scale": scale,
        "components": components,
        "coordinates": coords,
        "explained_variance_ratio": ratio,
    }


def transform_preprocessor(x: np.ndarray, prep: dict[str, np.ndarray]) -> np.ndarray:
    return ((x - prep["mean"]) / prep["scale"]) @ prep["components"].T


def fit_linear_svm(x: np.ndarray, labels: np.ndarray, c_value: float) -> dict[str, Any]:
    """L2-regularized, class-balanced squared-hinge linear SVM."""
    signed = np.where(labels == 1, 1.0, -1.0)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    sample_weight = np.asarray([0.5 / counts[label] for label in labels])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights, bias = parameters[:3], parameters[3]
        margins = 1.0 - signed * (x @ weights + bias)
        active = margins > 0
        loss = 0.5 * float(weights @ weights)
        loss += c_value * float(np.sum(sample_weight[active] * margins[active] ** 2))
        common = sample_weight[active] * signed[active] * margins[active]
        grad_w = weights - 2.0 * c_value * (x[active].T @ common)
        grad_b = -2.0 * c_value * float(common.sum())
        return loss, np.r_[grad_w, grad_b]

    result = minimize(
        objective,
        np.zeros(4, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"SVM optimization failed: {result.message}")
    return {
        "weights": result.x[:3],
        "bias": float(result.x[3]),
        "objective": float(result.fun),
        "iterations": int(result.nit),
    }


def predict_svm(x: np.ndarray, svm: dict[str, Any]) -> np.ndarray:
    return (x @ svm["weights"] + svm["bias"] >= 0).astype(np.int64)


def classification_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for actual, pred in zip(labels, predicted):
        confusion[int(actual), int(pred)] += 1
    recall_failure = confusion[0, 0] / max(confusion[0].sum(), 1)
    recall_success = confusion[1, 1] / max(confusion[1].sum(), 1)
    return {
        "accuracy": float(np.mean(labels == predicted)),
        "balanced_accuracy": float(0.5 * (recall_failure + recall_success)),
        "failure_recall": float(recall_failure),
        "success_recall": float(recall_success),
        "confusion_matrix_rows_actual_failure_success": confusion.tolist(),
    }


def grouped_leave_one_rollout_out(
    x: np.ndarray,
    labels: np.ndarray,
    rollout_ids: np.ndarray,
    c_value: float,
) -> dict[str, Any]:
    predicted = np.empty_like(labels)
    folds: list[dict[str, Any]] = []
    for rollout_id in np.unique(rollout_ids):
        test = rollout_ids == rollout_id
        train = ~test
        prep = fit_preprocessor(x[train])
        svm = fit_linear_svm(
            transform_preprocessor(x[train], prep), labels[train], c_value
        )
        fold_prediction = predict_svm(transform_preprocessor(x[test], prep), svm)
        predicted[test] = fold_prediction
        folds.append({
            "held_out_rollout": int(rollout_id),
            "label": int(labels[test][0]),
            "n_samples": int(test.sum()),
            "accuracy": float(np.mean(fold_prediction == labels[test])),
        })
    metrics = classification_metrics(labels, predicted)
    metrics["folds"] = folds
    metrics["protocol"] = (
        "leave one complete rollout out; standardization, PCA, and SVM are refit "
        "using only the other rollouts"
    )
    return metrics


def scatter_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(labels) <= maximum:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        target = int(round(maximum * len(indices) / len(labels)))
        selected.append(rng.choice(indices, min(target, len(indices)), replace=False))
    return np.sort(np.concatenate(selected))


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    padding = 0.08 * max(high - low, 1e-6)
    return low - padding, high + padding


def plot_plane(
    axis: Any,
    svm: dict[str, Any],
    limits: list[tuple[float, float]],
) -> None:
    weights = np.asarray(svm["weights"])
    solved_axis = int(np.argmax(np.abs(weights)))
    free_axes = [axis_index for axis_index in range(3) if axis_index != solved_axis]
    first = np.linspace(*limits[free_axes[0]], 24)
    second = np.linspace(*limits[free_axes[1]], 24)
    first_grid, second_grid = np.meshgrid(first, second)
    coordinates = [None, None, None]
    coordinates[free_axes[0]] = first_grid
    coordinates[free_axes[1]] = second_grid
    solved = -(
        svm["bias"]
        + weights[free_axes[0]] * first_grid
        + weights[free_axes[1]] * second_grid
    ) / weights[solved_axis]
    valid = (solved >= limits[solved_axis][0]) & (solved <= limits[solved_axis][1])
    coordinates[solved_axis] = np.where(valid, solved, np.nan)
    axis.plot_surface(
        coordinates[0], coordinates[1], coordinates[2],
        color=PLANE_COLOR, alpha=0.22, linewidth=0, antialiased=True,
    )


def plot_panel(
    axis: Any,
    coordinates: np.ndarray,
    labels: np.ndarray,
    rollout_ids: np.ndarray,
    variance_ratio: np.ndarray,
    svm: dict[str, Any],
    train_metrics: dict[str, Any],
    cv_metrics: dict[str, Any],
    title: str,
    annotate_rollout: bool,
    max_scatter: int,
    seed: int,
) -> None:
    selected = scatter_indices(labels, max_scatter, seed)
    for label, color, name in (
        (0, FAILURE_COLOR, "failure"),
        (1, SUCCESS_COLOR, "success"),
    ):
        mask = selected[labels[selected] == label]
        axis.scatter(
            coordinates[mask, 0], coordinates[mask, 1], coordinates[mask, 2],
            s=38 if annotate_rollout else 8,
            alpha=0.95 if annotate_rollout else 0.2,
            color=color,
            edgecolors="white" if annotate_rollout else "none",
            linewidths=0.5,
            depthshade=False,
        )
    if annotate_rollout:
        for index, rollout_id in enumerate(rollout_ids):
            axis.text(*coordinates[index], str(int(rollout_id)), fontsize=7)
    limits = [padded_limits(coordinates[:, dim]) for dim in range(3)]
    plot_plane(axis, svm, limits)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_xlabel(f"PC1\n{variance_ratio[0] * 100:.1f}%")
    axis.set_ylabel(f"PC2\n{variance_ratio[1] * 100:.1f}%")
    axis.set_zlabel(f"PC3\n{variance_ratio[2] * 100:.1f}%")
    axis.set_title(
        f"{title}\ntrain acc={train_metrics['accuracy']:.1%}, "
        f"rollout-LORO acc={cv_metrics['accuracy']:.1%}",
        fontsize=10,
    )


def analyze_dataset(
    name: str,
    snapshot_dir: Path,
    output_dir: Path,
    c_value: float,
    max_scatter: int,
    seed: int,
) -> dict[str, Any]:
    episodes = load_episodes(snapshot_dir)
    representations = make_representations(episodes)
    common_prefix = int(representations["common_prefix_length"])

    figure = plt.figure(figsize=(18, 6.6), layout="constrained")
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    metrics: dict[str, Any] = {}
    for panel_index, (rep_name, title, annotate) in enumerate(REPRESENTATIONS):
        x, labels, rollout_ids = representations[rep_name]
        prep = fit_preprocessor(x)
        coordinates = prep["coordinates"]
        svm = fit_linear_svm(coordinates, labels, c_value)
        train_metrics = classification_metrics(labels, predict_svm(coordinates, svm))
        cv_metrics = grouped_leave_one_rollout_out(x, labels, rollout_ids, c_value)
        metrics[rep_name] = {
            "n_samples": int(len(x)),
            "n_failure": int(np.sum(labels == 0)),
            "n_success": int(np.sum(labels == 1)),
            "pca_explained_variance_ratio": prep["explained_variance_ratio"].tolist(),
            "svm": {
                "kind": "class-balanced L2-regularized squared-hinge linear SVM",
                "c": c_value,
                "weights_in_3d_pca": np.asarray(svm["weights"]).tolist(),
                "bias_in_3d_pca": svm["bias"],
                "plane_equation": "w0*PC1 + w1*PC2 + w2*PC3 + bias = 0",
            },
            "training": train_metrics,
            "leave_one_rollout_out": cv_metrics,
        }
        plot_panel(
            axes[panel_index], coordinates, labels, rollout_ids,
            prep["explained_variance_ratio"], svm, train_metrics, cv_metrics,
            title, annotate, max_scatter, seed + panel_index,
        )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FAILURE_COLOR,
               label="failure", markersize=8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SUCCESS_COLOR,
               label="success", markersize=8),
        Line2D([0], [0], color=PLANE_COLOR, linewidth=8, alpha=0.35,
               label="linear SVM plane"),
    ]
    figure.legend(handles=handles, loc="outside upper right", frameon=False)
    figure.suptitle(
        f"3D PCA linear-SVM separation — {name}\n"
        f"first {common_prefix} inferences; grouped validation leaves out one rollout",
        fontsize=15,
    )
    output_png = output_dir / f"{name}_output_action_svm_3d.png"
    figure.savefig(output_png, dpi=190)
    figure.savefig(output_png.with_suffix(".pdf"))
    plt.close(figure)
    return {
        "snapshot_dir": str(snapshot_dir.resolve()),
        "n_rollouts": len(episodes),
        "n_success_rollouts": int(sum(episode.success for episode in episodes)),
        "n_failure_rollouts": int(sum(not episode.success for episode in episodes)),
        "common_prefix_inferences": common_prefix,
        "plot": str(output_png.resolve()),
        "representations": metrics,
    }


def main() -> None:
    args = parse_args()
    if args.c <= 0:
        raise ValueError("--c must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for dataset_index, (name, path) in enumerate(args.dataset):
        results[name] = analyze_dataset(
            name, Path(path), args.output_dir, args.c,
            args.max_scatter, args.seed + 1000 * dataset_index,
        )
    output = {
        "definition": (
            "Linear SVMs are fit in the same three-dimensional standardized PCA "
            "spaces as analyze_output_action_clusters.py. Accuracy is reported both "
            "in-sample and with grouped leave-one-rollout-out refitting."
        ),
        "datasets": results,
    }
    output_path = args.output_dir / "output_action_svm_3d_metrics.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "metrics": str(output_path.resolve()),
        "plots": {name: result["plot"] for name, result in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
