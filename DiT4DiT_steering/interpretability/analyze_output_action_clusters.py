#!/usr/bin/env python3
"""Compare output-action tokens from successful and failed rollouts.

The snapshot bundles are produced by the reproducible rollout runners under
``lqr/``.  Each inference stores an 8 x 7 ``predicted_env_actions`` array.
This script analyzes the arrays at three resolutions:

* token: one seven-dimensional action token;
* inference chunk: the complete flattened 8 x 7 prediction;
* rollout: the per-rollout mean prediction, using both the full trajectory and
  a common-length prefix to reduce episode-length confounding.

No scikit-learn dependency is required.  PCA, two-means clustering, ARI/NMI,
silhouette scores, and leave-one-rollout-out nearest-centroid classification
are implemented with NumPy/SciPy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist, pdist


ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
FAILURE_COLOR = "#D55E00"
SUCCESS_COLOR = "#0072B2"


@dataclass
class Episode:
    episode: int
    success: int
    chunks: np.ndarray  # [num_inferences, 8, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "SNAPSHOT_DIR"),
        required=True,
        help="Dataset name and directory containing ep*.pt bundles; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scatter", type=int, default=5000)
    parser.add_argument("--max-silhouette", type=int, default=2000)
    return parser.parse_args()


def load_episodes(snapshot_dir: Path) -> list[Episode]:
    paths = sorted(snapshot_dir.glob("ep*.pt"))
    if not paths:
        raise FileNotFoundError(f"No ep*.pt files found under {snapshot_dir}")
    episodes: list[Episode] = []
    for path in paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        chunks = np.stack(
            [np.asarray(row["predicted_env_actions"], dtype=np.float64) for row in bundle["inferences"]]
        )
        if chunks.ndim != 3 or chunks.shape[1:] != (8, 7):
            raise ValueError(f"Unexpected predicted_env_actions shape {chunks.shape} in {path}")
        episodes.append(
            Episode(
                episode=int(bundle.get("episode", len(episodes))),
                success=int(bool(bundle["success"])),
                chunks=chunks,
            )
        )
    labels = {ep.success for ep in episodes}
    if labels != {0, 1}:
        raise ValueError(f"Need both success and failure examples; found labels {labels}")
    return episodes


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x - mean) / scale, mean, scale


def pca_coordinates(x: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    z, _, _ = standardize(x)
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    n = min(n_components, vt.shape[0])
    coords = z @ vt[:n].T
    denom = max(len(z) - 1, 1)
    variance = singular**2 / denom
    ratio = variance[:n] / max(variance.sum(), 1e-12)
    if n < n_components:
        coords = np.pad(coords, ((0, 0), (0, n_components - n)))
        ratio = np.pad(ratio, (0, n_components - n))
    return coords, ratio


def stratified_subsample(y: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(y) <= max_n:
        return np.arange(len(y))
    indices: list[np.ndarray] = []
    classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(classes, counts):
        target = max(2, int(round(max_n * count / len(y))))
        cls_idx = np.flatnonzero(y == cls)
        indices.append(rng.choice(cls_idx, size=min(target, len(cls_idx)), replace=False))
    out = np.concatenate(indices)
    if len(out) > max_n:
        out = rng.choice(out, size=max_n, replace=False)
    return np.sort(out)


def silhouette_for_labels(x: np.ndarray, y: np.ndarray) -> float:
    """Mean silhouette using the success/failure labels as cluster labels."""
    scores = np.empty(len(x), dtype=np.float64)
    for cls in (0, 1):
        own = np.flatnonzero(y == cls)
        other = np.flatnonzero(y != cls)
        own_d = cdist(x[own], x[own])
        if len(own) > 1:
            a = own_d.sum(axis=1) / (len(own) - 1)
        else:
            a = np.zeros(len(own))
        b = cdist(x[own], x[other]).mean(axis=1)
        scores[own] = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    return float(scores.mean())


def kmeans2(x: np.ndarray, rng: np.random.Generator, n_init: int = 24) -> np.ndarray:
    best_labels: np.ndarray | None = None
    best_inertia = math.inf
    n = len(x)
    for _ in range(n_init):
        first = int(rng.integers(n))
        distances = np.sum((x - x[first]) ** 2, axis=1)
        total = distances.sum()
        second = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=distances / total))
        centers = x[[first, second]].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _iteration in range(200):
            new_labels = np.argmin(cdist(x, centers, metric="sqeuclidean"), axis=1)
            if _iteration and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in (0, 1):
                members = x[labels == k]
                centers[k] = members.mean(axis=0) if len(members) else x[int(rng.integers(n))]
        inertia = float(np.sum((x - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    assert best_labels is not None
    return best_labels


def comb2(n: np.ndarray | int) -> np.ndarray | float:
    arr = np.asarray(n, dtype=np.float64)
    return arr * (arr - 1.0) / 2.0


def adjusted_rand_index(y: np.ndarray, pred: np.ndarray) -> float:
    table = np.zeros((2, 2), dtype=np.int64)
    for a, b in zip(y, pred):
        table[int(a), int(b)] += 1
    nij = float(comb2(table).sum())
    ai = float(comb2(table.sum(axis=1)).sum())
    bj = float(comb2(table.sum(axis=0)).sum())
    total = float(comb2(len(y)))
    expected = ai * bj / max(total, 1e-12)
    upper = 0.5 * (ai + bj)
    return float((nij - expected) / max(upper - expected, 1e-12))


def normalized_mutual_information(y: np.ndarray, pred: np.ndarray) -> float:
    table = np.zeros((2, 2), dtype=np.float64)
    for a, b in zip(y, pred):
        table[int(a), int(b)] += 1
    pxy = table / table.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    hx = -sum(p * math.log(p) for p in px if p > 0)
    hy = -sum(p * math.log(p) for p in py if p > 0)
    return float(mi / max(math.sqrt(hx * hy), 1e-12))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    recalls = [float((pred[y == cls] == cls).mean()) for cls in (0, 1)]
    return float(np.mean(recalls))


def best_cluster_balanced_accuracy(y: np.ndarray, cluster: np.ndarray) -> float:
    return max(balanced_accuracy(y, cluster), balanced_accuracy(y, 1 - cluster))


def pairwise_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x0, x1 = x[y == 0], x[y == 1]
    between = float(cdist(x0, x1).mean())
    within0 = float(pdist(x0).mean()) if len(x0) > 1 else float("nan")
    within1 = float(pdist(x1).mean()) if len(x1) > 1 else float("nan")
    pooled = float(np.nanmean([within0, within1]))
    return {
        "between_success_failure_mean": between,
        "within_failure_mean": within0,
        "within_success_mean": within1,
        "between_to_within_ratio": between / max(pooled, 1e-12),
    }


def metric_summary(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    max_silhouette: int,
) -> dict[str, Any]:
    z, _, _ = standardize(x)
    mu0, mu1 = z[y == 0].mean(axis=0), z[y == 1].mean(axis=0)
    centroid_distance = float(np.linalg.norm(mu1 - mu0))
    spread0 = float(np.mean(np.sum((z[y == 0] - mu0) ** 2, axis=1)))
    spread1 = float(np.mean(np.sum((z[y == 1] - mu1) ** 2, axis=1)))
    pooled_rms = math.sqrt(0.5 * (spread0 + spread1))

    sample_idx = stratified_subsample(y, max_silhouette, rng)
    cluster = kmeans2(z, rng)
    result: dict[str, Any] = {
        "n_samples": int(len(x)),
        "n_failure": int((y == 0).sum()),
        "n_success": int((y == 1).sum()),
        "raw_centroid_l2": float(np.linalg.norm(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0))),
        "standardized_centroid_l2": centroid_distance,
        "centroid_distance_over_within_rms": centroid_distance / max(pooled_rms, 1e-12),
        "label_silhouette": silhouette_for_labels(z[sample_idx], y[sample_idx]),
        "kmeans_adjusted_rand_index": adjusted_rand_index(y, cluster),
        "kmeans_normalized_mutual_info": normalized_mutual_information(y, cluster),
        "kmeans_best_balanced_accuracy": best_cluster_balanced_accuracy(y, cluster),
        "pairwise_standardized": pairwise_summary(z[sample_idx], y[sample_idx]),
    }
    return result


def leave_one_episode_out_nearest_centroid(x: np.ndarray, y: np.ndarray) -> float:
    pred = np.empty_like(y)
    for i in range(len(x)):
        train = np.arange(len(x)) != i
        z_train, mean, scale = standardize(x[train])
        z_test = (x[i] - mean) / scale
        centers = np.stack([z_train[y[train] == cls].mean(axis=0) for cls in (0, 1)])
        pred[i] = int(np.argmin(np.linalg.norm(centers - z_test, axis=1)))
    return balanced_accuracy(y, pred)


def make_representations(episodes: list[Episode]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    token_x: list[np.ndarray] = []
    token_y: list[np.ndarray] = []
    token_ep: list[np.ndarray] = []
    chunk_x: list[np.ndarray] = []
    chunk_y: list[np.ndarray] = []
    chunk_ep: list[np.ndarray] = []

    common_prefix = min(len(ep.chunks) for ep in episodes)
    prefix_token_x: list[np.ndarray] = []
    prefix_token_y: list[np.ndarray] = []
    prefix_token_ep: list[np.ndarray] = []
    prefix_chunk_x: list[np.ndarray] = []
    prefix_chunk_y: list[np.ndarray] = []
    prefix_chunk_ep: list[np.ndarray] = []
    full_rollout: list[np.ndarray] = []
    prefix_rollout: list[np.ndarray] = []
    rollout_y: list[int] = []
    rollout_ep: list[int] = []

    for ep in episodes:
        n_inf = len(ep.chunks)
        token_x.append(ep.chunks.reshape(-1, 7))
        token_y.append(np.full(n_inf * 8, ep.success, dtype=np.int64))
        token_ep.append(np.full(n_inf * 8, ep.episode, dtype=np.int64))
        chunk_x.append(ep.chunks.reshape(n_inf, -1))
        chunk_y.append(np.full(n_inf, ep.success, dtype=np.int64))
        chunk_ep.append(np.full(n_inf, ep.episode, dtype=np.int64))
        prefix_chunks = ep.chunks[:common_prefix]
        prefix_token_x.append(prefix_chunks.reshape(-1, 7))
        prefix_token_y.append(np.full(common_prefix * 8, ep.success, dtype=np.int64))
        prefix_token_ep.append(np.full(common_prefix * 8, ep.episode, dtype=np.int64))
        prefix_chunk_x.append(prefix_chunks.reshape(common_prefix, -1))
        prefix_chunk_y.append(np.full(common_prefix, ep.success, dtype=np.int64))
        prefix_chunk_ep.append(np.full(common_prefix, ep.episode, dtype=np.int64))
        full_rollout.append(ep.chunks.mean(axis=0).reshape(-1))
        prefix_rollout.append(ep.chunks[:common_prefix].mean(axis=0).reshape(-1))
        rollout_y.append(ep.success)
        rollout_ep.append(ep.episode)

    return {
        "token": (np.concatenate(token_x), np.concatenate(token_y), np.concatenate(token_ep)),
        "chunk": (np.concatenate(chunk_x), np.concatenate(chunk_y), np.concatenate(chunk_ep)),
        "token_common_prefix": (
            np.concatenate(prefix_token_x),
            np.concatenate(prefix_token_y),
            np.concatenate(prefix_token_ep),
        ),
        "chunk_common_prefix": (
            np.concatenate(prefix_chunk_x),
            np.concatenate(prefix_chunk_y),
            np.concatenate(prefix_chunk_ep),
        ),
        "rollout_full": (np.stack(full_rollout), np.asarray(rollout_y), np.asarray(rollout_ep)),
        "rollout_common_prefix": (
            np.stack(prefix_rollout),
            np.asarray(rollout_y),
            np.asarray(rollout_ep),
        ),
        "common_prefix_length": common_prefix,
    }


def without_gripper(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 7:
        return x[:, :6]
    reshaped = x.reshape(len(x), 8, 7)
    return reshaped[:, :, :6].reshape(len(x), -1)


def scatter_indices(y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    return stratified_subsample(y, max_n, np.random.default_rng(seed))


def plot_pca2d(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    episodes: np.ndarray,
    title: str,
    max_scatter: int,
    seed: int,
    annotate_episode: bool = False,
) -> None:
    coords, ratio = pca_coordinates(x, 3)
    idx = scatter_indices(y, max_scatter, seed)
    for cls, color, label in [(0, FAILURE_COLOR, "failure"), (1, SUCCESS_COLOR, "success")]:
        mask = idx[y[idx] == cls]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=34 if annotate_episode else 10,
            alpha=0.9 if annotate_episode else 0.22,
            c=color,
            label=label,
            edgecolors="white" if annotate_episode else "none",
            linewidths=0.5,
        )
    if annotate_episode:
        for i, ep in enumerate(episodes):
            ax.annotate(str(ep), coords[i, :2], xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")
    ax.grid(alpha=0.2)


def plot_pca3d_panel(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    episodes: np.ndarray,
    title: str,
    max_scatter: int,
    seed: int,
    annotate_episode: bool = False,
) -> None:
    coords, ratio = pca_coordinates(x, 3)
    idx = scatter_indices(y, max_scatter, seed)
    for cls, color, label in [(0, FAILURE_COLOR, "failure"), (1, SUCCESS_COLOR, "success")]:
        mask = idx[y[idx] == cls]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            s=34 if annotate_episode else 7,
            alpha=0.9 if annotate_episode else 0.18,
            c=color,
            label=label,
            depthshade=False,
        )
    if annotate_episode:
        for i, ep in enumerate(episodes):
            ax.text(*coords[i, :3], str(ep), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(f"PC1\n{ratio[0] * 100:.1f}%")
    ax.set_ylabel(f"PC2\n{ratio[1] * 100:.1f}%")
    ax.set_zlabel(f"PC3\n{ratio[2] * 100:.1f}%")


def make_plots(
    name: str,
    representations: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
    max_scatter: int,
    seed: int,
) -> tuple[Path, Path]:
    token_x, token_y, token_ep = representations["token_common_prefix"]
    chunk_x, chunk_y, chunk_ep = representations["chunk_common_prefix"]
    rollout_x, rollout_y, rollout_ep = representations["rollout_common_prefix"]
    common_prefix = int(representations["common_prefix_length"])

    class_means = [rollout_x[rollout_y == cls].mean(axis=0).reshape(8, 7) for cls in (0, 1)]
    delta = class_means[1] - class_means[0]
    vmax = max(float(np.abs(delta).max()), 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout="constrained")
    plot_pca2d(
        axes[0, 0], rollout_x, rollout_y, rollout_ep,
        f"Rollout means: first {common_prefix} inferences", max_scatter, seed, True,
    )
    plot_pca2d(axes[0, 1], chunk_x, chunk_y, chunk_ep, "Inference chunks (8×7)", max_scatter, seed + 1)
    plot_pca2d(axes[1, 0], token_x, token_y, token_ep, "Individual action tokens (7D)", max_scatter, seed + 2)
    image = axes[1, 1].imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1, 1].set_title(f"Success − failure mean action\n(first {common_prefix} inferences; equal rollout weight)")
    axes[1, 1].set_xticks(np.arange(7), ACTION_NAMES, rotation=35, ha="right")
    axes[1, 1].set_yticks(np.arange(8), [f"token {i}" for i in range(8)])
    for row in range(8):
        for col in range(7):
            axes[1, 1].text(col, row, f"{delta[row, col]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes[1, 1], shrink=0.82, label="action units")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FAILURE_COLOR, label="failure", markersize=8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SUCCESS_COLOR, label="success", markersize=8),
    ]
    fig.legend(handles=handles, loc="outside upper right", frameon=False)
    fig.suptitle(f"Output action separation — {name}", fontsize=16)
    png = output_dir / f"{name}_output_action_clusters_2d.png"
    pdf = output_dir / f"{name}_output_action_clusters_2d.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    fig3d = plt.figure(figsize=(18, 6), layout="constrained")
    axes3d = [fig3d.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    plot_pca3d_panel(
        axes3d[0], rollout_x, rollout_y, rollout_ep,
        f"Rollout means\nfirst {common_prefix} inferences", max_scatter, seed, True,
    )
    plot_pca3d_panel(axes3d[1], chunk_x, chunk_y, chunk_ep, "Inference chunks", max_scatter, seed + 1)
    plot_pca3d_panel(axes3d[2], token_x, token_y, token_ep, "Action tokens", max_scatter, seed + 2)
    fig3d.legend(handles=handles, loc="outside upper right", frameon=False)
    fig3d.suptitle(f"3D PCA of output actions — {name}", fontsize=16)
    png3d = output_dir / f"{name}_output_action_clusters_3d.png"
    fig3d.savefig(png3d, dpi=180)
    plt.close(fig3d)
    return png, png3d


def analyze_dataset(
    name: str,
    path: Path,
    output_dir: Path,
    seed: int,
    max_scatter: int,
    max_silhouette: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episodes = load_episodes(path)
    reps = make_representations(episodes)
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "snapshot_dir": str(path.resolve()),
        "n_rollouts": len(episodes),
        "n_success": int(sum(ep.success for ep in episodes)),
        "n_failure": int(sum(1 - ep.success for ep in episodes)),
        "inferences_per_rollout": [len(ep.chunks) for ep in episodes],
        "mean_inferences_failure": float(np.mean([len(ep.chunks) for ep in episodes if not ep.success])),
        "mean_inferences_success": float(np.mean([len(ep.chunks) for ep in episodes if ep.success])),
        "common_prefix_inferences": int(reps["common_prefix_length"]),
        "representations": {},
    }

    for rep_name in (
        "token",
        "chunk",
        "token_common_prefix",
        "chunk_common_prefix",
        "rollout_full",
        "rollout_common_prefix",
    ):
        x, y, _episode_ids = reps[rep_name]
        summary = metric_summary(x, y, rng, max_silhouette)
        if rep_name.startswith("rollout"):
            summary["leave_one_rollout_out_nearest_centroid_balanced_accuracy"] = (
                leave_one_episode_out_nearest_centroid(x, y)
            )
        result["representations"][rep_name] = summary

        x_no_gripper = without_gripper(x)
        no_gripper = metric_summary(x_no_gripper, y, rng, max_silhouette)
        if rep_name.startswith("rollout"):
            no_gripper["leave_one_rollout_out_nearest_centroid_balanced_accuracy"] = (
                leave_one_episode_out_nearest_centroid(x_no_gripper, y)
            )
        result["representations"][f"{rep_name}_without_gripper"] = no_gripper

    rollout_x, rollout_y, _ = reps["rollout_common_prefix"]
    rollout_mats = rollout_x.reshape(len(rollout_x), 8, 7)
    rows: list[dict[str, Any]] = []
    for token_idx in range(8):
        for action_idx, action_name in enumerate(ACTION_NAMES):
            values = rollout_mats[:, token_idx, action_idx]
            fail = values[rollout_y == 0]
            success = values[rollout_y == 1]
            pooled_var = (
                ((len(fail) - 1) * fail.var(ddof=1) + (len(success) - 1) * success.var(ddof=1))
                / max(len(fail) + len(success) - 2, 1)
            )
            rows.append(
                {
                    "perturbation": name,
                    "token_index": token_idx,
                    "action_dimension": action_name,
                    "failure_mean": float(fail.mean()),
                    "success_mean": float(success.mean()),
                    "success_minus_failure": float(success.mean() - fail.mean()),
                    "cohens_d": float((success.mean() - fail.mean()) / max(math.sqrt(pooled_var), 1e-12)),
                }
            )

    plot2d, plot3d = make_plots(name, reps, result, output_dir, max_scatter, seed)
    result["plots"] = {"pca_2d": str(plot2d.resolve()), "pca_3d": str(plot3d.resolve())}
    return result, rows


def plot_metric_summary(results: dict[str, Any], output_dir: Path) -> Path:
    names = list(results)
    reps = ["token_common_prefix", "chunk_common_prefix", "rollout_common_prefix", "rollout_full"]
    rep_labels = ["token prefix", "chunk prefix", "rollout prefix", "rollout full*"]
    metrics = [
        ("label_silhouette", "Label silhouette\n(>0 suggests separation)"),
        ("kmeans_adjusted_rand_index", "K-means ARI\n(1 = exact labels)"),
        ("kmeans_best_balanced_accuracy", "K-means best balanced accuracy"),
        ("centroid_distance_over_within_rms", "Centroid distance / within RMS"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    width = 0.36
    xloc = np.arange(len(reps))
    colors = ["#6A51A3", "#238B45"]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for i, name in enumerate(names):
            vals = [results[name]["representations"][rep][metric] for rep in reps]
            ax.bar(xloc + (i - (len(names) - 1) / 2) * width, vals, width, label=name, color=colors[i])
        ax.set_xticks(xloc, rep_labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.7)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Can success and failure output actions form two clusters?  (*full rollout is length-confounded)", fontsize=16)
    path = output_dir / "output_action_cluster_metric_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for offset, (name, raw_path) in enumerate(args.dataset):
        result, rows = analyze_dataset(
            name=name,
            path=Path(raw_path),
            output_dir=args.output_dir,
            seed=args.seed + offset * 1000,
            max_scatter=args.max_scatter,
            max_silhouette=args.max_silhouette,
        )
        results[name] = result
        csv_rows.extend(rows)

    summary_plot = plot_metric_summary(results, args.output_dir)
    output = {
        "seed": args.seed,
        "definitions": {
            "raw_centroid_l2": "Euclidean distance between success/failure mean vectors in original action units.",
            "standardized_centroid_l2": "Centroid distance after standardizing every feature over all samples.",
            "centroid_distance_over_within_rms": "Standardized centroid distance divided by pooled within-class RMS radius.",
            "label_silhouette": "Silhouette using rollout success/failure as labels; near 0 means overlap, negative means many samples are closer to the other label.",
            "kmeans_adjusted_rand_index": "Agreement of unsupervised K-means clusters with success/failure, corrected for chance.",
            "kmeans_best_balanced_accuracy": "Best label permutation for the two K-means clusters; 0.5 is chance.",
            "leave_one_rollout_out_nearest_centroid_balanced_accuracy": "Held-out rollout prediction from class centroids fit to the other 19 rollouts.",
        },
        "datasets": results,
        "summary_plot": str(summary_plot.resolve()),
    }
    json_path = args.output_dir / "output_action_cluster_metrics.json"
    json_path.write_text(json.dumps(output, indent=2, allow_nan=True) + "\n")

    csv_path = args.output_dir / "output_action_success_failure_differences.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps({
        "metrics": str(json_path.resolve()),
        "differences": str(csv_path.resolve()),
        "summary_plot": str(summary_plot.resolve()),
        "datasets": {name: {"success": value["n_success"], "failure": value["n_failure"]} for name, value in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
