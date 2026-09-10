#!/usr/bin/env python
"""Plot baseline failure/success and post-steering DiT4DiT activations together.

The PCA plane follows the original interpretability script's idea: it is fit to
balanced success-minus-failure activation differences.  The LQR contrastive
direction is reconstructed from the saved SVD basis and drawn in that same
plane.  A JSON report quantifies whether the steered centroid moved toward the
baseline-success centroid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


COLORS = {
    "baseline_failure": "#ef6c00",
    "baseline_success": "#1976d2",
    "steered_failure": "#8e24aa",
    "steered_success": "#2e7d32",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-dir", type=Path, required=True)
    p.add_argument("--svd-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--step", type=int, default=0,
                   help="denoising step to display (default: 0)")
    p.add_argument("--blocks", default="auto",
                   help="comma-separated block ids, or auto=first/best/last")
    p.add_argument("--max-points", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_traces(rollout_dir: Path, step: int) -> dict[str, dict[int, np.ndarray]]:
    groups: dict[str, dict[int, list[np.ndarray]]] = {}
    paths = sorted((rollout_dir / "activations").glob("ep*__*.pt"))
    if not paths:
        raise FileNotFoundError(f"no activation traces under {rollout_dir / 'activations'}")
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        condition = str(payload["condition"])
        success = bool(payload["success"])
        label = f"{condition}_{'success' if success else 'failure'}"
        blocks = payload["block"].numpy().astype(np.int64)
        steps = payload["step"].numpy().astype(np.int64)
        acts = payload["activation"].float().numpy()
        for block in np.unique(blocks[steps == step]):
            mask = (steps == step) & (blocks == block)
            groups.setdefault(label, {}).setdefault(int(block), []).append(acts[mask])
    return {
        label: {block: np.concatenate(chunks, axis=0) for block, chunks in by_block.items()}
        for label, by_block in groups.items()
    }


def balanced_pca(success: np.ndarray, failure: np.ndarray, rng: np.random.Generator,
                 n_components: int = 3):
    n = min(len(success), len(failure))
    if n < 3:
        raise ValueError(f"need >=3 success and failure points, got {len(success)} and {len(failure)}")
    si = rng.choice(len(success), n, replace=False)
    fi = rng.choice(len(failure), n, replace=False)
    diffs = success[si] - failure[fi]
    mean = failure.mean(axis=0)
    centered = diffs - diffs.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components].T
    return mean, components


def project(points: np.ndarray, origin: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (points - origin) @ components


def load_contrastive_direction(svd_dir: Path, block: int, step: int) -> np.ndarray:
    cfg = json.loads((svd_dir / "config.json").read_text())
    summary = torch.load(svd_dir / "svd_summary.pt", map_location="cpu", weights_only=False)
    selected = [int(x) for x in cfg["selected_timesteps"]]
    if step not in selected:
        raise ValueError(f"step {step} is not in SVD selected_timesteps={selected}")
    t_pos = selected.index(step)
    partitions = [tuple(x) for x in cfg["partitions"]]
    part_idx = int(summary["layer_to_part"][block])
    l0, l1 = partitions[part_idx]
    k = int(cfg["k_target"])
    v_path = svd_dir / f"V_part{part_idx}_layers{l0}-{l1}_t{step}_k{k}.pt"
    V = torch.load(v_path, map_location="cpu", weights_only=False)["V"].float()
    c = summary["c_means"][block, t_pos].float()
    full = (V @ c).reshape(int(cfg["T_p_denoise"]), -1)
    # Trace activations are mean-pooled over the action horizon, so pool the
    # reconstructed contrastive direction identically.
    return full.mean(dim=0).numpy()


def subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return x
    return x[rng.choice(len(x), n, replace=False)]


def separation_score(success: np.ndarray, failure: np.ndarray) -> float:
    delta = np.linalg.norm(success.mean(0) - failure.mean(0))
    spread = np.sqrt(success.var(0).mean() + failure.var(0).mean()) + 1e-12
    return float(delta / spread)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or (args.rollout_dir / "interpretability")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    groups = load_traces(args.rollout_dir, args.step)
    if "baseline_success" not in groups or "baseline_failure" not in groups:
        raise ValueError(
            "the rollout must contain both successful and failed unsteered, perturbed baseline episodes"
        )
    common_blocks = sorted(set(groups["baseline_success"]) & set(groups["baseline_failure"]))
    if args.blocks == "auto":
        scores = {
            b: separation_score(groups["baseline_success"][b], groups["baseline_failure"][b])
            for b in common_blocks
        }
        interior = [b for b in common_blocks if b not in (common_blocks[0], common_blocks[-1])]
        best = max(interior, key=scores.get) if interior else common_blocks[0]
        blocks = list(dict.fromkeys([common_blocks[0], best, common_blocks[-1]]))
    else:
        blocks = [int(x) for x in args.blocks.split(",")]

    fig, axes = plt.subplots(1, len(blocks), figsize=(6.2 * len(blocks), 5.4), squeeze=False)
    plot3d: dict[int, dict[str, object]] = {}
    metrics = {"step": args.step, "blocks": {}, "definitions": {
        "toward_success_ratio": "same ratio in the displayed two-dimensional contrastive PCA plane",
        "toward_success_ratio_full": "1 - d(steered centroid, success centroid) / d(failure centroid, success centroid), evaluated in the full mean-pooled activation space",
        "positive_means": "steered centroid is closer to baseline success than baseline failure is",
        "steered_success_side_fraction": "fraction of steered points on the success side of a regularized LDA boundary fit only to baseline outcomes",
    }}

    for ax, block in zip(axes[0], blocks):
        success = groups["baseline_success"][block]
        failure = groups["baseline_failure"][block]
        origin, components_3d = balanced_pca(success, failure, rng)
        components = components_3d[:, :2]
        projected: dict[str, np.ndarray] = {}
        for label, by_block in groups.items():
            if block not in by_block:
                continue
            points = subsample(by_block[block], args.max_points, rng)
            projected[label] = project(points, origin, components)
            marker = "x" if label.startswith("steered") else "o"
            ax.scatter(projected[label][:, 0], projected[label][:, 1], s=13,
                       alpha=0.48, marker=marker, color=COLORS.get(label, "gray"),
                       label=label.replace("_", " "), rasterized=True)

        direction = load_contrastive_direction(args.svd_dir, block, args.step)
        direction_2d = direction @ components
        success_2d = project(success, origin, components)
        failure_2d = project(failure, origin, components)
        fail_center = failure_2d.mean(0)
        success_center = success_2d.mean(0)
        steered_all = np.concatenate(
            [groups[label][block] for label in ("steered_success", "steered_failure")
             if label in groups and block in groups[label]], axis=0
        )
        steered_2d = project(steered_all, origin, components)
        steer_center = steered_2d.mean(0)

        # Linear-discriminant success side, mirroring the separating-hyperplane
        # view used by the original interpretability plots.
        pooled = np.vstack([success_2d - success_center, failure_2d - fail_center])
        covariance = (pooled.T @ pooled) / max(len(pooled) - 2, 1)
        covariance += 1e-5 * np.eye(2)
        boundary_w = np.linalg.solve(covariance, success_center - fail_center)
        boundary_b = -0.5 * float((success_center + fail_center) @ boundary_w)
        xlim = np.array([
            min(failure_2d[:, 0].min(), success_2d[:, 0].min(), steered_2d[:, 0].min()),
            max(failure_2d[:, 0].max(), success_2d[:, 0].max(), steered_2d[:, 0].max()),
        ])
        if abs(boundary_w[1]) > 1e-10:
            ylim = -(boundary_w[0] * xlim + boundary_b) / boundary_w[1]
            ax.plot(xlim, ylim, "--", color="0.35", linewidth=1.0,
                    label="baseline outcome boundary")
        else:
            ax.axvline(-boundary_b / boundary_w[0], linestyle="--", color="0.35",
                       linewidth=1.0, label="baseline outcome boundary")
        norm = np.linalg.norm(direction_2d)
        plot_scale = 0.75 * np.linalg.norm(success_center - fail_center) / max(norm, 1e-12)
        ax.quiver(fail_center[0], fail_center[1], direction_2d[0] * plot_scale,
                  direction_2d[1] * plot_scale, angles="xy", scale_units="xy", scale=1,
                  width=0.012, color="black", label="SVD contrastive direction")
        ax.scatter(*success_center, s=150, marker="*", color=COLORS["baseline_success"],
                   edgecolor="black", linewidth=0.6)
        ax.scatter(*steer_center, s=115, marker="P", color="#1b5e20",
                   edgecolor="black", linewidth=0.6)
        d0 = float(np.linalg.norm(success_center - fail_center))
        d1 = float(np.linalg.norm(success_center - steer_center))
        shift = steer_center - fail_center
        cosine = float(np.dot(shift, direction_2d) /
                       ((np.linalg.norm(shift) * np.linalg.norm(direction_2d)) + 1e-12))
        ratio = float(1.0 - d1 / max(d0, 1e-12))
        success_center_full = success.mean(0)
        fail_center_full = failure.mean(0)
        steer_center_full = steered_all.mean(0)
        d0_full = float(np.linalg.norm(success_center_full - fail_center_full))
        d1_full = float(np.linalg.norm(success_center_full - steer_center_full))
        shift_full = steer_center_full - fail_center_full
        ratio_full = float(1.0 - d1_full / max(d0_full, 1e-12))
        cosine_full = float(np.dot(shift_full, direction) /
                            ((np.linalg.norm(shift_full) * np.linalg.norm(direction)) + 1e-12))
        success_3d = project(success, origin, components_3d)
        failure_3d = project(failure, origin, components_3d)
        steered_3d = project(steered_all, origin, components_3d)
        direction_3d = direction @ components_3d
        fail_center_3d = failure_3d.mean(0)
        success_center_3d = success_3d.mean(0)
        steer_center_3d = steered_3d.mean(0)
        d0_3d = float(np.linalg.norm(success_center_3d - fail_center_3d))
        d1_3d = float(np.linalg.norm(success_center_3d - steer_center_3d))
        shift_3d = steer_center_3d - fail_center_3d
        ratio_3d = float(1.0 - d1_3d / max(d0_3d, 1e-12))
        cosine_3d = float(np.dot(shift_3d, direction_3d) /
                          ((np.linalg.norm(shift_3d) * np.linalg.norm(direction_3d)) + 1e-12))
        steered_success_side = float(np.mean(steered_2d @ boundary_w + boundary_b >= 0))
        failure_success_side = float(np.mean(failure_2d @ boundary_w + boundary_b >= 0))
        success_success_side = float(np.mean(success_2d @ boundary_w + boundary_b >= 0))
        metrics["blocks"][str(block)] = {
            "baseline_success_points": int(len(success)),
            "baseline_failure_points": int(len(failure)),
            "steered_points": int(len(steered_all)),
            "baseline_failure_to_success_distance_2d": d0,
            "steered_to_success_distance_2d": d1,
            "toward_success_ratio": ratio,
            "steering_shift_contrastive_cosine_2d": cosine,
            "baseline_failure_to_success_distance_3d": d0_3d,
            "steered_to_success_distance_3d": d1_3d,
            "toward_success_ratio_3d": ratio_3d,
            "steering_shift_contrastive_cosine_3d": cosine_3d,
            "baseline_failure_to_success_distance_full": d0_full,
            "steered_to_success_distance_full": d1_full,
            "toward_success_ratio_full": ratio_full,
            "steering_shift_contrastive_cosine_full": cosine_full,
            "steered_success_side_fraction": steered_success_side,
            "baseline_success_success_side_fraction": success_success_side,
            "baseline_failure_success_side_fraction": failure_success_side,
        }
        ax.set_title(
            f"Block {block}\n2D toward success={ratio:+.2%}, "
            f"success-side={steered_success_side:.1%}, cos={cosine:+.2f}"
        )
        ax.set_xlabel("contrastive PC1")
        ax.set_ylabel("contrastive PC2")
        ax.grid(alpha=0.2, linestyle=":")
        plot3d[block] = {
            "origin": origin, "components": components_3d,
            "direction": direction, "groups": groups,
        }

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle("DiT4DiT perturbation: baseline outcomes and post-LQR activations")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(out_dir / "steering_activation_overlap.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "steering_activation_overlap.pdf", bbox_inches="tight")
    plt.close(fig)

    # A matching three-dimensional view. It uses the first three components of
    # the exact same balanced contrastive PCA fit as the 2D panels above.
    fig3 = plt.figure(figsize=(7.2 * len(blocks), 6.3))
    for panel, block in enumerate(blocks, start=1):
        ax3 = fig3.add_subplot(1, len(blocks), panel, projection="3d")
        ctx = plot3d[block]
        origin = ctx["origin"]
        components_3d = ctx["components"]
        for label, by_block in groups.items():
            if block not in by_block:
                continue
            points = subsample(by_block[block], args.max_points, rng)
            p3 = project(points, origin, components_3d)
            marker = "x" if label.startswith("steered") else "o"
            ax3.scatter(p3[:, 0], p3[:, 1], p3[:, 2], s=12, alpha=0.45,
                        marker=marker, color=COLORS.get(label, "gray"),
                        label=label.replace("_", " "), rasterized=True)
        success_3d = project(groups["baseline_success"][block], origin, components_3d)
        failure_3d = project(groups["baseline_failure"][block], origin, components_3d)
        steered_all = np.concatenate(
            [groups[label][block] for label in ("steered_success", "steered_failure")
             if label in groups and block in groups[label]], axis=0
        )
        steered_3d = project(steered_all, origin, components_3d)
        fc, sc, tc = failure_3d.mean(0), success_3d.mean(0), steered_3d.mean(0)
        d3 = ctx["direction"] @ components_3d
        arrow_scale = 0.75 * np.linalg.norm(sc - fc) / max(np.linalg.norm(d3), 1e-12)
        ax3.quiver(*fc, *(d3 * arrow_scale), color="black", linewidth=2.0,
                   arrow_length_ratio=0.12, label="SVD contrastive direction")
        ax3.scatter(*sc, s=170, marker="*", color=COLORS["baseline_success"],
                    edgecolor="black", linewidth=0.6)
        ax3.scatter(*tc, s=125, marker="P", color="#1b5e20",
                    edgecolor="black", linewidth=0.6)
        bm = metrics["blocks"][str(block)]
        ax3.set_title(f"Block {block}\n3D toward success={bm['toward_success_ratio_3d']:+.2%}, "
                      f"cos={bm['steering_shift_contrastive_cosine_3d']:+.2f}")
        ax3.set_xlabel("contrastive PC1")
        ax3.set_ylabel("contrastive PC2")
        ax3.set_zlabel("contrastive PC3")
        ax3.grid(alpha=0.2, linestyle=":")
    handles3, labels3 = fig3.axes[0].get_legend_handles_labels()
    unique3 = dict(zip(labels3, handles3))
    fig3.legend(unique3.values(), unique3.keys(), loc="lower center", ncol=3, frameon=False)
    fig3.suptitle("DiT4DiT perturbation: 3D activation projection after LQR steering")
    fig3.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig3.savefig(out_dir / "steering_activation_overlap_3d.png", dpi=180, bbox_inches="tight")
    fig3.savefig(out_dir / "steering_activation_overlap_3d.pdf", bbox_inches="tight")
    plt.close(fig3)
    (out_dir / "steering_activation_overlap_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
