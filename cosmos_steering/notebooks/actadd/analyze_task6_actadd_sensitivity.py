#!/usr/bin/env python
"""Aggregate and plot the paired task-6 ActAdd sensitivity evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
DISPLAY = {
    "raw_a": "unsteered",
    "plus_v": "+ contrastive",
    "minus_v": "− contrastive",
    "random": "random (norm matched)",
}
COLORS = {
    "raw_a": "#4C78A8",
    "plus_v": "#F58518",
    "minus_v": "#54A24B",
    "random": "#B279A2",
}


def percentile_band(values: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    width = max(len(value) for value in values)
    padded = np.full((len(values), width), np.nan, dtype=np.float64)
    for row, value in enumerate(values):
        padded[row, :len(value)] = value
    return (
        np.nanmedian(padded, axis=0),
        np.nanpercentile(padded, 25, axis=0),
        np.nanpercentile(padded, 75, axis=0),
    )


def bootstrap_rate_ci(binary: np.ndarray, *, seed: int = 20260917,
                      n_boot: int = 20000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if binary.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = rng.choice(binary, size=(n_boot, binary.size), replace=True).mean(axis=1)
    return float(binary.mean()), float(np.quantile(draws, 0.025)), \
        float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, nargs="+", required=True)
    args = parser.parse_args()

    records = []
    arrays = []
    for episode in args.expected_episodes:
        json_path = args.input_dir / f"episode_{episode:02d}.json"
        npz_path = args.input_dir / f"episode_{episode:02d}.npz"
        if not json_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(f"missing completed episode {episode}: {json_path} / {npz_path}")
        records.append(json.loads(json_path.read_text()))
        arrays.append(np.load(npz_path))

    probe_alphas = np.asarray(arrays[0]["probe_alphas"], dtype=np.float64)
    if not all(np.array_equal(np.asarray(data["probe_alphas"]), probe_alphas)
               for data in arrays):
        raise ValueError("probe alpha grids differ across episodes")
    zero_idx = int(np.flatnonzero(np.isclose(probe_alphas, 0.0))[0])
    probe_sets = [np.asarray(data["probe_actions"], dtype=np.float64)
                  for data in arrays if len(data["probe_actions"])]
    if not probe_sets:
        raise RuntimeError("no local fixed-observation probes were recorded")
    probes = np.concatenate(probe_sets, axis=0)  # [observations, alpha, 16, 7]
    delta = probes - probes[:, zero_idx:zero_idx + 1]
    probe_rms = np.sqrt(np.mean(delta ** 2, axis=(2, 3)))
    probe_action_dim_mean = delta.mean(axis=2).mean(axis=0)  # [alpha, action dim]

    med = np.median(probe_rms, axis=0)
    q25 = np.percentile(probe_rms, 25, axis=0)
    q75 = np.percentile(probe_rms, 75, axis=0)
    nonzero = ~np.isclose(probe_alphas, 0.0)
    rates = probe_rms[:, nonzero] / np.abs(probe_alphas[nonzero])[None, :]
    rate_med = np.median(rates, axis=0)
    rate_q25 = np.percentile(rates, 25, axis=0)
    rate_q75 = np.percentile(rates, 75, axis=0)

    divergence = {}
    for condition in ("plus_v", "minus_v", "random"):
        series = []
        for record, data in zip(records, arrays):
            prefix = int(record["prefix_env_steps"])
            full = np.asarray(data[f"{condition}__eef_divergence_vs_raw_m"],
                              dtype=np.float64)
            series.append(full[prefix:] * 100.0)
        divergence[condition] = percentile_band(series)

    success_stats = {}
    for condition in DISPLAY:
        values = np.asarray([
            bool(record["conditions"][condition]["success_within_window"])
            for record in records
        ], dtype=np.float64)
        success_stats[condition] = bootstrap_rate_ci(values)

    raw_repeat_action_max = max(
        float(record["comparisons_vs_raw_a"]["raw_b"]["action_max_abs"])
        for record in records
    )
    raw_repeat_eef_max_m = max(
        float(record["comparisons_vs_raw_a"]["raw_b"]["eef_divergence_max_m"])
        for record in records
    )
    anchor_replay_max_m = max(
        max(float(condition["anchor_replay_max_abs_error_m"])
            for condition in record["conditions"].values())
        for record in records
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(probe_alphas, med, marker="o", color="#4C78A8")
    ax.fill_between(probe_alphas, q25, q75, color="#4C78A8", alpha=0.2)
    ax.axvline(0.0, color="0.35", linewidth=1)
    ax.set_title("Same-observation action-chunk change")
    ax.set_xlabel("ActAdd α")
    ax.set_ylabel("RMS Δ action (16×7 LIBERO units)")

    ax = axes[0, 1]
    abs_alpha = np.abs(probe_alphas[nonzero])
    order = np.argsort(abs_alpha)
    ax.plot(abs_alpha[order], rate_med[order], marker="o", color="#F58518")
    ax.fill_between(abs_alpha[order], rate_q25[order], rate_q75[order],
                    color="#F58518", alpha=0.2)
    ax.set_xscale("log")
    ax.set_title("Finite-α sensitivity")
    ax.set_xlabel("|α|")
    ax.set_ylabel("RMS Δ action / |α|")

    ax = axes[1, 0]
    for condition in ("plus_v", "minus_v", "random"):
        median, low, high = divergence[condition]
        x = np.arange(len(median))
        ax.plot(x, median, label=DISPLAY[condition], color=COLORS[condition])
        ax.fill_between(x, low, high, color=COLORS[condition], alpha=0.18)
    ax.set_title("Closed-loop EEF divergence from unsteered branch")
    ax.set_xlabel("Environment steps after common prefix")
    ax.set_ylabel("EEF distance (cm)")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    conditions = list(DISPLAY)
    estimates = np.asarray([success_stats[name][0] for name in conditions])
    lows = np.asarray([success_stats[name][1] for name in conditions])
    highs = np.asarray([success_stats[name][2] for name in conditions])
    x = np.arange(len(conditions))
    ax.bar(x, estimates, color=[COLORS[name] for name in conditions])
    ax.errorbar(x, estimates, yerr=[estimates - lows, highs - estimates],
                fmt="none", color="0.15", capsize=4)
    ax.set_xticks(x, [DISPLAY[name] for name in conditions], rotation=18,
                  ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Task success within the fixed branch window")
    ax.set_ylabel("Fraction of held-out episodes")
    fig.suptitle(
        "Cosmos Policy activation-space ActAdd sensitivity — LIBERO-10 task 6, Gaussian noise",
        fontsize=15,
    )
    overview_path = args.input_dir / "sensitivity_overview.png"
    fig.savefig(overview_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    vmax = float(np.max(np.abs(probe_action_dim_mean)))
    vmax = max(vmax, 1e-9)
    image = ax.imshow(probe_action_dim_mean, aspect="auto", cmap="coolwarm",
                      vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(ACTION_NAMES)), ACTION_NAMES, rotation=30,
                  ha="right")
    ax.set_yticks(np.arange(len(probe_alphas)),
                  [f"{alpha:g}" for alpha in probe_alphas])
    ax.set_xlabel("LIBERO action dimension")
    ax.set_ylabel("ActAdd α")
    ax.set_title("Mean signed action change on identical observations")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Mean Δ action over observations and 16-token chunk")
    heatmap_path = args.input_dir / "action_dimension_heatmap.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    summary = {
        "n_episodes": len(records),
        "episodes": [int(record["episode"]) for record in records],
        "train_eval_split": {
            "eval_episodes": [int(record["episode"]) for record in records],
            "contrastive_vector_path": records[0]["v_path"],
        },
        "task_id": int(records[0]["task_id"]),
        "noise": records[0]["noise"],
        "anchor_chunk": int(records[0]["anchor_chunk"]),
        "branch_horizon_chunks": int(records[0]["branch_horizon_chunks"]),
        "rollout_alpha": float(records[0]["rollout_alpha"]),
        "probe_alphas": probe_alphas.tolist(),
        "n_fixed_observation_probes": int(probes.shape[0]),
        "probe_action_rms_median": med.tolist(),
        "probe_action_rms_q25": q25.tolist(),
        "probe_action_rms_q75": q75.tolist(),
        "success_within_window": {
            name: {"rate": success_stats[name][0],
                   "bootstrap_95_ci": [success_stats[name][1],
                                        success_stats[name][2]]}
            for name in conditions
        },
        "determinism_controls": {
            "raw_repeat_max_abs_action": raw_repeat_action_max,
            "raw_repeat_max_eef_divergence_m": raw_repeat_eef_max_m,
            "max_anchor_replay_error_m": anchor_replay_max_m,
        },
        "plots": {
            "overview": str(overview_path),
            "action_dimensions": str(heatmap_path),
        },
    }
    (args.input_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Cosmos task-6 ActAdd sensitivity",
        "",
        f"- Held-out eval episodes: {len(records)} ({summary['episodes']})",
        f"- Fixed-observation probes: {probes.shape[0]}",
        f"- Anchor / branch horizon: chunk {summary['anchor_chunk']} / "
        f"{summary['branch_horizon_chunks']} chunks",
        f"- Rollout alpha: {summary['rollout_alpha']}",
        f"- Raw-repeat max action error: {raw_repeat_action_max:.3e}",
        f"- Raw-repeat max EEF divergence: {raw_repeat_eef_max_m:.3e} m",
        f"- Max prefix replay error: {anchor_replay_max_m:.3e} m",
        "",
        "| condition | success in fixed window | bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for name in conditions:
        estimate, low, high = success_stats[name]
        lines.append(
            f"| {DISPLAY[name]} | {estimate:.1%} | [{low:.1%}, {high:.1%}] |"
        )
    lines.extend([
        "",
        f"- Overview: `{overview_path}`",
        f"- Action dimensions: `{heatmap_path}`",
    ])
    (args.input_dir / "summary.md").write_text("\n".join(lines) + "\n")
    for data in arrays:
        data.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
