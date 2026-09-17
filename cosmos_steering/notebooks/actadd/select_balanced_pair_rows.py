#!/usr/bin/env python
"""Select an equal number of temporally spread clean/noisy rows per episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows-per-episode", type=int, default=4)
    args = parser.parse_args()
    if args.rows_per_episode < 1:
        raise ValueError("--rows-per-episode must be positive")

    pos = _load(args.pair_dir / "positive.npz")
    neg = _load(args.pair_dir / "negative.npz")
    if tuple(pos) != tuple(neg):
        raise ValueError("positive and negative schemas differ")
    if not np.array_equal(pos["episode_idx"], neg["episode_idx"]):
        raise ValueError("positive/negative episode indices are not paired")
    if not np.array_equal(pos["inference_idx"], neg["inference_idx"]):
        raise ValueError("positive/negative inference indices are not paired")

    selected: list[int] = []
    selection = []
    for episode in sorted(np.unique(pos["episode_idx"]).tolist()):
        indices = np.flatnonzero(pos["episode_idx"] == episode)
        count = min(args.rows_per_episode, indices.size)
        local = np.unique(np.rint(np.linspace(0, indices.size - 1, count)).astype(int))
        chosen = indices[local]
        selected.extend(chosen.tolist())
        selection.append({
            "episode": int(episode),
            "inference_indices": pos["inference_idx"][chosen].astype(int).tolist(),
        })
    selected_arr = np.asarray(selected, dtype=np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "positive.npz",
        **{key: value[selected_arr] for key, value in pos.items()},
    )
    np.savez_compressed(
        args.out_dir / "negative.npz",
        **{key: value[selected_arr] for key, value in neg.items()},
    )
    source_manifest = json.loads((args.pair_dir / "manifest.json").read_text())
    manifest = {
        "source_pair_dir": str(args.pair_dir.resolve()),
        "suite": source_manifest["suite"],
        "task_id": int(source_manifest["task_id"]),
        "prompt": source_manifest["prompt"],
        "noise": source_manifest["noise"],
        "selection_rule": "equal per episode, evenly spaced over inference index",
        "rows_per_episode_requested": int(args.rows_per_episode),
        "selection": selection,
        "total_rows": int(selected_arr.size),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"selected {selected_arr.size} paired rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
