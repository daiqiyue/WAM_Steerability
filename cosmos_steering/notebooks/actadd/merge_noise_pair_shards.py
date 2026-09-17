#!/usr/bin/env python
"""Merge task-specific Gaussian clean/noisy pair shards deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _merge_kind(shards: list[Path], filename: str, out_path: Path) -> int:
    loaded = [_load_npz(shard / filename) for shard in shards]
    keys = tuple(loaded[0])
    for shard, arrays in zip(shards, loaded):
        if tuple(arrays) != keys:
            raise ValueError(f"schema mismatch in {shard / filename}")
    merged = {key: np.concatenate([arrays[key] for arrays in loaded], axis=0)
              for key in keys}
    order = np.lexsort((merged["inference_idx"], merged["episode_idx"]))
    merged = {key: value[order] for key, value in merged.items()}
    np.savez_compressed(out_path, **merged)
    return int(order.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, nargs="+", required=True)
    args = parser.parse_args()

    shards = sorted(
        path for path in args.shard_root.glob("episode_*")
        if (path / "positive.npz").is_file() and (path / "negative.npz").is_file()
    )
    if not shards:
        raise FileNotFoundError(f"no complete shards under {args.shard_root}")

    found_episodes: list[int] = []
    manifests = []
    for shard in shards:
        manifest_path = shard / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        manifests.append(manifest)
        found_episodes.extend(int(x) for x in manifest["episode_ids_run"])

    expected = sorted(set(args.expected_episodes))
    if sorted(found_episodes) != expected:
        raise RuntimeError(
            f"episode coverage mismatch: found={sorted(found_episodes)}, expected={expected}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_pos = _merge_kind(shards, "positive.npz", args.out_dir / "positive.npz")
    n_neg = _merge_kind(shards, "negative.npz", args.out_dir / "negative.npz")
    if n_pos != n_neg:
        raise RuntimeError(f"positive/negative row mismatch: {n_pos} vs {n_neg}")

    first = manifests[0]
    rollouts = sorted(
        [record for manifest in manifests for record in manifest.get("rollouts", [])],
        key=lambda record: int(record["episode"]),
    )
    merged_manifest = {
        "suite": first["suite"],
        "task_id": int(first["task_id"]),
        "prompt": first["prompt"],
        "resolution": int(first["resolution"]),
        "episode_ids_run": expected,
        "noise": first["noise"],
        "pairing": first["pairing"],
        "rollouts": rollouts,
        "totals": {
            "rollouts": len(rollouts),
            "successes": sum(bool(record["success"]) for record in rollouts),
            "paired_rows": n_pos,
        },
        "merged_from": [str(path.resolve()) for path in shards],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(merged_manifest, indent=2) + "\n"
    )
    print(f"merged {len(shards)} shards, {n_pos} paired rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
