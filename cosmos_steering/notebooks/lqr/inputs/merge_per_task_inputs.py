#!/usr/bin/env python
"""Merge per-task positive.npz / negative.npz / manifest.json shards from
collect_policy_inputs_gripper_xyz_perturbation_multitask.py (invoked once
per task in PARALLEL=1 mode) into a single combined dir matching the
schema you'd get from running all tasks in one process.

Usage:
    merge_per_task_inputs.py \\
        --in-dirs <dir1> <dir2> ... \\
        --out-dir <merged_dir>

Each <dirN> must contain positive.npz and negative.npz produced by the
multitask collector. NPZ keys are concatenated along axis 0; manifests
are merged with totals summed and rollouts / tasks unioned. The merged
output is byte-for-byte equivalent to what the serial collector would
have written (modulo rollout order, which is task-major when merged).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def _merge_npz(in_paths: list[Path], out_path: Path) -> int:
    arrs = [dict(np.load(p, allow_pickle=True)) for p in in_paths]
    keys = set(arrs[0].keys())
    for p, a in zip(in_paths[1:], arrs[1:]):
        if set(a.keys()) != keys:
            raise RuntimeError(f"schema mismatch at {p}: "
                               f"{sorted(a.keys())} vs {sorted(keys)}")
    merged = {k: np.concatenate([a[k] for a in arrs], axis=0) for k in keys}
    np.savez_compressed(out_path, **merged)
    n = int(next(iter(merged.values())).shape[0])
    sz_mb = out_path.stat().st_size / 1e6
    _log(f"wrote {out_path.name}  ({sz_mb:.1f} MB)  rows={n}")
    return n


def _merge_manifests(in_paths: list[Path], out_path: Path) -> None:
    mfs = [json.loads(p.read_text()) for p in in_paths]
    base = dict(mfs[0])
    totals_keys = ("rollouts", "rollout_successes", "rollout_failures",
                   "positive_rows", "negative_rows")
    totals = {k: 0 for k in totals_keys}
    seen_tids: set[int] = set()
    tasks: list[dict] = []
    task_ids: list[int] = []
    per_task_totals: list[dict] = []
    rollouts: list[dict] = []
    for mf in mfs:
        for k in totals_keys:
            totals[k] += int(mf.get("totals", {}).get(k, 0))
        for t in mf.get("tasks", []):
            tid = int(t["task_id"])
            if tid not in seen_tids:
                seen_tids.add(tid)
                tasks.append(t)
                task_ids.append(tid)
        per_task_totals.extend(mf.get("per_task_totals", []))
        rollouts.extend(mf.get("rollouts", []))
    base["task_ids"] = task_ids
    base["tasks"] = tasks
    base["per_task_totals"] = per_task_totals
    base["rollouts"] = rollouts
    base["totals"] = totals
    base["merged_from"] = [str(p.parent) for p in in_paths]
    out_path.write_text(json.dumps(base, indent=2))
    _log(f"wrote {out_path.name}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in-dirs", type=Path, nargs="+", required=True,
                    help="per-task shard dirs (each containing positive.npz "
                         "+ negative.npz + manifest.json)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="where to write the merged positive.npz / "
                         "negative.npz / manifest.json")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"merging {len(args.in_dirs)} shards -> {out_dir}")

    for d in args.in_dirs:
        for fn in ("positive.npz", "negative.npz"):
            if not (d / fn).exists():
                raise FileNotFoundError(f"{d}/{fn} missing")

    pos_paths = [d / "positive.npz" for d in args.in_dirs]
    neg_paths = [d / "negative.npz" for d in args.in_dirs]
    mf_paths = [d / "manifest.json" for d in args.in_dirs
                if (d / "manifest.json").exists()]

    n_pos = _merge_npz(pos_paths, out_dir / "positive.npz")
    n_neg = _merge_npz(neg_paths, out_dir / "negative.npz")
    if mf_paths:
        _merge_manifests(mf_paths, out_dir / "manifest.json")
    else:
        _log("no per-shard manifests found; skipping manifest merge")

    _log(f"merged totals: positive={n_pos} rows, negative={n_neg} rows")
    _log(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
