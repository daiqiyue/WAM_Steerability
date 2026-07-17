#!/usr/bin/env python3
"""Write live markdown tables for the three LingBot LQR sweeps."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_IDX_RE = re.compile(r"^idx(\d+)_")


@dataclass(frozen=True)
class SweepSpec:
    name: str
    out_root: Path
    output_md: Path
    task_ids: Tuple[int, ...]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _combo_idx(name: str) -> int:
    match = _IDX_RE.match(name)
    return int(match.group(1)) if match else 10**9


def _fmt_rate(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return ""


def _fmt_num(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return ""


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _collect_rows(out_root: Path) -> List[Tuple[int, str, Dict[str, Any]]]:
    if not out_root.is_dir():
        return []
    rows: List[Tuple[int, str, Dict[str, Any]]] = []
    for combo_dir in sorted(out_root.iterdir()):
        if not combo_dir.is_dir() or not combo_dir.name.startswith("idx"):
            continue
        summary = _load_json(combo_dir / "summary.json")
        if summary is None:
            continue
        rows.append((_combo_idx(combo_dir.name), combo_dir.name, summary))
    rows.sort(key=lambda row: (row[0], row[1]))
    return rows


def _variant_summary(summary: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    perturbed = summary.get("perturbed") or {}
    if not perturbed:
        return "", {}
    name = sorted(perturbed.keys())[0]
    data = perturbed.get(name) or {}
    return name, data


def _task_rate(task_data: Dict[str, Any], task_id: int) -> str:
    task_payload = task_data.get(str(task_id))
    if task_payload is None:
        return ""
    if isinstance(task_payload, dict) and "succ_rate" in task_payload:
        return _fmt_rate(task_payload.get("succ_rate"))
    if isinstance(task_payload, dict) and "tasks" in task_payload:
        nested = task_payload.get("tasks") or {}
        return _fmt_rate((nested.get(str(task_id)) or {}).get("succ_rate"))
    return ""


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _render_table(spec: SweepSpec) -> str:
    rows = _collect_rows(spec.out_root)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# {spec.name} LQR Sweep Live Table",
        "",
        f"- updated: {now}",
        f"- out_root: `{spec.out_root}`",
        f"- completed_combos: {len(rows)}",
        f"- task_ids: `{list(spec.task_ids)}`",
        "",
    ]

    task_cols = [f"task{task_id}" for task_id in spec.task_ids]
    columns = [
        "idx",
        "lambda",
        "q_scale",
        "r_scale",
        "r_tau",
        "avg_success",
        "variant",
    ] + task_cols + ["combo_dir"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")

    for idx, combo_name, summary in rows:
        lqr = summary.get("lqr") or {}
        variant_name, variant_data = _variant_summary(summary)
        task_data = variant_data.get("tasks") or {}
        row = [
            str(idx),
            _fmt_num(lqr.get("lambda_scale")),
            _fmt_num(lqr.get("q_scale")),
            _fmt_num(lqr.get("r_scale")),
            _fmt_num(lqr.get("r_scale_tau")),
            _fmt_rate(summary.get("avg_succ_rate_over_variants")),
            variant_name,
        ]
        row.extend(_task_rate(task_data, task_id) for task_id in spec.task_ids)
        row.append(combo_name)
        lines.append("| " + " | ".join(_md_escape(cell) for cell in row) + " |")

    if not rows:
        lines.extend(["", "_No completed `summary.json` files found yet._"])
    lines.append("")
    return "\n".join(lines)


def _default_specs(repo_root: Path) -> List[SweepSpec]:
    table_root = repo_root / "outputs/lqr_sweep/live_tables"
    return [
        SweepSpec(
            name="Gaussian",
            out_root=repo_root
            / "outputs/lqr_sweep/time_noPar_gaussian_20260528_115621_tasks0_1_6/eval_seed99",
            output_md=table_root / "gaussian_sweep_table.md",
            task_ids=(0, 1, 6),
        ),
        SweepSpec(
            name="Init Position",
            out_root=repo_root
            / "outputs/lqr_sweep/time_noPar_init_pos_20260528_113245_tasks1_2_7/eval_seed99",
            output_md=table_root / "init_pos_sweep_table.md",
            task_ids=(1, 2, 7),
        ),
        SweepSpec(
            name="Camera Position",
            out_root=repo_root
            / "outputs/lqr_sweep/time_noPar_camera_20260528_134057_tasks0_2_4/eval_seed99",
            output_md=table_root / "camera_sweep_table.md",
            task_ids=(0, 2, 4),
        ),
    ]


def write_tables(specs: Iterable[SweepSpec]) -> None:
    for spec in specs:
        text = _render_table(spec)
        _write_atomic(spec.output_md, text)
        print(f"[live-table] wrote {spec.output_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write live markdown tables for the three LQR sweeps.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--watch", action="store_true", help="Refresh tables repeatedly.")
    parser.add_argument("--interval-sec", type=float, default=60.0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    specs = _default_specs(repo_root)
    while True:
        write_tables(specs)
        if not args.watch:
            break
        time.sleep(max(float(args.interval_sec), 1.0))


if __name__ == "__main__":
    main()
