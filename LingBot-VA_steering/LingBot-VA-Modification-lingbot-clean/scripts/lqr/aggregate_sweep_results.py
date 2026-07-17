#!/usr/bin/env python3
"""Aggregate per-combo summary.json files from an LQR parameter sweep into a markdown report."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_IDX_RE = re.compile(r"^idx(\d+)_")


def _parse_combo_dir(name: str) -> Optional[int]:
    m = _IDX_RE.match(name)
    return int(m.group(1)) if m else None


def _load_summary(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _collect_rows(out_root: Path) -> List[Tuple[int, str, Dict[str, Any]]]:
    rows: List[Tuple[int, str, Dict[str, Any]]] = []
    for combo_dir in sorted(out_root.iterdir()):
        if not combo_dir.is_dir():
            continue
        summary_fp = combo_dir / "summary.json"
        if not summary_fp.is_file():
            continue
        summary = _load_summary(summary_fp)
        if summary is None:
            continue
        idx = _parse_combo_dir(combo_dir.name)
        sort_key = idx if idx is not None else 10**9
        rows.append((sort_key, combo_dir.name, summary))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def _variant_names(rows: List[Tuple[int, str, Dict[str, Any]]]) -> List[str]:
    names: List[str] = []
    seen = set()
    for _, _, summary in rows:
        perturbed = summary.get("perturbed") or {}
        for name in sorted(perturbed.keys()):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_rate(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _format_report(
    out_root: Path,
    rows: List[Tuple[int, str, Dict[str, Any]]],
    variant_names: List[str],
) -> str:
    lines: List[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append("# LQR Parameter Sweep Success Rates")
    lines.append("")
    lines.append(f"- generated: {now}")
    lines.append(f"- out_root: `{out_root.resolve()}`")
    lines.append(f"- combos_with_summary: {len(rows)}")
    if rows:
        first_summary = rows[0][2]
        lines.append(f"- perturb_spec: `{first_summary.get('perturb_spec', '')}`")
        lines.append(f"- libero_benchmark: `{first_summary.get('libero_benchmark', '')}`")
        task_ids = first_summary.get("task_ids")
        task_range = first_summary.get("task_range")
        if task_ids:
            lines.append(f"- task_ids: `{task_ids}`")
            lines.append("- task_selector: `explicit task_ids`")
        else:
            lines.append(f"- task_range: `{task_range}`")
            lines.append("- task_selector: `contiguous task_range`")
        lines.append(f"- trials_per_variant: `{first_summary.get('num_episodes', '')}`")
    lines.append("")

    base_cols = [
        "idx",
        "combo_dir",
        "lambda",
        "q_scale",
        "r_scale",
        "qf_scale",
        "avg_success_rate",
    ]
    header_cols = base_cols + variant_names
    lines.append("| " + " | ".join(_md_escape(c) for c in header_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cols) + " |")

    for idx, combo_name, summary in rows:
        lqr = summary.get("lqr") or {}
        perturbed = summary.get("perturbed") or {}
        row_vals = [
            str(idx) if idx is not None else "",
            combo_name,
            str(lqr.get("lambda_scale", "")),
            str(lqr.get("q_scale", "")),
            str(lqr.get("r_scale", "")),
            str(lqr.get("qf_scale", "")),
            _format_rate(summary.get("avg_succ_rate_over_variants")),
        ]
        for vname in variant_names:
            vdata = perturbed.get(vname) or {}
            row_vals.append(_format_rate(vdata.get("avg_succ_rate")))
        lines.append("| " + " | ".join(_md_escape(v) for v in row_vals) + " |")

    if not rows:
        lines.append("")
        lines.append("_No `summary.json` files found yet._")

    lines.append("")
    lines.append("Per-variant columns are `avg_succ_rate` over the selected tasks.")
    lines.append("`avg_success_rate` is the mean across all non-nominal perturb variants.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LQR sweep summary.json into a markdown report.")
    parser.add_argument("--out-root", type=str, required=True, help="Sweep output root (contains idx*_... dirs).")
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Report path (default: <out-root>/sweep_success_rates.md).",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=None,
        help="Deprecated alias for --output-md.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_dir():
        raise SystemExit(f"[error] out-root is not a directory: {out_root}")

    out_md = Path(args.output_md or args.output_txt) if (args.output_md or args.output_txt) else out_root / "sweep_success_rates.md"
    rows = _collect_rows(out_root)
    variant_names = _variant_names(rows)
    report = _format_report(out_root, rows, variant_names)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")
    print(f"[aggregate] wrote {out_md} ({len(rows)} combos)")


if __name__ == "__main__":
    main()
