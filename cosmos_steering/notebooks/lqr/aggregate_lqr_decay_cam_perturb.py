#!/usr/bin/env python
"""Aggregate LQR-decay cam-perturb sweep results into a markdown summary.

Scans <rollouts-root> for directories produced by
run_lqr_decay_cosmos_policy_cam_perturb.py (matching the _compute_config_tag
format), reads each one's results.json, counts steered-rollout successes,
and writes a markdown table plus marginal-success breakdowns.

Usage:
    python aggregate_lqr_decay_cam_perturb.py \
        --rollouts-root notebooks/lqr/rollouts/cam_perturb \
        --summary-md   notebooks/lqr/camera_position_sweep.md \
        --cam-mode random --cam-seed 99 --run-tag cam_perturb_decay
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Match the deterministic prefix produced by _compute_config_tag(); the
# remainder (after "...__cam{mode}_seed{N}__") is `{slug(prompt)}[__{run_tag}]`,
# which we split below using the known run_tag value.
_PREFIX_RE = re.compile(
    r"^(?P<suite>libero_\d+)__task(?P<task>\d+)__lqr_decay_camperturb__"
    r"lam(?P<lam>[\d\.]+)_q(?P<q>[\deE\.\+\-]+)_rinit(?P<ri>[\deE\.\+\-]+)"
    r"_rfin(?P<rf>[\deE\.\+\-]+)_tau(?P<tau>[\deE\.\+\-]+)_qf(?P<qf>[\deE\.\+\-]+)__"
    r"cam(?P<mode>[a-z]+)_seed(?P<seed>\d+)__(?P<rest>.+)$"
)


def parse_tag(name: str, run_tag: str | None):
    """Return a dict of parsed fields, or None if `name` doesn't match."""
    m = _PREFIX_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    rest = d["rest"]
    if run_tag:
        suffix = f"__{run_tag}"
        if rest.endswith(suffix):
            prompt = rest[: -len(suffix)]
            tag = run_tag
        else:
            # name doesn't carry the requested run_tag; treat as no tag and
            # skip in the caller.
            prompt, tag = rest, None
    else:
        prompt, tag = rest, None
    return {
        "lambda": float(d["lam"]),
        "q": float(d["q"]),
        "rinit": float(d["ri"]),
        "rfin": float(d["rf"]),
        "tau": float(d["tau"]),
        "qf": float(d["qf"]),
        "cam_mode": d["mode"],
        "cam_seed": int(d["seed"]),
        "prompt_slug": prompt,
        "run_tag": tag,
        "name": name,
    }


def _fmt_num(x: float) -> str:
    return f"{x:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-root", type=Path, required=True)
    ap.add_argument("--summary-md", type=Path, required=True)
    ap.add_argument("--cam-mode", default=None, help="filter by cam_mode (match | random | off)")
    ap.add_argument("--cam-seed", type=int, default=None, help="filter by cam_base_seed")
    ap.add_argument("--run-tag", default=None, help="filter by run_tag")
    args = ap.parse_args()

    if not args.rollouts_root.exists():
        raise SystemExit(f"rollouts root does not exist: {args.rollouts_root}")

    rows = []
    skipped_no_results = []
    skipped_parse = []
    for cfg_dir in sorted(args.rollouts_root.iterdir()):
        if not cfg_dir.is_dir():
            continue
        meta = parse_tag(cfg_dir.name, run_tag=args.run_tag)
        if meta is None:
            skipped_parse.append(cfg_dir.name)
            continue
        if args.cam_mode and meta["cam_mode"] != args.cam_mode:
            continue
        if args.cam_seed is not None and meta["cam_seed"] != args.cam_seed:
            continue
        if args.run_tag and meta["run_tag"] != args.run_tag:
            continue
        results_json = cfg_dir / "results.json"
        if not results_json.exists():
            skipped_no_results.append(cfg_dir.name)
            continue
        try:
            results = json.loads(results_json.read_text())
        except json.JSONDecodeError as e:
            print(f"WARN: bad results.json in {cfg_dir.name}: {e}")
            continue
        n_total = len(results)
        n_success = sum(1 for r in results if r.get("success"))
        steps = [r["env_steps"] for r in results if r.get("success")]
        avg_steps_success = (sum(steps) / len(steps)) if steps else None

        # Detect a baseline (unsteered) rollout next to the steered results.
        # baseline depends only on the camera perturbation, so it's the same
        # reference across every steered combo that shares (cam_mode, cam_seed).
        baseline_json = cfg_dir / "baseline" / "results.json"
        baseline_total = baseline_success = None
        baseline_avg_steps = None
        if baseline_json.exists():
            try:
                bres = json.loads(baseline_json.read_text())
                baseline_total = len(bres)
                baseline_success = sum(1 for r in bres if r.get("success"))
                bsteps = [r["env_steps"] for r in bres if r.get("success")]
                baseline_avg_steps = (sum(bsteps) / len(bsteps)) if bsteps else None
            except json.JSONDecodeError as e:
                print(f"WARN: bad baseline/results.json in {cfg_dir.name}: {e}")

        rows.append({
            **meta,
            "n_total": n_total,
            "n_success": n_success,
            "success_rate": n_success / n_total if n_total else 0.0,
            "avg_steps_success": avg_steps_success,
            "baseline_total": baseline_total,
            "baseline_success": baseline_success,
            "baseline_rate": (baseline_success / baseline_total)
                              if baseline_total else None,
            "baseline_avg_steps": baseline_avg_steps,
            "dir": str(cfg_dir),
        })

    rows.sort(key=lambda r: (-r["success_rate"], r["lambda"], r["q"], r["rinit"], r["tau"]))

    lines: list[str] = []
    lines.append("# LQR-decay cam-perturb hyperparameter sweep")
    lines.append("")
    lines.append(f"- rollouts root: `{args.rollouts_root}`")
    if rows:
        first = rows[0]
        lines.append(
            f"- filters: cam_mode=`{first['cam_mode']}`  cam_seed=`{first['cam_seed']}`  "
            f"run_tag=`{first['run_tag']}`  prompt slug=`{first['prompt_slug']}`"
        )
        lines.append(f"- r_fin (fixed): `{_fmt_num(first['rfin'])}`  qf (fixed): `{_fmt_num(first['qf'])}`")
    lines.append(f"- configurations with results.json: **{len(rows)}**")
    if skipped_no_results:
        lines.append(f"- configurations without results.json (jobs still running or failed): "
                     f"{len(skipped_no_results)}")

    # Baseline reference: when RUN_BASELINE was on for any combo, capture the
    # unsteered success rate. Baseline depends only on the camera perturbation
    # (no LQR), so all baseline runs at the same (cam_mode, cam_seed) should
    # match — report the union, plus a warning if they disagree.
    baseline_rows = [r for r in rows if r["baseline_total"] is not None]
    baseline_ref_rate = None
    if baseline_rows:
        total_eps = sum(r["baseline_total"] for r in baseline_rows)
        total_succ = sum(r["baseline_success"] for r in baseline_rows)
        baseline_ref_rate = total_succ / total_eps if total_eps else None
        # Report the single largest baseline as the reference (50-episode is
        # more informative than 10-episode).
        ref = max(baseline_rows, key=lambda r: r["baseline_total"])
        lines.append("")
        lines.append(f"- **baseline (unsteered policy under same camera perturbation)**: "
                     f"{ref['baseline_success']}/{ref['baseline_total']} = "
                     f"{ref['baseline_rate'] * 100:.0f}%  "
                     f"(avg {ref['baseline_avg_steps']:.1f} steps on successes)")
        if len(baseline_rows) > 1:
            distinct_rates = {round(r["baseline_rate"], 3) for r in baseline_rows}
            if len(distinct_rates) > 1:
                lines.append(
                    f"- note: {len(baseline_rows)} combos ran baselines and they disagree "
                    f"(rates: {sorted(distinct_rates)}). The reference above uses the "
                    f"one with the most episodes."
                )
            else:
                lines.append(
                    f"- ({len(baseline_rows)} combos ran baselines; all agree on rate)"
                )
    lines.append("")
    lines.append("Each row reports the number of successful steered rollouts out of the")
    lines.append("episodes attempted, sorted by success rate descending. `avg steps (succ)` is")
    lines.append("the mean env step count across successful episodes only.")
    lines.append("")
    # Show QF + R_FIN columns only if they vary across the loaded rows;
    # otherwise the columns are noise. With a single QF/R_FIN value the
    # markdown stays narrow.
    qf_values = {r["qf"] for r in rows} if rows else set()
    rfin_values = {r["rfin"] for r in rows} if rows else set()
    show_qf = len(qf_values) > 1
    show_rfin = len(rfin_values) > 1
    extra_cols_hdr = ""
    extra_cols_sep = ""
    if show_qf:
        extra_cols_hdr += " QF |"; extra_cols_sep += "---|"
    if show_rfin:
        extra_cols_hdr += " R_fin |"; extra_cols_sep += "---|"

    if baseline_ref_rate is not None:
        lines.append(f"`Δ vs base` = steered rate − baseline rate ({baseline_ref_rate * 100:.0f}%); "
                     "positive means steering helped.")
        lines.append("")
        lines.append(f"| λ | Q | r_init | τ |{extra_cols_hdr} success | rate | Δ vs base | avg steps (succ) |")
        lines.append(f"|---|---|--------|---|{extra_cols_sep}---------|------|-----------|------------------|")
    else:
        lines.append(f"| λ | Q | r_init | τ |{extra_cols_hdr} success | rate | avg steps (succ) |")
        lines.append(f"|---|---|--------|---|{extra_cols_sep}---------|------|------------------|")
    for r in rows:
        avg_str = f"{r['avg_steps_success']:.1f}" if r["avg_steps_success"] is not None else "—"
        extras = ""
        if show_qf:
            extras += f" {_fmt_num(r['qf'])} |"
        if show_rfin:
            extras += f" {_fmt_num(r['rfin'])} |"
        if baseline_ref_rate is not None:
            delta = (r["success_rate"] - baseline_ref_rate) * 100
            delta_str = f"{delta:+.0f}%"
            lines.append(
                f"| {r['lambda']:.2f} | {_fmt_num(r['q'])} | {_fmt_num(r['rinit'])} | "
                f"{_fmt_num(r['tau'])} |{extras} {r['n_success']}/{r['n_total']} | "
                f"{r['success_rate'] * 100:.0f}% | {delta_str} | {avg_str} |"
            )
        else:
            lines.append(
                f"| {r['lambda']:.2f} | {_fmt_num(r['q'])} | {_fmt_num(r['rinit'])} | "
                f"{_fmt_num(r['tau'])} |{extras} {r['n_success']}/{r['n_total']} | "
                f"{r['success_rate'] * 100:.0f}% | {avg_str} |"
            )

    def _marginal(field: str, title: str) -> list[str]:
        groups: dict[float, list[dict]] = {}
        for r in rows:
            groups.setdefault(r[field], []).append(r)
        out = [f"\n## Marginal over {title}", ""]
        out.append(f"| {title} | configs | episodes | successes | rate |")
        out.append("|---|---|---|---|---|")
        for k in sorted(groups):
            grp = groups[k]
            ep_total = sum(r["n_total"] for r in grp)
            ep_succ = sum(r["n_success"] for r in grp)
            rate = ep_succ / ep_total if ep_total else 0.0
            out.append(f"| {_fmt_num(k)} | {len(grp)} | {ep_total} | {ep_succ} | {rate * 100:.0f}% |")
        return out

    if rows:
        lines.extend(_marginal("lambda", "λ"))
        lines.extend(_marginal("q", "Q"))
        lines.extend(_marginal("rinit", "R_init"))
        lines.extend(_marginal("tau", "τ"))

    # Dedicated section for the QF × R_FIN orthogonal ablation: configs at
    # the canonical top point (λ=10, Q=10, R_init=20, τ=7) that vary only on
    # QF / R_FIN. Useful when you ran an orthogonal sweep on top of the main
    # (λ, Q, R_init, τ) grid.
    ABLATION_KEY = (10.0, 10.0, 20.0, 7.0)
    ablation_rows = [
        r for r in rows
        if (r["lambda"], r["q"], r["rinit"], r["tau"]) == ABLATION_KEY
    ]
    if len(ablation_rows) > 1 or (
        ablation_rows and (len({r["qf"] for r in ablation_rows}) > 1
                            or len({r["rfin"] for r in ablation_rows}) > 1)
    ):
        lines.append("")
        lines.append("## QF / R_FIN ablation (λ=10, Q=10, R_init=20, τ=7)")
        lines.append("")
        lines.append("Holds the main (λ, Q, R_init, τ) at the previous sweep's best point and")
        lines.append("varies only the terminal cost weight `QF` and the steering-saturation")
        lines.append("clamp `R_FIN`. Sorted by `QF`, then `R_FIN`.")
        lines.append("")
        ab_sorted = sorted(ablation_rows, key=lambda r: (r["qf"], r["rfin"]))
        if baseline_ref_rate is not None:
            lines.append("| QF | R_fin | success | rate | Δ vs base |")
            lines.append("|---|---|---------|------|-----------|")
            for r in ab_sorted:
                delta = (r["success_rate"] - baseline_ref_rate) * 100
                lines.append(
                    f"| {_fmt_num(r['qf'])} | {_fmt_num(r['rfin'])} | "
                    f"{r['n_success']}/{r['n_total']} | "
                    f"{r['success_rate'] * 100:.0f}% | {delta:+.0f}% |"
                )
        else:
            lines.append("| QF | R_fin | success | rate |")
            lines.append("|---|---|---------|------|")
            for r in ab_sorted:
                lines.append(
                    f"| {_fmt_num(r['qf'])} | {_fmt_num(r['rfin'])} | "
                    f"{r['n_success']}/{r['n_total']} | "
                    f"{r['success_rate'] * 100:.0f}% |"
                )

    if skipped_no_results:
        lines.append("")
        lines.append("## Configurations missing results.json")
        lines.append("")
        for name in skipped_no_results[:50]:
            lines.append(f"- `{name}`")
        if len(skipped_no_results) > 50:
            lines.append(f"- ... and {len(skipped_no_results) - 50} more")

    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.summary_md}  ({len(rows)} configurations)")
    if skipped_no_results:
        print(f"  ({len(skipped_no_results)} configurations missing results.json — "
              f"jobs may still be running)")


if __name__ == "__main__":
    main()
