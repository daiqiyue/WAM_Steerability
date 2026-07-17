import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.lqr.common import default_slurm_port, ensure_dir, maybe_load_yaml
from scripts.lqr.libero_eval_common import (
    apply_variant_overrides,
    build_client_cmd,
    collect_task_metrics,
    load_eval_variants,
    resolve_task_ranges,
    run_client_or_accept_complete,
    stop_process,
    wait_for_port,
)


def _build_server_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        "python",
        "scripts/lqr/patch_infer_with_lqr.py",
        "--config-name",
        args.config_name,
        "--port",
        str(args.port),
        "--svd-dir",
        str(args.svd_dir),
        "--jac-dir-act",
        args.jac_dir_act,
        "--lambda-scale",
        str(args.lambda_scale),
        "--q-scale",
        str(args.q_scale),
        "--r-scale",
        str(args.r_scale),
        "--r-scale-tau",
        str(args.r_scale_tau),
        "--r-scale-final",
        str(args.r_scale_final),
        "--max-chunks",
        str(args.max_chunks),
        "--qf-scale",
        str(args.qf_scale),
        "--inject-mode",
        str(args.inject_mode),
        "--save_root",
        args.save_root,
    ]
    return cmd


def _load_lqr_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    return maybe_load_yaml(path)


def _collect_task_metrics_for_ranges(
    out_dir: str,
    benchmark_name: str,
    task_ranges: List[List[int]],
) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    succ_rates: List[float] = []
    for task_range in task_ranges:
        metrics = collect_task_metrics(out_dir, benchmark_name, task_range)
        for task_id, data in metrics["tasks"].items():
            rows[task_id] = data
            succ_rates.append(float(data.get("succ_rate", 0.0)))
    avg = float(sum(succ_rates) / len(succ_rates)) if succ_rates else 0.0
    return {"tasks": rows, "avg_succ_rate": avg}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIBERO evaluation with LQR-steered server.")
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--libero-benchmark", type=str, default="libero_10")
    parser.add_argument("--task-range", type=int, nargs=2, default=[0, 1])
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit LIBERO task ids (e.g. 1 2 7 9). Overrides --task-range.",
    )
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--startup-wait-sec", type=int, default=240)
    parser.add_argument("--svd-dir", type=str, required=True)
    parser.add_argument("--jac-dir-act", type=str, default=None)
    parser.add_argument("--lqr-config", type=str, default=None)
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--q-scale", type=float, default=10000.0)
    parser.add_argument("--r-scale", type=float, default=75000.0)
    parser.add_argument("--r-scale-tau", type=float, default=3.0)
    parser.add_argument("--r-scale-final", type=float, default=1e9)
    parser.add_argument("--max-chunks", type=int, default=50)
    parser.add_argument("--qf-scale", type=float, default=1.0)
    parser.add_argument("--inject-mode", type=str, choices=["auto", "action", "video", "both"], default="auto")
    parser.add_argument("--perturb-spec", type=str, default=None)
    parser.add_argument(
        "--agentview-noise-seed-base",
        type=int,
        default=None,
        help="Override noise_seed_base for gaussian variants (the original eval uses 99).",
    )
    parser.add_argument(
        "--agentview-noise-sigma",
        type=float,
        default=None,
        help="Override sigma for gaussian variants (noise_extreme=90).",
    )
    parser.add_argument(
        "--gripper-xyz-base-seed",
        type=int,
        default=None,
        help="Override base_seed for init_position variants (the original eval uses 99).",
    )
    parser.add_argument(
        "--random-camera-base-seed",
        type=int,
        default=None,
        help="Override base_seed for camera variants (the original camera eval sweep uses 99).",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--save-root",
        dest="save_root",
        type=str,
        default="",
        help="Server debug tensor/video save root. Empty disables VA_Server debug outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing episode videos in each variant out-dir and continue until --num-episodes exist.",
    )
    parser.add_argument(
        "--client-episode-batch-size",
        type=int,
        default=1,
        help="Run the eval client in small resume batches to isolate native EGL/SIGABRT failures. Use 0 for one long client run.",
    )
    args = parser.parse_args()
    task_ranges = resolve_task_ranges(args.task_range, args.task_ids)

    if args.port is None:
        args.port = default_slurm_port()

    cfg = _load_lqr_config(args.lqr_config)
    args.lambda_scale = float(cfg.get("lambda_scale", args.lambda_scale))
    args.q_scale = float(cfg.get("q_scale", args.q_scale))
    args.r_scale = float(cfg.get("r_scale", args.r_scale))
    args.r_scale_tau = float(cfg.get("r_scale_tau", args.r_scale_tau))
    args.r_scale_final = float(cfg.get("r_scale_final", args.r_scale_final))
    args.max_chunks = int(cfg.get("max_chunks", args.max_chunks))
    args.qf_scale = float(cfg.get("qf_scale", args.qf_scale))
    cfg_inject_mode = cfg.get("inject_mode")
    if cfg_inject_mode is not None:
        args.inject_mode = str(cfg_inject_mode)
    if args.jac_dir_act is None and cfg.get("jac_dir_act"):
        args.jac_dir_act = str(cfg["jac_dir_act"])
    elif args.jac_dir_act is None:
        args.jac_dir_act = "A_tilde_lingbot"

    if not args.perturb_spec:
        raise ValueError("--perturb-spec is required (nominal eval is skipped; only perturbed variants are run).")

    variants = apply_variant_overrides(
        load_eval_variants(args.perturb_spec),
        gripper_xyz_base_seed=args.gripper_xyz_base_seed,
        agentview_noise_sigma=args.agentview_noise_sigma,
        agentview_noise_seed_base=args.agentview_noise_seed_base,
        random_camera_base_seed=args.random_camera_base_seed,
    )
    if not variants:
        raise ValueError(
            f"No non-nominal variants found in perturb spec: {args.perturb_spec}"
        )

    out_root = ensure_dir(args.out_dir)
    perturbed_root = ensure_dir(os.path.join(out_root, "perturbed"))

    server_cmd = _build_server_cmd(args)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", ".")
    print(f"[eval] starting lqr server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(server_cmd, env=env)
    try:
        if not wait_for_port("127.0.0.1", args.port, args.startup_wait_sec):
            raise RuntimeError(f"LQR server did not start at port {args.port}")

        variant_metrics = {}
        for variant in variants:
            name = str(variant.get("name", "variant"))
            out_i = ensure_dir(os.path.join(perturbed_root, name))
            for task_range in task_ranges:
                args.task_range = task_range
                pert_cmd = build_client_cmd(args, out_dir=out_i, variant=variant)
                print(f"[eval] perturbed({name}) client: {' '.join(pert_cmd)}")
                run_client_or_accept_complete(
                    pert_cmd,
                    env,
                    out_i,
                    args.libero_benchmark,
                    args.task_range,
                    args.num_episodes,
                    "eval",
                    args.client_episode_batch_size,
                )
            variant_metrics[name] = _collect_task_metrics_for_ranges(
                out_i, args.libero_benchmark, task_ranges
            )
    finally:
        stop_process(server_proc)

    vals = [v["avg_succ_rate"] for v in variant_metrics.values()]
    avg_variant = float(sum(vals) / len(vals)) if vals else 0.0
    summary = {
        "config_name": args.config_name,
        "svd_dir": args.svd_dir,
        "jac_dir_act": args.jac_dir_act,
        "perturb_spec": args.perturb_spec,
        "agentview_noise_sigma": args.agentview_noise_sigma,
        "agentview_noise_seed_base": args.agentview_noise_seed_base,
        "gripper_xyz_base_seed": args.gripper_xyz_base_seed,
        "random_camera_base_seed": args.random_camera_base_seed,
        "libero_benchmark": args.libero_benchmark,
        "task_range": None if args.task_ids else args.task_range,
        "task_ids": args.task_ids,
        "task_ranges": task_ranges,
        "num_episodes": args.num_episodes,
        "resume": args.resume,
        "server_save_root": args.save_root,
        "inject_mode": args.inject_mode,
        "lqr": {
            "lambda_scale": args.lambda_scale,
            "q_scale": args.q_scale,
            "r_scale": args.r_scale,
            "r_scale_tau": args.r_scale_tau,
            "r_scale_final": args.r_scale_final,
            "max_chunks": args.max_chunks,
            "qf_scale": args.qf_scale,
        },
        "perturbed": variant_metrics,
        "avg_succ_rate_over_variants": avg_variant,
    }
    out_fp = Path(out_root) / "summary.json"
    out_fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_fp}")


if __name__ == "__main__":
    main()
