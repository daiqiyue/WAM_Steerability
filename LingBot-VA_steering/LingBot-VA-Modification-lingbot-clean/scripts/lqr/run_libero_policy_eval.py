"""Run LIBERO evaluation with vanilla policy (no LQR) under perturbations."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from scripts.lqr.common import default_slurm_port, ensure_dir
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


def _build_policy_server_cmd(args: argparse.Namespace) -> list:
    return [
        "python",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "1",
        "--master_port",
        str(args.master_port),
        "wan_va/wan_va_server.py",
        "--config-name",
        args.config_name,
        "--port",
        str(args.port),
        "--save_root",
        args.save_root,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LIBERO perturbed rollout with vanilla policy (no LQR injection)."
    )
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--libero-benchmark", type=str, default="libero_10")
    parser.add_argument("--task-range", type=int, nargs=2, default=None)
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
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="torch.distributed master port for wan_va_server (default: 12000 + job_id % 20000).",
    )
    parser.add_argument("--startup-wait-sec", type=int, default=1200)
    parser.add_argument("--perturb-spec", type=str, required=True)
    parser.add_argument(
        "--gripper-xyz-base-seed",
        type=int,
        default=None,
        help="Override base_seed for init_position variants (the original eval uses 99).",
    )
    parser.add_argument(
        "--agentview-noise-sigma",
        type=float,
        default=None,
        help="Override Gaussian sigma for image noise variants (noise_extreme=90).",
    )
    parser.add_argument(
        "--agentview-noise-seed-base",
        type=int,
        default=None,
        help="Override noise RNG seed base (the original eval uses 99; collect uses 0).",
    )
    parser.add_argument(
        "--random-camera-base-seed",
        type=int,
        default=None,
        help="Override base_seed for camera variants (the original camera eval uses 99).",
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
    if args.task_ids:
        task_ranges = resolve_task_ranges([], args.task_ids)
    elif args.task_range is not None:
        task_ranges = resolve_task_ranges(args.task_range, None)
    else:
        task_ranges = [[0, 1]]

    if args.port is None:
        args.port = default_slurm_port()
    if args.master_port is None:
        job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
        args.master_port = 12000 + (job_id % 20000)

    variants = apply_variant_overrides(
        load_eval_variants(args.perturb_spec),
        gripper_xyz_base_seed=args.gripper_xyz_base_seed,
        agentview_noise_sigma=args.agentview_noise_sigma,
        agentview_noise_seed_base=args.agentview_noise_seed_base,
        random_camera_base_seed=args.random_camera_base_seed,
    )
    if not variants:
        raise ValueError(
            f"No perturbation variants found in perturb spec: {args.perturb_spec}"
        )

    out_root = ensure_dir(args.out_dir)
    perturbed_root = ensure_dir(os.path.join(out_root, "perturbed"))

    server_cmd = _build_policy_server_cmd(args)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", ".")
    print(f"[policy-eval] starting vanilla server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(server_cmd, env=env)
    try:
        if not wait_for_port("127.0.0.1", args.port, args.startup_wait_sec):
            raise RuntimeError(f"Policy server did not start at port {args.port}")

        variant_metrics: Dict[str, Any] = {}
        for variant in variants:
            name = str(variant.get("name", "variant"))
            out_i = ensure_dir(os.path.join(perturbed_root, name))
            task_metrics: Dict[str, Any] = {}
            for task_range in task_ranges:
                eval_args = argparse.Namespace(**vars(args))
                eval_args.task_range = task_range
                client_cmd = build_client_cmd(eval_args, out_dir=out_i, variant=variant)
                print(
                    f"[policy-eval] perturbed({name}) tasks=[{task_range[0]},{task_range[1]}) "
                    f"client: {' '.join(client_cmd)}"
                )
                run_client_or_accept_complete(
                    client_cmd,
                    env,
                    out_i,
                    args.libero_benchmark,
                    task_range,
                    args.num_episodes,
                    "policy-eval",
                    args.client_episode_batch_size,
                )
                task_metrics[str(task_range[0])] = collect_task_metrics(
                    out_i, args.libero_benchmark, task_range
                )
            vals = [v["avg_succ_rate"] for v in task_metrics.values()]
            variant_metrics[name] = {
                "tasks": task_metrics,
                "avg_succ_rate": float(sum(vals) / len(vals)) if vals else 0.0,
            }
    finally:
        stop_process(server_proc)

    vals = [v["avg_succ_rate"] for v in variant_metrics.values()]
    avg_variant = float(sum(vals) / len(vals)) if vals else 0.0
    summary = {
        "config_name": args.config_name,
        "policy": "vanilla",
        "perturb_spec": args.perturb_spec,
        "libero_benchmark": args.libero_benchmark,
        "task_ranges": task_ranges,
        "gripper_xyz_base_seed": args.gripper_xyz_base_seed,
        "agentview_noise_sigma": args.agentview_noise_sigma,
        "agentview_noise_seed_base": args.agentview_noise_seed_base,
        "random_camera_base_seed": args.random_camera_base_seed,
        "num_episodes": args.num_episodes,
        "resume": args.resume,
        "server_save_root": args.save_root,
        "perturbed": variant_metrics,
        "avg_succ_rate_over_variants": avg_variant,
    }
    out_fp = Path(out_root) / "summary.json"
    out_fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[policy-eval] wrote {out_fp}")


if __name__ == "__main__":
    main()
