"""Shared helpers for LIBERO policy and LQR eval launchers."""

import argparse
import json
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.lqr.common import maybe_load_yaml

_INIT_POS_KINDS = frozenset(
    {"init_position", "gripper_init", "init_pos", "gripper_xyz"}
)
_GAUSSIAN_KINDS = frozenset({"gaussian", "image_gaussian_noise", "noise"})
_CAMERA_KINDS = frozenset({"camera", "camera_view", "random_camera"})
_EPISODE_VIDEO_RE = re.compile(r"^(\d+)_(True|False)\.mp4$")


def wait_for_port(host: str, port: int, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def apply_variant_overrides(
    variants: List[Dict[str, Any]],
    gripper_xyz_base_seed: Optional[int] = None,
    agentview_noise_sigma: Optional[float] = None,
    agentview_noise_seed_base: Optional[int] = None,
    random_camera_base_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if (
        gripper_xyz_base_seed is None
        and agentview_noise_sigma is None
        and agentview_noise_seed_base is None
        and random_camera_base_seed is None
    ):
        return variants
    out: List[Dict[str, Any]] = []
    for variant in variants:
        patched = dict(variant)
        kind = str(patched.get("kind", patched.get("type", ""))).lower()
        if kind in _INIT_POS_KINDS and gripper_xyz_base_seed is not None:
            patched["base_seed"] = int(gripper_xyz_base_seed)
        if kind in _CAMERA_KINDS and random_camera_base_seed is not None:
            patched["base_seed"] = int(random_camera_base_seed)
        if kind in _GAUSSIAN_KINDS:
            if agentview_noise_sigma is not None:
                patched["sigma"] = float(agentview_noise_sigma)
            if agentview_noise_seed_base is not None:
                patched["noise_seed_base"] = int(agentview_noise_seed_base)
                patched["seed_base"] = int(agentview_noise_seed_base)
        out.append(patched)
    return out


def resolve_task_ranges(
    task_range: List[int],
    task_ids: Optional[List[int]],
) -> List[List[int]]:
    if task_ids:
        return [[int(t), int(t) + 1] for t in task_ids]
    return [task_range]


def load_eval_variants(perturb_spec: Optional[str]) -> List[Dict[str, Any]]:
    if not perturb_spec:
        return []
    spec = maybe_load_yaml(perturb_spec)
    if "perturbation" in spec:
        perturb = dict(spec["perturbation"])
        perturb.setdefault("name", perturb.get("kind", "perturbation"))
        return [perturb]
    variants = list(spec.get("variants", []))
    out = [v for v in variants if str(v.get("name", "")) != "nominal"]
    for v in out:
        if "kind" not in v:
            raise ValueError(
                "Eval perturb specs must use repo-root-style variant entries with explicit `kind`."
            )
    return out


def build_client_cmd(
    args: argparse.Namespace,
    out_dir: str,
    variant: Optional[Dict[str, Any]],
) -> List[str]:
    cmd = [
        "python",
        "evaluation/libero/client.py",
        "--libero-benchmark",
        args.libero_benchmark,
        "--port",
        str(args.port),
        "--test-num",
        str(args.num_episodes),
        "--task-range",
        str(args.task_range[0]),
        str(args.task_range[1]),
        "--out-dir",
        out_dir,
    ]
    if getattr(args, "resume", False):
        cmd += ["--resume"]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if variant:
        kind = str(variant.get("kind", variant.get("type", ""))).lower()
        if kind in _GAUSSIAN_KINDS:
            cmd += ["--agentview-noise-sigma", str(float(variant.get("sigma", 90.0)))]
            cmd += [
                "--agentview-noise-seed-base",
                str(int(variant.get("noise_seed_base", variant.get("seed_base", 0)))),
            ]
            cmd += ["--noise-apply-wrist"]
        if kind in _CAMERA_KINDS:
            cmd += [
                "--random-camera-pos-sigma",
                str(float(variant.get("pos_sigma_m", variant.get("pos_sigma", 0.10)))),
            ]
            cmd += [
                "--random-camera-rot-sigma-deg",
                str(float(variant.get("rot_sigma_deg", 8.0))),
            ]
            cmd += [
                "--random-camera-fov-sigma",
                str(float(variant.get("fov_sigma_deg", variant.get("fov_sigma", 5.0)))),
            ]
            cmd += ["--random-camera-base-seed", str(int(variant.get("base_seed", 42)))]
            cmd += ["--random-camera-name", str(variant.get("camera_name", "agentview"))]
            cmd += [
                "--random-camera-workspace-table-z",
                str(float(variant.get("workspace_table_z", 0.90))),
            ]
            cmd += [
                "--random-camera-workspace-visible-fraction",
                str(float(variant.get("workspace_visible_fraction", 0.55))),
            ]
            cmd += [
                "--random-camera-visibility-margin-px",
                str(int(variant.get("visibility_margin_px", 8))),
            ]
            cmd += [
                "--random-camera-image-size",
                str(int(variant.get("image_size", 128))),
            ]
            cmd += [
                "--random-camera-max-rejection-attempts",
                str(int(variant.get("max_rejection_attempts", 2000))),
            ]
            if not bool(variant.get("enforce_visibility", True)):
                cmd += ["--disable-random-camera-visibility"]
        if kind in _INIT_POS_KINDS:
            cmd += [
                "--gripper-xyz-preset",
                str(variant.get("preset", variant.get("name", "xyz_random_xlarge_3"))),
            ]
            cmd += ["--gripper-xyz-base-seed", str(int(variant.get("base_seed", 42)))]
    return cmd


def collect_task_metrics(
    out_dir: str,
    benchmark_name: str,
    task_range: List[int],
) -> Dict[str, Any]:
    rows = {}
    succ_rates = []
    for task_id in range(task_range[0], task_range[1]):
        fp = Path(out_dir) / f"{benchmark_name}_{task_id}.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        rows[str(task_id)] = data
        succ_rates.append(float(data.get("succ_rate", 0.0)))
    avg = float(sum(succ_rates) / len(succ_rates)) if succ_rates else 0.0
    return {"tasks": rows, "avg_succ_rate": avg}


def eval_outputs_complete(
    out_dir: str,
    benchmark_name: str,
    task_range: List[int],
    num_episodes: int,
) -> bool:
    for task_id in range(task_range[0], task_range[1]):
        fp = Path(out_dir) / f"{benchmark_name}_{task_id}.json"
        total = 0.0
        if fp.exists():
            data = json.loads(fp.read_text(encoding="utf-8"))
            total = max(total, float(data.get("total_num", 0.0)))
        total = max(total, _completed_episode_count(out_dir, benchmark_name, [task_id, task_id + 1]))
        if total < float(num_episodes):
            return False
    return True


def _completed_episode_count(
    out_dir: str,
    benchmark_name: str,
    task_range: List[int],
) -> float:
    total = 0.0
    for task_id in range(task_range[0], task_range[1]):
        task_total = 0.0
        fp = Path(out_dir) / f"{benchmark_name}_{task_id}.json"
        if fp.exists():
            data = json.loads(fp.read_text(encoding="utf-8"))
            task_total = max(task_total, float(data.get("total_num", 0.0)))

        benchmark_dir = Path(out_dir) / benchmark_name
        if benchmark_dir.is_dir():
            seen = set()
            for task_dir in benchmark_dir.glob(f"{task_id}_*"):
                if not task_dir.is_dir():
                    continue
                for video in task_dir.iterdir():
                    match = _EPISODE_VIDEO_RE.match(video.name)
                    if match:
                        seen.add(int(match.group(1)))
            task_total = max(task_total, float(len(seen)))
        total += task_total
    return total


def _with_resume_batch(cmd: List[str], batch_size: int) -> List[str]:
    out = list(cmd)
    if "--resume" not in out:
        out.append("--resume")
    if "--max-new-episodes" not in out:
        out += ["--max-new-episodes", str(int(batch_size))]
    return out


def run_client_or_accept_complete(
    cmd: List[str],
    env: Dict[str, str],
    out_dir: str,
    benchmark_name: str,
    task_range: List[int],
    num_episodes: int,
    log_prefix: str,
    episode_batch_size: int = 0,
) -> None:
    if episode_batch_size and int(episode_batch_size) > 0:
        batched_cmd = _with_resume_batch(cmd, int(episode_batch_size))
        failures_without_progress = 0
        while not eval_outputs_complete(out_dir, benchmark_name, task_range, num_episodes):
            before = _completed_episode_count(out_dir, benchmark_name, task_range)
            try:
                subprocess.run(batched_cmd, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                after = _completed_episode_count(out_dir, benchmark_name, task_range)
                if exc.returncode == -signal.SIGABRT and after > before:
                    failures_without_progress = 0
                    print(
                        f"[{log_prefix}] warning: client SIGABRT after progress "
                        f"({before:g} -> {after:g}); continuing."
                    )
                    continue
                if exc.returncode == -signal.SIGABRT and eval_outputs_complete(
                    out_dir,
                    benchmark_name,
                    task_range,
                    num_episodes,
                ):
                    print(
                        f"[{log_prefix}] warning: client exited with SIGABRT after "
                        "writing complete eval outputs; continuing."
                    )
                    return
                failures_without_progress += 1
                if failures_without_progress >= 2:
                    raise
                print(
                    f"[{log_prefix}] warning: client SIGABRT without JSON progress; "
                    "retrying once."
                )
                continue
            after = _completed_episode_count(out_dir, benchmark_name, task_range)
            if after <= before and not eval_outputs_complete(
                out_dir,
                benchmark_name,
                task_range,
                num_episodes,
            ):
                print(
                    f"[{log_prefix}] warning: client exited 0 but progress counter "
                    f"did not increase ({before:g} -> {after:g}); continuing. "
                    "This can happen when old prompt-named video dirs and new "
                    "hashed prompt dirs coexist."
                )
        return

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == -signal.SIGABRT and eval_outputs_complete(
            out_dir,
            benchmark_name,
            task_range,
            num_episodes,
        ):
            print(
                f"[{log_prefix}] warning: client exited with SIGABRT after "
                "writing complete eval outputs; continuing."
            )
            return
        raise
