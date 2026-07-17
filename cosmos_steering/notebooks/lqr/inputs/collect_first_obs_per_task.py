#!/usr/bin/env python
"""Capture the FIRST observation that the policy would see in episode 0 of
each LIBERO task, AFTER the gripper-XYZ perturbation has been applied.

Output is one tiny npz per task -- exactly the schema run_jacobians_full.py
expects via --inputs-npz / --obs-index 0:

    <OUT_DIR>/task{TID:02d}/inputs.npz
       primary_images (1, 256, 256, 3) uint8  (flip_images applied)
       wrist_images   (1, 256, 256, 3) uint8
       proprios       (1, 9)           float32
       task_id        (1,)             int32
       episode_idx    (1,)             int32  (always 0)
       inference_idx  (1,)             int32  (always 0)
       xyz_delta_m    (1, 3)           float32  (requested perturbation)
       achieved_xyz_delta_m (1, 3)     float32

A single manifest.json at <OUT_DIR>/manifest.json lists every task with its
prompt + perturbation sample. No model load / no get_action / no torch --
this is pure env initialization + the same perturbation pipeline used by
collect_policy_inputs_gripper_xyz_perturbation_multitask.py.

Usage:
    python collect_first_obs_per_task.py \\
        --out-dir notebooks/lqr/inputs/policy_inputs/libero_10__first_obs__xyz_random_xlarge_2__seed42 \\
        --task-ids 0 1 2 3 4 5 6 7 8 9 \\
        --preset xyz_random_xlarge_2 --base-seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_NOTEBOOKS_ROOT = _HERE.parent.parent
if str(_NOTEBOOKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_ROOT))
from _setup import setup_env  # noqa: E402

setup_env()
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from cosmos_policy.experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env, get_libero_dummy_action,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (  # noqa: E402
    PolicyEvalConfig, prepare_observation, TASK_MAX_STEPS,
)

# Reuse the perturbation classes + preset builder from the multitask collect
# script so the per-episode RNG matches bit-for-bit.
sys.path.insert(0, str(_HERE))
from collect_policy_inputs_gripper_xyz_perturbation_multitask import (  # noqa: E402
    build_presets,
)


def _log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--task-ids", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--episode-idx", type=int, default=0,
                    help="which init state to use (default 0 = first episode)")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--preset", type=str, default="xyz_random_xlarge_2")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--num-steps-wait", type=int, default=10,
                    help="dummy steps between perturbation and observation "
                         "capture; matches collect_policy_inputs_*.py")
    ap.add_argument("--prompt", type=str, default=None,
                    help="optional override applied to ALL tasks. Default uses "
                         "each libero task's built-in description.")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir: {out_dir}")

    presets = build_presets(args.base_seed)
    if args.preset not in presets:
        raise SystemExit(
            f"unknown --preset={args.preset!r}; pick one of {sorted(presets)}"
        )
    stress_test = presets[args.preset]
    _log(f"preset: {args.preset}  ({type(stress_test).__name__})  "
         f"manifest={stress_test.manifest()}")

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    eval_cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path="",
        t5_text_embeddings_path="",
        use_wrist_image=True, use_proprio=True,
        normalize_proprio=False, unnormalize_actions=False,
        chunk_size=16, num_open_loop_steps=16,
        trained_with_image_aug=False, use_jpeg_compression=False,
        flip_images=True,
        num_denoising_steps_action=1,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        task_suite_name=args.suite,
    )

    per_task_meta = []
    for task_id in args.task_ids:
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        ep = int(args.episode_idx)
        if ep >= init_states.shape[0]:
            raise SystemExit(
                f"--episode-idx {ep} >= n_init_states {init_states.shape[0]} "
                f"for task {task_id}"
            )

        env, base_desc = get_libero_env(task, "cosmos",
                                         resolution=args.resolution)
        prompt = args.prompt if args.prompt is not None else base_desc
        _log(f"--- task={task_id} ep={ep} prompt={prompt!r} ---")

        try:
            env.reset()
            obs = stress_test.set_init_state(env, init_states[ep],
                                              episode_idx=ep)
            stress_test.apply_to_env(env, episode_idx=ep)
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(
                    get_libero_dummy_action(eval_cfg.model_family)
                )
            observation = prepare_observation(
                obs, resize_size=args.resolution,
                flip_images=eval_cfg.flip_images,
            )
        finally:
            env.close()

        sample = dict(getattr(stress_test, "_last_sample", {}) or {})
        xyz_d = sample.get("xyz_delta_m") or [0.0, 0.0, 0.0]
        ach_d = sample.get("achieved_xyz_delta_m") or [0.0, 0.0, 0.0]
        xyz_str = ",".join(f"{v*1000:+5.1f}" for v in xyz_d)
        _log(f"  captured dxyz_mm=[{xyz_str}]  "
             f"prim={observation['primary_image'].shape} "
             f"wrist={observation['wrist_image'].shape} "
             f"proprio={np.asarray(observation['proprio']).shape}")

        task_dir = out_dir / f"task{task_id:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        npz_path = task_dir / "inputs.npz"
        np.savez_compressed(
            npz_path,
            primary_images=np.ascontiguousarray(
                observation["primary_image"][None])
                .astype(np.uint8, copy=False),
            wrist_images=np.ascontiguousarray(
                observation["wrist_image"][None])
                .astype(np.uint8, copy=False),
            proprios=np.asarray(observation["proprio"],
                                 dtype=np.float32)[None],
            task_id=np.asarray([task_id], dtype=np.int32),
            episode_idx=np.asarray([ep], dtype=np.int32),
            inference_idx=np.asarray([0], dtype=np.int32),
            xyz_delta_m=np.asarray([xyz_d], dtype=np.float32),
            achieved_xyz_delta_m=np.asarray([ach_d], dtype=np.float32),
        )
        _log(f"  wrote {npz_path}  ({npz_path.stat().st_size/1e3:.1f} kB)")
        per_task_meta.append({
            "task_id": int(task_id),
            "prompt": prompt,
            "episode_idx": ep,
            "n_init_states_available": int(init_states.shape[0]),
            "perturbation_sample": sample if sample else None,
            "npz": str(npz_path),
        })

    manifest = {
        "suite": args.suite,
        "task_ids": [int(t) for t in args.task_ids],
        "episode_idx": int(args.episode_idx),
        "preset": args.preset,
        "base_seed": int(args.base_seed),
        "stress_test": stress_test.manifest(),
        "num_steps_wait": int(args.num_steps_wait),
        "resolution": int(args.resolution),
        "prompt_override": args.prompt,
        "tasks": per_task_meta,
        "note": (
            "First observation each task's policy would see after "
            "applying the gripper xyz perturbation to episode_idx + "
            "num_steps_wait dummy steps. One inputs.npz per task with a "
            "single row at index 0; designed as drop-in --inputs-npz for "
            "run_jacobians_full.py."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(f"wrote {out_dir / 'manifest.json'}  "
         f"({len(per_task_meta)} tasks captured)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
