#!/usr/bin/env python
"""Paired task-6 ActAdd sensitivity evaluation under fixed Gaussian noise.

The contrastive direction is added to DiT block activations with Cosmos
Policy's native ``install_actadd_hooks`` implementation.  Every condition
replays an identical unsteered action prefix from the same LIBERO init state,
then branches under common-random-number image noise and policy sampling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_NOTEBOOKS_ROOT = _HERE.parent
if str(_NOTEBOOKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_ROOT))
from _setup import setup_env  # noqa: E402

setup_env()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from actadd.run_actadd_cosmos_policy import (  # noqa: E402
    install_actadd_hooks,
    load_steering_vectors,
)
from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    COSMOS_IMAGE_SIZE,
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (  # noqa: E402
    PolicyEvalConfig,
    prepare_observation,
)
from libero.libero import benchmark  # noqa: E402

CONDITIONS = {
    "raw_a": ("contrastive", 0.0),
    "raw_b": ("contrastive", 0.0),
    "plus_v": ("contrastive", 1.0),
    "minus_v": ("contrastive", -1.0),
    "random": ("random", 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument(
        "--prompt",
        default=("put the white mug on the plate and put the chocolate "
                 "pudding to the right of the plate"),
    )
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--max-env-steps", type=int, default=520)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--anchor-chunk", type=int, default=8)
    parser.add_argument("--branch-horizon-chunks", type=int, default=10,
                        help="0 means continue to max-env-steps")
    parser.add_argument("--raw-repeat-chunks", type=int, default=3)
    parser.add_argument("--rollout-alpha", type=float, default=0.1)
    parser.add_argument("--probe-chunks", type=int, nargs="+",
                        default=[8, 11, 14, 17])
    parser.add_argument("--probe-alphas", type=float, nargs="+",
                        default=[-1.0, -0.5, -0.1, -0.01, 0.0,
                                 0.01, 0.1, 0.5, 1.0])
    parser.add_argument("--noise-sigma", type=float, default=90.0)
    parser.add_argument("--noise-seed-base", type=int, default=99)
    parser.add_argument("--policy-seed", type=int, default=1)
    parser.add_argument("--random-direction-seed", type=int,
                        default=20260917)
    return parser.parse_args()


def fixed_gaussian_noise(image: np.ndarray, *, sigma: float, seed_base: int,
                         episode: int, chunk: int, camera: int) -> np.ndarray:
    """Stateless common-random-number noise for one camera and inference."""
    sequence = np.random.SeedSequence([seed_base, episode, chunk, camera])
    rng = np.random.default_rng(sequence)
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def prepare_noisy_observation(obs: dict, *, args: argparse.Namespace,
                              absolute_chunk: int) -> dict:
    observation = prepare_observation(
        obs, resize_size=COSMOS_IMAGE_SIZE, flip_images=True,
    )
    observation["primary_image"] = fixed_gaussian_noise(
        observation["primary_image"], sigma=args.noise_sigma,
        seed_base=args.noise_seed_base, episode=args.episode_id,
        chunk=absolute_chunk, camera=0,
    )
    observation["wrist_image"] = fixed_gaussian_noise(
        observation["wrist_image"], sigma=args.noise_sigma,
        seed_base=args.noise_seed_base, episode=args.episode_id,
        chunk=absolute_chunk, camera=1,
    )
    return observation


def make_norm_matched_random(v_per_layer: dict[int, torch.Tensor],
                             seed: int) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result = {}
    for layer, vector in sorted(v_per_layer.items()):
        random = torch.randn(vector.shape, generator=generator,
                             dtype=torch.float32)
        vector_norm = vector.float().norm()
        random_norm = random.norm().clamp_min(1e-12)
        result[layer] = random * (vector_norm / random_norm)
    return result


def object_positions(obs: dict) -> dict[str, list[float]]:
    result = {}
    for key, value in obs.items():
        array = np.asarray(value)
        if key.endswith("_pos") and array.shape == (3,) and "robot" not in key:
            result[key] = array.astype(float).tolist()
    return result


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    eval_cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=args.ckpt_path,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{args.ckpt_path}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{args.ckpt_path}/libero_t5_embeddings.pkl",
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        chunk_size=16,
        num_open_loop_steps=16,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=args.sampling_steps,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        task_suite_name=args.suite,
        suite="libero",
    )
    dataset_stats = load_dataset_stats(eval_cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(eval_cfg.t5_text_embeddings_path)
    model, _ = get_model(eval_cfg)

    v_per_layer = load_steering_vectors(args.v_path)
    n_blocks = len(model.net.blocks)
    if list(v_per_layer) == [-1]:
        v_per_layer = {layer: v_per_layer[-1] for layer in range(n_blocks)}
    invalid = [layer for layer in v_per_layer if not 0 <= layer < n_blocks]
    if invalid:
        raise ValueError(f"steering vector has invalid layers: {invalid}")
    random_per_layer = make_norm_matched_random(
        v_per_layer, args.random_direction_seed,
    )
    directions = {"contrastive": v_per_layer, "random": random_per_layer}

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.episode_id < len(init_states):
        raise ValueError(f"episode {args.episode_id} outside init-state range")
    env_horizon = args.num_steps_wait + args.max_env_steps + 20
    env, task_desc = get_libero_env(
        task, "cosmos", resolution=args.resolution, horizon=env_horizon,
    )

    def reset_env() -> dict:
        env.reset()
        obs = env.set_init_state(init_states[args.episode_id])
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
        return obs

    def infer(observation: dict, *, direction: str, alpha: float,
              absolute_chunk: int) -> np.ndarray:
        noisy = prepare_noisy_observation(
            observation, args=args, absolute_chunk=absolute_chunk,
        )
        enabled = [True]
        handles = install_actadd_hooks(
            model, directions[direction], float(alpha), enabled,
        )
        try:
            with torch.inference_mode():
                output = get_action(
                    eval_cfg, model, dataset_stats, noisy, args.prompt,
                    seed=args.policy_seed,
                    randomize_seed=False,
                    num_denoising_steps_action=args.sampling_steps,
                    generate_future_state_and_value_in_parallel=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        return np.asarray(output["actions"][:eval_cfg.num_open_loop_steps],
                          dtype=np.float32)

    # Build one deterministic unsteered prefix.  All branches replay these
    # exact actions, rather than re-inferring the prefix.
    obs = reset_env()
    prefix_actions: list[np.ndarray] = []
    prefix_eef = [np.asarray(obs["robot0_eef_pos"], dtype=np.float32)]
    prefix_success = False
    for chunk in range(args.anchor_chunk):
        chunk_actions = infer(
            obs, direction="contrastive", alpha=0.0, absolute_chunk=chunk,
        )
        for action in chunk_actions:
            obs, _, done, _ = env.step(action.tolist())
            prefix_actions.append(action.copy())
            prefix_eef.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
            if done:
                prefix_success = True
                break
        if prefix_success:
            break
    anchor_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    if prefix_success:
        raise RuntimeError(
            f"episode {args.episode_id} succeeded before anchor chunk "
            f"{args.anchor_chunk}; choose an earlier anchor"
        )

    probe_records: list[dict] = []
    probe_actions: list[np.ndarray] = []
    condition_arrays: dict[str, dict[str, np.ndarray]] = {}
    condition_summary: dict[str, dict] = {}

    def run_condition(name: str) -> None:
        direction_name, sign = CONDITIONS[name]
        alpha = args.rollout_alpha * sign
        local_obs = reset_env()
        eef_steps = [np.asarray(local_obs["robot0_eef_pos"], dtype=np.float32)]
        success = False
        executed = 0
        for action in prefix_actions:
            local_obs, _, done, _ = env.step(action.tolist())
            executed += 1
            eef_steps.append(
                np.asarray(local_obs["robot0_eef_pos"], dtype=np.float32)
            )
            if done:
                success = True
                break
        replay_anchor_eef = np.asarray(local_obs["robot0_eef_pos"],
                                       dtype=np.float32)
        anchor_error = float(np.max(np.abs(replay_anchor_eef - anchor_eef)))
        if success:
            raise RuntimeError(f"prefix replay unexpectedly ended in {name}")

        action_chunks = []
        branch_actions = []
        eef_chunks = [replay_anchor_eef.copy()]
        start_chunk = args.anchor_chunk
        max_chunks_by_steps = math.ceil(args.max_env_steps /
                                        eval_cfg.num_open_loop_steps)
        stop_chunk = max_chunks_by_steps
        if args.branch_horizon_chunks > 0:
            stop_chunk = min(stop_chunk,
                             start_chunk + args.branch_horizon_chunks)
        if name == "raw_b":
            stop_chunk = min(stop_chunk,
                             start_chunk + args.raw_repeat_chunks)

        for absolute_chunk in range(start_chunk, stop_chunk):
            if executed >= args.max_env_steps:
                break
            actions = None
            if name == "raw_a" and absolute_chunk in args.probe_chunks:
                probe_stack = []
                for probe_alpha in args.probe_alphas:
                    probed = infer(
                        local_obs, direction="contrastive",
                        alpha=probe_alpha, absolute_chunk=absolute_chunk,
                    )
                    probe_stack.append(probed)
                    if probe_alpha == 0.0:
                        actions = probed
                probe_actions.append(np.stack(probe_stack, axis=0))
                probe_records.append({
                    "absolute_chunk": int(absolute_chunk),
                    "eef_m": np.asarray(local_obs["robot0_eef_pos"],
                                        dtype=float).tolist(),
                })
            if actions is None:
                actions = infer(
                    local_obs, direction=direction_name, alpha=alpha,
                    absolute_chunk=absolute_chunk,
                )
            action_chunks.append(actions.copy())
            for action in actions:
                if executed >= args.max_env_steps:
                    break
                local_obs, _, done, _ = env.step(action.tolist())
                branch_actions.append(action.copy())
                executed += 1
                eef_steps.append(
                    np.asarray(local_obs["robot0_eef_pos"], dtype=np.float32)
                )
                if done:
                    success = True
                    break
            eef_chunks.append(
                np.asarray(local_obs["robot0_eef_pos"], dtype=np.float32)
            )
            if success:
                break

        action_chunks_arr = np.asarray(action_chunks, dtype=np.float32)
        branch_actions_arr = np.asarray(branch_actions, dtype=np.float32)
        eef_steps_arr = np.asarray(eef_steps, dtype=np.float32)
        eef_chunks_arr = np.asarray(eef_chunks, dtype=np.float32)
        condition_arrays[name] = {
            "action_chunks": action_chunks_arr,
            "branch_actions": branch_actions_arr,
            "eef_steps": eef_steps_arr,
            "eef_chunks": eef_chunks_arr,
        }
        condition_summary[name] = {
            "direction": direction_name,
            "alpha": float(alpha),
            "success_within_window": bool(success),
            "total_policy_env_steps": int(executed),
            "branch_env_steps": int(len(branch_actions_arr)),
            "branch_chunks": int(len(action_chunks_arr)),
            "anchor_replay_max_abs_error_m": anchor_error,
            "branch_eef_displacement_m": float(
                np.linalg.norm(eef_steps_arr[-1] - replay_anchor_eef)
            ),
            "branch_eef_path_length_m": path_length(
                eef_steps_arr[len(prefix_actions):]
            ),
            "final_eef_m": eef_steps_arr[-1].astype(float).tolist(),
            "final_object_positions_m": object_positions(local_obs),
        }
        print(
            f"[{name}] alpha={alpha:+g} chunks={len(action_chunks_arr)} "
            f"steps={len(branch_actions_arr)} success={success} "
            f"anchor_error={anchor_error:.3e}",
            flush=True,
        )

    for condition in CONDITIONS:
        run_condition(condition)

    comparisons = {}
    raw_eef = condition_arrays["raw_a"]["eef_steps"]
    raw_chunks = condition_arrays["raw_a"]["action_chunks"]
    for name in ("raw_b", "plus_v", "minus_v", "random"):
        other_eef = condition_arrays[name]["eef_steps"]
        n_steps = min(len(raw_eef), len(other_eef))
        eef_divergence = np.linalg.norm(
            other_eef[:n_steps] - raw_eef[:n_steps], axis=1,
        )
        other_chunks = condition_arrays[name]["action_chunks"]
        n_chunks = min(len(raw_chunks), len(other_chunks))
        if n_chunks:
            chunk_delta = other_chunks[:n_chunks] - raw_chunks[:n_chunks]
            action_rms = np.sqrt(np.mean(chunk_delta ** 2, axis=(1, 2)))
            action_max_abs = float(np.max(np.abs(chunk_delta)))
        else:
            action_rms = np.empty((0,), dtype=np.float32)
            action_max_abs = float("nan")
        condition_arrays[name]["eef_divergence_vs_raw_m"] = eef_divergence
        condition_arrays[name]["action_rms_vs_raw"] = action_rms
        comparisons[name] = {
            "aligned_env_steps": int(n_steps),
            "eef_divergence_final_m": float(eef_divergence[-1]),
            "eef_divergence_mean_m": float(eef_divergence.mean()),
            "eef_divergence_max_m": float(eef_divergence.max()),
            "action_chunk_rms_mean": (
                float(action_rms.mean()) if len(action_rms) else None
            ),
            "action_max_abs": action_max_abs,
        }

    arrays = {
        "prefix_actions": np.asarray(prefix_actions, dtype=np.float32),
        "prefix_eef_steps": np.asarray(prefix_eef, dtype=np.float32),
        "probe_alphas": np.asarray(args.probe_alphas, dtype=np.float32),
        "probe_chunks": np.asarray(
            [record["absolute_chunk"] for record in probe_records],
            dtype=np.int32,
        ),
        "probe_actions": (
            np.stack(probe_actions, axis=0).astype(np.float32)
            if probe_actions else np.empty((0, len(args.probe_alphas), 16, 7),
                                           dtype=np.float32)
        ),
    }
    for name, values in condition_arrays.items():
        for key, value in values.items():
            arrays[f"{name}__{key}"] = value

    stem = f"episode_{args.episode_id:02d}"
    npz_path = args.out_dir / f"{stem}.npz"
    json_path = args.out_dir / f"{stem}.json"
    np.savez_compressed(npz_path, **arrays)
    metadata = {
        "episode": int(args.episode_id),
        "suite": args.suite,
        "task_id": int(args.task_id),
        "libero_task_description": task_desc,
        "policy_prompt": args.prompt,
        "v_path": str(args.v_path.resolve()),
        "intervention_space": "DiT block output activation [B,T,H,W,D]",
        "hook_implementation": (
            "actadd.run_actadd_cosmos_policy.install_actadd_hooks"
        ),
        "hooked_layers": sorted(int(layer) for layer in v_per_layer),
        "vector_l2_per_layer": {
            str(layer): float(vector.float().norm())
            for layer, vector in sorted(v_per_layer.items())
        },
        "random_control": (
            "fixed isotropic random direction independently per layer, "
            "L2 norm matched to that layer's contrastive vector"
        ),
        "random_direction_seed": int(args.random_direction_seed),
        "noise": {
            "kind": "Gaussian pixel noise",
            "sigma_uint8": float(args.noise_sigma),
            "seed_base": int(args.noise_seed_base),
            "seed_recipe": (
                "SeedSequence([base, episode, absolute_chunk, camera]); "
                "same draw for every branch"
            ),
            "cameras": ["primary_image", "wrist_image"],
        },
        "policy_seed": int(args.policy_seed),
        "sampling_steps": int(args.sampling_steps),
        "anchor_chunk": int(args.anchor_chunk),
        "prefix_env_steps": int(len(prefix_actions)),
        "branch_horizon_chunks": int(args.branch_horizon_chunks),
        "rollout_alpha": float(args.rollout_alpha),
        "probe_alphas": [float(x) for x in args.probe_alphas],
        "probe_records": probe_records,
        "conditions": condition_summary,
        "comparisons_vs_raw_a": comparisons,
        "artifacts": {"npz": str(npz_path), "json": str(json_path)},
        "wall_time_s": float(time.time() - started),
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n")
    env.close()
    print(f"wrote {json_path} and {npz_path}", flush=True)


if __name__ == "__main__":
    main()
