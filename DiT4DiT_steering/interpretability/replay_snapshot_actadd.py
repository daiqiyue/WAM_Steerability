#!/usr/bin/env python
"""Branch a saved LIBERO inference snapshot into raw and ActAdd rollouts.

Both branches restore the same MuJoCo state, consume the exact saved first
observation, restore the Gaussian-noise RNG immediately before that inference,
and use the same policy/diffusion seed.  The steered branch adds
``alpha * unit_contrastive_direction`` at the requested DiT blocks and
denoising steps.  The output is a side-by-side video and a JSON/NPZ record of
the end-effector trajectories.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
LQR_ROOT = HERE.parent / "lqr"
if str(LQR_ROOT) not in sys.path:
    sys.path.insert(0, str(LQR_ROOT))

from reproducibility import clone_observation  # noqa: E402
from runtime_paths import configure_runtime  # noqa: E402

CODE_ROOT, LIBERO_HOME = configure_runtime(HERE.parent)
os.environ.setdefault("LIBERO_HOME", LIBERO_HOME)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--svd-dir", type=Path, required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-index", type=int, default=20)
    parser.add_argument(
        "--max-inferences", type=int, default=0,
        help="Maximum action chunks per branch; 0 uses the source rollout remainder.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--blocks", default="all")
    parser.add_argument("--steps", default="all")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--video-fps", type=int, default=30)
    return parser.parse_args()


def parse_int_selection(spec: str, available: list[int], label: str) -> list[int]:
    if spec.strip().lower() == "all":
        return list(available)
    values = [int(item) for item in spec.split(",") if item.strip()]
    invalid = sorted(set(values) - set(available))
    if invalid:
        raise ValueError(f"invalid {label} {invalid}; available={available}")
    return values


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def add_gaussian_noise(image: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    return np.clip(image + rng.normal(0.0, sigma, image.shape), 0, 255).astype(np.uint8)


class ActAddRuntime:
    """Forward hooks implementing fixed activation addition for DiT4DiT."""

    def __init__(
        self,
        *,
        action_dit,
        directions: dict[tuple[int, int], torch.Tensor],
        denoise_t_start: int,
        denoise_t_end: int,
        alpha: float,
    ):
        self.directions = directions
        self.denoise_t_start = int(denoise_t_start)
        self.denoise_t_end = int(denoise_t_end)
        self.alpha = float(alpha)
        self.pass_idx = -1
        self.enabled = False
        self.handles = []

        def pass_tick(_module, _args):
            self.pass_idx += 1

        self.handles.append(
            action_dit.transformer_blocks[0].register_forward_pre_hook(pass_tick)
        )
        for block_index, block in enumerate(action_dit.transformer_blocks):
            self.handles.append(
                block.register_forward_hook(self._make_hook(block_index))
            )

    def _make_hook(self, block_index: int):
        def hook(_module, _args, output):
            if not self.enabled:
                return None
            direction = self.directions.get((self.pass_idx, block_index))
            if direction is None:
                return None
            updated = output.clone()
            update = direction.to(device=output.device, dtype=output.dtype)
            updated[:, self.denoise_t_start:self.denoise_t_end, :] += (
                self.alpha * update.unsqueeze(0)
            )
            return updated

        return hook

    def reset_chunk(self) -> None:
        self.pass_idx = -1

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def load_unit_directions(
    svd_dir: Path,
    cfg: dict,
    summary: dict,
    blocks: list[int],
    steps: list[int],
    device: torch.device,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[str, float]]:
    selected_steps = [int(value) for value in cfg["selected_timesteps"]]
    step_position = {step: index for index, step in enumerate(selected_steps)}
    layer_to_part = [int(value) for value in summary["layer_to_part"]]
    partitions = [tuple(int(value) for value in pair) for pair in cfg["partitions"]]
    k = int(cfg["k_target"])
    v_cache: dict[tuple[int, int], torch.Tensor] = {}
    result = {}
    raw_norms = {}

    for step in steps:
        for block in blocks:
            partition = layer_to_part[block]
            low, high = partitions[partition]
            cache_key = (partition, step)
            if cache_key not in v_cache:
                path = svd_dir / f"V_part{partition}_layers{low}-{high}_t{step}_k{k}.pt"
                v_cache[cache_key] = torch.load(
                    path, map_location="cpu", weights_only=False
                )["V"].float()
            coefficients = summary["c_means"][block, step_position[step]].float()
            full = v_cache[cache_key] @ coefficients
            norm = full.norm().clamp(min=1e-12)
            direction = (full / norm).reshape(
                int(cfg.get("action_horizon", cfg["T_p_denoise"])),
                int(cfg.get("inner_dim", cfg["D_flat"] // cfg["T_p_denoise"])),
            )
            result[(step, block)] = direction.to(device=device, dtype=torch.bfloat16)
            raw_norms[f"step{step}_block{block}"] = float(norm)
    return result, raw_norms


@dataclass
class BranchResult:
    label: str
    success: bool
    frames: list[np.ndarray]
    inference_ids: list[int]
    noisy_images: list[np.ndarray]
    eef_positions: np.ndarray
    executed_actions: np.ndarray
    inference_count: int
    first_model_input_equal: bool
    first_model_input_max_abs_error: int


def render_branch_panel(
    frame: np.ndarray,
    noisy_model_image: np.ndarray,
    label: str,
    inference_index: int,
    eef_position: np.ndarray,
    eef_start: np.ndarray,
    finished: bool,
) -> np.ndarray:
    clean_size = 320
    inset_size = 160
    header = 54
    panel = np.zeros((header + clean_size, clean_size + inset_size, 3), dtype=np.uint8)

    clean = cv2.resize(
        np.ascontiguousarray(np.flipud(frame)),
        (clean_size, clean_size),
        interpolation=cv2.INTER_AREA,
    )
    panel[header:, :clean_size] = np.clip(clean, 0, 255).astype(np.uint8)

    noisy = np.clip(noisy_model_image, 0, 255).astype(np.uint8)
    midpoint = noisy.shape[1] // 2
    noisy_agent = cv2.resize(
        noisy[:, :midpoint], (inset_size, inset_size), interpolation=cv2.INTER_AREA
    )
    noisy_wrist = cv2.resize(
        noisy[:, midpoint:], (inset_size, inset_size), interpolation=cv2.INTER_AREA
    )
    panel[header:header + inset_size, clean_size:] = noisy_agent
    panel[header + inset_size:, clean_size:] = noisy_wrist

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, label, (10, 22), font, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        panel, f"inference #{inference_index:03d}", (10, 45),
        font, 0.47, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.rectangle(panel, (0, header + clean_size - 48),
                  (clean_size - 1, header + clean_size - 1), (0, 0, 0), -1)
    delta_cm = 100.0 * (eef_position - eef_start)
    cv2.putText(
        panel,
        f"EEF delta cm: [{delta_cm[0]:+.2f}, {delta_cm[1]:+.2f}, {delta_cm[2]:+.2f}]",
        (7, header + clean_size - 27), font, 0.39, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel, f"distance: {np.linalg.norm(delta_cm):.2f} cm",
        (7, header + clean_size - 9), font, 0.39, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.rectangle(panel, (clean_size, header + inset_size - 24),
                  (clean_size + inset_size - 1, header + inset_size - 1), (0, 0, 0), -1)
    cv2.putText(panel, "noisy agent input", (clean_size + 5, header + inset_size - 8),
                font, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (clean_size, header + clean_size - 24),
                  (clean_size + inset_size - 1, header + clean_size - 1), (0, 0, 0), -1)
    cv2.putText(panel, "noisy wrist input", (clean_size + 5, header + clean_size - 8),
                font, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    if finished:
        cv2.putText(panel, "FINISHED", (clean_size - 105, 22), font, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def write_comparison_video(
    raw: BranchResult,
    steered: BranchResult,
    path: Path,
    fps: int,
    alpha: float,
) -> None:
    n_frames = max(len(raw.frames), len(steered.frames))
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1
    )
    for index in range(n_frames):
        raw_index = min(index, len(raw.frames) - 1)
        steer_index = min(index, len(steered.frames) - 1)
        raw_panel = render_branch_panel(
            raw.frames[raw_index], raw.noisy_images[raw_index],
            "RAW (ActAdd off)", raw.inference_ids[raw_index],
            raw.eef_positions[raw_index], raw.eef_positions[0],
            index >= len(raw.frames),
        )
        steer_panel = render_branch_panel(
            steered.frames[steer_index], steered.noisy_images[steer_index],
            f"STEERED (ActAdd alpha={alpha:g})", steered.inference_ids[steer_index],
            steered.eef_positions[steer_index], steered.eef_positions[0],
            index >= len(steered.frames),
        )
        separator = np.full((raw_panel.shape[0], 4, 3), 255, dtype=np.uint8)
        writer.append_data(np.concatenate([raw_panel, separator, steer_panel], axis=1))
    writer.close()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.alpha):
        raise ValueError("--alpha must be finite")
    if args.max_inferences < 0:
        raise ValueError("--max-inferences must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    matching = [
        row for row in snapshot["inferences"]
        if int(row["inference_idx"]) == args.inference_index
    ]
    if len(matching) != 1:
        raise ValueError(
            f"expected one inference #{args.inference_index}, found {len(matching)}"
        )
    source = matching[0]
    rng_state = source["model_input"].get("noise_rng_state_before")
    if rng_state is None:
        raise ValueError("snapshot does not contain noise_rng_state_before")

    source_action_count = int(source["executed_action_count"])
    source_total_actions = int(len(snapshot["executed_actions"]))
    remaining_actions = max(0, source_total_actions - source_action_count)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this comparison requires a CUDA GPU")
    policy_seed = int(snapshot["policy_seed"])
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    cfg = json.loads((args.svd_dir / "config.json").read_text())
    summary = torch.load(
        args.svd_dir / "svd_summary.pt", map_location="cpu", weights_only=False
    )
    n_blocks = int(summary["c_means"].shape[0])
    available_steps = [int(value) for value in cfg["selected_timesteps"]]
    blocks = parse_int_selection(args.blocks, list(range(n_blocks)), "blocks")
    steps = parse_int_selection(args.steps, available_steps, "steps")

    import DiT4DiT.model.framework.DiT4DiT  # noqa: F401
    from DiT4DiT.model.framework.base_framework import baseframework
    from DiT4DiT.model.framework.share_tools import read_mode_config
    from run_lqr_dit4dit_noised import _import_shared

    *_, run_denoising_loop = _import_shared()
    print(f"[model] loading {args.ckpt_path} on {device}", flush=True)
    model = baseframework.from_pretrained(args.ckpt_path).to(device).eval()
    action_model = model.action_model
    action_dit = action_model.model
    _, norm_stats = read_mode_config(args.ckpt_path)
    action_stats = norm_stats[next(iter(norm_stats))]["action"]
    action_high = np.asarray(action_stats["max"], dtype=np.float32)
    action_low = np.asarray(action_stats["min"], dtype=np.float32)
    action_mask = np.asarray(
        action_stats.get("mask", np.ones(len(action_high), dtype=bool)), dtype=bool
    )
    state_dim = int(action_model.config.state_dim)
    action_horizon = int(action_model.action_horizon)
    if remaining_actions == 0:
        remaining_actions = action_horizon
    max_inferences = (
        args.max_inferences
        if args.max_inferences > 0
        else int(math.ceil(remaining_actions / action_horizon))
    )

    directions, raw_direction_norms = load_unit_directions(
        args.svd_dir, cfg, summary, blocks, steps, device
    )
    runtime = ActAddRuntime(
        action_dit=action_dit,
        directions=directions,
        denoise_t_start=int(cfg["denoise_t_start"]),
        denoise_t_end=int(cfg["denoise_t_end"]),
        alpha=args.alpha,
    )
    print(
        f"[actadd] alpha={args.alpha:g}, blocks={blocks}, steps={steps}, "
        f"sites={len(directions)}", flush=True,
    )

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite_name = str(snapshot["suite"])
    task_id = int(snapshot["task_id"])
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        horizon=max(remaining_actions, max_inferences * action_horizon) + 20,
    )
    env.seed(int(snapshot["env_seed"]))

    noise_sigma = float(snapshot["noise_sigma"])
    prompt = str(snapshot["prompt"])
    saved_observation = clone_observation(source["observation"])
    saved_model_image = np.asarray(source["model_input"]["model_image"]).copy()
    saved_sim_state = np.asarray(source["sim_state"], dtype=np.float64).copy()

    def policy(obs: dict, rng: np.random.Generator, steered: bool):
        primary_raw = np.ascontiguousarray(
            obs["agentview_image"][::-1, ::-1]
        ).astype(np.float32)
        wrist_raw = np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1, ::-1]
        ).astype(np.float32)
        primary_raw = add_gaussian_noise(primary_raw, rng, noise_sigma)
        wrist_raw = add_gaussian_noise(wrist_raw, rng, noise_sigma)
        primary = cv2.resize(primary_raw, (224, 224), interpolation=cv2.INTER_AREA)
        wrist = cv2.resize(wrist_raw, (224, 224), interpolation=cv2.INTER_AREA)
        model_image = np.concatenate([primary, wrist], axis=1)

        eef_pos = obs["robot0_eef_pos"].astype(np.float32)
        quaternion = obs["robot0_eef_quat"].astype(np.float32).copy()
        quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
        denominator = np.sqrt(max(0.0, 1.0 - quaternion[3] ** 2))
        axis_angle = (
            np.zeros(3, dtype=np.float32)
            if math.isclose(denominator, 0.0)
            else (
                quaternion[:3] * 2.0 * math.acos(float(quaternion[3])) / denominator
            ).astype(np.float32)
        )
        gripper = obs["robot0_gripper_qpos"].astype(np.float32)
        proprio = np.concatenate([eef_pos, axis_angle, gripper])
        sine, cosine = np.sin(proprio[None]), np.cos(proprio[None])
        state = np.stack([sine, cosine], axis=-1).reshape(1, -1).astype(np.float32)
        if state.shape[-1] < state_dim:
            state = np.pad(state, ((0, 0), (0, state_dim - state.shape[-1])))

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                backbone_inputs = model.backbone_interface.build_cosmos_inputs(
                    images=[[model_image]], instructions=[prompt]
                )
                backbone_output = model.backbone_interface(
                    **backbone_inputs,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )
                visual_language = backbone_output.hidden_states[-1]
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(
                device=device, dtype=visual_language.dtype
            )
            with torch.autocast("cuda", dtype=torch.float32):
                runtime.enabled = bool(steered)
                runtime.reset_chunk()
                normalized = run_denoising_loop(
                    action_model,
                    visual_language,
                    state_tensor,
                    seed=policy_seed,
                    num_steps=int(cfg["sampling_steps"]),
                )

        normalized_np = normalized[0].cpu().numpy()
        normalized7 = np.clip(normalized_np[:, :7], -1.0, 1.0)
        raw = np.where(
            action_mask,
            0.5 * (normalized7 + 1.0) * (action_high - action_low) + action_low,
            normalized7,
        )
        raw[:, 6] = np.where(normalized_np[:, 6] < 0.5, 1.0, -1.0)
        return raw.astype(np.float32), model_image

    def run_branch(label: str, steered: bool) -> BranchResult:
        env.reset()
        restored_observation = env.set_init_state(saved_sim_state)
        restored_eef_error = float(np.max(np.abs(
            np.asarray(restored_observation["robot0_eef_pos"])
            - np.asarray(saved_observation["robot0_eef_pos"])
        )))
        print(f"[{label}] restored eef max error={restored_eef_error:.3e}", flush=True)
        obs = clone_observation(saved_observation)
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(rng_state)

        frames = [obs["agentview_image"].copy()]
        inference_ids = [args.inference_index]
        eef_positions = [obs["robot0_eef_pos"].astype(np.float64).copy()]
        noisy_images: list[np.ndarray] = []
        actions_executed = []
        success = False
        first_equal = False
        first_max_error = -1
        inference_count = 0

        while inference_count < max_inferences and len(actions_executed) < remaining_actions:
            inference_index = args.inference_index + inference_count
            actions, model_image = policy(obs, rng, steered)
            if inference_count == 0:
                difference = np.abs(
                    model_image.astype(np.int16) - saved_model_image.astype(np.int16)
                )
                first_equal = bool(np.array_equal(model_image, saved_model_image))
                first_max_error = int(difference.max())
                print(
                    f"[{label}] first noisy model input exact={first_equal} "
                    f"max_abs_error={first_max_error}", flush=True,
                )
            if not noisy_images:
                noisy_images.append(model_image.copy())
            for action in actions:
                if len(actions_executed) >= remaining_actions:
                    break
                actions_executed.append(action.copy())
                obs, _, done, _ = env.step(action.tolist())
                frames.append(obs["agentview_image"].copy())
                inference_ids.append(inference_index)
                noisy_images.append(model_image.copy())
                eef_positions.append(obs["robot0_eef_pos"].astype(np.float64).copy())
                if done:
                    success = True
                    break
            inference_count += 1
            if success:
                break

        return BranchResult(
            label=label,
            success=success,
            frames=frames,
            inference_ids=inference_ids,
            noisy_images=noisy_images,
            eef_positions=np.stack(eef_positions),
            executed_actions=(
                np.stack(actions_executed).astype(np.float32)
                if actions_executed else np.empty((0, 7), dtype=np.float32)
            ),
            inference_count=inference_count,
            first_model_input_equal=first_equal,
            first_model_input_max_abs_error=first_max_error,
        )

    try:
        raw_result = run_branch("raw", steered=False)
        steered_result = run_branch("steered", steered=True)
    finally:
        runtime.close()
        env.close()

    video_path = args.output_dir / (
        f"gaussian_noise_ep{int(snapshot['episode']):02d}_inference"
        f"{args.inference_index:03d}_raw_vs_actadd.mp4"
    )
    write_comparison_video(
        raw_result, steered_result, video_path, args.video_fps, args.alpha
    )

    npz_path = video_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        raw_eef_positions=raw_result.eef_positions,
        steered_eef_positions=steered_result.eef_positions,
        raw_executed_actions=raw_result.executed_actions,
        steered_executed_actions=steered_result.executed_actions,
        raw_inference_ids=np.asarray(raw_result.inference_ids, dtype=np.int32),
        steered_inference_ids=np.asarray(steered_result.inference_ids, dtype=np.int32),
    )

    def branch_summary(result: BranchResult) -> dict:
        start = result.eef_positions[0]
        end = result.eef_positions[-1]
        delta = end - start
        return {
            "success": result.success,
            "inferences_executed": result.inference_count,
            "environment_actions_executed": int(len(result.executed_actions)),
            "start_eef_position_m": start,
            "end_eef_position_m": end,
            "eef_delta_xyz_m": delta,
            "eef_delta_xyz_cm": 100.0 * delta,
            "eef_start_to_end_distance_m": float(np.linalg.norm(delta)),
            "eef_start_to_end_distance_cm": float(100.0 * np.linalg.norm(delta)),
            "first_model_input_exactly_reproduced": result.first_model_input_equal,
            "first_model_input_max_abs_error_uint8": result.first_model_input_max_abs_error,
        }

    report = {
        "definition": (
            "Both branches restore the same saved MuJoCo state and exact observation; "
            "raw disables ActAdd and steered adds alpha times the unit reconstructed "
            "contrastive direction at every selected site. EEF displacement is straight-"
            "line start-to-end displacement, not trajectory arc length."
        ),
        "source_snapshot": str(args.snapshot.resolve()),
        "source_episode": int(snapshot["episode"]),
        "source_condition": snapshot["condition"],
        "start_inference_index": args.inference_index,
        "source_env_step": int(source["env_step"]),
        "noise_sigma_uint8": noise_sigma,
        "noise_seed": int(snapshot["noise_seed"]),
        "policy_seed": policy_seed,
        "actadd": {
            "alpha": args.alpha,
            "direction_normalization": "unit L2 norm independently at each block/step",
            "blocks": blocks,
            "denoising_steps": steps,
            "injection_sites": len(directions),
            "raw_contrastive_direction_norms": raw_direction_norms,
        },
        "comparison_horizon": {
            "source_remaining_actions": remaining_actions,
            "requested_max_inferences": max_inferences,
            "action_horizon": action_horizon,
        },
        "raw": branch_summary(raw_result),
        "steered": branch_summary(steered_result),
        "video": str(video_path.resolve()),
        "trajectories_npz": str(npz_path.resolve()),
    }
    report_path = video_path.with_suffix(".json")
    report_path.write_text(json.dumps(jsonable(report), indent=2))
    print(json.dumps(jsonable(report), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
