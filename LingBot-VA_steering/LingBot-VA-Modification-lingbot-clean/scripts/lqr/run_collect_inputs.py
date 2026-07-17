import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.distributed as dist
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from wan_va.utils import init_logger

from scripts.lqr.common import maybe_load_yaml, parse_int_list


@dataclass
class _CallCtx:
    action_mode: bool
    step_idx: int


class LingbotActivationTracer:
    def __init__(self, layers: List[int], selected_timesteps: List[int], mode: str) -> None:
        self.layers = set(layers)
        self.selected_timesteps = set(selected_timesteps)
        self.mode = mode
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.current: Optional[_CallCtx] = None
        self.captured: Dict[Tuple[int, int], torch.Tensor] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def reset_chunk(self) -> None:
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.current = None
        self.captured = {}

    def begin_call(self, action_mode: bool) -> None:
        if action_mode:
            step_idx = self.action_step_idx
            self.action_step_idx += 1
        else:
            step_idx = self.video_step_idx
            self.video_step_idx += 1
        self.current = _CallCtx(action_mode=bool(action_mode), step_idx=step_idx)

    def end_call(self) -> None:
        self.current = None

    def _mode_allow(self, action_mode: bool) -> bool:
        if self.mode == "both":
            return True
        if self.mode == "action":
            return action_mode
        return not action_mode

    def register_hooks(self, transformer_model: torch.nn.Module) -> None:
        for idx, block in enumerate(transformer_model.blocks):
            if idx not in self.layers:
                continue
            self._hook_handles.append(block.register_forward_hook(self._hook_fn(idx)))

    def _hook_fn(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self.current is None:
                return output
            if not self._mode_allow(self.current.action_mode):
                return output
            if self.current.step_idx not in self.selected_timesteps:
                return output
            self.captured[(layer_idx, self.current.step_idx)] = output[0].detach().reshape(-1).float().cpu()
            return output

        return hook

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def _ensure_dist_env() -> None:
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(12355 + (os.getpid() % 1000)))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def _build_server(config_name: str):
    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.wan_va_server import VA_Server

    _ensure_dist_env()
    if not dist.is_initialized():
        init_distributed(world_size=1, local_rank=0, rank=0)
    cfg = copy.deepcopy(VA_CONFIGS[config_name])
    cfg.save_root = os.environ.get("LQR_SERVER_SAVE_ROOT", "")
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    return VA_Server(cfg)


def _extract_obs(raw_obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    agentview = np.ascontiguousarray(raw_obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1])
    return agentview, eye_in_hand


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"cannot normalize near-zero vector: {v.tolist()}")
    return v / n


def _axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    h = angle_rad / 2.0
    return np.array([np.cos(h), axis[0] * np.sin(h), axis[1] * np.sin(h), axis[2] * np.sin(h)], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _rotate_vector(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    axis = _normalize(axis)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


def _rotmat_to_quat_wxyz(rotmat: np.ndarray) -> np.ndarray:
    m = np.asarray(rotmat, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _find_body_pos_by_keywords(env_in: OffScreenRenderEnv, include_keywords: List[str]):
    model = env_in.sim.model
    data = env_in.sim.data
    include_keywords = tuple(k.lower() for k in include_keywords)
    candidates = []
    for body_id in range(model.nbody):
        body_name = model.body_id2name(body_id)
        if not body_name:
            continue
        body_name_l = body_name.lower()
        if all(k in body_name_l for k in include_keywords):
            candidates.append((len(body_name), body_id, body_name))
    if not candidates:
        return None, None
    _, body_id, body_name = sorted(candidates, key=lambda x: (x[0], x[2]))[0]
    return np.asarray(data.body_xpos[body_id], dtype=np.float64).copy(), body_name


def _lookat_quat_wxyz(cam_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    forward = _normalize(target_pos - cam_pos)
    cam_z = -forward
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cam_x = np.cross(world_up, cam_z)
    if np.linalg.norm(cam_x) < 1e-6:
        cam_x = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), cam_z)
    cam_x = _normalize(cam_x)
    cam_y = _normalize(np.cross(cam_z, cam_x))
    rotmat = np.column_stack([cam_x, cam_y, cam_z])
    return _rotmat_to_quat_wxyz(rotmat)


def _infer_agentview_orbit_center_and_target(env_in: OffScreenRenderEnv):
    robot_base_pos, _ = _find_body_pos_by_keywords(env_in, ["robot0", "base"])
    if robot_base_pos is None:
        robot_base_pos, _ = _find_body_pos_by_keywords(env_in, ["robot0"])
    table_pos, _ = _find_body_pos_by_keywords(env_in, ["table"])
    if robot_base_pos is not None and table_pos is not None:
        center = 0.5 * (robot_base_pos + table_pos)
        target = center.copy()
        target[2] = max(robot_base_pos[2], table_pos[2]) + 0.08
        return center, target
    if table_pos is not None:
        center = table_pos.copy()
        target = table_pos.copy()
        target[2] += 0.10
        return center, target
    if robot_base_pos is not None:
        center = robot_base_pos.copy()
        target = robot_base_pos.copy()
        target[2] += 0.18
        return center, target
    return None, None


def _apply_agentview_camera_rotation(env_in: OffScreenRenderEnv, rotate_deg: Optional[float], rotate_axis: str = "z") -> None:
    if rotate_deg in (None, 0):
        return
    axis_by_name = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if rotate_axis not in axis_by_name:
        raise ValueError(f"camera axis must be x/y/z, got {rotate_axis}")
    axis = axis_by_name[rotate_axis]
    cam_id = env_in.sim.model.camera_name2id("agentview")
    old_cam_pos = np.asarray(env_in.sim.model.cam_pos[cam_id], dtype=np.float64).copy()
    old_quat = np.asarray(env_in.sim.model.cam_quat[cam_id], dtype=np.float64)
    center, target = _infer_agentview_orbit_center_and_target(env_in)
    if center is None or target is None:
        dq = _axis_angle_to_quat(axis, np.deg2rad(float(rotate_deg)))
        new_q = _quat_multiply(dq, old_quat)
        env_in.sim.model.cam_quat[cam_id] = new_q / np.linalg.norm(new_q)
        env_in.sim.forward()
        return
    rel_vec = old_cam_pos - center
    new_cam_pos = center + _rotate_vector(rel_vec, axis, np.deg2rad(float(rotate_deg)))
    env_in.sim.model.cam_pos[cam_id] = new_cam_pos
    env_in.sim.model.cam_quat[cam_id] = _lookat_quat_wxyz(new_cam_pos, target)
    env_in.sim.forward()


def _apply_eef_delta_preposition(
    env_in: OffScreenRenderEnv,
    raw_obs: Dict[str, Any],
    cam_keys: Optional[List[str]] = None,
    video_frames: Optional[List[np.ndarray]] = None,
    eef_delta: Optional[List[float]] = None,
    max_steps: int = 80,
    step_size: float = 0.01,
    tolerance: float = 0.01,
):
    if eef_delta is None:
        return raw_obs
    start_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64)
    target_pos = start_pos + np.asarray(eef_delta, dtype=np.float64)
    obs = raw_obs
    for _ in range(int(max_steps)):
        cur_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        err = target_pos - cur_pos
        dist = float(np.linalg.norm(err))
        if dist <= float(tolerance):
            break
        delta = err if dist <= float(step_size) else err / dist * float(step_size)
        action = np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, _, done, _ = env_in.step(action)
        if video_frames is not None and cam_keys is not None:
            obs_view = _obs_payload_from_raw(obs, cam_keys)["obs"][0]
            video_frames.append(_compose_video_frame(obs_view[cam_keys[0]], obs_view[cam_keys[1]]))
        if done:
            break
    return obs


def _construct_env(env_args: Dict[str, Any]) -> OffScreenRenderEnv:
    for _ in range(5):
        try:
            return OffScreenRenderEnv(**env_args)
        except Exception:
            continue
    raise RuntimeError("failed to construct OffScreenRenderEnv")


def _obs_payload_from_raw(raw_obs: Dict[str, Any], cam_keys: List[str]) -> Dict[str, List[Dict[str, np.ndarray]]]:
    agent, eye = _extract_obs(raw_obs)
    return {"obs": [{cam_keys[0]: agent, cam_keys[1]: eye}]}


def _apply_agentview_noise(
    agent_img: np.ndarray,
    sigma: Optional[float],
    rng: Optional[np.random.Generator],
) -> np.ndarray:
    if sigma is None:
        return agent_img
    sigma_f = float(sigma)
    if sigma_f <= 0:
        return agent_img
    if rng is None:
        rng = np.random.default_rng(seed=0)
    noise = rng.normal(loc=0.0, scale=sigma_f, size=agent_img.shape).astype(np.float32)
    return np.clip(agent_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _apply_agentview_noise_to_obs(
    obs_dict: Dict[str, np.ndarray],
    cam_keys: List[str],
    sigma: Optional[float],
    rng: Optional[np.random.Generator],
) -> Dict[str, np.ndarray]:
    if sigma is None or float(sigma) <= 0:
        return obs_dict
    if rng is None:
        rng = np.random.default_rng(seed=0)
    out = {cam_keys[0]: obs_dict[cam_keys[0]].copy(), cam_keys[1]: obs_dict[cam_keys[1]].copy()}
    out[cam_keys[0]] = _apply_agentview_noise(out[cam_keys[0]], sigma=sigma, rng=rng)
    return out


def _obs_payload_from_raw_with_perturb(
    raw_obs: Dict[str, Any],
    cam_keys: List[str],
    image_noise_sigma: Optional[float],
    rng: Optional[np.random.Generator],
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    agent, eye = _extract_obs(raw_obs)
    agent = _apply_agentview_noise(agent, sigma=image_noise_sigma, rng=rng)
    return {"obs": [{cam_keys[0]: agent, cam_keys[1]: eye}]}


def _compose_video_frame(agent_img: np.ndarray, eye_img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([agent_img, eye_img], axis=1))


def _write_video_mp4(frames: List[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    final_frames: List[np.ndarray] = []
    for frame in frames:
        if frame.shape[:2] != (h, w):
            raise ValueError(f"inconsistent frame shape in video frames: expected {(h, w)}, got {frame.shape[:2]}")
        final_frames.append(frame.astype(np.uint8, copy=False))
    imageio.mimsave(str(path), final_frames, fps=int(fps))


def _variant_list(spec_path: Optional[Path]) -> List[Dict[str, Any]]:
    if spec_path is None:
        return [{"name": "nominal"}]
    spec = maybe_load_yaml(str(spec_path))
    variants = spec.get("variants", [])
    if not variants:
        raise ValueError("perturb spec must contain non-empty `variants`")
    out = []
    for idx, v in enumerate(variants):
        if "name" not in v:
            v["name"] = f"variant_{idx}"
        out.append(v)
    return out


def _run_one_trajectory(
    server,
    tracer: LingbotActivationTracer,
    env: OffScreenRenderEnv,
    init_state: np.ndarray,
    prompt: str,
    cam_keys: List[str],
    top_k: int,
    variant: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
    save_video: bool = True,
    max_env_steps: int = 800,
):
    env.reset()
    raw_obs = env.set_init_state(init_state)
    for _ in range(5):
        raw_obs, _, _, _ = env.step([0.0] * 7)
    _apply_agentview_camera_rotation(
        env,
        rotate_deg=variant.get("camera_rotate_deg"),
        rotate_axis=str(variant.get("camera_axis", "z")),
    )
    if variant.get("camera_rotate_deg") not in (None, 0):
        raw_obs, _, _, _ = env.step([0.0] * 7)

    server._reset(prompt=prompt)
    first_obs = _obs_payload_from_raw(raw_obs, cam_keys)["obs"][0]
    done = False
    infer_idx = 0
    first = True
    captures: List[Dict[str, Any]] = []
    video_frames: List[np.ndarray] = []
    raw_obs = _apply_eef_delta_preposition(
        env,
        raw_obs,
        cam_keys=cam_keys if save_video else None,
        video_frames=video_frames if save_video else None,
        eef_delta=variant.get("eef_delta"),
        max_steps=int(variant.get("eef_preposition_steps", 80)),
        step_size=float(variant.get("eef_step_size", 0.01)),
        tolerance=float(variant.get("eef_tolerance", 0.01)),
    )
    first_obs = _obs_payload_from_raw(raw_obs, cam_keys)["obs"][0]

    while env.env.timestep < max_env_steps:
        infer_obs = _apply_agentview_noise_to_obs(
            first_obs,
            cam_keys,
            sigma=variant.get("image_noise_sigma"),
            rng=rng,
        )
        obs_payload = {"obs": [infer_obs]}
        frame_st_id = int(server.frame_st_id)
        tracer.reset_chunk()
        actions, _ = server._infer(obs_payload, frame_st_id=frame_st_id)
        if infer_idx < top_k:
            captures.append(
                {
                    "inference_idx_in_traj": infer_idx,
                    "frame_st_id": frame_st_id,
                    "activations": dict(tracer.captured),
                }
            )

        key_frame_list: List[Dict[str, np.ndarray]] = []
        assert actions.shape[2] % 4 == 0, f"Unexpected action horizon: {actions.shape}"
        action_per_frame = int(actions.shape[2] // 4)
        start_idx = 1 if first else 0
        for i in range(start_idx, actions.shape[1]):
            for j in range(actions.shape[2]):
                ee_action = actions[:, i, j]
                raw_obs, _, done, _ = env.step(ee_action)
                if done:
                    break
                if (j + 1) % action_per_frame == 0:
                    key_frame_obs = _obs_payload_from_raw(raw_obs, cam_keys)["obs"][0]
                    key_frame_list.append(
                        _apply_agentview_noise_to_obs(
                            key_frame_obs,
                            cam_keys=cam_keys,
                            sigma=variant.get("image_noise_sigma"),
                            rng=rng,
                        )
                    )
                    if save_video:
                        video_frames.append(_compose_video_frame(key_frame_obs[cam_keys[0]], key_frame_obs[cam_keys[1]]))
            if done:
                break
        first = False
        infer_idx += 1
        if done:
            break
        if key_frame_list:
            server._compute_kv_cache({"obs": key_frame_list, "state": actions})

    return bool(done), infer_idx, captures, video_frames


def main() -> None:
    # Ensure logger handlers are installed so server-side VRAM logs are visible.
    init_logger()
    parser = argparse.ArgumentParser(description="Collect trajectory-topK activation records for LQR contrastive pairs.")
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--libero-benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--top-k-inference-per-traj", type=int, default=10)
    parser.add_argument("--selected-timesteps", type=str, default="0,10,20,30,40")
    parser.add_argument("--layers", type=str, default="", help="Comma list, empty means all layers.")
    parser.add_argument("--mode", choices=["video", "action", "both"], default="action")
    parser.add_argument("--perturb-spec", type=Path, required=True)
    parser.add_argument("--target-variants", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--disable-video", action="store_true", help="Disable trajectory video generation in collect stage.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.out_dir / "trajectory_records"
    records_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    bench_cls = benchmark.get_benchmark_dict()[args.libero_benchmark]
    bench = bench_cls()
    task = bench.get_task(args.task_id)
    task_lang = task.language
    init_states = bench.get_task_init_states(args.task_id)
    env_args = {
        "bddl_file_name": bench.get_task_bddl_file_path(args.task_id),
        "camera_heights": 128,
        "camera_widths": 128,
    }

    server = _build_server(args.config_name)
    # `_configure_model` only freezes the transformer; the VAE and text encoder are
    # loaded via HF `from_pretrained` with default `requires_grad=True`. When this
    # script bypasses `VA_Server.infer` (which is decorated with `@torch.no_grad()`)
    # and calls `_infer` / `_compute_kv_cache` directly, the VAE forward inside
    # `_encode_obs` (which sits outside the inner `with torch.no_grad():` blocks)
    # builds an autograd graph. The streaming VAE's `feat_cache` and the persisted
    # `self.init_latent` then keep that graph alive across chunks, leaking VAE
    # activations until OOM. Force everything to eval/no-grad to mirror the
    # behaviour of the production server path.
    server.vae.eval().requires_grad_(False)
    server.text_encoder.eval().requires_grad_(False)
    if getattr(server, "streaming_vae_half", None) is not None:
        server.streaming_vae_half.vae.eval().requires_grad_(False)
    transformer = server.transformer
    n_layers = len(transformer.blocks)
    layers = parse_int_list(args.layers) if args.layers.strip() else list(range(n_layers))
    selected_timesteps = parse_int_list(args.selected_timesteps)
    tracer = LingbotActivationTracer(layers=layers, selected_timesteps=selected_timesteps, mode=args.mode)
    tracer.register_hooks(transformer)

    original_forward = transformer.forward

    def patched_forward(*f_args, **f_kwargs):
        tracer.begin_call(action_mode=bool(f_kwargs.get("action_mode", False)))
        try:
            return original_forward(*f_args, **f_kwargs)
        finally:
            tracer.end_call()

    transformer.forward = patched_forward

    variants = _variant_list(args.perturb_spec)
    if args.target_variants:
        keep = {x.strip() for x in args.target_variants.split(",") if x.strip()}
        variants = [v for v in variants if str(v["name"]) in keep]
    if not variants:
        raise ValueError("No variants selected for collection.")

    cam_keys = list(server.job_config.obs_cam_keys)
    rows: List[Dict[str, Any]] = []
    save_video = not bool(args.disable_video)
    try:
        for variant in variants:
            variant_name = str(variant["name"])
            env = _construct_env(env_args)
            try:
                for episode_idx in range(int(args.num_episodes)):
                    init_state = init_states[episode_idx % init_states.shape[0]]
                    rng_seed = int(args.seed) + episode_idx
                    rng = np.random.default_rng(seed=rng_seed)
                    success, infer_calls, captures, video_frames = _run_one_trajectory(
                        server=server,
                        tracer=tracer,
                        env=env,
                        init_state=init_state,
                        prompt=task_lang,
                        cam_keys=cam_keys,
                        top_k=int(args.top_k_inference_per_traj),
                        variant=variant,
                        rng=rng,
                        save_video=save_video,
                    )
                    rec_name = f"{variant_name}__ep{episode_idx:04d}.pt"
                    rec_path = records_dir / rec_name
                    video_path: Optional[Path] = None
                    if save_video and video_frames:
                        safe_variant = variant_name.replace("/", "_")
                        variant_video_dir = videos_dir / safe_variant
                        variant_video_dir.mkdir(parents=True, exist_ok=True)
                        outcome = "success" if success else "fail"
                        video_path = variant_video_dir / f"ep{episode_idx:04d}_{outcome}.mp4"
                        _write_video_mp4(video_frames, video_path, fps=int(args.video_fps))
                    torch.save(
                        {
                            "variant_name": variant_name,
                            "is_nominal": bool(variant_name == "nominal"),
                            "task_id": int(args.task_id),
                            "episode_idx": int(episode_idx),
                            "trajectory_success": bool(success),
                            "num_infer_calls": int(infer_calls),
                            "captures": captures,
                            "selected_timesteps": selected_timesteps,
                            "layers": layers,
                            "mode": args.mode,
                            "prompt": task_lang,
                            "video_path": str(video_path.resolve()) if video_path is not None else None,
                        },
                        rec_path,
                    )
                    rows.append(
                        {
                            "variant_name": variant_name,
                            "is_nominal": bool(variant_name == "nominal"),
                            "task_id": int(args.task_id),
                            "episode_idx": int(episode_idx),
                            "trajectory_success": bool(success),
                            "num_infer_calls": int(infer_calls),
                            "num_captured": int(len(captures)),
                            "path": str(rec_path.resolve()),
                            "video_path": str(video_path.resolve()) if video_path is not None else None,
                        }
                    )
                    print(
                        f"[collect] variant={variant_name} ep={episode_idx} success={success} "
                        f"infer_calls={infer_calls} captured={len(captures)} video={'yes' if video_path else 'no'}"
                    )
            finally:
                env.close()
    finally:
        tracer.close()
        transformer.forward = original_forward

    summary = {
        "config_name": args.config_name,
        "libero_benchmark": args.libero_benchmark,
        "task_id": int(args.task_id),
        "task_language": task_lang,
        "num_episodes": int(args.num_episodes),
        "top_k_inference_per_traj": int(args.top_k_inference_per_traj),
        "selected_timesteps": selected_timesteps,
        "layers": layers,
        "mode": args.mode,
        "video_fps": int(args.video_fps),
        "video_enabled": bool(save_video),
        "variants": variants,
        "records": rows,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[collect] wrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
