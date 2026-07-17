import numpy as np
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
import argparse
from libero.libero import benchmark
import time
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.utils import write_json
import os
import imageio
import cv2
import re
import hashlib

from scripts.lqr.perturbations import RandomCameraViewPerturbation, build_gripper_xyz_preset


_EPISODE_VIDEO_RE = re.compile(r"^(\d+)_(True|False)\.mp4$")
_OPEN_ENVS = []


def _task_dir_name(task_idx, prompt, max_prompt_chars=64):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(prompt)).strip("_")
    slug = slug[:max_prompt_chars].strip("_") or "task"
    digest = hashlib.sha1(str(prompt).encode("utf-8")).hexdigest()[:8]
    return f"{int(task_idx)}_{slug}_{digest}"


def _apply_agentview_noise_to_obs(
    obs_dict,
    sigma,
    rng: np.random.Generator,
    apply_wrist=True,
):
    if sigma is None:
        return obs_dict
    sigma_f = float(sigma)
    if sigma_f <= 0:
        return obs_dict
    out = {
        "observation.images.agentview_rgb": obs_dict["observation.images.agentview_rgb"].copy(),
        "observation.images.eye_in_hand_rgb": obs_dict["observation.images.eye_in_hand_rgb"].copy(),
    }
    img = out["observation.images.agentview_rgb"]
    noise = rng.normal(loc=0.0, scale=sigma_f, size=img.shape).astype(np.float32)
    out["observation.images.agentview_rgb"] = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if apply_wrist:
        wrist = out["observation.images.eye_in_hand_rgb"]
        wrist_noise = rng.normal(loc=0.0, scale=sigma_f, size=wrist.shape).astype(np.float32)
        out["observation.images.eye_in_hand_rgb"] = np.clip(wrist.astype(np.float32) + wrist_noise, 0, 255).astype(np.uint8)
    return out


def save_video(
    real_obs_list,
    save_path,
    fps=15,
    video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"],
    phase_labels=None,
):
    if not real_obs_list:
        print("❌ No real observation frames")
        return
    if phase_labels is not None and len(phase_labels) != len(real_obs_list):
        raise ValueError(
            f"phase_labels length ({len(phase_labels)}) must match "
            f"real_obs_list length ({len(real_obs_list)})"
        )

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = []
    for obs in real_obs_list:
        frame = np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        final_frames.append(frame)

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def _axis_angle_to_quat(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half_angle = angle_rad / 2.0
    return np.array(
        [
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
        ],
        dtype=np.float64,
    )


def _quat_multiply(quat_a, quat_b):
    aw, ax, ay, az = quat_a
    bw, bx, by, bz = quat_b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _normalize(vec, eps=1e-8):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {vec.tolist()}")
    return vec / norm


def _rotate_vector(vec, axis, angle_rad):
    """
    Rodrigues rotation formula for rotating a 3D vector around a world axis.
    """
    vec = np.asarray(vec, dtype=np.float64)
    axis = _normalize(axis)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return vec * cos_a + np.cross(axis, vec) * sin_a + axis * np.dot(axis, vec) * (1.0 - cos_a)


def _rotmat_to_quat_wxyz(rotmat):
    """
    Convert a 3x3 rotation matrix to MuJoCo wxyz quaternion.
    """
    m = np.asarray(rotmat, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
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
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _lookat_quat_wxyz(cam_pos, target_pos, world_up=np.array([0.0, 0.0, 1.0], dtype=np.float64)):
    """
    Build camera quaternion so that the camera forward axis points to target_pos.
    MuJoCo camera looks along local -Z, with local +Y as up.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    forward = _normalize(target_pos - cam_pos)
    cam_z = -forward

    world_up = np.asarray(world_up, dtype=np.float64)
    cam_x = np.cross(world_up, cam_z)
    if np.linalg.norm(cam_x) < 1e-6:
        # Degenerate when forward is parallel to world_up
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cam_x = np.cross(alt_up, cam_z)
    cam_x = _normalize(cam_x)
    cam_y = _normalize(np.cross(cam_z, cam_x))
    rotmat = np.column_stack([cam_x, cam_y, cam_z])
    return _rotmat_to_quat_wxyz(rotmat)


def _find_body_pos_by_keywords(env_in, include_keywords):
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


def _infer_agentview_orbit_center_and_target(env_in):
    robot_base_pos, robot_name = _find_body_pos_by_keywords(env_in, ["robot0", "base"])
    if robot_base_pos is None:
        robot_base_pos, robot_name = _find_body_pos_by_keywords(env_in, ["robot0"])

    table_pos, table_name = _find_body_pos_by_keywords(env_in, ["table"])

    if robot_base_pos is not None and table_pos is not None:
        center = 0.5 * (robot_base_pos + table_pos)
        target = center.copy()
        target[2] = max(robot_base_pos[2], table_pos[2]) + 0.08
        reason = f"robot={robot_name}, table={table_name}"
        return center, target, reason
    if table_pos is not None:
        center = table_pos.copy()
        target = table_pos.copy()
        target[2] += 0.10
        reason = f"table={table_name}"
        return center, target, reason
    if robot_base_pos is not None:
        center = robot_base_pos.copy()
        target = robot_base_pos.copy()
        target[2] += 0.18
        reason = f"robot={robot_name}"
        return center, target, reason
    return None, None, "no robot/table body found"


def apply_agentview_camera_rotation(env_in, rotate_deg=None, rotate_axis="z"):
    if rotate_deg is None or rotate_deg == 0:
        return

    axis_by_name = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if rotate_axis not in axis_by_name:
        raise ValueError(f"agentview_camera_rotate_axis must be one of x/y/z, got {rotate_axis}")

    cam_id = env_in.sim.model.camera_name2id("agentview")
    old_cam_pos = np.asarray(env_in.sim.model.cam_pos[cam_id], dtype=np.float64).copy()
    old_quat = np.asarray(env_in.sim.model.cam_quat[cam_id], dtype=np.float64)
    center_pos, target_pos, anchor_reason = _infer_agentview_orbit_center_and_target(env_in)
    if center_pos is None or target_pos is None:
        delta_quat = _axis_angle_to_quat(axis_by_name[rotate_axis], np.deg2rad(rotate_deg))
        new_quat = _quat_multiply(delta_quat, old_quat)
        new_quat = new_quat / np.linalg.norm(new_quat)
        env_in.sim.model.cam_quat[cam_id] = new_quat
        env_in.sim.forward()
        print(
            f"Fallback self-rotation (anchor not found: {anchor_reason}): "
            f"{rotate_deg} deg around {rotate_axis}-axis, quat {old_quat.tolist()} -> {new_quat.tolist()}"
        )
        return

    orbit_axis = axis_by_name[rotate_axis]
    rel_vec = old_cam_pos - center_pos
    new_cam_pos = center_pos + _rotate_vector(rel_vec, orbit_axis, np.deg2rad(rotate_deg))
    new_quat = _lookat_quat_wxyz(new_cam_pos, target_pos)
    env_in.sim.model.cam_pos[cam_id] = new_cam_pos
    env_in.sim.model.cam_quat[cam_id] = new_quat
    env_in.sim.forward()
    print(
        f"Orbited agentview camera by {rotate_deg} deg around {rotate_axis}-axis "
        f"(anchor: {anchor_reason}). cam_pos {old_cam_pos.tolist()} -> {new_cam_pos.tolist()}, "
        f"cam_quat {old_quat.tolist()} -> {new_quat.tolist()}, lookat={target_pos.tolist()}"
    )


def init_single_env(env_in, init_state, init_perturb=None, episode_idx=0):
    env_in.reset()
    if init_perturb is None:
        obs = env_in.set_init_state(init_state)
    else:
        obs = init_perturb.set_init_state(env_in, init_state, episode_idx=episode_idx)
    for _ in range(10):
        obs, _, _, _ = env_in.step([0., 0., 0., 0., 0., 0., -1.])
    return obs


def _noise_rng_for_episode(seed_base, episode_idx):
    """Match the repo root's EpisodeNoise: rng(seed=noise_seed_base + episode_idx)."""
    return np.random.default_rng(seed=int(seed_base) + int(episode_idx))


def apply_eef_delta_preposition(
    env_in,
    raw_obs,
    eef_delta=None,
    max_steps=120,
    step_size=1.0,
    tolerance=0.01,
    video_obs_list=None,
    phase_labels=None,
):
    if eef_delta is None:
        return raw_obs
    if max_steps <= 0:
        raise ValueError(f"eef_preposition_steps must be positive, got {max_steps}")
    if step_size <= 0:
        raise ValueError(f"eef_step_size must be positive, got {step_size}")
    if tolerance <= 0:
        raise ValueError(f"eef_tolerance must be positive, got {tolerance}")

    start_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64)
    target_pos = start_pos + np.asarray(eef_delta, dtype=np.float64)
    if target_pos.shape != (3,):
        raise ValueError(f"eef_delta must have 3 values, got {eef_delta}")

    print(
        f"EEF preposition start={start_pos.tolist()} "
        f"target={target_pos.tolist()} delta={list(eef_delta)}"
    )
    final_error = float(np.linalg.norm(target_pos - start_pos))
    steps_used = 0
    obs = raw_obs
    for step_idx in range(max_steps):
        current_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        error = target_pos - current_pos
        dist = float(np.linalg.norm(error))
        final_error = dist
        if dist <= tolerance:
            break

        delta = error
        if dist > step_size:
            delta = error / dist * step_size

        action = np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, _, done, _ = env_in.step(action)
        steps_used = step_idx + 1
        if video_obs_list is not None:
            video_obs_list.append(_extract_obs(obs))
        if phase_labels is not None:
            phase_labels.append("preposition")
        if done:
            print("EEF preposition reached env done; continuing to inference from latest obs.")
            break

    final_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    final_error = float(np.linalg.norm(target_pos - final_pos))
    print(
        f"EEF preposition final={final_pos.tolist()} "
        f"error={final_error:.6f} steps={steps_used}"
    )
    return obs


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(
    model,
    libero_benchmark,
    task_idx,
    out_dir,
    episode_idx,
    prompt_override=None,
    agentview_noise_sigma=None,
    agentview_noise_seed_base=0,
    noise_apply_wrist=True,
    random_camera_pos_sigma=None,
    random_camera_rot_sigma_deg=8.0,
    random_camera_fov_sigma=5.0,
    random_camera_base_seed=42,
    random_camera_name="agentview",
    random_camera_enforce_visibility=True,
    random_camera_workspace_table_z=0.90,
    random_camera_workspace_visible_fraction=0.55,
    random_camera_visibility_margin_px=8,
    random_camera_image_size=128,
    random_camera_max_rejection_attempts=2000,
    gripper_xyz_preset=None,
    gripper_xyz_base_seed=42,
    close_env=False,
    fixed_first_obs=False,
):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    if prompt_override is not None:
        prompt = prompt_override
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    init_perturb = None
    if gripper_xyz_preset:
        init_perturb = build_gripper_xyz_preset(
            str(gripper_xyz_preset),
            base_seed=int(gripper_xyz_base_seed),
        )
    raw_obs = init_single_env(
        cur_env,
        init_states[episode_idx % init_states.shape[0]],
        init_perturb=init_perturb,
        episode_idx=episode_idx,
    )
    if random_camera_pos_sigma is not None:
        cam_perturb = RandomCameraViewPerturbation(
            pos_sigma=float(random_camera_pos_sigma),
            rot_sigma_rad=float(np.radians(float(random_camera_rot_sigma_deg))),
            fov_sigma=float(random_camera_fov_sigma),
            base_seed=int(random_camera_base_seed),
            camera_name=str(random_camera_name),
            enforce_visibility=bool(random_camera_enforce_visibility),
            workspace_table_z=float(random_camera_workspace_table_z),
            workspace_visible_fraction=float(random_camera_workspace_visible_fraction),
            visibility_margin_px=int(random_camera_visibility_margin_px),
            image_size=int(random_camera_image_size),
            max_rejection_attempts=int(random_camera_max_rejection_attempts),
            name_hint="cam_random_large",
        )
        cam_perturb.apply_to_env(cur_env, episode_idx=episode_idx)
        raw_obs, _, _, _ = cur_env.step([0.] * 7)
        print(f"Applied camera perturbation: {cam_perturb.manifest()} sample={getattr(cam_perturb, '_last_sample', {})}")
    full_obs_list = []
    phase_labels = []
    noise_rng_infer = _noise_rng_for_episode(agentview_noise_seed_base, episode_idx)
    noise_rng_cache = _noise_rng_for_episode(agentview_noise_seed_base + 1000003, episode_idx)
    first_obs = _extract_obs(raw_obs)
    current_obs = first_obs
    print(f"Prompt: {prompt}")
    ret = model.infer(dict(reset=True, prompt=prompt))

    done = False
    first = True
    while cur_env.env.timestep < 800:
        source_obs = first_obs if fixed_first_obs else current_obs
        infer_obs = _apply_agentview_noise_to_obs(
            source_obs,
            sigma=agentview_noise_sigma,
            rng=noise_rng_infer,
            apply_wrist=noise_apply_wrist,
        )
        ret = model.infer(dict(obs=infer_obs, prompt=prompt))
        action = ret['action']

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                observes, done = env_one_step(cur_env, ee_action)
                current_obs = observes
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    noisy_obs = _apply_agentview_noise_to_obs(
                        observes,
                        sigma=agentview_noise_sigma,
                        rng=noise_rng_cache,
                        apply_wrist=noise_apply_wrist,
                    )
                    full_obs_list.append(noisy_obs)
                    phase_labels.append("inference")
                    key_frame_list.append(noisy_obs)

            if done:
                break

        first = False

        if done:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    out_file = Path(out_dir) / libero_benchmark / _task_dir_name(task_idx, prompt) / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"],
        phase_labels=phase_labels,
    )

    if close_env:
        cur_env.close()
    else:
        # robosuite/EGL cleanup can abort the Python process on some Slurm
        # nodes. Keep envs alive for the short eval client lifetime instead.
        _OPEN_ENVS.append(cur_env)
    return done


def run(
    libero_benchmark,
    port,
    out_dir,
    test_num,
    task_range=None,
    prompt=None,
    agentview_noise_sigma=None,
    agentview_noise_seed_base=0,
    noise_apply_wrist=True,
    random_camera_pos_sigma=None,
    random_camera_rot_sigma_deg=8.0,
    random_camera_fov_sigma=5.0,
    random_camera_base_seed=42,
    random_camera_name="agentview",
    random_camera_enforce_visibility=True,
    random_camera_workspace_table_z=0.90,
    random_camera_workspace_visible_fraction=0.55,
    random_camera_visibility_margin_px=8,
    random_camera_image_size=128,
    random_camera_max_rejection_attempts=2000,
    gripper_xyz_preset=None,
    gripper_xyz_base_seed=42,
    resume=False,
    close_envs=False,
    max_new_episodes=None,
    fixed_first_obs=False,
):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    model = WebsocketClientPolicy(port=port)

    for task_idx in progress_bar:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        task_prompt = benchmark_instance.get_task(task_idx).language
        if prompt is not None:
            task_prompt = prompt

        completed = {}
        if resume:
            video_dir = Path(out_dir) / libero_benchmark / _task_dir_name(task_idx, task_prompt)
            if video_dir.is_dir():
                for file_name in os.listdir(video_dir):
                    match = _EPISODE_VIDEO_RE.match(file_name)
                    if not match:
                        continue
                    ep_idx = int(match.group(1))
                    if ep_idx < test_num:
                        completed[ep_idx] = match.group(2) == "True"

        succ_num = float(sum(1 for done in completed.values() if done))
        completed_num = len(completed)
        episode_list = [ep_idx for ep_idx in range(test_num) if ep_idx not in completed]
        if max_new_episodes is not None and int(max_new_episodes) > 0:
            episode_list = episode_list[: int(max_new_episodes)]

        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(
                model,
                libero_benchmark,
                task_idx,
                out_dir,
                episode_idx,
                prompt,
                agentview_noise_sigma=agentview_noise_sigma,
                agentview_noise_seed_base=agentview_noise_seed_base,
                noise_apply_wrist=noise_apply_wrist,
                random_camera_pos_sigma=random_camera_pos_sigma,
                random_camera_rot_sigma_deg=random_camera_rot_sigma_deg,
                random_camera_fov_sigma=random_camera_fov_sigma,
                random_camera_base_seed=random_camera_base_seed,
                random_camera_name=random_camera_name,
                random_camera_enforce_visibility=random_camera_enforce_visibility,
                random_camera_workspace_table_z=random_camera_workspace_table_z,
                random_camera_workspace_visible_fraction=random_camera_workspace_visible_fraction,
                random_camera_visibility_margin_px=random_camera_visibility_margin_px,
                random_camera_image_size=random_camera_image_size,
                random_camera_max_rejection_attempts=random_camera_max_rejection_attempts,
                gripper_xyz_preset=gripper_xyz_preset,
                gripper_xyz_base_seed=gripper_xyz_base_seed,
                close_env=close_envs,
                fixed_first_obs=fixed_first_obs,
            )
            succ_num += res_i
            completed_num += 1
            succ_rate = succ_num / completed_num
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {completed_num}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "succ_num": succ_num,
                "total_num": float(completed_num),
                "succ_rate": succ_rate,
                }, out_file
            )

        if resume and not episode_list:
            succ_rate = succ_num / completed_num if completed_num else 0.0
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {completed_num}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "succ_num": succ_num,
                "total_num": float(completed_num),
                "succ_rate": succ_rate,
                }, out_file
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for the task (overrides benchmark prompt)",
    )
    parser.add_argument(
        "--agentview-noise-sigma",
        type=float,
        default=None,
        help="Additive gaussian sigma (pixel units) on third-person agentview image before policy inference.",
    )
    parser.add_argument(
        "--agentview-noise-seed-base",
        type=int,
        default=0,
        help="Per-episode gaussian noise seed base.",
    )
    parser.add_argument(
        "--noise-apply-wrist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply gaussian noise to the wrist image as well as agentview. Default matches the repo root's noise_extreme.",
    )
    parser.add_argument("--random-camera-pos-sigma", type=float, default=None)
    parser.add_argument("--random-camera-rot-sigma-deg", type=float, default=8.0)
    parser.add_argument("--random-camera-fov-sigma", type=float, default=5.0)
    parser.add_argument("--random-camera-base-seed", type=int, default=42)
    parser.add_argument("--random-camera-name", type=str, default="agentview")
    parser.add_argument("--random-camera-workspace-table-z", type=float, default=0.90)
    parser.add_argument("--random-camera-workspace-visible-fraction", type=float, default=0.55)
    parser.add_argument("--random-camera-visibility-margin-px", type=int, default=8)
    parser.add_argument("--random-camera-image-size", type=int, default=128)
    parser.add_argument("--random-camera-max-rejection-attempts", type=int, default=2000)
    parser.add_argument("--disable-random-camera-visibility", action="store_true")
    parser.add_argument(
        "--gripper-xyz-preset",
        type=str,
        default=None,
        help="Repo root's gripper XYZ init perturbation preset, e.g. xyz_random_xlarge_3.",
    )
    parser.add_argument("--gripper-xyz-base-seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing episode videos in out-dir and continue until --test-num episodes exist.",
    )
    parser.add_argument(
        "--close-envs",
        action="store_true",
        help="Close robosuite envs after each episode. Disabled by default to avoid native EGL aborts.",
    )
    parser.add_argument(
        "--max-new-episodes",
        type=int,
        default=None,
        help="With --resume, run at most this many missing episodes before exiting.",
    )
    parser.add_argument(
        "--fixed-first-obs",
        action="store_true",
        help="Use the legacy eval protocol that feeds the initial observation to every main infer chunk.",
    )
    args = parser.parse_args()
    args.random_camera_enforce_visibility = not bool(args.disable_random_camera_visibility)
    delattr(args, "disable_random_camera_visibility")
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
