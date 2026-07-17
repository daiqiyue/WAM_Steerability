import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from scripts.lqr.common import maybe_load_yaml
from scripts.lqr.perturbations import (
    AGENTVIEW_KEY,
    WRIST_KEY,
    StressTest,
    build_perturbation,
)
from scripts.lqr.run_collect_inputs import _build_server, _construct_env, _compose_video_frame


def _extract_obs(raw_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {
        AGENTVIEW_KEY: np.ascontiguousarray(raw_obs[AGENTVIEW_KEY][::-1]),
        WRIST_KEY: np.ascontiguousarray(raw_obs[WRIST_KEY][::-1]),
    }


def _obs_payload(raw_obs: Dict[str, Any], cam_keys: List[str], perturb: Optional[StressTest] = None, episode_idx: int = 0, inference_idx: int = 0):
    imgs = _extract_obs(raw_obs)
    if perturb is not None:
        imgs = perturb.transform_observation(imgs, episode_idx=episode_idx, inference_idx=inference_idx)
    return {"obs": [{cam_keys[0]: imgs[AGENTVIEW_KEY], cam_keys[1]: imgs[WRIST_KEY]}]}


def _store_from_raw(raw_obs: Dict[str, Any], cam_keys: List[str], perturb: Optional[StressTest] = None, episode_idx: int = 0, inference_idx: int = 0):
    obs = _obs_payload(raw_obs, cam_keys, perturb=perturb, episode_idx=episode_idx, inference_idx=inference_idx)["obs"][0]
    return {
        "primary_image": np.ascontiguousarray(obs[cam_keys[0]]),
        "wrist_image": np.ascontiguousarray(obs[cam_keys[1]]),
        "proprio": _proprio(raw_obs),
    }


def _proprio(raw_obs: Dict[str, Any]) -> np.ndarray:
    parts = []
    for key, width in (("robot0_gripper_qpos", 2), ("robot0_eef_pos", 3), ("robot0_eef_quat", 4)):
        if key in raw_obs:
            arr = np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)
        else:
            arr = np.zeros(width, dtype=np.float32)
        if arr.size < width:
            arr = np.pad(arr, (0, width - arr.size))
        parts.append(arr[:width])
    return np.concatenate(parts, axis=0).astype(np.float32)


def _get_sim_state(env) -> np.ndarray:
    if hasattr(env, "get_sim_state"):
        return np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()


def _dummy_action() -> List[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _init_env(env: OffScreenRenderEnv, init_state: np.ndarray, perturb: Optional[StressTest], episode_idx: int, wait_steps: int) -> Dict[str, Any]:
    env.reset()
    if perturb is None:
        raw_obs = env.set_init_state(init_state)
    else:
        raw_obs = perturb.set_init_state(env, init_state, episode_idx=episode_idx)
    if perturb is not None:
        perturb.apply_to_env(env, episode_idx=episode_idx)
    for _ in range(wait_steps):
        raw_obs, _, _, _ = env.step(_dummy_action())
    return raw_obs


def _run_server_infer(server, obs_payload: Dict[str, Any], frame_st_id: int):
    with torch.no_grad():
        return server._infer(obs_payload, frame_st_id=frame_st_id)


def _compute_cache(server, payload: Dict[str, Any]):
    with torch.no_grad():
        return server._compute_kv_cache(payload)


def _record_drive(server, env, init_state, prompt, cam_keys, max_env_steps, episode_idx, wait_steps, perturb=None):
    raw_obs = _init_env(env, init_state, perturb=perturb, episode_idx=episode_idx, wait_steps=wait_steps)
    server._reset(prompt=prompt)
    records = []
    done = False
    infer_idx = 0
    first = True
    while env.env.timestep < max_env_steps:
        frame_st_id = int(server.frame_st_id)
        records.append({
            "sim_state": _get_sim_state(env),
            "raw_obs": raw_obs,
            "stored": _store_from_raw(raw_obs, cam_keys),
            "inference_idx": infer_idx,
            "frame_st_id": frame_st_id,
        })
        actions, _ = _run_server_infer(server, _obs_payload(raw_obs, cam_keys), frame_st_id=frame_st_id)
        key_frames = []
        action_per_frame = int(actions.shape[2] // 4)
        start_idx = 1 if first else 0
        for i in range(start_idx, actions.shape[1]):
            for j in range(actions.shape[2]):
                raw_obs, _, done, _ = env.step(actions[:, i, j])
                if done:
                    break
                if (j + 1) % action_per_frame == 0:
                    key_frames.append(_obs_payload(raw_obs, cam_keys)["obs"][0])
            if done:
                break
        first = False
        infer_idx += 1
        if done:
            break
        if key_frames:
            _compute_cache(server, {"obs": key_frames, "state": actions})
    return bool(done), records


def _record_clean_drive(server, env_pos, init_state, prompt, cam_keys, max_env_steps, episode_idx, wait_steps):
    return _record_drive(
        server,
        env_pos,
        init_state,
        prompt,
        cam_keys,
        max_env_steps,
        episode_idx,
        wait_steps,
        perturb=None,
    )


def _replay_states(env_neg, init_state, records, perturb, cam_keys, episode_idx, wait_steps):
    _init_env(env_neg, init_state, perturb=perturb, episode_idx=episode_idx, wait_steps=0)
    # Keep the camera/model perturbation after reset; each set_init_state only changes physics state.
    if wait_steps:
        pass
    rows = []
    for rec in records:
        raw_obs = env_neg.set_init_state(rec["sim_state"])
        rows.append(_store_from_raw(raw_obs, cam_keys))
    return rows


def _rollout_neg_drive(server, env_neg, env_pos, init_state, prompt, perturb, cam_keys, max_env_steps, episode_idx, wait_steps):
    raw_obs = _init_env(env_neg, init_state, perturb=perturb, episode_idx=episode_idx, wait_steps=wait_steps)
    env_pos.reset()
    env_pos.set_init_state(init_state)
    server._reset(prompt=prompt)
    pos_rows, neg_rows = [], []
    done = False
    infer_idx = 0
    first = True
    while env_neg.env.timestep < max_env_steps:
        driver_state = _get_sim_state(env_neg)
        pos_raw = env_pos.set_init_state(driver_state)
        pos_rows.append(_store_from_raw(pos_raw, cam_keys))
        neg_rows.append(_store_from_raw(raw_obs, cam_keys))
        frame_st_id = int(server.frame_st_id)
        actions, _ = _run_server_infer(server, _obs_payload(raw_obs, cam_keys), frame_st_id=frame_st_id)
        key_frames = []
        action_per_frame = int(actions.shape[2] // 4)
        start_idx = 1 if first else 0
        for i in range(start_idx, actions.shape[1]):
            for j in range(actions.shape[2]):
                raw_obs, _, done, _ = env_neg.step(actions[:, i, j])
                if done:
                    break
                if (j + 1) % action_per_frame == 0:
                    key_frames.append(_obs_payload(raw_obs, cam_keys)["obs"][0])
            if done:
                break
        first = False
        infer_idx += 1
        if done:
            break
        if key_frames:
            _compute_cache(server, {"obs": key_frames, "state": actions})
    return bool(done), pos_rows, neg_rows


def _rollout_init_pair(server, env_pos, env_neg, init_state, prompt, perturb, cam_keys, max_env_steps, episode_idx, wait_steps):
    pos_success, pos_records = _record_clean_drive(server, env_pos, init_state, prompt, cam_keys, max_env_steps, episode_idx, wait_steps)
    neg_raw = _init_env(env_neg, init_state, perturb=perturb, episode_idx=episode_idx, wait_steps=wait_steps)
    server._reset(prompt=prompt)
    neg_rows = []
    done = False
    first = True
    while env_neg.env.timestep < max_env_steps and len(neg_rows) < len(pos_records):
        neg_rows.append(_store_from_raw(neg_raw, cam_keys))
        actions, _ = _run_server_infer(server, _obs_payload(neg_raw, cam_keys), frame_st_id=int(server.frame_st_id))
        key_frames = []
        action_per_frame = int(actions.shape[2] // 4)
        start_idx = 1 if first else 0
        for i in range(start_idx, actions.shape[1]):
            for j in range(actions.shape[2]):
                neg_raw, _, done, _ = env_neg.step(actions[:, i, j])
                if done:
                    break
                if (j + 1) % action_per_frame == 0:
                    key_frames.append(_obs_payload(neg_raw, cam_keys)["obs"][0])
            if done:
                break
        first = False
        if done:
            break
        if key_frames:
            _compute_cache(server, {"obs": key_frames, "state": actions})
    n = min(len(pos_records), len(neg_rows))
    return bool(pos_success), bool(done), [r["stored"] for r in pos_records[:n]], neg_rows[:n]


def _stack(rows: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {
        "primary_images": np.stack([r["primary_image"] for r in rows], axis=0),
        "wrist_images": np.stack([r["wrist_image"] for r in rows], axis=0),
        "proprios": np.stack([r["proprio"] for r in rows], axis=0).astype(np.float32),
    }


def _first_perturb_from_spec(path: Path) -> Dict[str, Any]:
    spec = maybe_load_yaml(str(path))
    if "perturbation" in spec:
        return dict(spec["perturbation"])
    variants = list(spec.get("variants", []))
    for v in variants:
        name = str(v.get("name", ""))
        if name == "nominal":
            continue
        out = dict(v)
        if "kind" not in out:
            raise ValueError(
                "LQR perturb specs must use repo-root-style fields: "
                "`perturbation: {kind: gaussian|camera|init_position, ...}` "
                "or variant entries with an explicit `kind`."
            )
        return out
    return {"kind": "none", "name": "nominal"}


def _write_video(frames: List[np.ndarray], path: Path, fps: int):
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=int(fps))


def _write_paired_outputs(
    out_dir: Path,
    prompt: str,
    pos_rows: List[Dict[str, np.ndarray]],
    neg_rows: List[Dict[str, np.ndarray]],
    ep_idx: List[int],
    inf_idx: List[int],
    drive_source: List[int],
) -> Optional[float]:
    if not pos_rows or not neg_rows:
        return None
    pos = _stack(pos_rows)
    neg = _stack(neg_rows)
    ep_arr = np.asarray(ep_idx, dtype=np.int32)
    inf_arr = np.asarray(inf_idx, dtype=np.int32)
    ds_arr = np.asarray(drive_source, dtype=np.int32)
    cfg_arr = np.zeros_like(ep_arr, dtype=np.int32)
    np.savez_compressed(out_dir / "positive.npz", **pos, episode_idx=ep_arr, inference_idx=inf_arr, drive_source=ds_arr, config_idx=cfg_arr)
    np.savez_compressed(out_dir / "negative.npz", **neg, episode_idx=ep_arr, inference_idx=inf_arr, drive_source=ds_arr, config_idx=cfg_arr)
    (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    return float(np.max(np.abs(pos["proprios"] - neg["proprios"])))


def _write_init_outputs(
    out_dir: Path,
    prompt: str,
    pos_rows: List[Dict[str, np.ndarray]],
    neg_rows: List[Dict[str, np.ndarray]],
    summaries: List[Dict[str, Any]],
) -> Optional[Dict[str, int]]:
    if not pos_rows and not neg_rows:
        return None

    pos_ep_idx, pos_inf_idx, neg_ep_idx, neg_inf_idx = [], [], [], []
    for row in summaries:
        ep = int(row["episode"])
        n_inf = int(row["n_inferences"])
        if bool(row.get("success", False)):
            pos_ep_idx.extend([ep] * n_inf)
            pos_inf_idx.extend(range(n_inf))
        else:
            neg_ep_idx.extend([ep] * n_inf)
            neg_inf_idx.extend(range(n_inf))

    pos_ep_arr = np.asarray(pos_ep_idx, dtype=np.int32)
    pos_inf_arr = np.asarray(pos_inf_idx, dtype=np.int32)
    neg_ep_arr = np.asarray(neg_ep_idx, dtype=np.int32)
    neg_inf_arr = np.asarray(neg_inf_idx, dtype=np.int32)
    if pos_rows:
        pos = _stack(pos_rows)
        np.savez_compressed(
            out_dir / "positive.npz",
            **pos,
            episode_idx=pos_ep_arr,
            inference_idx=pos_inf_arr,
            drive_source=np.zeros_like(pos_ep_arr, dtype=np.int32),
            config_idx=np.zeros_like(pos_ep_arr, dtype=np.int32),
            success=np.ones_like(pos_ep_arr, dtype=np.int32),
        )
    if neg_rows:
        neg = _stack(neg_rows)
        np.savez_compressed(
            out_dir / "negative.npz",
            **neg,
            episode_idx=neg_ep_arr,
            inference_idx=neg_inf_arr,
            drive_source=np.zeros_like(neg_ep_arr, dtype=np.int32),
            config_idx=np.zeros_like(neg_ep_arr, dtype=np.int32),
            success=np.zeros_like(neg_ep_arr, dtype=np.int32),
        )
    (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "paired_obs_fixed_prompt",
                "checkpoint": True,
                "task_language": prompt,
                "prompt": prompt,
                "positive_rows": int(len(pos_rows)),
                "negative_rows": int(len(neg_rows)),
                "rollouts": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"positive_rows": int(len(pos_rows)), "negative_rows": int(len(neg_rows))}


def main():
    parser = argparse.ArgumentParser(description="Collect repo-root-style paired LingBot LIBERO observations for LQR.")
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--libero-benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--n-pos-rollouts", type=int, default=None)
    parser.add_argument("--n-neg-rollouts", type=int, default=None)
    parser.add_argument("--perturb-spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-env-steps", type=int, default=800)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument(
        "--close-envs",
        action="store_true",
        help="Explicitly close robosuite environments before writing outputs. Disabled by default because EGL/MuJoCo can abort during close on some Slurm nodes.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    perturb_cfg = _first_perturb_from_spec(args.perturb_spec)
    perturb = build_perturbation(perturb_cfg)
    perturb_kind = type(perturb).__name__
    is_camera = perturb_kind == "RandomCameraViewPerturbation"
    is_noise = perturb_kind == "ImageGaussianNoise"
    is_init = perturb_kind in {"GripperXYZPerturbation", "RandomGripperXYZPerturbation"}

    bench = benchmark.get_benchmark_dict()[args.libero_benchmark]()
    task = bench.get_task(args.task_id)
    prompt = task.language
    init_states = bench.get_task_init_states(args.task_id)
    env_args = {
        "bddl_file_name": bench.get_task_bddl_file_path(args.task_id),
        "camera_heights": 128,
        "camera_widths": 128,
    }

    server = _build_server(args.config_name)
    server.vae.eval().requires_grad_(False)
    server.text_encoder.eval().requires_grad_(False)
    if getattr(server, "streaming_vae_half", None) is not None:
        server.streaming_vae_half.vae.eval().requires_grad_(False)
    cam_keys = list(server.job_config.obs_cam_keys)

    n_pos = int(args.n_pos_rollouts if args.n_pos_rollouts is not None else args.num_episodes)
    n_neg = int(args.n_neg_rollouts if args.n_neg_rollouts is not None else args.num_episodes)
    max_ep = min(max(n_pos, n_neg), int(init_states.shape[0]))

    env_pos = _construct_env(env_args)
    env_neg = _construct_env(env_args)
    pos_rows: List[Dict[str, np.ndarray]] = []
    neg_rows: List[Dict[str, np.ndarray]] = []
    ep_idx: List[int] = []
    inf_idx: List[int] = []
    drive_source: List[int] = []
    summaries: List[Dict[str, Any]] = []
    videos: List[np.ndarray] = []

    if is_noise:
        for ep in range(min(n_pos, max_ep)):
            print(f"[collect-pairs] noise clean episode {ep + 1}/{min(n_pos, max_ep)}", flush=True)
            success, records = _record_clean_drive(server, env_pos, init_states[ep], prompt, cam_keys, args.max_env_steps, ep, args.wait_steps)
            for rec in records:
                pos_rows.append(rec["stored"])
                neg_rows.append(_store_from_raw(rec["raw_obs"], cam_keys, perturb=perturb, episode_idx=ep, inference_idx=rec["inference_idx"]))
                ep_idx.append(ep)
                inf_idx.append(int(rec["inference_idx"]))
                drive_source.append(0)
            summaries.append({"episode": ep, "drive_source": 0, "success": bool(success), "n_inferences": len(records)})
            max_dproprio = _write_paired_outputs(args.out_dir, prompt, pos_rows, neg_rows, ep_idx, inf_idx, drive_source)
            print(f"[collect-pairs] checkpoint rows={len(ep_idx)} paired_proprio_max_abs_diff={max_dproprio:.3e}", flush=True)
    elif is_camera:
        for ep in range(min(n_pos, max_ep)):
            print(f"[collect-pairs] camera clean-drive episode {ep + 1}/{min(n_pos, max_ep)}", flush=True)
            success, records = _record_clean_drive(server, env_pos, init_states[ep], prompt, cam_keys, args.max_env_steps, ep, args.wait_steps)
            neg_stored = _replay_states(env_neg, init_states[ep], records, perturb, cam_keys, ep, args.wait_steps)
            for rec, neg in zip(records, neg_stored):
                pos_rows.append(rec["stored"])
                neg_rows.append(neg)
                ep_idx.append(ep)
                inf_idx.append(int(rec["inference_idx"]))
                drive_source.append(1)
            summaries.append({"episode": ep, "drive_source": 1, "success": bool(success), "n_inferences": len(records), "perturb_sample": getattr(perturb, "_last_sample", {})})
            max_dproprio = _write_paired_outputs(args.out_dir, prompt, pos_rows, neg_rows, ep_idx, inf_idx, drive_source)
            print(f"[collect-pairs] checkpoint rows={len(ep_idx)} paired_proprio_max_abs_diff={max_dproprio:.3e}", flush=True)
        for ep in range(min(n_neg, max_ep)):
            print(f"[collect-pairs] camera perturbed-drive episode {ep + 1}/{min(n_neg, max_ep)}", flush=True)
            success, pos_stored, neg_stored = _rollout_neg_drive(server, env_neg, env_pos, init_states[ep], prompt, perturb, cam_keys, args.max_env_steps, ep, args.wait_steps)
            for j, (pos, neg) in enumerate(zip(pos_stored, neg_stored)):
                pos_rows.append(pos)
                neg_rows.append(neg)
                ep_idx.append(ep)
                inf_idx.append(j)
                drive_source.append(0)
            summaries.append({"episode": ep, "drive_source": 0, "success": bool(success), "n_inferences": len(pos_stored), "perturb_sample": getattr(perturb, "_last_sample", {})})
            max_dproprio = _write_paired_outputs(args.out_dir, prompt, pos_rows, neg_rows, ep_idx, inf_idx, drive_source)
            print(f"[collect-pairs] checkpoint rows={len(ep_idx)} paired_proprio_max_abs_diff={max_dproprio:.3e}", flush=True)
    elif is_init:
        for ep in range(min(n_pos, max_ep)):
            print(f"[collect-pairs] init-position perturbed episode {ep + 1}/{min(n_pos, max_ep)}", flush=True)
            success, records = _record_drive(
                server,
                env_neg,
                init_states[ep],
                prompt,
                cam_keys,
                args.max_env_steps,
                ep,
                args.wait_steps,
                perturb=perturb,
            )
            target_rows = pos_rows if success else neg_rows
            for rec in records:
                target_rows.append(rec["stored"])
                ep_idx.append(ep)
                inf_idx.append(int(rec["inference_idx"]))
                drive_source.append(0)
            summaries.append(
                {
                    "episode": ep,
                    "drive_source": 0,
                    "success": bool(success),
                    "n_inferences": len(records),
                    "perturb_sample": getattr(perturb, "_last_sample", {}),
                }
            )
            checkpoint = _write_init_outputs(args.out_dir, prompt, pos_rows, neg_rows, summaries)
            if checkpoint is None:
                print(
                    f"[collect-pairs] checkpoint waiting for both success/failure buckets: "
                    f"positive_rows={len(pos_rows)} negative_rows={len(neg_rows)}",
                    flush=True,
                )
            else:
                print(
                    f"[collect-pairs] checkpoint positive_rows={checkpoint['positive_rows']} "
                    f"negative_rows={checkpoint['negative_rows']}",
                    flush=True,
                )
    else:
        raise ValueError("nominal perturbation does not produce contrastive pairs")

    if args.close_envs:
        env_pos.close()
        env_neg.close()

    if is_init:
        pos_count = len(pos_rows)
        neg_count = len(neg_rows)
        pos_ep_idx, pos_inf_idx, neg_ep_idx, neg_inf_idx = [], [], [], []
        for row in summaries:
            ep = int(row["episode"])
            n_inf = int(row["n_inferences"])
            if bool(row.get("success", False)):
                pos_ep_idx.extend([ep] * n_inf)
                pos_inf_idx.extend(range(n_inf))
            else:
                neg_ep_idx.extend([ep] * n_inf)
                neg_inf_idx.extend(range(n_inf))
        pos = _stack(pos_rows) if pos_rows else None
        neg = _stack(neg_rows) if neg_rows else None
        if pos is None or neg is None:
            raise RuntimeError(
                "init_position repo-root-style collection needs both successful and failed "
                f"perturbed rollouts; got positive_rows={pos_count}, negative_rows={neg_count}"
            )
        pos_ep_arr = np.asarray(pos_ep_idx, dtype=np.int32)
        pos_inf_arr = np.asarray(pos_inf_idx, dtype=np.int32)
        neg_ep_arr = np.asarray(neg_ep_idx, dtype=np.int32)
        neg_inf_arr = np.asarray(neg_inf_idx, dtype=np.int32)
        np.savez_compressed(
            args.out_dir / "positive.npz",
            **pos,
            episode_idx=pos_ep_arr,
            inference_idx=pos_inf_arr,
            drive_source=np.zeros_like(pos_ep_arr, dtype=np.int32),
            config_idx=np.zeros_like(pos_ep_arr, dtype=np.int32),
            success=np.ones_like(pos_ep_arr, dtype=np.int32),
        )
        np.savez_compressed(
            args.out_dir / "negative.npz",
            **neg,
            episode_idx=neg_ep_arr,
            inference_idx=neg_inf_arr,
            drive_source=np.zeros_like(neg_ep_arr, dtype=np.int32),
            config_idx=np.zeros_like(neg_ep_arr, dtype=np.int32),
            success=np.zeros_like(neg_ep_arr, dtype=np.int32),
        )
        ep_arr = np.concatenate([pos_ep_arr, neg_ep_arr])
        ds_arr = np.zeros_like(ep_arr, dtype=np.int32)
    else:
        if not pos_rows or not neg_rows:
            raise RuntimeError(f"no paired rows collected: positive_rows={len(pos_rows)}, negative_rows={len(neg_rows)}")
        pos = _stack(pos_rows)
        neg = _stack(neg_rows)
        ep_arr = np.asarray(ep_idx, dtype=np.int32)
        inf_arr = np.asarray(inf_idx, dtype=np.int32)
        ds_arr = np.asarray(drive_source, dtype=np.int32)
        cfg_arr = np.zeros_like(ep_arr, dtype=np.int32)
        np.savez_compressed(args.out_dir / "positive.npz", **pos, episode_idx=ep_arr, inference_idx=inf_arr, drive_source=ds_arr, config_idx=cfg_arr)
        np.savez_compressed(args.out_dir / "negative.npz", **neg, episode_idx=ep_arr, inference_idx=inf_arr, drive_source=ds_arr, config_idx=cfg_arr)
    (args.out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    if is_init:
        max_dproprio = None
    else:
        max_dproprio = float(np.max(np.abs(pos["proprios"] - neg["proprios"])))
    manifest = {
        "mode": "paired_obs_fixed_prompt",
        "config_name": args.config_name,
        "libero_benchmark": args.libero_benchmark,
        "task_id": int(args.task_id),
        "task_language": prompt,
        "prompt": prompt,
        "perturb_spec": str(args.perturb_spec),
        "perturbation": perturb.manifest(),
        "pairing": (
            "Gaussian pairs use clean drives and post-hoc noised negatives. "
            "Camera pairs use the repo root's two-pass same-state replay: clean "
            "camera drives positive rollouts and replays states into the "
            "perturbed-camera env, then perturbed camera drives negative "
            "rollouts and replays states into the clean env. Init-position "
            "outputs follow the repo root's task07 semantics: positive.npz contains rows "
            "from successful gripper-perturbed rollouts and negative.npz contains "
            "rows from failed gripper-perturbed rollouts; run "
            "scripts/lqr/pair_inputs_by_similarity.py before SVD."
        ),
        "requested_rollouts": {
            "n_pos_rollouts": int(n_pos),
            "n_neg_rollouts": int(n_neg),
            "max_episode_index_used": int(max_ep - 1) if max_ep > 0 else -1,
        },
        "image_layout": "HWC uint8, vertically flipped from LIBERO raw observations",
        "proprio_layout": "concat(robot0_gripper_qpos[2], robot0_eef_pos[3], robot0_eef_quat[4])",
        "total_paired_rows": int(ep_arr.shape[0]),
        "drive_source_distribution": {
            "0_count": int((ds_arr == 0).sum()),
            "1_count": int((ds_arr == 1).sum()),
        },
        "paired_proprio_max_abs_diff": max_dproprio,
        "rollouts": summaries,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[collect-pairs] wrote {args.out_dir / 'positive.npz'}")
    print(f"[collect-pairs] wrote {args.out_dir / 'negative.npz'}")
    if max_dproprio is None:
        print(f"[collect-pairs] rows={ep_arr.shape[0]} unpaired_init_position_outputs=1")
    else:
        print(f"[collect-pairs] rows={ep_arr.shape[0]} paired_proprio_max_abs_diff={max_dproprio:.3e}")


if __name__ == "__main__":
    main()
