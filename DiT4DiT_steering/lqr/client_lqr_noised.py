#!/usr/bin/env python3
"""LIBERO client for LQR-steered noise rollouts.

Runs in the libero conda env (Python 3.8). Connects to server_lqr_noised.py
(dit4dit env) via WebSocket. Applies Gaussian pixel noise client-side before
sending observations; the server handles DiT4DiT inference + LQR steering.

Usage (called by run_lqr_dit4dit_noised.sh):
    python client_lqr_noised.py \
        --host 127.0.0.1 --port 5700 \
        --rank 0 --world-size 4 \
        --n-episodes 50 --noise-sigma 75 \
        --task-id 1 --suite libero_10 \
        --out-dir .../rollouts/...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DIT4DIT_CODE_ROOT = os.environ.get(
    "DIT4DIT_CODE_ROOT", os.environ.get("DIT4DIT_ROOT", str(Path(__file__).resolve().parents[2]))
)
if DIT4DIT_CODE_ROOT not in sys.path:
    sys.path.insert(0, DIT4DIT_CODE_ROOT)

LIBERO_HOME = os.environ.get("LIBERO_HOME", "")
if LIBERO_HOME and LIBERO_HOME not in sys.path:
    sys.path.insert(0, LIBERO_HOME)

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from runtime_paths import load_libero_init_states

IMAGE_SIZE   = 224
DUMMY_ACTION = [0.0] * 6 + [-1.0]


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host",       default="127.0.0.1")
    ap.add_argument("--port",       type=int, default=5700)
    ap.add_argument("--rank",       type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--task-id",    type=int, default=1)
    ap.add_argument("--suite",      default="libero_10")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--noise-sigma",     type=float, default=75.0)
    ap.add_argument("--noise-seed-base", type=int,   default=0)
    ap.add_argument("--num-steps-wait",  type=int,   default=10)
    ap.add_argument("--max-env-steps",   type=int,   default=1000)
    ap.add_argument("--run-baseline",    action="store_true")
    ap.add_argument("--no-baseline",     dest="run_baseline", action="store_false")
    ap.add_argument("--seed",       type=int, default=1)
    ap.add_argument("--tag",        type=str, default="seed1")
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--video-fps",  type=int,  default=30)
    ap.add_argument("--save-video", action="store_true", default=False)
    return ap.parse_args()


# ── Observation helpers ───────────────────────────────────────────────────────
def _quat2axisangle(quat):
    q = quat.copy()
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def _encode_state(obs, max_state_dim):
    eef_pos  = obs["robot0_eef_pos"].astype(np.float32)
    axisangle = _quat2axisangle(obs["robot0_eef_quat"])
    gripper  = obs["robot0_gripper_qpos"].astype(np.float32)
    proprio  = np.concatenate([eef_pos, axisangle, gripper])
    sin_s = np.sin(proprio[None]); cos_s = np.cos(proprio[None])
    state = np.stack([sin_s, cos_s], axis=-1).reshape(1, -1).astype(np.float32)
    pad = max_state_dim - state.shape[-1]
    if pad > 0:
        state = np.pad(state, ((0, 0), (0, pad)), "constant")
    return state


def _prepare_image(obs, noise_rng=None, sigma=0.0):
    primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist   = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    noised_video_frame = None
    if noise_rng is not None and sigma > 0:
        primary = np.clip(primary.astype(np.float32) + noise_rng.normal(0, sigma, primary.shape),
                          0, 255).astype(np.uint8)
        wrist   = np.clip(wrist.astype(np.float32)   + noise_rng.normal(0, sigma, wrist.shape),
                          0, 255).astype(np.uint8)
        # undo the double-flip so this frame matches the raw orientation used in `frames`
        noised_video_frame = primary[::-1, ::-1].copy()
    primary = cv2.resize(primary, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    wrist   = cv2.resize(wrist,   (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return np.concatenate([primary, wrist], axis=1), noised_video_frame


def _unnormalize(norm_actions, meta):
    hi   = np.array(meta["action_high"],  dtype=np.float32)
    lo   = np.array(meta["action_low"],   dtype=np.float32)
    mask = np.array(meta["action_mask"],  dtype=bool)
    actions = np.clip(norm_actions[:, :7], -1.0, 1.0).copy()
    raw = np.where(mask, 0.5 * (actions + 1.0) * (hi - lo) + lo, actions)
    raw[:, 6] = np.where(norm_actions[:, 6] < 0.5, 1.0, -1.0)
    return raw


# ── Rollout ───────────────────────────────────────────────────────────────────
def rollout(env, client, meta, init_state, prompt, ep_idx, args, *, steer):
    env.reset()
    obs = env.set_init_state(init_state)
    terminated_early = False
    for _ in range(args.num_steps_wait):
        obs, _, terminated_early, _ = env.step(DUMMY_ACTION)
        if terminated_early:
            break

    rng = (np.random.default_rng(seed=args.noise_seed_base + ep_idx)
           if args.noise_sigma > 0 else None)
    num_open_loop = int(meta["num_open_loop"])
    max_state_dim = int(meta["max_state_dim"])

    queue = []
    frames = [obs["agentview_image"].copy()]
    noised_frames = [obs["agentview_image"].copy()]  # no model call yet; placeholder matches clean
    current_noised = None  # noised agentview from last model call, in raw orientation
    success = False
    first_call = True
    chunk_idx = 0

    for t in range(args.max_env_steps):
        if terminated_early:
            break
        if not queue:
            concat_img, noised_video = _prepare_image(obs, rng, args.noise_sigma)
            if noised_video is not None:
                current_noised = noised_video
            state_enc  = _encode_state(obs, max_state_dim)
            response   = client.predict_action({
                "examples":      [{"image": [concat_img], "lang": prompt, "state": state_enc}],
                "steer":         bool(steer),
                "reset_episode": first_call,
                "chunk_idx":     chunk_idx,
            })
            first_call = False
            chunk_idx += 1
            data = response.get("data", response)
            norm = data["normalized_actions"][0]           # (chunk, action_dim)
            raw  = _unnormalize(norm, meta)
            queue = [raw[i] for i in range(min(num_open_loop, len(raw)))]

        a = queue.pop(0)
        try:
            obs, _, done, _ = env.step(a.tolist())
        except ValueError:
            break  # env already terminated (e.g. task done mid open-loop chunk)
        frames.append(obs["agentview_image"].copy())
        # repeat the noised frame from the last model call (shows what drove this open-loop chunk)
        noised_frames.append(current_noised if current_noised is not None
                             else obs["agentview_image"].copy())
        if done:
            success = True
            break

    has_noise = args.noise_sigma > 0
    return success, t + args.num_steps_wait + 1, frames, noised_frames if has_noise else None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite  = benchmark.get_benchmark_dict()[args.suite]()
    task        = task_suite.get_task(args.task_id)
    init_states = load_libero_init_states(task_suite, args.task_id)
    if args.n_episodes > init_states.shape[0]:
        raise ValueError(f"--n-episodes {args.n_episodes} > {init_states.shape[0]}")
    task_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    prompt = task.language

    env = OffScreenRenderEnv(bddl_file_name=str(task_bddl),
                              camera_heights=args.resolution, camera_widths=args.resolution)
    env.seed(args.seed)

    print(f"Connecting to server at {args.host}:{args.port} ...", flush=True)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    meta   = client.get_server_metadata()
    print(f"Connected. num_open_loop={meta['num_open_loop']}  "
          f"max_state_dim={meta['max_state_dim']}", flush=True)

    my_episodes = list(range(args.rank, args.n_episodes, args.world_size))
    print(f"[rank {args.rank}/{args.world_size}] "
          f"episodes: {my_episodes}  task: {prompt!r}", flush=True)

    results = []
    baseline_dir = args.out_dir / "baseline"
    if args.run_baseline and args.save_video:
        baseline_dir.mkdir(parents=True, exist_ok=True)

    def _run(ep, *, steer, label):
        t0 = time.time()
        success, env_steps, frames, noised_frames = rollout(env, client, meta, init_states[ep],
                                                            prompt, ep, args, steer=steer)
        dt = time.time() - t0
        tag = "SUCCESS" if success else "FAILURE"
        if args.save_video:
            import imageio
            vdir = (args.out_dir if steer else baseline_dir)
            suffix = "__baseline" if not steer else ""
            for old in vdir.glob(f"ep{ep:02d}--*{suffix}.mp4"):
                old.unlink()
            vp = vdir / f"ep{ep:02d}--{tag}{suffix}.mp4"
            writer = imageio.get_writer(str(vp), fps=args.video_fps)
            for i, f in enumerate(frames):
                clean = np.flipud(f)
                if noised_frames is not None:
                    tiled = np.concatenate([clean, np.flipud(noised_frames[i])], axis=1)
                    writer.append_data(tiled)
                else:
                    writer.append_data(clean)
            writer.close()
        print(f"[ep {ep:2d}] {label:8s} {tag:7s}  steps={env_steps:4d}  {dt:.1f}s", flush=True)
        return {"success": bool(success), "env_steps": int(env_steps), "wall_time_s": float(dt)}

    for ep in my_episodes:
        rec = _run(ep, steer=True,  label="steered")
        bl  = _run(ep, steer=False, label="baseline") if args.run_baseline else None
        results.append({"episode": ep, "rank": args.rank, **rec, "baseline": bl})

    env.close()
    client.close()

    rp = args.out_dir / f"results_rank{args.rank}.json"
    rp.write_text(json.dumps(results, indent=2))
    n_succ = sum(r["success"] for r in results)
    print(f"[rank {args.rank}] DONE: {n_succ}/{len(results)} steered  → {rp}", flush=True)


if __name__ == "__main__":
    main()
