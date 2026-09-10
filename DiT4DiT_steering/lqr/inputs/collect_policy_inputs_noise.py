#!/usr/bin/env python
"""Collect DiT4DiT contrastive (positive, negative) input pairs using
Gaussian image noise as the perturbation.

Analogue of the repo root's notebooks/lqr/inputs/collect_policy_inputs_noise.py
adapted for DiT4DiT's action-generation model.

Runs CLEAN rollouts under PROMPT and records every policy input
(primary_image, wrist_image, proprio) at each inference step.
Negative observations are then constructed **post-hoc** by adding i.i.d.
Gaussian pixel noise (σ = NOISE_SIGMA in uint8 units) to both camera images,
with per-episode seeding for reproducibility.

Key difference from collect_policy_inputs_gripper_xyz_perturbation.py:
  - NO physical gripper perturbation.
  - Positive and negative npzs are 1-to-1 PAIRED on every row (same MuJoCo
    state, only images differ). No separate pairing step is needed before SVD.

Output layout:
    <OUT_DIR>/
      positive.npz     # clean images at every captured pose
      negative.npz     # same rows + Gaussian noise on both cameras
      manifest.json

NPZ schema (identical keys in positive.npz and negative.npz):
    primary_images : (N, 256, 256, 3) uint8  (raw flipped, NOT resized)
    wrist_images   : (N, 256, 256, 3) uint8
    proprios       : (N, 8) float32           (eef_pos + axisangle + gripper_qpos)
    episode_idx    : (N,) int32
    inference_idx  : (N,) int32
    drive_source   : (N,) int32               (0 for all rows)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LOCAL_DIT4DIT_ROOT = _HERE.parent.parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from runtime_paths import configure_runtime, load_libero_init_states  # noqa: E402

_DIT4DIT_ROOT, LIBERO_HOME = configure_runtime(_LOCAL_DIT4DIT_ROOT)
os.environ.setdefault("LIBERO_HOME", LIBERO_HOME)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DIT4DIT_ROOT = _DIT4DIT_ROOT
CKPT_DEFAULT = os.environ.get(
    "CKPT_PATH",
    str(DIT4DIT_ROOT / "checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt"),
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def capture_obs(obs: dict) -> dict:
    """Return raw (flipped, not resized) images and raw proprio."""
    return {
        "primary_image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist_image":   np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "proprio": np.concatenate([
            obs["robot0_eef_pos"].astype(np.float32),
            _quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
            obs["robot0_gripper_qpos"].astype(np.float32),
        ]),
    }


def obs_to_model_input(captured: dict, prompt: str, image_size: int = 224,
                        max_state_dim: int = 16, noise_sigma: float = 0.0,
                        rng: np.random.Generator | None = None) -> dict:
    """Build a DiT4DiT example dict from a captured obs.

    If noise_sigma > 0, adds Gaussian pixel noise to both images.
    """
    primary = captured["primary_image"].copy().astype(np.float32)
    wrist   = captured["wrist_image"].copy().astype(np.float32)
    if noise_sigma > 0.0 and rng is not None:
        primary = np.clip(primary + rng.normal(0.0, noise_sigma, primary.shape), 0, 255)
        wrist   = np.clip(wrist   + rng.normal(0.0, noise_sigma, wrist.shape),   0, 255)
    primary = cv2.resize(primary.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_AREA)
    wrist   = cv2.resize(wrist.astype(np.uint8),   (image_size, image_size), interpolation=cv2.INTER_AREA)
    concat_img = np.concatenate([primary, wrist], axis=1)

    proprio = captured["proprio"]
    sin_s = np.sin(proprio[None])
    cos_s = np.cos(proprio[None])
    state_enc = np.stack([sin_s, cos_s], axis=-1).reshape(1, -1).astype(np.float32)
    pad = max_state_dim - state_enc.shape[-1]
    if pad > 0:
        state_enc = np.pad(state_enc, ((0, 0), (0, pad)), "constant")

    return {"image": [concat_img], "lang": prompt, "state": state_enc}


def unnormalize_actions(norm_np: np.ndarray, action_stats: dict) -> np.ndarray:
    mask = action_stats.get("mask", np.ones(norm_np.shape[-1], dtype=bool))
    hi = np.array(action_stats["max"], dtype=np.float32)
    lo = np.array(action_stats["min"], dtype=np.float32)
    n  = np.clip(norm_np[:, :len(hi)], -1.0, 1.0)
    return np.where(mask, 0.5 * (n + 1.0) * (hi - lo) + lo, n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir",     type=Path, required=True)
    ap.add_argument("--suite",       type=str,  default="libero_10")
    ap.add_argument("--task-id",     type=int,  required=True)
    ap.add_argument("--n-episodes",  type=int,  default=50)
    ap.add_argument("--resolution",  type=int,  default=256)
    ap.add_argument("--prompt",      type=str,  default=None)
    ap.add_argument("--noise-sigma", type=float, default=75.0,
                    help="Gaussian noise σ in uint8 pixel units for negative.npz "
                         "(default 75). Applied to both cameras post-hoc.")
    ap.add_argument("--ckpt-path",   type=str,  default=CKPT_DEFAULT)
    ap.add_argument("--num-open-loop-steps", type=int, default=8)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir: {out_dir}")
    _log(f"noise_sigma={args.noise_sigma}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from pathlib import Path as _Path

    task_suite  = benchmark.get_benchmark_dict()[args.suite]()
    task        = task_suite.get_task(args.task_id)
    init_states = load_libero_init_states(task_suite, args.task_id)
    n_episodes  = min(args.n_episodes, int(init_states.shape[0]))
    prompt      = args.prompt if args.prompt else task.language
    _log(f"suite={args.suite}  task={args.task_id}  episodes={n_episodes}")
    _log(f"prompt: {prompt!r}")

    suite_max_steps = {
        "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
        "libero_10": 520, "libero_90": 400,
    }.get(args.suite, 520)

    task_bddl = _Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl),
        camera_heights=args.resolution, camera_widths=args.resolution,
    )
    env.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"loading DiT4DiT from {args.ckpt_path} on {device} ...")
    import DiT4DiT.model.framework.DiT4DiT  # register framework before from_pretrained
    from DiT4DiT.model.framework.base_framework import baseframework
    from DiT4DiT.model.framework.share_tools import read_mode_config
    model       = baseframework.from_pretrained(args.ckpt_path).to(device).eval()
    action_model = model.action_model
    _, norm_stats = read_mode_config(args.ckpt_path)
    unnorm_key   = next(iter(norm_stats))
    action_stats = norm_stats[unnorm_key]["action"]
    _log(f"model ready; unnorm_key={unnorm_key!r}")

    max_state_dim = action_model.config.state_dim

    all_primary, all_wrist, all_proprio = [], [], []
    all_episode_idx, all_inference_idx   = [], []
    rollout_summaries = []

    _log("=== run loop starting ===")
    for ep in range(n_episodes):
        t0 = time.time()
        env.reset()
        obs = env.set_init_state(init_states[ep])
        for _ in range(10):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        queue: deque = deque(maxlen=args.num_open_loop_steps)
        inputs: list = []
        success = False
        t = 0
        while t < suite_max_steps:
            if not queue:
                captured = capture_obs(obs)
                inputs.append(captured)
                example = obs_to_model_input(captured, prompt, max_state_dim=max_state_dim)
                with torch.inference_mode():
                    result = model.predict_action([example])
                norm_np = result["normalized_actions"][0]  # (action_horizon, action_dim)
                raw = unnormalize_actions(norm_np, action_stats)
                raw[:, 6] = np.where(norm_np[:, 6] < 0.5, 1.0, -1.0)
                for a in raw[:args.num_open_loop_steps]:
                    queue.append(a)
            a = queue.popleft()
            obs, _, done, _ = env.step(a.tolist())
            if done:
                success = True
                break
            t += 1

        for inf_idx, rec in enumerate(inputs):
            all_primary.append(rec["primary_image"])
            all_wrist.append(rec["wrist_image"])
            all_proprio.append(rec["proprio"])
            all_episode_idx.append(ep)
            all_inference_idx.append(inf_idx)

        dt = time.time() - t0
        tag = "SUCCESS" if success else "FAILURE"
        _log(f"ep {ep:2d}  {tag:7s}  steps={t:4d}  inferences={len(inputs):3d}  {dt:6.1f}s")
        rollout_summaries.append({
            "episode": ep, "success": bool(success),
            "env_steps": int(t), "n_inferences": len(inputs), "wall_time_s": round(dt, 2),
        })
        torch.cuda.empty_cache()

    env.close()

    pos_primary  = np.stack(all_primary,  axis=0)
    pos_wrist    = np.stack(all_wrist,    axis=0)
    proprio_arr  = np.stack(all_proprio,  axis=0)
    episode_arr  = np.asarray(all_episode_idx,  dtype=np.int32)
    inference_arr = np.asarray(all_inference_idx, dtype=np.int32)
    drive_arr    = np.zeros_like(episode_arr, dtype=np.int32)

    # ---- Generate negatives post-hoc: same rows, noised images ----
    neg_primary = np.empty_like(pos_primary)
    neg_wrist   = np.empty_like(pos_wrist)
    current_ep, rng = -1, None
    for row in range(pos_primary.shape[0]):
        ep = int(episode_arr[row])
        if ep != current_ep:
            rng = np.random.default_rng(seed=ep)
            current_ep = ep
        p = pos_primary[row].astype(np.float32)
        neg_primary[row] = np.clip(
            p + rng.normal(0.0, args.noise_sigma, p.shape), 0, 255
        ).astype(np.uint8)
        w = pos_wrist[row].astype(np.float32)
        neg_wrist[row] = np.clip(
            w + rng.normal(0.0, args.noise_sigma, w.shape), 0, 255
        ).astype(np.uint8)

    abs_diff = np.abs(neg_primary.astype(np.float32) - pos_primary.astype(np.float32))

    POSITIVE_NPZ  = out_dir / "positive.npz"
    NEGATIVE_NPZ  = out_dir / "negative.npz"
    MANIFEST_JSON = out_dir / "manifest.json"

    def _save(path, primary, wrist):
        np.savez_compressed(
            path,
            primary_images=primary, wrist_images=wrist,
            proprios=proprio_arr,
            episode_idx=episode_arr, inference_idx=inference_arr,
            drive_source=drive_arr,
        )
        _log(f"wrote {path.name}  ({path.stat().st_size/1e6:.1f} MB)  rows={primary.shape[0]}")

    _save(POSITIVE_NPZ, pos_primary, pos_wrist)
    _save(NEGATIVE_NPZ, neg_primary, neg_wrist)

    n_succ = sum(r["success"] for r in rollout_summaries)
    manifest = {
        "model": "DiT4DiT", "ckpt_path": args.ckpt_path,
        "suite": args.suite, "task_id": args.task_id,
        "n_episodes": n_episodes, "resolution": args.resolution, "prompt": prompt,
        "noise": {
            "kind": "ImageGaussianNoise",
            "sigma": float(args.noise_sigma),
            "apply_to": ["primary_image", "wrist_image"],
            "per_episode_seed": True,
            "seed_recipe": "np.random.default_rng(seed=int(episode_idx))",
            "clipped_to": [0, 255], "output_dtype": "uint8",
        },
        "pairing": (
            "Row i in positive.npz and row i in negative.npz share the same "
            "MuJoCo state. Negatives are post-hoc noised images. "
            "NO pairing step is needed before SVD — feed directly to "
            "run_partition_svd_pairs_no_action.sh as POS_NPZ/NEG_NPZ."
        ),
        "image_layout": "HWC uint8, flipped 180 deg, NOT resized or concatenated",
        "proprio_layout": "eef_pos[3] + axisangle(eef_quat)[3] + gripper_qpos[2] = 8-dim float32",
        "totals": {
            "rollouts": n_episodes, "rollout_successes": n_succ,
            "rollout_failures": n_episodes - n_succ,
            "total_inferences": int(pos_primary.shape[0]),
        },
        "pixel_diff_stats": {
            "primary_abs_diff_mean": float(abs_diff.mean()),
            "primary_abs_diff_max":  float(abs_diff.max()),
        },
        "rollouts": rollout_summaries,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    _log(f"wrote {MANIFEST_JSON.name}")
    _log(f"success rate: {n_succ}/{n_episodes} = {100*n_succ/n_episodes:.1f}%")
    _log("Pairs are already aligned. No pairing step needed.")
    _log(f"Next: run_partition_svd_pairs_no_action.sh POS_NPZ={POSITIVE_NPZ} NEG_NPZ={NEGATIVE_NPZ}")


if __name__ == "__main__":
    main()
