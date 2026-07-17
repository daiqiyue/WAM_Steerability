#!/usr/bin/env python
"""Multi-task variant of collect_policy_inputs_gripper_xyz_perturbation.py.

Same gripper-XYZ perturbation stress test, same N rollouts, same positive
/ negative bucketing — but the rollouts span MULTIPLE libero_10 tasks
(default: task 0, 2, 4, 5) instead of a single task. Each task contributes
--n-episodes rollouts and all captured policy inputs are concatenated into
one positive.npz and one negative.npz.

Scene / prompt / objects per task are unchanged — the only stress applied
is the RandomGripperXYZPerturbation from 11_gripper_xyz_perturbation.ipynb
(default preset: xyz_random_xlarge_2, sigma=15cm isotropic, per-episode
Gaussian sampled via SeedSequence([base_seed, episode_idx])).

For each --task-id and each of --n-episodes init_states we:
  1. perturb the gripper's initial Cartesian (x, y, z) per the preset,
  2. run a Cosmos-Policy rollout under that task's prompt,
  3. capture every prepare_observation packed dict the policy consumed,
  4. label all captured rows by rollout outcome AND source task_id.

Successful rollouts -> positive.npz; failed rollouts -> negative.npz.

NOTE on pairing: this stage produces UNPAIRED npzs (positive and
negative have different row counts and rows are NOT 1-1 matched at
identical states). They are NOT directly consumable by
run_partition_svd_pairs*.py. Use notebooks/lqr/svd/pair_inputs_by_similarity.py
to produce a paired, SVD-compatible version.

NOTE on RNG: the per-episode xyz delta is sampled from
SeedSequence([base_seed, episode_idx]) inside the stress test. Because
episode_idx restarts at 0 for each task, the same xyz delta is applied
to (task_a, ep_k) and (task_b, ep_k). This matches what you'd get by
running the single-task script 4 times with the same base_seed.

Output layout:

    <OUT_DIR>/
      positive.npz       # rows from all successful rollouts (across tasks)
      negative.npz       # rows from all failed rollouts (across tasks)
      manifest.json

NPZ schema (shared by positive.npz and negative.npz):
    primary_images : (N, 256, 256, 3) uint8  (HWC, flip_images applied)
    wrist_images   : (N, 256, 256, 3) uint8
    proprios       : (N, 9) float32
    task_id        : (N,) int32              (source libero task id)
    episode_idx    : (N,) int32              (per-task rollout index)
    inference_idx  : (N,) int32              (position within source rollout)
    drive_source   : (N,) int32              (fixed to 0; single drive campaign)
    success        : (N,) int32              (1 for positive.npz, 0 for negative.npz)
    xyz_delta_m    : (N, 3) float32          (requested per-episode shift)
    achieved_xyz_delta_m : (N, 3) float32    (measured EE shift)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# Match notebook _setup.py — must run BEFORE any cosmos_policy import. This
# script lives at notebooks/lqr/inputs/<this>.py; notebooks/_setup.py is two
# dirs up.
_HERE = Path(__file__).resolve().parent
_NOTEBOOKS_ROOT = _HERE.parent.parent
if str(_NOTEBOOKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_ROOT))
from _setup import setup_env  # noqa: E402

setup_env()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import torch as _torch  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from cosmos_policy.experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env, get_libero_dummy_action,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (  # noqa: E402
    PolicyEvalConfig, prepare_observation, TASK_MAX_STEPS,
)
from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    get_action, get_model, load_dataset_stats, init_t5_text_embeddings_cache,
)


def _log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


# ====================================================================
# GripperXYZPerturbation — copied verbatim from
# notebooks/stress_test/11_gripper_xyz_perturbation.ipynb
# ====================================================================

AGENTVIEW_KEY = "agentview_image"

EE_SITE_CANDIDATES = (
    "gripper0_grip_site", "robot0_grip_site", "grip_site", "ee_site",
)
EE_BODY_CANDIDATES = (
    "robot0_right_hand", "right_hand", "gripper0_eef", "gripper0_hand",
)
OSC_OUTPUT_MAX_M = 0.05


def _resolve_problem(env):
    cur, seen = env, set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "sim"):
            return cur
        cur = getattr(cur, "env", None)
    raise RuntimeError("could not locate problem on env")


def _resolve_ee_target(sim):
    site_names = set(sim.model.site_names)
    for n in EE_SITE_CANDIDATES:
        if n in site_names:
            return n, None
    body_names = set(sim.model.body_names)
    for n in EE_BODY_CANDIDATES:
        if n in body_names:
            return None, n
    raise RuntimeError("no EE site/body found")


def _ee_pos(sim, site_name=None, body_name=None):
    if site_name is not None:
        return sim.data.site_xpos[sim.model.site_name2id(site_name)].copy()
    return sim.data.body_xpos[sim.model.body_name2id(body_name)].copy()


def _shift_gripper_xyz_via_actions(env, target_pos, *, site_name=None, body_name=None,
                                    gripper_action=-1.0, max_steps=30, tol_m=2e-3,
                                    output_max_m=OSC_OUTPUT_MAX_M):
    problem = _resolve_problem(env)
    sim = problem.sim
    obs = None
    n_used = 0
    achieved = _ee_pos(sim, site_name=site_name, body_name=body_name)
    for step in range(max_steps):
        cur = _ee_pos(sim, site_name=site_name, body_name=body_name)
        err = target_pos - cur
        if np.linalg.norm(err) < tol_m:
            break
        action_xyz = np.clip(err / output_max_m, -1.0, 1.0)
        action = [float(action_xyz[0]), float(action_xyz[1]), float(action_xyz[2]),
                  0.0, 0.0, 0.0, float(gripper_action)]
        obs, _, _, _ = env.step(action)
        n_used = step + 1
        achieved = _ee_pos(sim, site_name=site_name, body_name=body_name)
    residual = float(np.linalg.norm(target_pos - achieved))
    return obs, achieved, residual, n_used


class StressTest:
    slug: str = "stock"
    def transform_task(self, task, output_dir=None): return task
    def set_init_state(self, env, init_state, episode_idx=0): return env.set_init_state(init_state)
    def apply_to_env(self, env, episode_idx=0): return None
    def transform_task_desc(self, desc, env=None): return desc
    def manifest(self) -> dict: return {"kind": type(self).__name__, "slug": self.slug}


@dataclass
class GripperXYZPerturbation(StressTest):
    xyz_delta: Tuple[float, float, float] = (0., 0., 0.)
    gripper_action: float = -1.0
    pre_shift_settle_steps: int = 10
    shift_max_steps: int = 30
    shift_tol_m: float = 2e-3
    output_max_m: float = OSC_OUTPUT_MAX_M
    post_shift_pause_steps: int = 10
    name_hint: str = "gripper_xyz"
    _last_sample: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def slug(self) -> str:
        xyz_str = ",".join(f"{a:+.3f}" for a in self.xyz_delta)
        h = hashlib.md5(f"xyz={xyz_str}|g={self.gripper_action:+.1f}".encode()).hexdigest()[:6]
        return f"{self.name_hint}_{h}"

    def _resolve_xyz(self, episode_idx):
        return tuple(float(v) for v in self.xyz_delta)

    def set_init_state(self, env, init_state, episode_idx=0):
        problem = _resolve_problem(env)
        sim = problem.sim
        site_name, body_name = _resolve_ee_target(sim)

        baseline_obs = env.set_init_state(init_state)
        for _ in range(self.pre_shift_settle_steps):
            baseline_obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

        sim.forward()
        baseline_ee = _ee_pos(sim, site_name=site_name, body_name=body_name)

        dxyz = self._resolve_xyz(episode_idx)
        target_ee = baseline_ee + np.asarray(dxyz, dtype=np.float64)

        if np.allclose(dxyz, 0.0):
            achieved_ee = baseline_ee
            residual    = 0.0
            n_used      = 0
            final_obs   = baseline_obs
        else:
            final_obs, achieved_ee, residual, n_used = _shift_gripper_xyz_via_actions(
                env, target_ee,
                site_name=site_name, body_name=body_name,
                gripper_action=self.gripper_action,
                max_steps=self.shift_max_steps,
                tol_m=self.shift_tol_m,
                output_max_m=self.output_max_m,
            )

        pause_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(self.gripper_action)]
        for _ in range(self.post_shift_pause_steps):
            final_obs, _, _, _ = env.step(pause_action)

        self._last_sample = {
            "episode_idx": int(episode_idx),
            "xyz_delta_m":       [float(v) for v in dxyz],
            "baseline_ee_pos":   [float(v) for v in baseline_ee],
            "target_ee_pos":     [float(v) for v in target_ee],
            "achieved_ee_pos":   [float(v) for v in achieved_ee],
            "achieved_xyz_delta_m": [float(a - b) for a, b in zip(achieved_ee, baseline_ee)],
            "shift_residual_m":  float(residual),
            "shift_steps_used":  int(n_used),
            "gripper_action":    float(self.gripper_action),
        }
        if residual > max(self.shift_tol_m * 5, 5e-3):
            _log(f"  [shift warn] ep{episode_idx} residual={residual*1000:.2f} mm "
                 f"after {n_used} steps; target may be blocked or out of reach.")
        return final_obs

    def manifest(self) -> dict:
        return {
            "kind": "GripperXYZPerturbation", "slug": self.slug,
            "xyz_delta_m":     [float(v) for v in self.xyz_delta],
            "gripper_action":  float(self.gripper_action),
            "pre_shift_settle_steps":  int(self.pre_shift_settle_steps),
            "post_shift_pause_steps":  int(self.post_shift_pause_steps),
            "shift": {
                "max_steps":    int(self.shift_max_steps),
                "tol_m":        float(self.shift_tol_m),
                "output_max_m": float(self.output_max_m),
            },
        }


@dataclass
class RandomGripperXYZPerturbation(GripperXYZPerturbation):
    sigma_xyz_m: object = 0.02
    base_seed: int = 0

    def _sigma_vec(self):
        s = self.sigma_xyz_m
        if np.isscalar(s):
            return np.array([float(s)] * 3)
        return np.array([float(v) for v in s])

    @property
    def slug(self) -> str:
        sig = self._sigma_vec()
        sig_str = ",".join(f"{v:.4f}" for v in sig)
        payload = f"sigma=[{sig_str}]|seed{self.base_seed}"
        h = hashlib.md5(payload.encode()).hexdigest()[:6]
        return f"{self.name_hint}_seed{self.base_seed}_{h}"

    def _resolve_xyz(self, episode_idx):
        ss  = np.random.SeedSequence([int(self.base_seed), int(episode_idx)])
        rng = np.random.default_rng(ss)
        sig = self._sigma_vec()
        return tuple(float(v) for v in rng.normal(0.0, sig))

    def manifest(self) -> dict:
        sig = self._sigma_vec()
        return {
            "kind": "RandomGripperXYZPerturbation", "slug": self.slug,
            "sampling": "per-episode Gaussian, seed=SeedSequence([base_seed, episode_idx])",
            "sigma_xyz_m":    [float(v) for v in sig],
            "base_seed":      int(self.base_seed),
            "gripper_action": float(self.gripper_action),
            "pre_shift_settle_steps":  int(self.pre_shift_settle_steps),
            "post_shift_pause_steps":  int(self.post_shift_pause_steps),
            "shift": {
                "max_steps":    int(self.shift_max_steps),
                "tol_m":        float(self.shift_tol_m),
                "output_max_m": float(self.output_max_m),
            },
        }


def build_presets(base_seed: int):
    """Same presets as 11_gripper_xyz_perturbation.ipynb."""
    return {
        "xyz_+x_2cm": GripperXYZPerturbation(xyz_delta=( 0.02,  0.00,  0.00), name_hint="xyz_+x_2cm"),
        "xyz_-x_2cm": GripperXYZPerturbation(xyz_delta=(-0.02,  0.00,  0.00), name_hint="xyz_-x_2cm"),
        "xyz_+y_2cm": GripperXYZPerturbation(xyz_delta=( 0.00,  0.02,  0.00), name_hint="xyz_+y_2cm"),
        "xyz_-y_2cm": GripperXYZPerturbation(xyz_delta=( 0.00, -0.02,  0.00), name_hint="xyz_-y_2cm"),
        "xyz_+z_2cm": GripperXYZPerturbation(xyz_delta=( 0.00,  0.00,  0.02), name_hint="xyz_+z_2cm"),
        "xyz_-z_2cm": GripperXYZPerturbation(xyz_delta=( 0.00,  0.00, -0.02), name_hint="xyz_-z_2cm"),
        "xyz_diag_+1cm": GripperXYZPerturbation(xyz_delta=( 0.01,  0.01,  0.01), name_hint="xyz_diag_+1cm"),
        "xyz_diag_+3cm": GripperXYZPerturbation(xyz_delta=( 0.03,  0.03,  0.03), name_hint="xyz_diag_+3cm"),
        "xyz_random_small":  RandomGripperXYZPerturbation(sigma_xyz_m=0.01, base_seed=base_seed, name_hint="xyz_random_small"),
        "xyz_random_medium": RandomGripperXYZPerturbation(sigma_xyz_m=0.02, base_seed=base_seed, name_hint="xyz_random_medium"),
        "xyz_random_large":  RandomGripperXYZPerturbation(sigma_xyz_m=0.04, base_seed=base_seed, name_hint="xyz_random_large"),
        "xyz_random_xlarge": RandomGripperXYZPerturbation(sigma_xyz_m=0.06, base_seed=base_seed, name_hint="xyz_random_xlarge"),
        "xyz_random_xlarge_2": RandomGripperXYZPerturbation(sigma_xyz_m=0.15, base_seed=base_seed, name_hint="xyz_random_xlarge_2"),
        "xyz_random_xlarge_3": RandomGripperXYZPerturbation(sigma_xyz_m=0.10, base_seed=base_seed, name_hint="xyz_random_xlarge_3"),
        "xyz_random_horizontal": RandomGripperXYZPerturbation(
            sigma_xyz_m=(0.03, 0.03, 0.005), base_seed=base_seed, name_hint="xyz_random_horizontal",
        ),
        "xyz_random": RandomGripperXYZPerturbation(
            sigma_xyz_m=(0.1, 0.1, 0.1), base_seed=base_seed, name_hint="xyz_random",
        ),
    }


# ====================================================================
# Rollout that captures policy inputs at every inference call
# ====================================================================

def rollout_collect_inputs(env, init_state, prompt, stress_test,
                            eval_cfg, model, dataset_stats, max_env_steps,
                            *, episode_idx=0, num_steps_wait=10):
    env.reset()
    obs = stress_test.set_init_state(env, init_state, episode_idx=episode_idx)
    stress_test.apply_to_env(env, episode_idx=episode_idx)
    effective_desc = stress_test.transform_task_desc(prompt, env=env)

    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(eval_cfg.model_family))

    queue = deque(maxlen=eval_cfg.num_open_loop_steps)
    inputs = []
    success = False
    t = 0
    while t < max_env_steps:
        if not queue:
            observation = prepare_observation(
                obs, resize_size=224, flip_images=eval_cfg.flip_images,
            )
            inputs.append({
                "primary_image": np.ascontiguousarray(observation["primary_image"]),
                "wrist_image":   np.ascontiguousarray(observation["wrist_image"]),
                "proprio":       np.asarray(observation["proprio"], dtype=np.float32),
            })
            out = get_action(
                eval_cfg, model, dataset_stats, observation, effective_desc,
                num_denoising_steps_action=eval_cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            for a in out["actions"][:eval_cfg.num_open_loop_steps]:
                queue.append(np.asarray(a, dtype=np.float32))
        a = queue.popleft()
        obs, _, done, _ = env.step(a.tolist())
        if done:
            success = True
            break
        t += 1
    return success, t + num_steps_wait, inputs


# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="where to write positive.npz / negative.npz / manifest.json")
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--task-ids", type=int, nargs="+", default=[0, 2, 4, 5],
                    help="list of libero task IDs to roll out (e.g. --task-ids 0 2 4 5)")
    ap.add_argument("--n-episodes", type=int, default=50,
                    help="number of init_states (N settings) to roll out PER TASK; "
                         "capped per-task by suite.get_task_init_states(task).shape[0]")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--preset", type=str, default="xyz_random_xlarge_2",
                    help="gripper perturbation preset name "
                         "(see 11_gripper_xyz_perturbation.ipynb)")
    ap.add_argument("--base-seed", type=int, default=42,
                    help="base seed for the per-episode RNG of random presets")
    ap.add_argument("--prompt", type=str, default=None,
                    help="override prompt; applied to ALL tasks. "
                         "Default uses each libero env's built-in description.")
    ap.add_argument("--ckpt-path", type=str,
                    default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir: {out_dir}")

    presets = build_presets(args.base_seed)
    if args.preset not in presets:
        raise ValueError(
            f"unknown --preset={args.preset!r}; pick one of {sorted(presets)}"
        )
    stress_test = presets[args.preset]
    _log(f"preset: {args.preset}  ({type(stress_test).__name__})  "
         f"manifest={stress_test.manifest()}")

    # ---------------- libero suite (shared) ----------------
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    max_env_steps = TASK_MAX_STEPS[args.suite]
    _log(f"libero suite={args.suite} task_ids={args.task_ids}  "
         f"n_episodes_per_task={args.n_episodes}  max_env_steps={max_env_steps}")

    # ---------------- model (built once, reused across tasks) ----------------
    eval_cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=args.ckpt_path,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{args.ckpt_path}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{args.ckpt_path}/libero_t5_embeddings.pkl",
        use_wrist_image=True, use_proprio=True, normalize_proprio=True, unnormalize_actions=True,
        chunk_size=16, num_open_loop_steps=16, trained_with_image_aug=True,
        use_jpeg_compression=True, flip_images=True,
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1, num_denoising_steps_value=1,
        task_suite_name=args.suite,
    )
    dataset_stats = load_dataset_stats(eval_cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(eval_cfg.t5_text_embeddings_path)
    model, _ = get_model(eval_cfg)
    _log("model ready")

    # ---------------- accumulators ----------------
    all_primary, all_wrist, all_proprio = [], [], []
    all_task_id, all_episode_idx, all_inference_idx = [], [], []
    all_success = []
    all_xyz_delta, all_achieved_xyz_delta = [], []
    rollout_summaries = []
    per_task_meta = []

    # ---------------- main loop over tasks ----------------
    _log("=== run loop starting ===")
    for task_id in args.task_ids:
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        n_avail = int(init_states.shape[0])
        n_episodes = min(args.n_episodes, n_avail)
        if n_episodes < args.n_episodes:
            _log(f"WARNING: task{task_id} --n-episodes={args.n_episodes} > available "
                 f"init_states={n_avail}; using {n_episodes}")

        env, base_task_desc = get_libero_env(task, "cosmos", resolution=args.resolution)
        prompt = args.prompt if args.prompt is not None else base_task_desc
        _log(f"--- task={task_id}  n_episodes={n_episodes}  prompt={prompt!r} ---")

        per_task_meta.append({
            "task_id": int(task_id),
            "prompt": prompt,
            "n_init_states_available": n_avail,
            "n_episodes": int(n_episodes),
        })

        for ep in range(n_episodes):
            t0 = time.time()
            try:
                success, env_steps, inputs = rollout_collect_inputs(
                    env, init_states[ep], prompt, stress_test,
                    eval_cfg, model, dataset_stats, max_env_steps,
                    episode_idx=ep,
                )
            except Exception as e:
                _log(f"task{task_id} ep{ep:02d} CRASHED: {type(e).__name__}: {e}")
                continue
            dt = time.time() - t0
            sample = dict(getattr(stress_test, "_last_sample", {}) or {})
            xyz_d = sample.get("xyz_delta_m") or [0.0, 0.0, 0.0]
            ach_d = sample.get("achieved_xyz_delta_m") or [0.0, 0.0, 0.0]

            for inf_idx, rec in enumerate(inputs):
                all_primary.append(rec["primary_image"])
                all_wrist.append(rec["wrist_image"])
                all_proprio.append(rec["proprio"])
                all_task_id.append(int(task_id))
                all_episode_idx.append(ep)
                all_inference_idx.append(inf_idx)
                all_success.append(int(success))
                all_xyz_delta.append(xyz_d)
                all_achieved_xyz_delta.append(ach_d)

            tag = "SUCCESS" if success else "FAILURE"
            xyz_str = ",".join(f"{v*1000:+5.1f}" for v in xyz_d)
            _log(f"task{task_id} ep{ep:02d} {tag} steps={env_steps:4d} inf={len(inputs):3d} "
                 f"{dt:6.1f}s  dxyz_mm=[{xyz_str}]")
            rollout_summaries.append({
                "task_id": int(task_id),
                "episode": ep,
                "success": bool(success),
                "env_steps": int(env_steps),
                "n_inferences": len(inputs),
                "wall_time_s": dt,
                "sample": sample if sample else None,
            })
            _torch.cuda.empty_cache()

        env.close()

    if not all_episode_idx:
        raise RuntimeError("no rollouts produced any inputs")

    # ---------------- stack ----------------
    primary_arr = np.stack(all_primary, axis=0)
    wrist_arr   = np.stack(all_wrist,   axis=0)
    proprio_arr = np.stack(all_proprio, axis=0)
    task_arr      = np.asarray(all_task_id,       dtype=np.int32)
    episode_arr   = np.asarray(all_episode_idx,   dtype=np.int32)
    inference_arr = np.asarray(all_inference_idx, dtype=np.int32)
    success_arr   = np.asarray(all_success,       dtype=np.int32)
    xyz_arr       = np.asarray(all_xyz_delta,        dtype=np.float32)
    ach_xyz_arr   = np.asarray(all_achieved_xyz_delta, dtype=np.float32)

    pos_mask = success_arr == 1
    neg_mask = success_arr == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    _log(f"total inferences: {primary_arr.shape[0]}  "
         f"positive(success)={n_pos}  negative(failure)={n_neg}")

    # ---------------- save ----------------
    POSITIVE_NPZ  = out_dir / "positive.npz"
    NEGATIVE_NPZ  = out_dir / "negative.npz"
    MANIFEST_JSON = out_dir / "manifest.json"

    def _save(out_npz, mask):
        np.savez_compressed(
            out_npz,
            primary_images=primary_arr[mask],
            wrist_images=wrist_arr[mask],
            proprios=proprio_arr[mask],
            task_id=task_arr[mask],
            episode_idx=episode_arr[mask],
            inference_idx=inference_arr[mask],
            drive_source=np.zeros(int(mask.sum()), dtype=np.int32),
            success=success_arr[mask],
            xyz_delta_m=xyz_arr[mask],
            achieved_xyz_delta_m=ach_xyz_arr[mask],
        )
        _log(f"wrote {out_npz.name}  ({out_npz.stat().st_size/1e6:.1f} MB)  "
             f"rows={int(mask.sum())}")

    _save(POSITIVE_NPZ, pos_mask)
    _save(NEGATIVE_NPZ, neg_mask)

    n_succ = sum(r["success"] for r in rollout_summaries)

    per_task_totals = []
    for tm in per_task_meta:
        tid = tm["task_id"]
        task_rows = [r for r in rollout_summaries if r["task_id"] == tid]
        task_succ = sum(r["success"] for r in task_rows)
        pos_rows = int(((task_arr == tid) & pos_mask).sum())
        neg_rows = int(((task_arr == tid) & neg_mask).sum())
        per_task_totals.append({
            "task_id": tid,
            "rollouts": len(task_rows),
            "rollout_successes": task_succ,
            "rollout_failures":  len(task_rows) - task_succ,
            "positive_rows": pos_rows,
            "negative_rows": neg_rows,
        })

    manifest = {
        "suite": args.suite,
        "task_ids": [int(t) for t in args.task_ids],
        "tasks": per_task_meta,
        "n_episodes_per_task": int(args.n_episodes),
        "resolution": args.resolution,
        "prompt_override": args.prompt,
        "preset": args.preset,
        "base_seed": args.base_seed,
        "stress_test": stress_test.manifest(),
        "pairing": (
            "UNPAIRED: positive.npz holds rows from SUCCESSFUL rollouts, "
            "negative.npz holds rows from FAILED rollouts. Row counts differ "
            "and rows are NOT 1-to-1 matched at identical states. Rows span "
            "multiple tasks — see task_id field in each npz. Do NOT feed these "
            "directly to run_partition_svd_pairs*.py; instead run "
            "notebooks/lqr/svd/pair_inputs_by_similarity.py to produce a "
            "paired, SVD-compatible version."
        ),
        "image_layout": "HWC uint8, flip_images=True applied at capture time",
        "proprio_layout": ("concat(robot0_gripper_qpos[2], robot0_eef_pos[3], "
                          "robot0_eef_quat[4]) -> shape (9,) float32"),
        "rng_note": ("xyz delta is sampled from SeedSequence([base_seed, "
                     "episode_idx]); episode_idx restarts at 0 per task, so "
                     "the same xyz delta is applied to (task_a, ep_k) and "
                     "(task_b, ep_k). Matches running the single-task script "
                     "4 times with the same base_seed."),
        "drive_sources": [
            {"code": 0, "name": "gripper_xyz_perturbed",
             "desc": "env perturbed via RandomGripperXYZPerturbation, "
                     "then policy steps cleanly under per-task prompt"},
        ],
        "sets": {
            "positive": {"out_npz": str(POSITIVE_NPZ),
                          "role": "rows from successful rollouts"},
            "negative": {"out_npz": str(NEGATIVE_NPZ),
                          "role": "rows from failed rollouts"},
        },
        "totals": {
            "rollouts": len(rollout_summaries),
            "rollout_successes": n_succ,
            "rollout_failures":  len(rollout_summaries) - n_succ,
            "positive_rows": n_pos,
            "negative_rows": n_neg,
        },
        "per_task_totals": per_task_totals,
        "rollouts": rollout_summaries,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    _log(f"wrote {MANIFEST_JSON.name}")

    # ---------------- summary ----------------
    _log("=== summary ===")
    _log(f"  rollouts             : {len(rollout_summaries)} "
         f"({n_succ} success / {len(rollout_summaries) - n_succ} failure)")
    for t in per_task_totals:
        _log(f"    task{t['task_id']:>2}: rollouts={t['rollouts']} "
             f"({t['rollout_successes']} success / {t['rollout_failures']} failure) "
             f"  pos_rows={t['positive_rows']}  neg_rows={t['negative_rows']}")
    _log(f"  positive (success)   : {n_pos} rows -> {POSITIVE_NPZ}")
    _log(f"  negative (failure)   : {n_neg} rows -> {NEGATIVE_NPZ}")
    _log(f"  manifest             : {MANIFEST_JSON}")
    _log("Next: run notebooks/lqr/svd/pair_inputs_by_similarity.sh to "
         "produce paired NPZs.")


if __name__ == "__main__":
    main()
