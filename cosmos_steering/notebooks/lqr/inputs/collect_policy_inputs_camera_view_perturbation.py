#!/usr/bin/env python
"""Collect Cosmos-Policy contrastive (positive, negative) input pairs across
a camera-view-perturbation stress-test: env_pos is the un-perturbed
agentview rendering and env_neg has the cam_random_large perturbation
applied (visibility-constrained so the workspace stays in frame).

Mirrors collect_policy_inputs_object_pair_full_sweep.py / object_pairs.ipynb's
2-pass methodology, but:

  * env_pos and env_neg share the *same* BDDL — only the agentview camera
    pose/FOV differs. There is no SceneRetargetTask, no expand_pos_to_neg
    — both envs have identical (nq, nv), so we just call
    `env_neg.set_init_state(env_pos.get_sim_state())` to replay.
  * The perturbation is the visibility-constrained RandomCameraViewPerturbation
    from notebooks/stress_test/08_camera_view_perturbation.ipynb's
    cam_random_large preset (defined inline below).

Pass 1 records env_pos sim states + obs (drives `get_action`); Pass 2a
replays each recorded sim state into env_neg and captures the perturbed
render (drive_source=1, pos_drives); Pass 2b drives env_neg with `get_action`
fed perturbed obs and captures env_pos's clean render at the same MuJoCo
state (drive_source=0, neg_drives).

Parallelism: --world-size shards episodes across ranks. Each rank handles
episodes whose `(ep + rank_offset) % world_size == rank` (i.e. round-robin
on episode index for both N_POS and N_NEG independently). Each rank writes
shard NPZs + meta.json under <scratch>/rank{rank}/. The 'finalize' mode
concatenates all shards into the unified positive.npz / negative.npz +
per-config subdir + manifest.json that the SVD scripts under
notebooks/lqr/svd/ consume directly.

Output layout (after finalize):

    <OUT_DIR>/
      positive.npz                # unified, all rows across all ranks
      negative.npz                # unified, paired row-for-row with positive
      manifest.json
      <name_hint>/                # per-config subdir (SVD-script compatible)
        positive.npz
        negative.npz
        prompt.txt
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

# Match notebooks/_setup.py — must run BEFORE any cosmos_policy import.
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
from scipy.spatial.transform import Rotation as R  # noqa: E402

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


# ====================================================================
# Logging — direct fd writes survive nbconvert / sbatch's buffering.
# ====================================================================

def _log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


# ====================================================================
# Camera-view perturbation (copied verbatim from
# notebooks/stress_test/08_camera_view_perturbation.ipynb's cell 4, with
# the visibility-constrained rejection sampler. Keep in sync if that
# notebook's classes change.)
# ====================================================================

def _resolve_sim(env):
    cur, seen = env, set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "sim"):
            return cur.sim
        cur = getattr(cur, "env", None)
    raise RuntimeError("could not locate sim on env")


def _project_world_points(points, cam_pos, cam_rot, fovy_deg, image_size):
    R_wc = cam_rot.as_matrix()
    p_w = np.asarray(points, dtype=float).reshape(-1, 3)
    p_c = (p_w - cam_pos[None, :]) @ R_wc
    depth = -p_c[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    f = (image_size / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    u = image_size / 2.0 + f * (p_c[:, 0] / safe)
    v = image_size / 2.0 - f * (p_c[:, 1] / safe)
    return u, v, depth


@dataclass
class CameraViewPerturbation:
    dpos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    drot_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dfov: float = 0.0
    camera_name: str = "agentview"
    name_hint: str = "cam"
    _baseline_pos: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _baseline_quat: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _baseline_fov: Optional[float] = field(default=None, init=False, repr=False)
    _last_sample: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def slug(self) -> str:
        dx, dy, dz = self.dpos
        rx, ry, rz = self.drot_euler
        payload = (f"p{dx:+.3f},{dy:+.3f},{dz:+.3f}|"
                   f"r{rx:+.3f},{ry:+.3f},{rz:+.3f}|f{self.dfov:+.2f}")
        h = hashlib.md5(payload.encode()).hexdigest()[:6]
        return f"{self.name_hint}_{h}"

    def _resolve_pose(self, episode_idx):
        return self.dpos, self.drot_euler, self.dfov

    def _on_baseline_captured(self, env):
        pass

    def apply_to_env(self, env, episode_idx=0):
        sim = _resolve_sim(env)
        cam_id = sim.model.camera_name2id(self.camera_name)
        if self._baseline_pos is None:
            self._baseline_pos = sim.model.cam_pos[cam_id].copy()
            self._baseline_quat = sim.model.cam_quat[cam_id].copy()
            self._baseline_fov = float(sim.model.cam_fovy[cam_id])
            self._on_baseline_captured(env)

        dpos, drot_euler, dfov = self._resolve_pose(episode_idx)
        self._last_sample = {
            "episode_idx": int(episode_idx),
            "dpos": [float(v) for v in dpos],
            "drot_euler_rad": [float(v) for v in drot_euler],
            "drot_euler_deg": [float(np.degrees(v)) for v in drot_euler],
            "dfov_deg": float(dfov),
        }

        sim.model.cam_pos[cam_id] = self._baseline_pos + np.asarray(dpos, dtype=float)
        bw, bx, by, bz = self._baseline_quat
        base_rot = R.from_quat([bx, by, bz, bw])
        delta = R.from_euler("xyz", drot_euler, degrees=False)
        new_rot = delta * base_rot
        nx, ny, nz, nw = new_rot.as_quat()
        sim.model.cam_quat[cam_id] = np.array([nw, nx, ny, nz],
                                              dtype=sim.model.cam_quat.dtype)
        sim.model.cam_fovy[cam_id] = self._baseline_fov + float(dfov)
        sim.forward()

    def manifest(self) -> dict:
        return {
            "kind": "CameraViewPerturbation", "slug": self.slug,
            "camera_name": self.camera_name,
            "dpos": list(map(float, self.dpos)),
            "drot_euler_rad": list(map(float, self.drot_euler)),
            "drot_euler_deg": [float(np.degrees(a)) for a in self.drot_euler],
            "dfov_deg": float(self.dfov),
        }


@dataclass
class RandomCameraViewPerturbation(CameraViewPerturbation):
    pos_sigma: float = 0.10
    rot_sigma_rad: float = float(np.radians(8.0))
    fov_sigma: float = 5.0
    base_seed: int = 42
    enforce_visibility: bool = True
    workspace_center: Optional[Tuple[float, float, float]] = None
    workspace_half_extent: Optional[Tuple[float, float, float]] = None
    workspace_table_z: float = 0.90
    workspace_visible_fraction: float = 0.55
    visibility_margin_px: int = 8
    image_size: int = 256
    max_rejection_attempts: int = 2000
    _workspace_corners: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _resolved_workspace_center: Optional[Tuple[float, float, float]] = field(
        default=None, init=False, repr=False)
    _resolved_workspace_half_extent: Optional[Tuple[float, float, float]] = field(
        default=None, init=False, repr=False)
    _last_rejection_count: int = field(default=0, init=False, repr=False)

    @property
    def slug(self) -> str:
        payload = (f"pos{self.pos_sigma:.4f}|rot{self.rot_sigma_rad:.4f}|"
                   f"fov{self.fov_sigma:.3f}|seed{self.base_seed}|"
                   f"vis{int(self.enforce_visibility)}")
        h = hashlib.md5(payload.encode()).hexdigest()[:6]
        suffix = "_vis" if self.enforce_visibility else ""
        return f"{self.name_hint}{suffix}_seed{self.base_seed}_{h}"

    def _on_baseline_captured(self, env):
        if not self.enforce_visibility:
            return
        bw, bx, by, bz = self._baseline_quat
        base_rot = R.from_quat([bx, by, bz, bw])
        R_wc = base_rot.as_matrix()
        view_dir = R_wc @ np.array([0.0, 0.0, -1.0])
        if abs(view_dir[2]) >= 1e-6:
            t = (self.workspace_table_z - self._baseline_pos[2]) / view_dir[2]
            target = self._baseline_pos + t * view_dir
            distance = abs(float(t))
        else:
            target = np.array([0.0, 0.0, self.workspace_table_z])
            distance = max(abs(self._baseline_pos[2] - self.workspace_table_z), 0.5)
        center = (self.workspace_center if self.workspace_center is not None
                  else (float(target[0]), float(target[1]),
                        float(self.workspace_table_z + 0.05)))
        if self.workspace_half_extent is not None:
            half = tuple(map(float, self.workspace_half_extent))
        else:
            visible_half = distance * np.tan(np.radians(self._baseline_fov) / 2.0)
            xy_half = float(visible_half * self.workspace_visible_fraction)
            half = (xy_half, xy_half, 0.08)
        self._resolved_workspace_center = tuple(map(float, center))
        self._resolved_workspace_half_extent = tuple(map(float, half))
        cx, cy, cz = center
        hx, hy, hz = half
        self._workspace_corners = np.array(
            [[cx + sx * hx, cy + sy * hy, cz + sz * hz]
             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=float,
        )

    def _candidate_camera(self, dpos, drot_euler, dfov):
        cam_pos = self._baseline_pos + np.asarray(dpos, dtype=float)
        bw, bx, by, bz = self._baseline_quat
        base_rot = R.from_quat([bx, by, bz, bw])
        delta = R.from_euler("xyz", drot_euler, degrees=False)
        new_rot = delta * base_rot
        return cam_pos, new_rot, self._baseline_fov + float(dfov)

    def _visibility_ok(self, dpos, drot_euler, dfov):
        if self._workspace_corners is None:
            return True
        cam_pos, new_rot, fovy = self._candidate_camera(dpos, drot_euler, dfov)
        u, v, depth = _project_world_points(
            self._workspace_corners, cam_pos, new_rot, fovy, self.image_size,
        )
        m = float(self.visibility_margin_px)
        return bool(
            np.all(depth > 0.05) and
            np.all(u >= m) and np.all(u <= self.image_size - m) and
            np.all(v >= m) and np.all(v <= self.image_size - m)
        )

    def _resolve_pose(self, episode_idx):
        ss = np.random.SeedSequence([int(self.base_seed), int(episode_idx)])
        rng = np.random.default_rng(ss)
        if not self.enforce_visibility:
            dpos = tuple(float(v) for v in rng.normal(0.0, self.pos_sigma, size=3))
            drot_euler = tuple(float(v) for v in rng.normal(0.0, self.rot_sigma_rad, size=3))
            dfov = float(rng.normal(0.0, self.fov_sigma))
            self._last_rejection_count = 0
            return dpos, drot_euler, dfov
        for attempt in range(self.max_rejection_attempts):
            dpos = tuple(float(v) for v in rng.normal(0.0, self.pos_sigma, size=3))
            drot_euler = tuple(float(v) for v in rng.normal(0.0, self.rot_sigma_rad, size=3))
            dfov = float(rng.normal(0.0, self.fov_sigma))
            if self._visibility_ok(dpos, drot_euler, dfov):
                self._last_rejection_count = attempt
                return dpos, drot_euler, dfov
        self._last_rejection_count = self.max_rejection_attempts
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0

    def manifest(self) -> dict:
        out = {
            "kind": "RandomCameraViewPerturbation", "slug": self.slug,
            "camera_name": self.camera_name,
            "sampling": "per-episode Gaussian, seed=SeedSequence([base_seed, episode_idx])",
            "pos_sigma_m": float(self.pos_sigma),
            "rot_sigma_rad": float(self.rot_sigma_rad),
            "rot_sigma_deg": float(np.degrees(self.rot_sigma_rad)),
            "fov_sigma_deg": float(self.fov_sigma),
            "base_seed": int(self.base_seed),
            "enforce_visibility": bool(self.enforce_visibility),
        }
        if self.enforce_visibility:
            out.update({
                "visibility_margin_px": int(self.visibility_margin_px),
                "image_size": int(self.image_size),
                "max_rejection_attempts": int(self.max_rejection_attempts),
                "workspace_table_z": float(self.workspace_table_z),
                "workspace_visible_fraction": float(self.workspace_visible_fraction),
            })
            if self._resolved_workspace_center is not None:
                out["resolved_workspace_center"] = list(map(float, self._resolved_workspace_center))
                out["resolved_workspace_half_extent"] = list(map(float, self._resolved_workspace_half_extent))
        return out


# ====================================================================
# Obs / rollout helpers
# ====================================================================

def _store_obs(obs_packed):
    return {
        "primary_image": np.ascontiguousarray(obs_packed["primary_image"]),
        "wrist_image":   np.ascontiguousarray(obs_packed["wrist_image"]),
        "proprio":       np.asarray(obs_packed["proprio"], dtype=np.float32),
    }


def record_pos_rollout(env_pos, init_state, prompt, eval_cfg, model, dataset_stats,
                       max_env_steps, *, num_steps_wait=10):
    """Drive env_pos (clean cam). Per inference, snapshot the env_pos sim state
    + the obs that the policy saw."""
    env_pos.reset()
    obs = env_pos.set_init_state(init_state)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env_pos.step(get_libero_dummy_action(eval_cfg.model_family))

    queue = deque(maxlen=eval_cfg.num_open_loop_steps)
    records = []
    success = False
    t = 0
    while t < max_env_steps:
        if not queue:
            obs_packed = prepare_observation(
                obs, resize_size=224, flip_images=eval_cfg.flip_images)
            records.append({
                "sim_state": np.asarray(env_pos.get_sim_state(), dtype=np.float64).copy(),
                "obs_stored": _store_obs(obs_packed),
            })
            out = get_action(
                eval_cfg, model, dataset_stats, obs_packed, prompt,
                num_denoising_steps_action=eval_cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            for a in out["actions"][:eval_cfg.num_open_loop_steps]:
                queue.append(np.asarray(a, dtype=np.float32))
        a = queue.popleft()
        obs, _, done, _ = env_pos.step(a.tolist())
        if done:
            success = True
            break
        t += 1
    return success, t + num_steps_wait, records


def render_pos_states_in_neg(env_neg, init_state, records, perturbation, episode_idx,
                             eval_cfg):
    """Replay each captured env_pos sim state into env_neg (perturbed cam)
    and capture the rendered obs."""
    env_neg.reset()
    env_neg.set_init_state(init_state)
    # Apply the perturbation *after* reset so it lands on the freshly-built
    # MjSim model. set_init_state on the same MjSim does not reset
    # sim.model.cam_pos, so the perturbation persists across the replay loop.
    perturbation.apply_to_env(env_neg, episode_idx=episode_idx)
    neg_stored = []
    for rec in records:
        obs_neg = env_neg.set_init_state(rec["sim_state"])
        obs_neg_packed = prepare_observation(
            obs_neg, resize_size=224, flip_images=eval_cfg.flip_images)
        neg_stored.append(_store_obs(obs_neg_packed))
    return neg_stored


def rollout_collect_neg_drives(init_state, env_neg, env_pos, perturbation, prompt,
                                eval_cfg, model, dataset_stats, max_env_steps,
                                episode_idx, *, num_steps_wait=10):
    """Drive env_neg (perturbed cam). Per inference, capture both the
    perturbed obs (what the policy sees) and the clean obs that env_pos
    would render at the same MuJoCo state (via state injection)."""
    env_neg.reset()
    obs_driver = env_neg.set_init_state(init_state)
    perturbation.apply_to_env(env_neg, episode_idx=episode_idx)
    env_pos.reset()
    env_pos.set_init_state(init_state)
    for _ in range(num_steps_wait):
        obs_driver, _, _, _ = env_neg.step(
            get_libero_dummy_action(eval_cfg.model_family))

    queue = deque(maxlen=eval_cfg.num_open_loop_steps)
    neg_inputs, pos_inputs = [], []
    success = False
    t = 0
    while t < max_env_steps:
        if not queue:
            neg_packed = prepare_observation(
                obs_driver, resize_size=224, flip_images=eval_cfg.flip_images)
            driver_state_flat = env_neg.get_sim_state()
            obs_replay = env_pos.set_init_state(driver_state_flat)
            pos_packed = prepare_observation(
                obs_replay, resize_size=224, flip_images=eval_cfg.flip_images)
            neg_inputs.append(_store_obs(neg_packed))
            pos_inputs.append(_store_obs(pos_packed))
            out = get_action(
                eval_cfg, model, dataset_stats, neg_packed, prompt,
                num_denoising_steps_action=eval_cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            for a in out["actions"][:eval_cfg.num_open_loop_steps]:
                queue.append(np.asarray(a, dtype=np.float32))
        a = queue.popleft()
        obs_driver, _, done, _ = env_neg.step(a.tolist())
        if done:
            success = True
            break
        t += 1
    return success, t + num_steps_wait, neg_inputs, pos_inputs


# ====================================================================
# Sharded NPZ I/O
# ====================================================================

_PER_CFG_KEYS = (
    "primary_images", "wrist_images", "proprios",
    "episode_idx", "inference_idx", "drive_source", "config_idx",
)


def _stack_rows(rows):
    """rows: list of dicts with primary_image / wrist_image / proprio
    return dict of stacked arrays."""
    return {
        "primary_images": np.stack([r["primary_image"] for r in rows], axis=0),
        "wrist_images":   np.stack([r["wrist_image"]   for r in rows], axis=0),
        "proprios":       np.stack([r["proprio"]       for r in rows], axis=0),
    }


def _save_shard(shard_dir: Path, pos_rows, neg_rows,
                episode_idx_list, inference_idx_list,
                drive_source_list, summaries):
    shard_dir.mkdir(parents=True, exist_ok=True)
    if not pos_rows:
        # Empty shard — write an empty marker but still emit meta.json so
        # finalize knows the rank ran.
        (shard_dir / "meta.json").write_text(json.dumps({
            "n_rows": 0, "summaries": summaries,
        }, indent=2))
        return 0
    pos_arrs = _stack_rows(pos_rows)
    neg_arrs = _stack_rows(neg_rows)
    np.savez_compressed(
        shard_dir / "positive.npz",
        **pos_arrs,
        episode_idx=np.asarray(episode_idx_list, dtype=np.int32),
        inference_idx=np.asarray(inference_idx_list, dtype=np.int32),
        drive_source=np.asarray(drive_source_list, dtype=np.int32),
        config_idx=np.zeros(len(episode_idx_list), dtype=np.int32),
    )
    np.savez_compressed(
        shard_dir / "negative.npz",
        **neg_arrs,
        episode_idx=np.asarray(episode_idx_list, dtype=np.int32),
        inference_idx=np.asarray(inference_idx_list, dtype=np.int32),
        drive_source=np.asarray(drive_source_list, dtype=np.int32),
        config_idx=np.zeros(len(episode_idx_list), dtype=np.int32),
    )
    (shard_dir / "meta.json").write_text(json.dumps({
        "n_rows": len(pos_rows),
        "summaries": summaries,
    }, indent=2))
    return len(pos_rows)


# ====================================================================
# collect mode (one rank)
# ====================================================================

def run_collect(args):
    """Run a single sbatch-array rank. Builds env_pos + env_neg, processes its
    shard of episodes, writes shards to <scratch>/rank{R}/."""
    rank = args.rank
    ws = args.world_size
    assert 0 <= rank < ws

    scratch = args.scratch_dir.resolve()
    shard_dir = scratch / f"rank{rank}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _log(f"rank={rank}/{ws}  shard_dir={shard_dir}")

    # ---------- libero suite + model ----------
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    max_env_steps = TASK_MAX_STEPS[args.suite]
    _log(f"libero suite={args.suite} task={args.task_id}  "
         f"init_states={init_states.shape[0]}  max_env_steps={max_env_steps}")

    n_pos = args.n_pos_rollouts
    n_neg = args.n_neg_rollouts
    n_max = max(n_pos, n_neg)
    if n_max > init_states.shape[0]:
        raise ValueError(
            f"requested max(n_pos={n_pos}, n_neg={n_neg}) = {n_max} > "
            f"available init_states = {init_states.shape[0]}"
        )

    eval_cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=args.ckpt_path,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{args.ckpt_path}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{args.ckpt_path}/libero_t5_embeddings.pkl",
        use_wrist_image=True, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True,
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

    # ---------- envs (one clean, one perturbed) ----------
    _log("building env_pos (clean cam)")
    env_pos, base_task_desc = get_libero_env(task, "cosmos", resolution=args.resolution)
    _log("building env_neg (perturbed cam)")
    env_neg, _ = get_libero_env(task, "cosmos", resolution=args.resolution)
    prompt = base_task_desc

    perturbation = RandomCameraViewPerturbation(
        pos_sigma=args.pos_sigma,
        rot_sigma_rad=float(np.radians(args.rot_sigma_deg)),
        fov_sigma=args.fov_sigma,
        base_seed=args.cam_base_seed,
        enforce_visibility=not args.disable_visibility,
        workspace_table_z=args.workspace_table_z,
        workspace_visible_fraction=args.workspace_visible_fraction,
        visibility_margin_px=args.visibility_margin_px,
        image_size=args.resolution,
        max_rejection_attempts=args.max_rejection_attempts,
        name_hint=args.preset_name,
    )

    # ---------- accumulators (per-rank shard) ----------
    pos_rows, neg_rows = [], []
    ep_list, inf_list, ds_list = [], [], []
    summaries = []

    # ---------- Pass 1 + Pass 2a for assigned pos episodes ----------
    pos_eps = [ep for ep in range(n_pos) if (ep % ws) == rank]
    neg_eps = [ep for ep in range(n_neg) if (ep % ws) == rank]
    _log(f"assigned pos episodes: {pos_eps}")
    _log(f"assigned neg episodes: {neg_eps}")

    for ep in pos_eps:
        _log(f"ep={ep} pass1 (record env_pos) START")
        t0 = time.time()
        success, env_steps, records = record_pos_rollout(
            env_pos, init_states[ep], prompt,
            eval_cfg, model, dataset_stats, max_env_steps,
        )
        dt = time.time() - t0
        tag = "SUCCESS" if success else "FAILURE"
        _log(f"ep={ep} pass1 {tag} steps={env_steps} inf={len(records)} {dt:.1f}s")
        _torch.cuda.empty_cache()

        # Pass 2a: replay each pos sim state into env_neg.
        _log(f"ep={ep} pass2a (replay into env_neg) START inf={len(records)}")
        t0 = time.time()
        neg_stored = render_pos_states_in_neg(
            env_neg, init_states[ep], records, perturbation, episode_idx=ep,
            eval_cfg=eval_cfg,
        )
        dt = time.time() - t0
        assert len(neg_stored) == len(records)
        sample = dict(perturbation._last_sample)
        rej = int(perturbation._last_rejection_count)
        _log(f"ep={ep} pass2a replayed inf={len(records)} {dt:.1f}s  "
             f"sample={sample}  rejections={rej}")

        for inf_idx, (rec, n_st) in enumerate(zip(records, neg_stored)):
            p_st = rec["obs_stored"]
            pos_rows.append(p_st)
            neg_rows.append(n_st)
            ep_list.append(ep)
            inf_list.append(inf_idx)
            ds_list.append(1)  # pos_drives

        summaries.append({
            "rank": rank, "episode": ep, "drive_source": 1,
            "drive_name": "pos_drives",
            "pass1_success": bool(success), "pass1_env_steps": int(env_steps),
            "n_inferences": len(records),
            "wall_time_s_pass1_and_2a": dt + (time.time() - t0),
            "perturb_sample": sample, "rejection_count": rej,
        })

    # ---------- Pass 2b for assigned neg episodes ----------
    for ep in neg_eps:
        _log(f"ep={ep} pass2b (neg-drives) START")
        t0 = time.time()
        success, env_steps, neg_inputs, pos_inputs = rollout_collect_neg_drives(
            init_states[ep], env_neg, env_pos, perturbation, prompt,
            eval_cfg, model, dataset_stats, max_env_steps,
            episode_idx=ep,
        )
        dt = time.time() - t0
        sample = dict(perturbation._last_sample)
        rej = int(perturbation._last_rejection_count)
        tag = "SUCCESS" if success else "FAILURE"
        _log(f"ep={ep} pass2b {tag} steps={env_steps} inf={len(neg_inputs)} {dt:.1f}s "
             f" sample={sample}  rejections={rej}")
        assert len(neg_inputs) == len(pos_inputs)

        for inf_idx, (n_rec, p_rec) in enumerate(zip(neg_inputs, pos_inputs)):
            pos_rows.append(p_rec)
            neg_rows.append(n_rec)
            ep_list.append(ep)
            inf_list.append(inf_idx)
            ds_list.append(0)  # neg_drives

        summaries.append({
            "rank": rank, "episode": ep, "drive_source": 0,
            "drive_name": "neg_drives",
            "success": bool(success), "env_steps": int(env_steps),
            "n_inferences": len(neg_inputs),
            "wall_time_s": dt,
            "perturb_sample": sample, "rejection_count": rej,
        })
        _torch.cuda.empty_cache()

    env_pos.close()
    env_neg.close()

    n_saved = _save_shard(shard_dir, pos_rows, neg_rows,
                          ep_list, inf_list, ds_list, summaries)
    _log(f"rank={rank} DONE; shard rows={n_saved} -> {shard_dir}")


# ====================================================================
# finalize mode (login node)
# ====================================================================

def run_finalize(args):
    """Concatenate all rank shards into the unified positive.npz / negative.npz
    + per-config subdir + manifest.json."""
    scratch = args.scratch_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(scratch.glob("rank*/positive.npz"))
    if not shards:
        raise FileNotFoundError(f"no rank shards under {scratch}")
    _log(f"finalize: found {len(shards)} shards in {scratch}")

    # Load each shard, concat all rows.
    pos_chunks, neg_chunks = [], []
    ep_arr, inf_arr, ds_arr = [], [], []
    summaries = []
    total_rows = 0
    for shard_path in shards:
        rank_dir = shard_path.parent
        pos = dict(np.load(rank_dir / "positive.npz", allow_pickle=False))
        neg = dict(np.load(rank_dir / "negative.npz", allow_pickle=False))
        meta = json.loads((rank_dir / "meta.json").read_text())
        summaries.extend(meta.get("summaries", []))
        n = int(pos["episode_idx"].shape[0])
        if n == 0:
            continue
        pos_chunks.append(pos)
        neg_chunks.append(neg)
        ep_arr.append(pos["episode_idx"])
        inf_arr.append(pos["inference_idx"])
        ds_arr.append(pos["drive_source"])
        total_rows += n
        _log(f"  loaded {rank_dir.name}: {n} rows")

    if total_rows == 0:
        raise RuntimeError("all shards empty — nothing to finalize")

    ep_arr = np.concatenate(ep_arr)
    inf_arr = np.concatenate(inf_arr)
    ds_arr = np.concatenate(ds_arr)
    # Deterministic order: by (drive_source DESC, episode, inference). pos_drives
    # (ds=1) before neg_drives (ds=0) — same as object_pairs convention.
    order = np.lexsort((inf_arr, ep_arr, -ds_arr))

    def _concat_key(chunks, key):
        return np.concatenate([c[key] for c in chunks], axis=0)

    pos_primary = _concat_key(pos_chunks, "primary_images")[order]
    pos_wrist   = _concat_key(pos_chunks, "wrist_images")[order]
    pos_proprio = _concat_key(pos_chunks, "proprios")[order]
    neg_primary = _concat_key(neg_chunks, "primary_images")[order]
    neg_wrist   = _concat_key(neg_chunks, "wrist_images")[order]
    neg_proprio = _concat_key(neg_chunks, "proprios")[order]
    ep_arr  = ep_arr[order]
    inf_arr = inf_arr[order]
    ds_arr  = ds_arr[order]
    cfg_arr = np.zeros_like(ep_arr, dtype=np.int32)

    # Pairing sanity check: proprios should match row-for-row.
    max_dproprio = float(np.max(np.abs(pos_proprio - neg_proprio)))
    _log(f"paired proprio max |Δ| = {max_dproprio:.3e}  "
         f"(should be ~0; same MuJoCo state on both sides)")

    name_hint = args.preset_name
    prompts_lookup = np.array([args.prompt], dtype=object)
    name_hints_lookup = np.array([name_hint], dtype=object)

    POSITIVE_NPZ = out_dir / "positive.npz"
    NEGATIVE_NPZ = out_dir / "negative.npz"
    MANIFEST_JSON = out_dir / "manifest.json"

    np.savez_compressed(
        POSITIVE_NPZ,
        primary_images=pos_primary, wrist_images=pos_wrist, proprios=pos_proprio,
        episode_idx=ep_arr, inference_idx=inf_arr, drive_source=ds_arr,
        config_idx=cfg_arr,
        prompts=prompts_lookup, name_hints=name_hints_lookup,
    )
    np.savez_compressed(
        NEGATIVE_NPZ,
        primary_images=neg_primary, wrist_images=neg_wrist, proprios=neg_proprio,
        episode_idx=ep_arr, inference_idx=inf_arr, drive_source=ds_arr,
        config_idx=cfg_arr,
        prompts=prompts_lookup, name_hints=name_hints_lookup,
    )
    _log(f"wrote {POSITIVE_NPZ.name} ({POSITIVE_NPZ.stat().st_size/1e6:.1f} MB) "
         f"primary={pos_primary.shape}")
    _log(f"wrote {NEGATIVE_NPZ.name} ({NEGATIVE_NPZ.stat().st_size/1e6:.1f} MB) "
         f"primary={neg_primary.shape}")

    # Per-config subdir (compatible with run_partition_svd_pairs_no_action.sh).
    sub_dir = out_dir / name_hint
    sub_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sub_dir / "positive.npz",
        primary_images=pos_primary, wrist_images=pos_wrist, proprios=pos_proprio,
        episode_idx=ep_arr, inference_idx=inf_arr, drive_source=ds_arr,
        config_idx=cfg_arr,
    )
    np.savez_compressed(
        sub_dir / "negative.npz",
        primary_images=neg_primary, wrist_images=neg_wrist, proprios=neg_proprio,
        episode_idx=ep_arr, inference_idx=inf_arr, drive_source=ds_arr,
        config_idx=cfg_arr,
    )
    (sub_dir / "prompt.txt").write_text(args.prompt + "\n")
    _log(f"per-config subdir written: {sub_dir}")

    # Manifest.
    n_pos_rows = int((ds_arr == 1).sum())
    n_neg_rows = int((ds_arr == 0).sum())
    manifest = {
        "suite": args.suite,
        "task_id": args.task_id,
        "resolution": args.resolution,
        "n_pos_rollouts_requested": args.n_pos_rollouts,
        "n_neg_rollouts_requested": args.n_neg_rollouts,
        "world_size_collect": args.world_size,
        "perturb": {
            "preset_name": args.preset_name,
            "base_seed": args.cam_base_seed,
            "pos_sigma_m": args.pos_sigma,
            "rot_sigma_deg": args.rot_sigma_deg,
            "fov_sigma_deg": args.fov_sigma,
            "enforce_visibility": not args.disable_visibility,
            "workspace_table_z": args.workspace_table_z,
            "workspace_visible_fraction": args.workspace_visible_fraction,
            "visibility_margin_px": args.visibility_margin_px,
            "max_rejection_attempts": args.max_rejection_attempts,
        },
        "selection_rule": (
            "all init_states[0..max(n_pos,n_neg)-1]; pass1+pass2a over "
            "init_states[0..n_pos-1], pass2b over init_states[0..n_neg-1]"
        ),
        "pairing": (
            "row i in positive.npz and row i in negative.npz share the same "
            "MuJoCo state at capture time. drive_source=0 = env_neg drove "
            "(perturbed-cam policy), env_pos was state-injected; drive_source=1 "
            "= env_pos drove (clean-cam policy), env_neg was state-injected by "
            "replaying env_pos.get_sim_state() into env_neg.set_init_state()."
        ),
        "image_layout": "HWC uint8, flip_images=True applied at capture time",
        "proprio_layout": ("concat(robot0_gripper_qpos[2], robot0_eef_pos[3], "
                            "robot0_eef_quat[4]) -> shape (9,) float32"),
        "drive_sources": [
            {"code": 0, "name": "neg_drives",
             "desc": "env_neg (perturbed cam) drives; env_pos state-injected"},
            {"code": 1, "name": "pos_drives",
             "desc": "env_pos (clean cam) drives; env_neg state-injected via captured sim_state"},
        ],
        "policy_prompt": args.prompt,
        "sets": {
            "positive": {"out_npz": str(POSITIVE_NPZ),
                          "role": "clean-cam render at every captured pose"},
            "negative": {"out_npz": str(NEGATIVE_NPZ),
                          "role": "perturbed-cam render at every captured pose"},
        },
        "per_config_subdir": str(sub_dir),
        "rollouts": summaries,
        "total_paired_rows": int(pos_primary.shape[0]),
        "n_pos_drives_rows": n_pos_rows,
        "n_neg_drives_rows": n_neg_rows,
        "paired_proprio_max_abs_diff": max_dproprio,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    _log(f"wrote manifest -> {MANIFEST_JSON}")

    if not args.keep_scratch:
        _log(f"cleaning scratch {scratch}")
        # Be conservative: only remove rank* subdirs we wrote.
        import shutil
        for shard_path in shards:
            shutil.rmtree(shard_path.parent, ignore_errors=True)


# ====================================================================
# main
# ====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["collect", "finalize"], required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)

    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--keep-scratch", action="store_true")

    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--ckpt-path", type=str,
                    default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")

    ap.add_argument("--n-pos-rollouts", type=int, default=10,
                    help="how many episodes for pass1+pass2a (env_pos drives)")
    ap.add_argument("--n-neg-rollouts", type=int, default=10,
                    help="how many episodes for pass2b (env_neg drives)")
    ap.add_argument("--prompt", type=str, default=None,
                    help="policy prompt; default = libero task description "
                         "(set in finalize manifest to whatever was used)")

    # Camera perturbation knobs (defaults match the cam_random_large preset
    # in notebooks/stress_test/08_camera_view_perturbation.ipynb).
    ap.add_argument("--cam-base-seed", type=int, default=42)
    ap.add_argument("--pos-sigma", type=float, default=0.10)
    ap.add_argument("--rot-sigma-deg", type=float, default=8.0)
    ap.add_argument("--fov-sigma", type=float, default=5.0)
    ap.add_argument("--disable-visibility", action="store_true",
                    help="turn off the workspace-AABB rejection sampler")
    ap.add_argument("--workspace-table-z", type=float, default=0.90)
    ap.add_argument("--workspace-visible-fraction", type=float, default=0.55)
    ap.add_argument("--visibility-margin-px", type=int, default=8)
    ap.add_argument("--max-rejection-attempts", type=int, default=2000)
    ap.add_argument("--preset-name", type=str, default="cam_random_large")

    args = ap.parse_args()

    # In collect mode we need a libero env to derive the default prompt;
    # if the user didn't pass one, fall back to the suite's task description.
    if args.prompt is None:
        task_suite = benchmark.get_benchmark_dict()[args.suite]()
        task = task_suite.get_task(args.task_id)
        args.prompt = task.language
        _log(f"prompt unset; using libero task description: {args.prompt!r}")

    if args.mode == "collect":
        run_collect(args)
    elif args.mode == "finalize":
        run_finalize(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
