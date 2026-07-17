#!/usr/bin/env python
"""Run LQR-decay-steered Cosmos-Policy rollouts on a *perturbed-camera*
LIBERO scene. Direct port of notebooks/lqr/run_lqr_decay_cosmos_policy.ipynb
with two changes:

  1. The agentview camera is perturbed per-episode by the visibility-
     constrained RandomCameraViewPerturbation (cam_random_large defaults).
     The perturbation samples are seeded by (CAM_BASE_SEED, episode_idx)
     so setting CAM_BASE_SEED equal to the base seed used by
     collect_policy_inputs_camera_view_perturbation.py reproduces the
     exact perturbations of the negative training rollouts; pick a
     different seed for an independent random perturbation (CAM_MODE knob).

  2. The episode loop supports --rank / --world-size sharding. Each
     rank loads the model + SVD V cache + precomputes K matrices, then
     runs episodes assigned by round-robin: `ep % world_size == rank`.
     Per-rank video files land directly under OUT_DIR (filenames carry the
     episode index, so multiple ranks don't collide). A separate
     --mode finalize call concatenates per-rank results into one
     results.json + a unified manifest.

The hooks, K precomputation, R_SCALE(c) schedule, and baseline-rollout
behavior are all inherited verbatim from the source notebook.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# ---- env setup BEFORE importing cosmos_policy / libero / matplotlib ----
_HERE = Path(__file__).resolve().parent
_NOTEBOOKS_ROOT = _HERE.parent
if str(_NOTEBOOKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_ROOT))
from _setup import setup_env  # noqa: E402

setup_env()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from cosmos_policy.experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env, get_libero_dummy_action,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (  # noqa: E402
    PolicyEvalConfig, prepare_observation, TASK_MAX_STEPS,
)
from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    COSMOS_IMAGE_SIZE, get_action, get_model,
    init_t5_text_embeddings_cache, load_dataset_stats,
)


def _log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


# ====================================================================
# Camera-view perturbation (copy of the visibility-constrained sampler
# from notebooks/stress_test/08_camera_view_perturbation.ipynb).
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
        sim.model.cam_quat[cam_id] = np.array(
            [nw, nx, ny, nz], dtype=sim.model.cam_quat.dtype)
        sim.model.cam_fovy[cam_id] = self._baseline_fov + float(dfov)
        sim.forward()


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
    _last_rejection_count: int = field(default=0, init=False, repr=False)

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


# ====================================================================
# Module-level state populated by the rollout run (mirrors the notebook's
# globals so the steering hooks can read/write them).
# ====================================================================

_DEVICE: torch.device = torch.device("cpu")
_DEVICE_ID: int = 0
_LAMBDA: float = 1.0
_RUN_T0: float = 0.0

# Caches.
_V_CPU_CACHE: dict = {}
_V_GPU_CACHE: "OrderedDict" = OrderedDict()
_swap_stats = {"cpu_loads": 0, "gpu_swaps": 0, "gpu_hits": 0, "cpu_to_gpu_s": 0.0}

# Hook-shared mutable state.
state = {"pass_idx": -1, "u_step_pending": None}
in_ad = {"flag": False}
steering = {"enabled": True, "chunk_idx": 0}
u_norm_log: list = []

# Filled in by main(): SVD config / V_for / _lqr stack.
_SVD_CFG: dict = {}
_PARTITIONS: list = []
_LAYER_TO_PART: list = []
_SEL_T: list = []
_SEL_IDX_OF: dict = {}
_T_DIFF: int = 0
_SAMPLING_STEPS: int = 0
_L: int = 0
_R_DIM: int = 0
_T_P_DENOISE: int = 0
_DENOISE_T_START: int = 0
_DENOISE_T_END: int = 0
_H_P: int = 0
_W_P: int = 0
_D: int = 0
_SVD_DIR: Optional[Path] = None
_V_DEVICE: torch.device = torch.device("cpu")
_V_DTYPE = torch.bfloat16
_MAX_GPU_PARTITIONS: int = 0

_LQR = {}
_K_INTRA_PER_CHUNK: Optional[torch.Tensor] = None
_K_STEP_PER_CHUNK:  Optional[torch.Tensor] = None
_R_SCALE_SCHEDULE: list = []
_MAX_CHUNKS: int = 50


# --------------------------------------------------------------------
# V cache (CPU pre-load + on-demand CPU->GPU promotion).
# --------------------------------------------------------------------

def _v_file(p_idx: int, t_id: int) -> Path:
    a, b = _PARTITIONS[p_idx]
    pattern = f"V_part{p_idx}_layers{a}-{b}_t{t_id}_k*.pt"
    matches = sorted(_SVD_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no V file matching {pattern} in {_SVD_DIR}")
    preferred = [m for m in matches if m.name.endswith(f"_k{_SVD_CFG['k_target']}.pt")]
    return (preferred or matches)[0]


def _load_V_cpu(p_idx: int, t_id: int) -> torch.Tensor:
    key = (p_idx, t_id)
    V = _V_CPU_CACHE.get(key)
    if V is not None:
        return V
    fp = _v_file(p_idx, t_id)
    _log(f"  disk-load {fp.name} ({fp.stat().st_size/1e9:.2f} GB) -> CPU {_V_DTYPE}")
    t0 = time.time()
    raw = torch.load(fp, map_location="cpu", weights_only=False)["V"]
    V = raw.to(dtype=_V_DTYPE).contiguous()
    del raw
    _V_CPU_CACHE[key] = V
    _swap_stats["cpu_loads"] += 1
    _log(f"    CPU resident (p={p_idx}, t={t_id}): {tuple(V.shape)}  "
         f"~{V.element_size()*V.numel()/1e9:.2f} GB  ({time.time()-t0:.1f}s)")
    return V


def V_for(layer_idx: int, t_id: int) -> torch.Tensor:
    p_idx = _LAYER_TO_PART[layer_idx]
    key = (p_idx, t_id)
    V = _V_GPU_CACHE.get(key)
    if V is not None:
        _V_GPU_CACHE.move_to_end(key)
        _swap_stats["gpu_hits"] += 1
        return V
    V_cpu = _load_V_cpu(p_idx, t_id)
    if _V_DEVICE.type == "cpu":
        _V_GPU_CACHE[key] = V_cpu
        return V_cpu
    while len(_V_GPU_CACHE) >= _MAX_GPU_PARTITIONS:
        _, ev_V = _V_GPU_CACHE.popitem(last=False)
        del ev_V
        torch.cuda.empty_cache()
        _swap_stats["gpu_swaps"] += 1
    t0 = time.time()
    V = V_cpu.to(device=_V_DEVICE, non_blocking=False).contiguous()
    if _V_DEVICE.type == "cuda":
        torch.cuda.synchronize(_V_DEVICE)
    _swap_stats["cpu_to_gpu_s"] += time.time() - t0
    _V_GPU_CACHE[key] = V
    return V


# --------------------------------------------------------------------
# Steering hooks.
# --------------------------------------------------------------------

def _is_selected_step(pass_idx: int):
    if pass_idx < 0:
        return None, None
    step = pass_idx  # PASSES_PER_STEP == 1
    if step not in _SEL_IDX_OF:
        return None, None
    return step, _SEL_IDX_OF[step]


def _pass_tick(_block, _args):
    if not in_ad["flag"]:
        state["pass_idx"] += 1
    return None


def make_intra_hook(l_in: int):
    def hook(block, args, output):
        if in_ad["flag"] or not steering["enabled"]:
            return None
        step, sel = _is_selected_step(state["pass_idx"])
        if step is None:
            return None
        z_full = output[0, _DENOISE_T_START:_DENOISE_T_END, :, :, :].detach().reshape(-1)
        z_dt = output.dtype
        V_in = V_for(l_in, step)
        in_ad["flag"] = True
        try:
            x_proj = (z_full.to(V_in.dtype) @ V_in).float()
            v_fp = _LQR["v"][l_in, sel]
            mu_fp = _LQR["mu"][l_in, sel]
            K_fp = _LQR["K_intra"][sel, l_in]
            alpha = _LAMBDA * mu_fp - v_fp @ x_proj
            u_tilde = K_fp @ (alpha * v_fp)
            u_norm_log.append((sel, l_in, float(u_tilde.norm()), float(alpha)))
        finally:
            in_ad["flag"] = False
        del V_in
        V_out = V_for(l_in + 1, step)
        in_ad["flag"] = True
        try:
            u_full = V_out @ u_tilde.to(V_out.dtype)
        finally:
            in_ad["flag"] = False
        u_add = u_full.to(z_dt).reshape(_T_P_DENOISE, _H_P, _W_P, _D)
        output[0, _DENOISE_T_START:_DENOISE_T_END, :, :, :] = (
            output[0, _DENOISE_T_START:_DENOISE_T_END, :, :, :] + u_add
        )
        return output
    return hook


def cross_step_compute(block, args, output):
    if in_ad["flag"] or not steering["enabled"]:
        return None
    step, sel = _is_selected_step(state["pass_idx"])
    if step is None or sel >= _T_DIFF - 1:
        return None
    z_full = output[0, _DENOISE_T_START:_DENOISE_T_END, :, :, :].detach().reshape(-1)
    V_in = V_for(_L - 1, step)
    in_ad["flag"] = True
    try:
        x_proj = (z_full.to(V_in.dtype) @ V_in).float()
        v_fp = _LQR["v"][_L - 1, sel]
        mu_fp = _LQR["mu"][_L - 1, sel]
        K_fp = _LQR["K_step"][sel]
        alpha = _LAMBDA * mu_fp - v_fp @ x_proj
        u_tilde = K_fp @ (alpha * v_fp)
        u_norm_log.append((sel, -1, float(u_tilde.norm()), float(alpha)))
        state["u_step_pending"] = {"src_sel": sel, "u_tilde": u_tilde.detach()}
    finally:
        in_ad["flag"] = False
    return None


def cross_step_apply(block, args, output):
    if in_ad["flag"] or not steering["enabled"]:
        return None
    step, sel = _is_selected_step(state["pass_idx"])
    pending = state["u_step_pending"]
    if step is None or pending is None or sel == 0 or pending["src_sel"] != sel - 1:
        return None
    u_tilde = pending["u_tilde"]
    V_dest = V_for(0, step)
    in_ad["flag"] = True
    try:
        u_full = V_dest @ u_tilde.to(V_dest.dtype)
    finally:
        in_ad["flag"] = False
    u_add = u_full.to(output.dtype).reshape(_T_P_DENOISE, _H_P, _W_P, _D)
    output[0, _DENOISE_T_START:_DENOISE_T_END, :, :, :] = (
        output[0, _DENOISE_T_START:_DENOISE_T_END, :, :, :] + u_add
    )
    state["u_step_pending"] = None
    return output


# --------------------------------------------------------------------
# K precomputation (chained Riccati per chunk).
# --------------------------------------------------------------------

def _lqr_no_B(A_t, Q, R_, S_T):
    Tn, n, _ = A_t.shape
    S = torch.zeros(Tn + 1, n, n, dtype=A_t.dtype, device=A_t.device)
    K = torch.zeros(Tn,     n, n, dtype=A_t.dtype, device=A_t.device)
    S[Tn] = S_T
    for k in reversed(range(Tn)):
        Ak = A_t[k]
        P = S[k + 1] + R_[k]
        F = S[k + 1] @ Ak
        G = Q[k] + Ak.transpose(-2, -1) @ S[k + 1] @ Ak
        Kk = torch.linalg.solve(P, F)
        K[k] = Kk
        Snew = G - F.transpose(-2, -1) @ Kk
        S[k] = 0.5 * (Snew + Snew.transpose(-2, -1))
    return K


def _swap_K_for_chunk(c: int) -> None:
    c_eff = min(c, _MAX_CHUNKS - 1)
    _LQR["K_intra"].copy_(_K_INTRA_PER_CHUNK[c_eff].to(_V_DEVICE, dtype=torch.float32))
    if _K_STEP_PER_CHUNK.numel():
        _LQR["K_step"].copy_(_K_STEP_PER_CHUNK[c_eff].to(_V_DEVICE, dtype=torch.float32))


# --------------------------------------------------------------------
# Rollout helpers.
# --------------------------------------------------------------------

def save_video(frames, path, fps):
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        # LIBERO's agentview is rendered upside-down; flip for human viewing.
        writer.append_data(np.flipud(frame))
    writer.close()


def save_paired_video(perturbed_frames, clean_frames, path, fps):
    """Stack perturbed (left, what the policy saw) and clean (right,
    visualization-only) frames side-by-side per frame and write an mp4.
    Both lists must have the same length; missing tail frames on one side
    are filled with black so the video doesn't truncate mid-episode."""
    assert len(perturbed_frames) == len(clean_frames), (
        f"frame count mismatch: perturbed={len(perturbed_frames)} clean={len(clean_frames)}"
    )
    writer = imageio.get_writer(path, fps=fps)
    for p, c in zip(perturbed_frames, clean_frames):
        # LIBERO's agentview is rendered upside-down; flip for human viewing.
        p = np.flipud(p)
        c = np.flipud(c)
        # Pad to same height if they ever differ (shouldn't for matched resolutions).
        if p.shape[0] != c.shape[0]:
            h = max(p.shape[0], c.shape[0])
            def _pad(x):
                if x.shape[0] == h:
                    return x
                pad = np.zeros((h - x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype)
                return np.concatenate([x, pad], axis=0)
            p, c = _pad(p), _pad(c)
        writer.append_data(np.concatenate([p, c], axis=1))
    writer.close()


def rollout(env, init_state, task_desc, policy_fn, eval_cfg, max_env_steps,
            perturbation, *, episode_idx=0, num_steps_wait=10, env_clean=None):
    """Drive `env` (perturbed cam) with `policy_fn`. If `env_clean` is given,
    state-inject env's MuJoCo state into env_clean after every step and grab
    its agentview render alongside — purely for side-by-side visualization;
    env_clean is never touched by the policy."""
    env.reset()
    obs = env.set_init_state(init_state)
    if perturbation is not None:
        perturbation.apply_to_env(env, episode_idx=episode_idx)
    if env_clean is not None:
        env_clean.reset()
        env_clean.set_init_state(init_state)

    def _render_clean():
        if env_clean is None:
            return None
        obs_c = env_clean.set_init_state(env.get_sim_state())
        return obs_c["agentview_image"].copy()

    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(eval_cfg.model_family))
    queue = deque(maxlen=eval_cfg.num_open_loop_steps)
    frames = [obs["agentview_image"].copy()]
    clean_frames = [_render_clean()] if env_clean is not None else None
    success = False
    t = 0
    while t < max_env_steps:
        if not queue:
            observation = prepare_observation(
                obs, resize_size=COSMOS_IMAGE_SIZE, flip_images=eval_cfg.flip_images,
            )
            actions = policy_fn(observation, task_desc)
            for a in actions[:eval_cfg.num_open_loop_steps]:
                queue.append(np.asarray(a, dtype=np.float32))
        a = queue.popleft()
        obs, _, done, _ = env.step(a.tolist())
        frames.append(obs["agentview_image"].copy())
        if env_clean is not None:
            clean_frames.append(_render_clean())
        if done:
            success = True
            break
        t += 1
    return success, t + num_steps_wait, frames, clean_frames


# ====================================================================
# Output dir / config tag (mirrors the notebook's scheme).
# ====================================================================

def _slug(s: str, n: int = 24) -> str:
    return s[:n].strip().lower().replace(" ", "_").replace("/", "_").replace(".", "_")


def _compute_config_tag(args, prompt: str) -> str:
    base = (
        f"{args.suite}__task{args.task_id:02d}__lqr_decay_camperturb__"
        f"lam{args.lambda_:.2f}_q{args.q_scale:g}_rinit{args.r_scale_init:g}"
        f"_rfin{args.r_scale_final:g}_tau{args.r_scale_tau:g}_qf{args.qf_scale:g}__"
        f"cam{args.cam_mode}_seed{args.cam_base_seed}__{_slug(prompt)}"
    )
    if args.run_tag:
        return f"{base}__{args.run_tag}"
    return base


# ====================================================================
# Main per-rank work.
# ====================================================================

def run_rollouts(args):
    global _DEVICE, _DEVICE_ID, _LAMBDA, _SVD_CFG, _PARTITIONS, _LAYER_TO_PART
    global _SEL_T, _SEL_IDX_OF, _T_DIFF, _SAMPLING_STEPS, _L, _R_DIM
    global _T_P_DENOISE, _DENOISE_T_START, _DENOISE_T_END, _H_P, _W_P, _D
    global _SVD_DIR, _V_DEVICE, _MAX_GPU_PARTITIONS, _RUN_T0
    global _LQR, _K_INTRA_PER_CHUNK, _K_STEP_PER_CHUNK, _R_SCALE_SCHEDULE
    global _MAX_CHUNKS

    _RUN_T0 = time.time()
    _DEVICE = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    _DEVICE_ID = _DEVICE.index if _DEVICE.type == "cuda" else 0
    assert _DEVICE.type == "cuda", f"Cosmos-Policy requires CUDA; got {_DEVICE}"
    _log(f"rank={args.rank}/{args.world_size}  device={_DEVICE}")

    # ---- load SVD config + A_tilde + V tiles ----
    _SVD_DIR = Path(args.svd_dir)
    assert _SVD_DIR.exists(), f"SVD_DIR not found: {_SVD_DIR}"
    JAC_DIR_ACT = _SVD_DIR / args.jac_dir_act
    A_TILDE_FULL = JAC_DIR_ACT / "A_tilde__full.pt"
    assert A_TILDE_FULL.exists(), f"A_TILDE_FULL not found: {A_TILDE_FULL}"

    _SVD_CFG = json.loads((_SVD_DIR / "config.json").read_text())
    _SEL_T = list(_SVD_CFG["selected_timesteps"])
    _T_DIFF = len(_SEL_T)
    _SAMPLING_STEPS = _SVD_CFG["sampling_steps"]
    _L = _SVD_CFG["L"]
    _R_DIM = _SVD_CFG["k_target"]
    _PARTITIONS = [tuple(p) for p in _SVD_CFG["partitions"]]
    _SEL_IDX_OF = {t: i for i, t in enumerate(_SEL_T)}
    _log(f"sel_t={_SEL_T} T_diff={_T_DIFF} sampling_steps={_SAMPLING_STEPS} "
         f"L={_L} r={_R_DIM} D={_SVD_CFG['D']:,}")

    raw = torch.load(A_TILDE_FULL, map_location="cpu", weights_only=False)
    A_dict = raw.get("A_tilde", {})
    B_dict = raw.get("B_tilde", {})
    JAC_PROMPT = raw.get("prompt", "<unknown>")
    _log(f"A_tilde dict: {len(A_dict)} entries  B_tilde: {len(B_dict)}  "
         f"jac_prompt={JAC_PROMPT!r}")

    A_tilde = torch.zeros(_T_DIFF, _L - 1, _R_DIM, _R_DIM, dtype=torch.float32)
    for (t, l_in), Atl in A_dict.items():
        if t in _SEL_IDX_OF:
            A_tilde[_SEL_IDX_OF[t], l_in] = Atl.float()
    if len(B_dict) > 0:
        B_tilde = torch.zeros(_T_DIFF - 1, _R_DIM, _R_DIM, dtype=torch.float32)
        for (t,), Bt in B_dict.items():
            if t in _SEL_IDX_OF and _SEL_IDX_OF[t] < _T_DIFF - 1:
                B_tilde[_SEL_IDX_OF[t]] = Bt.float()
    else:
        _log("[note] B_tilde empty; cross-step transitions zero.")
        B_tilde = torch.zeros(max(_T_DIFF - 1, 0), _R_DIM, _R_DIM, dtype=torch.float32)

    summary = torch.load(_SVD_DIR / "svd_summary.pt", map_location="cpu", weights_only=False)
    c_means = summary["c_means"].float()  # (L, T_sel, k)
    tilde_mu = c_means.norm(dim=-1)
    tilde_v = c_means / tilde_mu.unsqueeze(-1).clamp(min=1e-12)
    _LAYER_TO_PART = list(summary["layer_to_part"])

    # ---- model load ----
    ckpt_path = _SVD_CFG.get("ckpt_path", "nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    config_name = _SVD_CFG.get(
        "config_name", "cosmos_predict2_2b_480p_libero__inference_only")
    eval_cfg = PolicyEvalConfig(
        config=config_name,
        ckpt_path=ckpt_path,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{ckpt_path}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{ckpt_path}/libero_t5_embeddings.pkl",
        use_wrist_image=True, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True,
        chunk_size=16, num_open_loop_steps=16, trained_with_image_aug=True,
        use_jpeg_compression=True, flip_images=True,
        num_denoising_steps_action=_SAMPLING_STEPS,
        num_denoising_steps_future_state=1, num_denoising_steps_value=1,
        task_suite_name=args.suite,
    )
    _log(f"loading dataset stats + T5 cache + Cosmos-Policy model on cuda:{_DEVICE_ID}")
    dataset_stats = load_dataset_stats(eval_cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(eval_cfg.t5_text_embeddings_path, worker_id=0)
    model, _ = get_model(eval_cfg)
    n_blocks = len(model.net.blocks)
    assert n_blocks == _L, f"got {n_blocks} DiT blocks; SVD config says L={_L}"
    gc.collect(); torch.cuda.empty_cache()

    # ---- token-space dims ----
    _T_P_DENOISE = _SVD_CFG["T_p_denoise"]
    _DENOISE_T_START = _SVD_CFG["denoise_t_start"]
    _DENOISE_T_END = _SVD_CFG["denoise_t_end"]
    _H_P = _SVD_CFG["H_p"]
    _W_P = _SVD_CFG["W_p"]
    _D = _SVD_CFG["D"]

    # ---- V cache pre-load ----
    _V_DEVICE = _DEVICE
    _MAX_GPU_PARTITIONS = len(_PARTITIONS) * len(_SEL_T)
    _log("CPU pre-load of all V (partition, t) tiles ...")
    for p_idx in range(len(_PARTITIONS)):
        for t_id in _SEL_T:
            _load_V_cpu(p_idx, t_id)
    V_for(_PARTITIONS[0][0], _SEL_T[0])

    # ---- LQR knobs + per-chunk K precomputation ----
    _LAMBDA = args.lambda_
    Q_SCALE = args.q_scale
    R_SCALE_INIT = args.r_scale_init
    R_SCALE_TAU = args.r_scale_tau
    R_SCALE_FINAL = args.r_scale_final
    QF_SCALE = args.qf_scale
    _MAX_CHUNKS = args.max_chunks
    assert R_SCALE_TAU > 0 and R_SCALE_FINAL >= R_SCALE_INIT and _MAX_CHUNKS >= 1
    _R_SCALE_SCHEDULE = [
        min(R_SCALE_FINAL, R_SCALE_INIT * math.exp(c / R_SCALE_TAU))
        for c in range(_MAX_CHUNKS)
    ]
    _log(f"Q_SCALE={Q_SCALE:g} R_SCALE_INIT={R_SCALE_INIT:g} TAU={R_SCALE_TAU:g} "
         f"FINAL={R_SCALE_FINAL:g} QF_SCALE={QF_SCALE:g} MAX_CHUNKS={_MAX_CHUNKS}")

    K_total = _T_DIFF * _L - 1 if _T_DIFF > 0 else 0
    _LQR_DEVICE = _DEVICE
    _LQR_DTYPE = torch.float64
    I_r = torch.eye(_R_DIM, dtype=_LQR_DTYPE, device=_LQR_DEVICE)
    Q_chain = (Q_SCALE * I_r).expand(K_total, _R_DIM, _R_DIM).contiguous()
    S_T_chain = (QF_SCALE * I_r).contiguous()
    A_tilde_dev = A_tilde.to(device=_LQR_DEVICE, dtype=_LQR_DTYPE)
    B_tilde_dev = B_tilde.to(device=_LQR_DEVICE, dtype=_LQR_DTYPE)
    A_chain = torch.zeros(K_total, _R_DIM, _R_DIM, dtype=_LQR_DTYPE, device=_LQR_DEVICE)
    for t in range(_T_DIFF):
        for l in range(_L - 1):
            A_chain[t * _L + l] = A_tilde_dev[t, l]
        if t < _T_DIFF - 1 and B_tilde_dev.numel():
            A_chain[t * _L + (_L - 1)] = B_tilde_dev[t]

    _K_INTRA_PER_CHUNK = torch.zeros(_MAX_CHUNKS, _T_DIFF, _L - 1, _R_DIM, _R_DIM,
                                     dtype=torch.float32)
    _K_STEP_PER_CHUNK = torch.zeros(_MAX_CHUNKS, max(_T_DIFF - 1, 0), _R_DIM, _R_DIM,
                                    dtype=torch.float32)
    t0 = time.time()
    for c, r_scale_c in enumerate(_R_SCALE_SCHEDULE):
        R_chain = (r_scale_c * I_r).expand(K_total, _R_DIM, _R_DIM).contiguous()
        K_chain = _lqr_no_B(A_chain, Q_chain, R_chain, S_T_chain)
        if _LQR_DEVICE.type == "cuda":
            torch.cuda.synchronize(_LQR_DEVICE)
        K_chain = K_chain.float().cpu()
        for t in range(_T_DIFF):
            for l in range(_L - 1):
                _K_INTRA_PER_CHUNK[c, t, l] = K_chain[t * _L + l]
            if t < _T_DIFF - 1:
                _K_STEP_PER_CHUNK[c, t] = K_chain[t * _L + (_L - 1)]
        del R_chain, K_chain
    _log(f"per-chunk K computed in {time.time()-t0:.2f}s")
    del A_chain, Q_chain, S_T_chain, I_r, A_tilde_dev, B_tilde_dev
    if _LQR_DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    _LQR.update({
        "K_intra": _K_INTRA_PER_CHUNK[0].to(device=_V_DEVICE, dtype=torch.float32),
        "K_step": (
            _K_STEP_PER_CHUNK[0].to(device=_V_DEVICE, dtype=torch.float32)
            if _K_STEP_PER_CHUNK.numel() else _K_STEP_PER_CHUNK[0]
        ),
        "v": tilde_v.to(device=_V_DEVICE, dtype=torch.float32),
        "mu": tilde_mu.to(device=_V_DEVICE, dtype=torch.float32),
    })

    # ---- libero env + init states ----
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    assert args.n_episodes <= init_states.shape[0]
    env, task_desc_libero = get_libero_env(task, "cosmos", resolution=args.resolution)
    max_env_steps = TASK_MAX_STEPS[args.suite]
    PROMPT = args.prompt or task_desc_libero
    _log(f"env ready  prompt={PROMPT!r}  max_steps={max_env_steps}")

    # When --side-by-side-video is on, build a second env with no perturbation
    # to render the clean view alongside. State is injected from the perturbed
    # env on every step, so this second env never affects policy inputs.
    env_clean = None
    if args.side_by_side_video and args.cam_mode != "off":
        env_clean, _ = get_libero_env(task, "cosmos", resolution=args.resolution)
        _log("env_clean (unperturbed) built for side-by-side video")

    # ---- camera perturbation ----
    if args.cam_mode == "off":
        perturbation = None
        _log("CAM_MODE=off  (no camera perturbation; matches baseline notebook)")
    else:
        perturbation = RandomCameraViewPerturbation(
            pos_sigma=args.cam_pos_sigma,
            rot_sigma_rad=float(np.radians(args.cam_rot_sigma_deg)),
            fov_sigma=args.cam_fov_sigma,
            base_seed=args.cam_base_seed,
            enforce_visibility=not args.cam_disable_visibility,
            workspace_table_z=args.cam_workspace_table_z,
            workspace_visible_fraction=args.cam_workspace_visible_fraction,
            visibility_margin_px=args.cam_visibility_margin_px,
            image_size=args.resolution,
            max_rejection_attempts=args.cam_max_rejection_attempts,
            name_hint=args.cam_preset_name,
        )
        _log(f"CAM_MODE={args.cam_mode}  base_seed={args.cam_base_seed}  "
             f"preset={args.cam_preset_name}")

    # ---- output dir + manifest ----
    config_tag = _compute_config_tag(args, PROMPT)
    rollouts_root = Path("notebooks/lqr/rollouts")
    if args.rollout_subdir:
        rollouts_root = rollouts_root / args.rollout_subdir.strip("/")
    out_dir = rollouts_root / config_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir -> {out_dir.resolve()}")

    # The first rank writes the master manifest (subsequent ranks skip).
    if args.rank == 0:
        manifest = {
            "suite": args.suite,
            "task_id": args.task_id,
            "task_desc_libero": task_desc_libero,
            "policy_prompt": PROMPT,
            "n_episodes": args.n_episodes,
            "world_size": args.world_size,
            "max_env_steps": max_env_steps,
            "lambda": args.lambda_,
            "Q_SCALE": Q_SCALE,
            "R_SCALE_INIT": R_SCALE_INIT,
            "R_SCALE_TAU": R_SCALE_TAU,
            "R_SCALE_FINAL": R_SCALE_FINAL,
            "QF_SCALE": QF_SCALE,
            "max_chunks": _MAX_CHUNKS,
            "r_scale_schedule": _R_SCALE_SCHEDULE,
            "seed": int(_SVD_CFG.get("seed", 42)),
            "sel_t": _SEL_T,
            "sampling_steps": _SAMPLING_STEPS,
            "jac_prompt": JAC_PROMPT,
            "svd_dir": str(_SVD_DIR),
            "jac_dir_act": args.jac_dir_act,
            "resolution": args.resolution,
            "run_tag": args.run_tag,
            "rollout_subdir": args.rollout_subdir,
            "cam_mode": args.cam_mode,
            "cam_base_seed": args.cam_base_seed,
            "cam_preset_name": args.cam_preset_name,
            "cam_pos_sigma_m": args.cam_pos_sigma,
            "cam_rot_sigma_deg": args.cam_rot_sigma_deg,
            "cam_fov_sigma_deg": args.cam_fov_sigma,
            "cam_workspace_table_z": args.cam_workspace_table_z,
            "cam_workspace_visible_fraction": args.cam_workspace_visible_fraction,
            "cam_visibility_margin_px": args.cam_visibility_margin_px,
            "cam_disable_visibility": args.cam_disable_visibility,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _log(f"wrote master manifest.json")

    # ---- register hooks ----
    handles = [
        model.net.blocks[0].register_forward_pre_hook(_pass_tick),
        model.net.blocks[0].register_forward_hook(cross_step_apply),
    ]
    for l_in in range(_L - 1):
        handles.append(model.net.blocks[l_in + 1].register_forward_hook(make_intra_hook(l_in)))
    handles.append(model.net.blocks[_L - 1].register_forward_hook(cross_step_compute))
    _log(f"registered {len(handles)} steering hooks")

    SEED = int(_SVD_CFG.get("seed", 42))

    def steered_policy_fn(observation, desc):
        state["pass_idx"] = -1
        state["u_step_pending"] = None
        c = steering["chunk_idx"]
        _swap_K_for_chunk(c)
        steering["chunk_idx"] += 1
        with torch.inference_mode():
            out = get_action(
                eval_cfg, model, dataset_stats, observation, desc,
                seed=SEED, randomize_seed=False,
                num_denoising_steps_action=_SAMPLING_STEPS,
                generate_future_state_and_value_in_parallel=True,
            )
        return out["actions"]

    # ---- steered rollouts (this rank's shard) ----
    assigned = [ep for ep in range(args.n_episodes) if (ep % args.world_size) == args.rank]
    _log(f"rank {args.rank} assigned episodes: {assigned}")
    results = []
    for ep in assigned:
        steering["chunk_idx"] = 0
        steering["enabled"] = True
        t0 = time.time()
        success, env_steps, frames, clean_frames = rollout(
            env, init_states[ep], PROMPT, steered_policy_fn,
            eval_cfg, max_env_steps, perturbation, episode_idx=ep,
            env_clean=env_clean,
        )
        tag = "SUCCESS" if success else "FAILURE"
        mp4 = out_dir / f"ep{ep:02d}--{tag}.mp4"
        if clean_frames is not None:
            save_paired_video(frames, clean_frames, mp4, args.video_fps)
        else:
            save_video(frames, mp4, args.video_fps)
        dt = time.time() - t0
        rec = {
            "episode": ep, "success": bool(success), "env_steps": int(env_steps),
            "wall_time_s": dt, "video_path": str(mp4),
            "chunks_consumed": steering["chunk_idx"],
            "rank": args.rank,
            "side_by_side": clean_frames is not None,
        }
        if perturbation is not None:
            rec["perturb_sample"] = dict(perturbation._last_sample)
            rec["rejection_count"] = int(perturbation._last_rejection_count)
        _log(f"ep {ep:2d}: {tag:7s} steps={env_steps:4d} {dt:6.1f}s -> {mp4.name}")
        results.append(rec)

    # Per-rank results shard.
    shard_dir = out_dir / "_rank_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"results_rank{args.rank:02d}.json").write_text(
        json.dumps({"steered": results}, indent=2))

    # ---- baseline (unsteered) for this rank's episodes ----
    if args.run_baseline:
        n_hooks_stripped = 0
        for _b in model.net.blocks:
            n_hooks_stripped += len(_b._forward_pre_hooks) + len(_b._forward_hooks)
            _b._forward_pre_hooks.clear()
            _b._forward_hooks.clear()
        _V_GPU_CACHE.clear()
        _LQR.clear()
        gc.collect(); torch.cuda.empty_cache()
        _log(f"baseline: stripped {n_hooks_stripped} hooks; re-rolling unsteered")
        baseline_dir = out_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        def vanilla_policy_fn(observation, desc):
            with torch.inference_mode():
                out = get_action(
                    eval_cfg, model, dataset_stats, observation, desc,
                    seed=SEED, randomize_seed=False,
                    num_denoising_steps_action=_SAMPLING_STEPS,
                    generate_future_state_and_value_in_parallel=True,
                )
            return out["actions"]

        baseline_results = []
        for ep in assigned:
            t0 = time.time()
            success, env_steps, frames, clean_frames = rollout(
                env, init_states[ep], PROMPT, vanilla_policy_fn,
                eval_cfg, max_env_steps, perturbation, episode_idx=ep,
                env_clean=env_clean,
            )
            tag = "SUCCESS" if success else "FAILURE"
            mp4 = baseline_dir / f"ep{ep:02d}--{tag}.mp4"
            if clean_frames is not None:
                save_paired_video(frames, clean_frames, mp4, args.video_fps)
            else:
                save_video(frames, mp4, args.video_fps)
            dt = time.time() - t0
            brec = {
                "episode": ep, "success": bool(success), "env_steps": int(env_steps),
                "wall_time_s": dt, "video_path": str(mp4),
                "rank": args.rank,
                "side_by_side": clean_frames is not None,
            }
            if perturbation is not None:
                brec["perturb_sample"] = dict(perturbation._last_sample)
                brec["rejection_count"] = int(perturbation._last_rejection_count)
            _log(f"baseline ep {ep:2d}: {tag:7s} steps={env_steps:4d} {dt:6.1f}s")
            baseline_results.append(brec)
        (shard_dir / f"baseline_results_rank{args.rank:02d}.json").write_text(
            json.dumps({"baseline": baseline_results}, indent=2))

    env.close()
    if env_clean is not None:
        env_clean.close()
    _log(f"rank {args.rank} DONE")
    return out_dir


# ====================================================================
# finalize: merge per-rank result shards into one results.json.
# ====================================================================

def run_finalize(args):
    PROMPT = args.prompt or "<libero default>"
    config_tag = _compute_config_tag(args, PROMPT)
    rollouts_root = Path("notebooks/lqr/rollouts")
    if args.rollout_subdir:
        rollouts_root = rollouts_root / args.rollout_subdir.strip("/")
    out_dir = rollouts_root / config_tag
    shard_dir = out_dir / "_rank_shards"
    assert shard_dir.exists(), f"no shard dir: {shard_dir}"

    all_steered = []
    for shard in sorted(shard_dir.glob("results_rank*.json")):
        all_steered.extend(json.loads(shard.read_text())["steered"])
    all_steered.sort(key=lambda r: r["episode"])
    (out_dir / "results.json").write_text(json.dumps(all_steered, indent=2))
    n_succ = sum(1 for r in all_steered if r["success"])
    _log(f"steered: {n_succ}/{len(all_steered)} succeeded")

    baseline_shards = sorted(shard_dir.glob("baseline_results_rank*.json"))
    if baseline_shards:
        all_baseline = []
        for shard in baseline_shards:
            all_baseline.extend(json.loads(shard.read_text())["baseline"])
        all_baseline.sort(key=lambda r: r["episode"])
        baseline_dir = out_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / "results.json").write_text(json.dumps(all_baseline, indent=2))
        n_b = sum(1 for r in all_baseline if r["success"])
        _log(f"baseline: {n_b}/{len(all_baseline)} succeeded")

    _log(f"finalize done -> {out_dir.resolve()}")


# ====================================================================
# main
# ====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["rollouts", "finalize"], required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)

    ap.add_argument("--svd-dir", type=str, required=True)
    ap.add_argument("--jac-dir-act", type=str, required=True,
                    help="subdir of svd-dir produced by run_jacobians_full.sh")

    # Rollout knobs.
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--prompt", type=str, default=None,
                    help="policy prompt; default = libero task description")
    ap.add_argument("--run-baseline", action="store_true", default=False,
                    help="also run unsteered baseline with same prompt+seed+perturbation")
    ap.add_argument("--no-baseline", action="store_true", default=False)
    ap.add_argument("--side-by-side-video", action="store_true", default=False,
                    help="render an unperturbed view alongside the perturbed (policy-driving) "
                         "view; clean view is state-injected, never feeds the policy. "
                         "No-op when --cam-mode=off.")

    # LQR knobs (decay variant).
    ap.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    ap.add_argument("--q-scale", type=float, default=10000.0)
    ap.add_argument("--r-scale-init", type=float, default=10.0)
    ap.add_argument("--r-scale-tau", type=float, default=3.0)
    ap.add_argument("--r-scale-final", type=float, default=1e9)
    ap.add_argument("--qf-scale", type=float, default=1.0)
    ap.add_argument("--max-chunks", type=int, default=50)

    # Camera perturbation knobs.
    ap.add_argument("--cam-mode", choices=["match", "random", "off"], default="match",
                    help="match: same perturbation seed as input collection (default 42); "
                         "random: independent seed; off: no perturbation")
    ap.add_argument("--cam-base-seed", type=int, default=42)
    ap.add_argument("--cam-preset-name", type=str, default="cam_random_large")
    ap.add_argument("--cam-pos-sigma", type=float, default=0.10)
    ap.add_argument("--cam-rot-sigma-deg", type=float, default=8.0)
    ap.add_argument("--cam-fov-sigma", type=float, default=5.0)
    ap.add_argument("--cam-disable-visibility", action="store_true")
    ap.add_argument("--cam-workspace-table-z", type=float, default=0.90)
    ap.add_argument("--cam-workspace-visible-fraction", type=float, default=0.55)
    ap.add_argument("--cam-visibility-margin-px", type=int, default=8)
    ap.add_argument("--cam-max-rejection-attempts", type=int, default=2000)

    # OUT_DIR / naming.
    ap.add_argument("--run-tag", type=str, default="cam_perturb")
    ap.add_argument("--rollout-subdir", type=str, default="")

    args = ap.parse_args()
    # Normalize baseline flag.
    args.run_baseline = args.run_baseline and not args.no_baseline

    if args.mode == "rollouts":
        run_rollouts(args)
    elif args.mode == "finalize":
        run_finalize(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
