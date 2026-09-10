#!/usr/bin/env python
"""Activation-LQR (A-LQR) rollout of DiT4DiT under per-episode RANDOM GRIPPER
(x,y,z) PERTURBATION applied at the start of each rollout.

Analogue of the repo root's notebooks/lqr/run_lqr_cosmos_policy_gripper_xyz.py
adapted for DiT4DiT's FlowmatchingActionHead / BasicTransformerBlock.

Key differences from the Cosmos version
-----------------------------------------
1. Model: DiT4DiT loaded via baseframework.from_pretrained.
2. Block access: model.action_model.model.transformer_blocks (not model.net.blocks).
3. Activation shape: (B, T_seq, inner_dim); action tokens at [denoise_t_start:denoise_t_end].
4. No spatial H_p/W_p dimensions: u_add is reshaped as (action_horizon, inner_dim).
5. Denoising loop: run manually (run_denoising_loop) so hooks fire during the loop.
6. VLM backbone called first; result cached per inference step.
7. Action unnormalization: min-max from dataset_statistics.json.
8. No T5 cache, no CFG.
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

# -----------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_LOCAL_DIT4DIT_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from runtime_paths import configure_runtime, load_libero_init_states  # noqa: E402
from reproducibility import (  # noqa: E402
    annotate_inference, clone_observation, flattened_sim_state,
)

_DIT4DIT_ROOT, LIBERO_HOME = configure_runtime(_LOCAL_DIT4DIT_ROOT)
os.environ.setdefault("LIBERO_HOME", LIBERO_HOME)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DIT4DIT_ROOT = _DIT4DIT_ROOT
CKPT_DEFAULT = os.environ.get(
    "CKPT_PATH",
    str(DIT4DIT_ROOT / "checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt"),
)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--phase", choices=["rollout", "merge"], default="rollout")
    ap.add_argument("--world-size", type=int,
                    default=int(os.environ.get("WORLD_SIZE", 1)))
    ap.add_argument("--rank", type=int,
                    default=int(os.environ.get("RANK", 0)))

    ap.add_argument("--svd-dir", type=Path, required=True)
    ap.add_argument("--jac-dir-act", type=str, required=True,
                    help="A_tilde subdir under --svd-dir (must contain A_tilde__full.pt)")

    ap.add_argument("--preset",    type=str, default="xyz_random_xlarge_3")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--gripper-action",        type=float, default=-1.0)
    ap.add_argument("--pre-shift-settle-steps", type=int,  default=10)
    ap.add_argument("--shift-max-steps",        type=int,  default=30)
    ap.add_argument("--shift-tol-m",            type=float, default=2e-3)
    ap.add_argument("--post-shift-pause-steps", type=int,  default=10)

    ap.add_argument("--lambda-scale", type=float, default=1.0,  dest="lambda_scale")
    ap.add_argument("--q-scale",      type=float, default=10000.0, dest="q_scale")
    ap.add_argument("--r-scale",      type=float, default=75000.0, dest="r_scale")
    ap.add_argument("--r-scale-tau",  type=float, default=3.0,  dest="r_scale_tau")
    ap.add_argument("--r-scale-final",type=float, default=1e9,  dest="r_scale_final")
    ap.add_argument("--max-chunks",   type=int,   default=50,   dest="max_chunks")
    ap.add_argument("--qf-scale",     type=float, default=1.0,  dest="qf_scale")

    ap.add_argument("--prompt",      type=str, required=True)
    ap.add_argument("--n-episodes",  type=int, default=10)
    ap.add_argument("--suite",       type=str, default="libero_10")
    ap.add_argument("--task-id",     type=int, default=0)
    ap.add_argument("--resolution",  type=int, default=256)
    ap.add_argument("--video-fps",   type=int, default=30)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--max-env-steps",  type=int, default=1000, dest="max_env_steps")
    ap.add_argument("--run-baseline",   dest="run_baseline", action="store_true",  default=False)
    ap.add_argument("--no-baseline",    dest="run_baseline", action="store_false")
    ap.add_argument("--baseline-only",  dest="baseline_only", action="store_true", default=False)

    ap.add_argument("--ckpt-path", type=str, default=CKPT_DEFAULT)
    ap.add_argument("--seed",      type=int, default=None)

    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    ap.add_argument("--save-activations", action="store_true",
                    help="save post-hook block activations for baseline/steered plots")
    ap.add_argument("--save-inference-snapshots", action="store_true",
                    help="save exact observation/model input at every policy inference")
    ap.add_argument("--tag", type=str, default=None)
    return ap.parse_args()


# -----------------------------------------------------------------------
# Gripper XYZ perturbation (identical to the repo root's original version)
# -----------------------------------------------------------------------

EE_SITE_CANDIDATES = (
    "gripper0_grip_site", "robot0_grip_site", "grip_site", "ee_site",
)
EE_BODY_CANDIDATES = (
    "robot0_right_hand", "right_hand", "gripper0_eef", "gripper0_hand",
)
OSC_OUTPUT_MAX_M = 0.05
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


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
    obs, n_used = None, 0
    achieved = _ee_pos(sim, site_name=site_name, body_name=body_name)
    for step in range(max_steps):
        cur = _ee_pos(sim, site_name=site_name, body_name=body_name)
        err = target_pos - cur
        if np.linalg.norm(err) < tol_m:
            break
        action_xyz = np.clip(err / output_max_m, -1.0, 1.0)
        action = [float(action_xyz[i]) for i in range(3)] + [0., 0., 0., float(gripper_action)]
        obs, _, _, _ = env.step(action)
        n_used = step + 1
        achieved = _ee_pos(sim, site_name=site_name, body_name=body_name)
    return obs, achieved, float(np.linalg.norm(target_pos - achieved)), n_used


@dataclass
class GripperXYZPerturbation:
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
            baseline_obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        sim.forward()
        baseline_ee = _ee_pos(sim, site_name=site_name, body_name=body_name)
        dxyz = self._resolve_xyz(episode_idx)
        target_ee = baseline_ee + np.asarray(dxyz, dtype=np.float64)

        if np.allclose(dxyz, 0.0):
            achieved_ee, residual, n_used, final_obs = baseline_ee, 0.0, 0, baseline_obs
        else:
            final_obs, achieved_ee, residual, n_used = _shift_gripper_xyz_via_actions(
                env, target_ee, site_name=site_name, body_name=body_name,
                gripper_action=self.gripper_action, max_steps=self.shift_max_steps,
                tol_m=self.shift_tol_m, output_max_m=self.output_max_m,
            )

        pause_action = [0., 0., 0., 0., 0., 0., float(self.gripper_action)]
        for _ in range(self.post_shift_pause_steps):
            final_obs, _, _, _ = env.step(pause_action)

        self._last_sample = {
            "episode_idx": int(episode_idx),
            "xyz_delta_m": [float(v) for v in dxyz],
            "achieved_xyz_delta_m": [float(a - b) for a, b in zip(achieved_ee, baseline_ee)],
            "shift_residual_m": float(residual),
        }
        if residual > max(self.shift_tol_m * 5, 5e-3):
            print(f"  [shift warn] ep{episode_idx} residual={residual*1000:.2f} mm", flush=True)
        return final_obs

    def manifest(self) -> dict:
        return {"kind": "GripperXYZPerturbation", "slug": self.slug,
                "xyz_delta_m": list(self.xyz_delta)}


@dataclass
class RandomGripperXYZPerturbation(GripperXYZPerturbation):
    sigma_xyz_m: object = 0.02
    base_seed: int = 0

    def _sigma_vec(self):
        s = self.sigma_xyz_m
        return np.array([float(s)] * 3) if np.isscalar(s) else np.array([float(v) for v in s])

    @property
    def slug(self) -> str:
        sig = self._sigma_vec()
        payload = f"sigma=[{','.join(f'{v:.4f}' for v in sig)}]|seed{self.base_seed}"
        h = hashlib.md5(payload.encode()).hexdigest()[:6]
        return f"{self.name_hint}_seed{self.base_seed}_{h}"

    def _resolve_xyz(self, episode_idx):
        ss = np.random.SeedSequence([int(self.base_seed), int(episode_idx)])
        rng = np.random.default_rng(ss)
        return tuple(float(v) for v in rng.normal(0.0, self._sigma_vec()))

    def manifest(self) -> dict:
        sig = self._sigma_vec()
        return {"kind": "RandomGripperXYZPerturbation", "slug": self.slug,
                "sigma_xyz_m": sig.tolist(), "base_seed": self.base_seed}


def build_presets(args) -> dict:
    common  = dict(gripper_action=args.gripper_action,
                   pre_shift_settle_steps=args.pre_shift_settle_steps,
                   shift_max_steps=args.shift_max_steps,
                   shift_tol_m=args.shift_tol_m,
                   post_shift_pause_steps=args.post_shift_pause_steps)
    rcommon = dict(common, base_seed=args.base_seed)
    return {
        "xyz_random_small":    RandomGripperXYZPerturbation(sigma_xyz_m=0.01, name_hint="xyz_random_small",    **rcommon),
        "xyz_random_medium":   RandomGripperXYZPerturbation(sigma_xyz_m=0.02, name_hint="xyz_random_medium",   **rcommon),
        "xyz_random_large":    RandomGripperXYZPerturbation(sigma_xyz_m=0.04, name_hint="xyz_random_large",    **rcommon),
        "xyz_random_xlarge":   RandomGripperXYZPerturbation(sigma_xyz_m=0.06, name_hint="xyz_random_xlarge",   **rcommon),
        "xyz_random_xlarge_2": RandomGripperXYZPerturbation(sigma_xyz_m=0.15, name_hint="xyz_random_xlarge_2", **rcommon),
        "xyz_random_xlarge_3": RandomGripperXYZPerturbation(sigma_xyz_m=0.10, name_hint="xyz_random_xlarge_3", **rcommon),
        "xyz_random_horizontal": RandomGripperXYZPerturbation(sigma_xyz_m=(0.03, 0.03, 0.005),
                                                               name_hint="xyz_random_horizontal", **rcommon),
        "xyz_random":          RandomGripperXYZPerturbation(sigma_xyz_m=(0.1, 0.1, 0.1),
                                                             name_hint="xyz_random", **rcommon),
    }


# -----------------------------------------------------------------------
# V-tile cache (identical API to Cosmos version)
# -----------------------------------------------------------------------

class VCache:
    def __init__(self, svd_dir, partitions, layer_to_part, sel_t, k_target,
                 device, dtype=torch.bfloat16, max_gpu_tiles=None):
        self.svd_dir = Path(svd_dir)
        self.partitions = partitions
        self.layer_to_part = layer_to_part
        self.sel_t = list(sel_t)
        self.k_target = k_target
        self.device = device
        self.dtype = dtype
        self.max_gpu = max_gpu_tiles or (len(partitions) * len(sel_t))
        self._cpu = {}
        self._gpu = OrderedDict()
        self.stats = {"cpu_loads": 0, "gpu_swaps": 0, "gpu_hits": 0, "cpu_to_gpu_s": 0.0}

    def _vfile(self, p_idx, t_id):
        a, b = self.partitions[p_idx]
        pattern = f"V_part{p_idx}_layers{a}-{b}_t{t_id}_k*.pt"
        matches = sorted(self.svd_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no V file matching {pattern} in {self.svd_dir}")
        preferred = [m for m in matches if m.name.endswith(f"_k{self.k_target}.pt")]
        return (preferred or matches)[0]

    def _load_cpu(self, p_idx, t_id):
        key = (p_idx, t_id)
        if key in self._cpu:
            return self._cpu[key]
        fp = self._vfile(p_idx, t_id)
        print(f"  disk-load {fp.name} ({fp.stat().st_size/1e9:.2f} GB) -> CPU {self.dtype} ...", flush=True)
        t0 = time.time()
        raw = torch.load(fp, map_location="cpu", weights_only=False)["V"]
        V = raw.to(dtype=self.dtype).contiguous()
        del raw
        self._cpu[key] = V
        self.stats["cpu_loads"] += 1
        print(f"    CPU resident (p={p_idx}, t={t_id}): {tuple(V.shape)}  "
              f"~{V.element_size()*V.numel()/1e9:.2f} GB  ({time.time()-t0:.1f}s)", flush=True)
        return V

    def for_layer(self, layer_idx, t_id):
        p_idx = self.layer_to_part[layer_idx]
        key = (p_idx, t_id)
        V = self._gpu.get(key)
        if V is not None:
            self._gpu.move_to_end(key)
            self.stats["gpu_hits"] += 1
            return V
        V_cpu = self._load_cpu(p_idx, t_id)
        if self.device.type == "cpu":
            self._gpu[key] = V_cpu
            return V_cpu
        while len(self._gpu) >= self.max_gpu:
            _, ev = self._gpu.popitem(last=False)
            del ev
            torch.cuda.empty_cache()
            self.stats["gpu_swaps"] += 1
        t0 = time.time()
        V = V_cpu.to(device=self.device, non_blocking=False).contiguous()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stats["cpu_to_gpu_s"] += time.time() - t0
        self._gpu[key] = V
        return V

    def preload(self, partition_t_pairs):
        for p_idx, t_id in partition_t_pairs:
            self._load_cpu(p_idx, t_id)


# -----------------------------------------------------------------------
# Chained Riccati (identical to Cosmos version)
# -----------------------------------------------------------------------

def chained_riccati_per_chunk(A_tilde, B_tilde, q_scale, r_scale_schedule,
                               qf_scale, device):
    T_diff, L_minus1, r, _ = A_tilde.shape
    L = L_minus1 + 1
    K_total = T_diff * L - 1 if T_diff > 0 else 0
    n_chunks = len(r_scale_schedule)

    _dtype = torch.float64
    I_r = torch.eye(r, dtype=_dtype, device=device)
    Q_chain = (q_scale  * I_r).expand(K_total, r, r).contiguous()
    S_T     = (qf_scale * I_r).contiguous()

    A_dev = A_tilde.to(device=device, dtype=_dtype)
    B_dev = B_tilde.to(device=device, dtype=_dtype)
    A_chain = torch.zeros(K_total, r, r, dtype=_dtype, device=device)
    for t in range(T_diff):
        for l in range(L - 1):
            A_chain[t * L + l] = A_dev[t, l]
        if t < T_diff - 1 and B_dev.numel():
            A_chain[t * L + (L - 1)] = B_dev[t]

    K_intra_per_chunk = torch.zeros(n_chunks, T_diff, L - 1, r, r, dtype=torch.float32)
    K_step_per_chunk  = torch.zeros(n_chunks, max(T_diff - 1, 0), r, r, dtype=torch.float32)

    for c, r_scale_c in enumerate(r_scale_schedule):
        R_chain = (r_scale_c * I_r).expand(K_total, r, r).contiguous()
        Tn = A_chain.shape[0]
        S = torch.zeros(Tn + 1, r, r, dtype=_dtype, device=device)
        K = torch.zeros(Tn, r, r, dtype=_dtype, device=device)
        S[Tn] = S_T
        for k in reversed(range(Tn)):
            Ak = A_chain[k]
            P  = S[k + 1] + R_chain[k]
            F  = S[k + 1] @ Ak
            G  = Q_chain[k] + Ak.transpose(-2, -1) @ S[k + 1] @ Ak
            Kk = torch.linalg.solve(P, F)
            K[k] = Kk
            Snew = G - F.transpose(-2, -1) @ Kk
            S[k] = 0.5 * (Snew + Snew.transpose(-2, -1))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        K_chain = K.float().cpu()
        for t in range(T_diff):
            for l in range(L - 1):
                K_intra_per_chunk[c, t, l] = K_chain[t * L + l]
            if t < T_diff - 1:
                K_step_per_chunk[c, t] = K_chain[t * L + (L - 1)]
        del R_chain, K, S, K_chain

    del A_chain, Q_chain, A_dev, B_dev, I_r
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return K_intra_per_chunk, K_step_per_chunk


# -----------------------------------------------------------------------
# Steering runtime + hooks
#
# DiT4DiT difference vs Cosmos:
#   - Block output shape: (B, T_seq, inner_dim) instead of (B, T_p, H_p, W_p, D)
#   - u_add reshaped as (action_horizon, inner_dim) and added to output[0, dt_start:dt_end, :]
# -----------------------------------------------------------------------

class SteeringRuntime:
    def __init__(self, *, L, T_diff, sel_t, denoise_t_start, denoise_t_end,
                 T_p_denoise, inner_dim, sampling_steps, lambda_scale, vcache, lqr):
        self.L = L
        self.T_diff = T_diff
        self.sel_t = list(sel_t)
        self.sel_idx_of = {t: i for i, t in enumerate(self.sel_t)}
        self.denoise_t_start = denoise_t_start
        self.denoise_t_end   = denoise_t_end
        self.T_p_denoise     = T_p_denoise     # = action_horizon
        self.inner_dim       = inner_dim
        self.sampling_steps  = sampling_steps
        self.lambda_scale    = float(lambda_scale)
        self.vcache          = vcache
        self.lqr             = lqr

        self.pass_idx         = -1
        self.u_step_pending   = None
        self.in_ad            = False
        self.u_norm_log       = []
        self.steering_enabled = True
        self.trace_enabled    = False
        self.trace_label      = None
        self.activation_trace = []

    def reset_chunk(self):
        self.pass_idx = -1
        self.u_step_pending = None

    def is_selected_step(self, pass_idx):
        if pass_idx < 0:
            return None, None
        step = pass_idx  # PASSES_PER_STEP = 1
        sel = self.sel_idx_of.get(step)
        if sel is None:
            return None, None
        return step, sel

    def begin_trace(self, label: str):
        self.trace_enabled = True
        self.trace_label = label
        self.activation_trace.clear()

    def finish_trace(self):
        self.trace_enabled = False
        trace = self.activation_trace
        self.activation_trace = []
        return trace


def install_lqr_hooks(action_dit, rt: SteeringRuntime):
    L = rt.L
    handles = []

    def _pass_tick(_block, _args):
        if not rt.in_ad:
            rt.pass_idx += 1

    def make_intra_hook(l_in):
        def hook(block, args, output):
            if rt.in_ad or not rt.steering_enabled:
                return None
            step, sel = rt.is_selected_step(rt.pass_idx)
            if step is None:
                return None
            # output: (B, T_seq, inner_dim)
            z_full = (
                output[0, rt.denoise_t_start:rt.denoise_t_end, :]
                .detach().reshape(-1)  # (D_flat,)
            )
            z_dt = output.dtype
            V_in = rt.vcache.for_layer(l_in, step)
            rt.in_ad = True
            try:
                x_proj  = (z_full.to(V_in.dtype) @ V_in).float()
                v_fp    = rt.lqr["v"][l_in, sel]
                mu_fp   = rt.lqr["mu"][l_in, sel]
                K_fp    = rt.lqr["K_intra"][sel, l_in]
                alpha   = rt.lambda_scale * mu_fp - v_fp @ x_proj
                u_tilde = K_fp @ (alpha * v_fp)
                rt.u_norm_log.append((sel, l_in, float(u_tilde.norm()), float(alpha)))
            finally:
                rt.in_ad = False
            del V_in
            V_out = rt.vcache.for_layer(l_in + 1, step)
            rt.in_ad = True
            try:
                u_full = V_out @ u_tilde.to(V_out.dtype)
            finally:
                rt.in_ad = False
            # Reshape for (B, T_seq, inner_dim) instead of Cosmos's 5-D layout
            u_add = u_full.to(z_dt).reshape(rt.T_p_denoise, rt.inner_dim)
            output[0, rt.denoise_t_start:rt.denoise_t_end, :] = (
                output[0, rt.denoise_t_start:rt.denoise_t_end, :] + u_add
            )
            return output
        return hook

    def cross_step_compute(block, args, output):
        if rt.in_ad or not rt.steering_enabled:
            return None
        step, sel = rt.is_selected_step(rt.pass_idx)
        if step is None or sel >= rt.T_diff - 1:
            return None
        z_full = (
            output[0, rt.denoise_t_start:rt.denoise_t_end, :]
            .detach().reshape(-1)
        )
        V_in = rt.vcache.for_layer(L - 1, step)
        rt.in_ad = True
        try:
            x_proj  = (z_full.to(V_in.dtype) @ V_in).float()
            v_fp    = rt.lqr["v"][L - 1, sel]
            mu_fp   = rt.lqr["mu"][L - 1, sel]
            K_fp    = rt.lqr["K_step"][sel]
            alpha   = rt.lambda_scale * mu_fp - v_fp @ x_proj
            u_tilde = K_fp @ (alpha * v_fp)
            rt.u_norm_log.append((sel, -1, float(u_tilde.norm()), float(alpha)))
            rt.u_step_pending = {"src_sel": sel, "u_tilde": u_tilde.detach()}
        finally:
            rt.in_ad = False
        return None

    def cross_step_apply(block, args, output):
        if rt.in_ad or not rt.steering_enabled:
            return None
        step, sel = rt.is_selected_step(rt.pass_idx)
        pending = rt.u_step_pending
        if step is None or pending is None or sel == 0 or pending["src_sel"] != sel - 1:
            return None
        u_tilde = pending["u_tilde"]
        V_dest = rt.vcache.for_layer(0, step)
        rt.in_ad = True
        try:
            u_full = V_dest @ u_tilde.to(V_dest.dtype)
        finally:
            rt.in_ad = False
        u_add = u_full.to(output.dtype).reshape(rt.T_p_denoise, rt.inner_dim)
        output[0, rt.denoise_t_start:rt.denoise_t_end, :] = (
            output[0, rt.denoise_t_start:rt.denoise_t_end, :] + u_add
        )
        rt.u_step_pending = None
        return output

    handles.append(action_dit.transformer_blocks[0].register_forward_pre_hook(_pass_tick))
    handles.append(action_dit.transformer_blocks[0].register_forward_hook(cross_step_apply))
    for l_in in range(L - 1):
        handles.append(
            action_dit.transformer_blocks[l_in + 1].register_forward_hook(
                make_intra_hook(l_in)
            )
        )
    handles.append(action_dit.transformer_blocks[L - 1].register_forward_hook(cross_step_compute))
    return handles


def install_activation_trace_hooks(action_dit, rt: SteeringRuntime):
    """Capture post-hook action-token activations for plotting.

    Register these hooks *after* ``install_lqr_hooks``.  PyTorch executes
    forward hooks in registration order, so the captured steered points include
    the actual LQR update while baseline points are observed unchanged.
    """

    handles = []
    for block_idx, block in enumerate(action_dit.transformer_blocks):
        def _capture(_module, _args, output, _block_idx=block_idx):
            if not rt.trace_enabled or rt.in_ad:
                return None
            step, _ = rt.is_selected_step(rt.pass_idx)
            if step is None:
                return None
            act = output[0, rt.denoise_t_start:rt.denoise_t_end, :]
            rt.activation_trace.append({
                "block": int(_block_idx),
                "step": int(step),
                "activation": act.detach().float().mean(dim=0).cpu(),
            })
            return None

        handles.append(block.register_forward_hook(_capture))
    return handles


# -----------------------------------------------------------------------
# Merge phase
# -----------------------------------------------------------------------

def run_merge_phase(args):
    out_dir = args.out_dir.resolve()
    pattern = sorted(out_dir.glob("results_rank*.json"))
    if not pattern:
        raise FileNotFoundError(f"no results_rank*.json under {out_dir}")
    merged = []
    for fp in pattern:
        merged.extend(json.loads(fp.read_text()))
    merged.sort(key=lambda r: r["episode"])
    seen, dedup = set(), []
    for r in merged:
        if r["episode"] not in seen:
            seen.add(r["episode"])
            dedup.append(r)
    merged = dedup
    (out_dir / "results.json").write_text(json.dumps(merged, indent=2))
    n_succ = sum(r["success"] for r in merged)
    n_base = sum((r.get("baseline") or {}).get("success", False) for r in merged)
    print(f"[merge] {len(merged)} episodes -> {out_dir/'results.json'}", flush=True)
    print(f"[merge] steered:  {n_succ}/{len(merged)}", flush=True)
    if any(r.get("baseline") is not None for r in merged):
        print(f"[merge] baseline: {n_base}/{len(merged)}", flush=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["totals"] = {
            "n_episodes": len(merged),
            "n_success_steered": int(n_succ),
            "n_success_baseline": int(n_base) if any(r.get("baseline") is not None for r in merged) else None,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


# -----------------------------------------------------------------------
# Rollout phase
# -----------------------------------------------------------------------

def run_rollout_phase(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.out_dir.resolve()
    rank, world_size = int(args.rank), int(args.world_size)
    assert 0 <= rank < world_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_id = device.index if device.index is not None else 0

    # ---- SVD config ----
    svd_dir  = args.svd_dir.resolve()
    cfg = json.loads((svd_dir / "config.json").read_text())
    sel_t           = list(cfg["selected_timesteps"])
    T_diff          = len(sel_t)
    sampling_steps  = int(cfg["sampling_steps"])
    L               = int(cfg["L"])
    r               = int(cfg["k_target"])
    partitions      = [tuple(p) for p in cfg["partitions"]]
    D_flat          = int(cfg["D_flat"])
    T_p_denoise     = int(cfg["T_p_denoise"])
    denoise_t_start = int(cfg["denoise_t_start"])
    denoise_t_end   = int(cfg["denoise_t_end"])
    action_horizon  = cfg.get("action_horizon", T_p_denoise)
    inner_dim       = cfg.get("inner_dim", D_flat // action_horizon)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[rank {rank}/{world_size}] svd: {svd_dir}")
    print(f"[svd] sel_t={sel_t}  T_diff={T_diff}  L={L}  r={r}  "
          f"sampling_steps={sampling_steps}")
    print(f"[svd] partitions={partitions}  D_flat={D_flat:,}  "
          f"action_horizon={action_horizon}  inner_dim={inner_dim}")

    # ---- A_tilde ----
    jac_dir = svd_dir / args.jac_dir_act
    a_tilde_full = jac_dir / "A_tilde__full.pt"
    if not a_tilde_full.exists():
        raise FileNotFoundError(f"A_tilde missing: {a_tilde_full}")
    raw = torch.load(a_tilde_full, map_location="cpu", weights_only=False)
    A_dict, B_dict = raw.get("A_tilde", {}), raw.get("B_tilde", {})
    jac_prompt = raw.get("prompt", "<unknown>")
    print(f"[jac] A_tilde entries: {len(A_dict)} (expected {T_diff*(L-1)})  prompt={jac_prompt!r}")

    sel_idx_of = {t: i for i, t in enumerate(sel_t)}
    A_tilde = torch.zeros(T_diff, L - 1, r, r, dtype=torch.float32)
    for (t, l_in), Atl in A_dict.items():
        if t in sel_idx_of:
            A_tilde[sel_idx_of[t], l_in] = Atl.float()
    have_B = len(B_dict) > 0
    B_tilde = (torch.zeros(max(T_diff - 1, 0), r, r, dtype=torch.float32)
               if not have_B else torch.stack([B_dict[(t,)].float()
                                               for t in sel_t[:-1]], dim=0))

    # ---- LQR setpoint (c_means / tilde_v) ----
    summary = torch.load(svd_dir / "svd_summary.pt", map_location="cpu", weights_only=False)
    c_means  = summary["c_means"].float()
    tilde_mu = c_means.norm(dim=-1)
    tilde_v  = c_means / tilde_mu.unsqueeze(-1).clamp(min=1e-12)
    layer_to_part = list(summary["layer_to_part"])

    # ---- R_SCALE schedule ----
    assert args.r_scale_tau > 0 and args.r_scale_final >= args.r_scale and args.max_chunks >= 1
    r_scale_schedule = [
        min(args.r_scale_final, args.r_scale * math.exp(c / args.r_scale_tau))
        for c in range(args.max_chunks)
    ]
    print(f"[lqr] λ={args.lambda_scale:g}  Q={args.q_scale:g}  Qf={args.qf_scale:g}")
    print(f"[lqr] R_SCALE: init={args.r_scale:g}  tau={args.r_scale_tau:g}  "
          f"final={args.r_scale_final:g}  max_chunks={args.max_chunks}")
    t0 = time.time()
    K_intra_per_chunk, K_step_per_chunk = chained_riccati_per_chunk(
        A_tilde, B_tilde, args.q_scale, r_scale_schedule, args.qf_scale, device,
    )
    print(f"[lqr] chained Riccati: {time.time()-t0:.2f}s")
    K_intra = K_intra_per_chunk[0]
    K_step  = K_step_per_chunk[0]

    # ---- Stress test ----
    presets = build_presets(args)
    if args.preset not in presets:
        raise ValueError(f"unknown --preset={args.preset!r}; choices: {sorted(presets)}")
    stress_test = presets[args.preset]
    print(f"[stress] preset={args.preset}  ({type(stress_test).__name__})")

    # ---- Load DiT4DiT model ----
    import DiT4DiT.model.framework.DiT4DiT  # register framework before from_pretrained
    from DiT4DiT.model.framework.base_framework import baseframework
    from DiT4DiT.model.framework.share_tools import read_mode_config
    print(f"[model] loading DiT4DiT from {args.ckpt_path} on {device} ...")
    model        = baseframework.from_pretrained(args.ckpt_path).to(device).eval()
    action_model = model.action_model
    action_dit   = action_model.model

    n_blocks = len(action_dit.transformer_blocks)
    if n_blocks != L:
        raise RuntimeError(f"model has {n_blocks} blocks; SVD L={L}")

    _, norm_stats  = read_mode_config(args.ckpt_path)
    unnorm_key     = next(iter(norm_stats))
    action_stats   = norm_stats[unnorm_key]["action"]
    action_high    = np.array(action_stats["max"], dtype=np.float32)
    action_low     = np.array(action_stats["min"], dtype=np.float32)
    action_mask    = np.array(action_stats.get("mask", np.ones(len(action_high), dtype=bool)))
    print(f"[model] unnorm_key={unnorm_key!r}")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- V cache ----
    vcache = VCache(svd_dir, partitions, layer_to_part, sel_t, r,
                    device=device, dtype=torch.bfloat16)
    print(f"[v] preloading {len(partitions)} × {len(sel_t)} V tiles ...")
    vcache.preload([(p, t) for p in range(len(partitions)) for t in sel_t])
    vcache.for_layer(0, sel_t[0])

    # ---- LQR push ----
    lqr = {
        "K_intra": K_intra.to(device=device, dtype=torch.float32),
        "K_step":  K_step.to(device=device, dtype=torch.float32) if K_step.numel() else K_step,
        "v":       tilde_v.to(device=device, dtype=torch.float32),
        "mu":      tilde_mu.to(device=device, dtype=torch.float32),
    }

    def _swap_K_for_chunk(c: int):
        c_eff = min(c, args.max_chunks - 1)
        lqr["K_intra"].copy_(K_intra_per_chunk[c_eff].to(device=device, dtype=torch.float32))
        if K_step_per_chunk.numel():
            lqr["K_step"].copy_(K_step_per_chunk[c_eff].to(device=device, dtype=torch.float32))

    # ---- Hooks ----
    rt = SteeringRuntime(
        L=L, T_diff=T_diff, sel_t=sel_t,
        denoise_t_start=denoise_t_start, denoise_t_end=denoise_t_end,
        T_p_denoise=T_p_denoise, inner_dim=inner_dim,
        sampling_steps=sampling_steps,
        lambda_scale=args.lambda_scale, vcache=vcache, lqr=lqr,
    )
    handles = install_lqr_hooks(action_dit, rt)
    if args.save_activations:
        handles.extend(install_activation_trace_hooks(action_dit, rt))
    print(f"[hooks] {len(handles)} hooks (1 tick + 1 cross_apply + {L-1} intra + 1 cross_compute)")

    # ---- Env ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from pathlib import Path as _Path

    task_suite  = benchmark.get_benchmark_dict()[args.suite]()
    task        = task_suite.get_task(args.task_id)
    init_states = load_libero_init_states(task_suite, args.task_id)
    if args.n_episodes > init_states.shape[0]:
        raise ValueError(f"--n-episodes {args.n_episodes} > {init_states.shape[0]}")

    suite_max_steps = {
        "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
        "libero_10": 520, "libero_90": 400,
    }.get(args.suite, 520)
    max_env_steps = int(args.max_env_steps)

    extra_for_shift = (args.pre_shift_settle_steps + args.shift_max_steps
                       + args.post_shift_pause_steps + 10)
    env_horizon = args.num_steps_wait + max_env_steps + extra_for_shift

    task_bddl = (
        _Path(get_libero_path("bddl_files"))
        / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        # Bound the env's internal horizon to cover warmup + main loop + shift
        # extras. robosuite defaults to horizon=1000 and RAISES on the step after
        # it auto-terminates; LIBERO's wrapper only returns done on task success
        # (not horizon), so without this the loop steps past termination and hits
        # "executing action in terminated episode". env_horizon was computed for
        # exactly this purpose above but was not being passed.
        horizon=env_horizon,
    )
    env.seed(42)
    task_desc_libero = task.language
    print(f"[env] {args.suite} task {args.task_id:02d}  desc={task_desc_libero!r}  "
          f"max_steps={max_env_steps}")
    if args.prompt != task_desc_libero:
        print(f"  note: --prompt differs from LIBERO task desc; steering uses --prompt.")

    # SVD scripts path (for run_denoising_loop import)
    sys.path.insert(0, str(_HERE / "svd"))
    from run_partition_svd_pairs_no_action import run_denoising_loop

    # ---- Policy fn ----
    NUM_OPEN_LOOP = action_model.action_horizon  # = 8

    def _obs_to_inputs(obs):
        IMAGE_SIZE = 224
        primary = cv2.resize(
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA,
        )
        wrist = cv2.resize(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
            (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA,
        )
        concat_img = np.concatenate([primary, wrist], axis=1)

        import math
        eef_pos  = obs["robot0_eef_pos"].astype(np.float32)
        eef_quat = obs["robot0_eef_quat"].astype(np.float32)
        q = eef_quat.copy()
        q[3] = np.clip(q[3], -1.0, 1.0)
        den = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
        axisangle = (np.zeros(3) if math.isclose(den, 0.0)
                     else (q[:3] * 2.0 * math.acos(q[3])) / den)
        gripper = obs["robot0_gripper_qpos"].astype(np.float32)
        proprio = np.concatenate([eef_pos, axisangle.astype(np.float32), gripper])

        sin_s = np.sin(proprio[None])
        cos_s = np.cos(proprio[None])
        state_enc = np.stack([sin_s, cos_s], axis=-1).reshape(1, -1).astype(np.float32)
        max_state_dim = action_model.config.state_dim
        pad = max_state_dim - state_enc.shape[-1]
        if pad > 0:
            state_enc = np.pad(state_enc, ((0, 0), (0, pad)), "constant")

        return [concat_img], state_enc

    def policy_fn(obs):
        """Run one DiT4DiT inference step; returns list of actions."""
        batch_images, state_enc = _obs_to_inputs(obs)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bi = model.backbone_interface.build_cosmos_inputs(
                    images=[batch_images], instructions=[args.prompt],
                )
                bout = model.backbone_interface(
                    **bi, output_hidden_states=True, output_attentions=False, return_dict=True,
                )
                vl_embs = bout.hidden_states[-1]  # (B, seq_len, H)

            state_t = torch.from_numpy(state_enc).unsqueeze(0).to(
                device=device, dtype=vl_embs.dtype
            )  # (B, 1, state_dim)

            with torch.autocast("cuda", dtype=torch.float32):
                rt.reset_chunk()
                norm_actions = run_denoising_loop(
                    action_model, vl_embs, state_t, seed=seed, num_steps=sampling_steps,
                )  # (B, action_horizon, action_dim)

        norm_np = norm_actions[0].cpu().numpy()  # (action_horizon, action_dim)
        # Unnormalize first 7 dims
        norm7 = np.clip(norm_np[:, :7], -1.0, 1.0)
        raw = np.where(
            action_mask,
            0.5 * (norm7 + 1.0) * (action_high - action_low) + action_low,
            norm7,
        )
        # Binarize gripper. LIBERO convention: -1.0 = open, +1.0 = close.
        # Model's normalized output >0.5 means "open" -> -1.0; <=0.5 -> +1.0.
        # This matches binarize_gripper() in the gripper_xyz collector and
        # _binarize_gripper_open in eval_libero.py. (Previously inverted, which
        # made the gripper do the opposite of the policy's intent -> 0% success.)
        raw[:, 6] = np.where(norm_np[:, 6] < 0.5, 1.0, -1.0)
        inference_input = {
            "model_image": batch_images[0].copy(),
            "state_encoded": state_enc.copy(),
        }
        return [raw[i] for i in range(NUM_OPEN_LOOP)], inference_input

    # ---- Video helper ----
    def save_video(frames, inference_ids, path, fps, flip_ud=True):
        writer = imageio.get_writer(str(path), fps=fps)
        for frame, inference_idx in zip(frames, inference_ids):
            shown = np.flipud(frame) if flip_ud else frame
            writer.append_data(annotate_inference(shown, inference_idx))
        writer.close()

    # ---- Rollout ----
    def rollout(ep_idx, init_state, *, steered: bool):
        env.reset()
        obs = stress_test.set_init_state(env, init_state, episode_idx=ep_idx)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        rt.steering_enabled = bool(steered)
        rt.reset_chunk()
        rt.u_norm_log.clear()
        if args.save_activations:
            rt.begin_trace("steered" if steered else "baseline")

        queue = deque(maxlen=NUM_OPEN_LOOP)
        frames = [obs["agentview_image"].copy()]
        frame_inference_ids = [0]
        inference_snapshots = []
        executed_actions = []
        inference_idx = -1
        chunk_idx = 0
        success = False
        t = 0
        while t < max_env_steps:
            if not queue:
                if steered:
                    _swap_K_for_chunk(chunk_idx)
                    if chunk_idx == 0 or chunk_idx % 5 == 0:
                        r_now = r_scale_schedule[min(chunk_idx, args.max_chunks - 1)]
                        print(f"    chunk {chunk_idx:3d}: R_SCALE={r_now:.2e}", flush=True)
                inference_idx += 1
                sim_state = flattened_sim_state(env) if args.save_inference_snapshots else None
                exact_obs = clone_observation(obs) if args.save_inference_snapshots else None
                actions, inference_input = policy_fn(obs)
                if args.save_inference_snapshots:
                    inference_snapshots.append({
                        "inference_idx": int(inference_idx),
                        "env_step": int(t),
                        "executed_action_count": int(len(executed_actions)),
                        "observation": exact_obs,
                        "sim_state": sim_state,
                        "model_input": inference_input,
                        "predicted_env_actions": np.stack(actions).astype(np.float32),
                        "policy_seed": int(seed),
                        "lqr_chunk_idx": int(chunk_idx),
                        "lqr_r_scale": float(
                            r_scale_schedule[min(chunk_idx, args.max_chunks - 1)]
                        ) if steered else None,
                    })
                for a in actions:
                    queue.append(np.asarray(a, dtype=np.float32))
                chunk_idx += 1
            a = queue.popleft()
            executed_actions.append(a.copy())
            obs, _, done, _ = env.step(a.tolist())
            frames.append(obs["agentview_image"].copy())
            frame_inference_ids.append(inference_idx)
            if done:
                success = True
                break
            t += 1
        trace = rt.finish_trace() if args.save_activations else []
        return (success, t + args.num_steps_wait, frames, frame_inference_ids,
                chunk_idx, trace, inference_snapshots, executed_actions)

    my_episodes = list(range(rank, args.n_episodes, world_size))
    print(f"[rank {rank}/{world_size}] handling {len(my_episodes)} episodes: {my_episodes}")

    results = []
    baseline_dir = out_dir / "baseline"
    if (args.run_baseline or args.baseline_only) and args.save_video:
        baseline_dir.mkdir(parents=True, exist_ok=True)

    def _do_run(ep, label):
        steered = (label == "steered")
        suffix = "" if steered else "__baseline"
        t0 = time.time()
        (success, env_steps, frames, frame_inference_ids, n_chunks, trace,
         inference_snapshots, executed_actions) = rollout(
            ep, init_states[ep], steered=steered
        )
        dt = time.time() - t0
        tag = "SUCCESS" if success else "FAILURE"
        video_path = None
        if args.save_video:
            video_dir = baseline_dir if not steered else out_dir
            video_path = video_dir / f"ep{ep:02d}--{tag}{suffix}.mp4"
            save_video(frames, frame_inference_ids, video_path, fps=args.video_fps)
        sample = dict(getattr(stress_test, "_last_sample", {})) or None
        snapshot_path = None
        if args.save_inference_snapshots:
            snapshot_dir = out_dir / "inference_snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / f"ep{ep:02d}__{label}.pt"
            torch.save({
                "format_version": 1,
                "episode": int(ep), "condition": label, "success": bool(success),
                "suite": args.suite, "task_id": int(args.task_id), "prompt": args.prompt,
                "initial_state": np.asarray(init_states[ep]).copy(),
                "env_seed": 42, "policy_seed": int(seed),
                "perturbation": sample,
                "num_steps_wait": int(args.num_steps_wait),
                "inferences": inference_snapshots,
                "executed_actions": (
                    np.stack(executed_actions).astype(np.float32)
                    if executed_actions else np.empty((0, 7), dtype=np.float32)
                ),
                "note": "observation is the exact pre-policy LIBERO obs; model_input is the exact resized policy input",
            }, snapshot_path)
        activation_path = None
        if args.save_activations:
            activation_dir = out_dir / "activations"
            activation_dir.mkdir(parents=True, exist_ok=True)
            activation_path = activation_dir / f"ep{ep:02d}__{label}.pt"
            acts = torch.stack([row["activation"] for row in trace]).half()
            torch.save({
                "episode": int(ep), "condition": label, "success": bool(success),
                "block": torch.tensor([row["block"] for row in trace], dtype=torch.int16),
                "step": torch.tensor([row["step"] for row in trace], dtype=torch.int16),
                "activation": acts,
                "pooling": "mean over action-token horizon after all registered steering hooks",
            }, activation_path)
        xyz_str = ("[" + ",".join(f"{v*1000:+5.1f}" for v in (sample or {}).get("xyz_delta_m", [0,0,0])) + "]"
                   if sample else "[--]")
        print(f"[ep {ep:2d}] {label:8s} {tag:7s}  steps={env_steps:4d}  "
              f"chunks={n_chunks:3d}  dxyz_mm={xyz_str}  {dt:6.1f}s", flush=True)
        return {"success": bool(success), "env_steps": int(env_steps),
                "n_chunks": int(n_chunks), "wall_time_s": float(dt),
                "video_path": str(video_path) if video_path else None,
                "snapshot_path": str(snapshot_path) if snapshot_path else None,
                "activation_path": str(activation_path) if activation_path else None,
                "perturbation": sample}

    t_total = time.time()
    for ep in my_episodes:
        if args.baseline_only:
            baseline_rec = _do_run(ep, "baseline")
            results.append({"episode": ep, "rank": rank, **baseline_rec, "baseline": None})
        else:
            steered_rec  = _do_run(ep, "steered")
            baseline_rec = _do_run(ep, "baseline") if args.run_baseline else None
            results.append({"episode": ep, "rank": rank, **steered_rec, "baseline": baseline_rec})

    for h in handles:
        h.remove()
    env.close()

    results_path = out_dir / f"results_rank{rank}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[rank {rank}] wrote {results_path}", flush=True)

    if rank == 0:
        manifest = {
            "tag": args.tag,
            "lqr": {
                "lambda":          float(args.lambda_scale),
                "Q_SCALE":         float(args.q_scale),
                "R_SCALE_INIT":    float(args.r_scale),
                "R_SCALE_TAU":     float(args.r_scale_tau),
                "R_SCALE_FINAL":   float(args.r_scale_final),
                "max_chunks":      int(args.max_chunks),
                "r_scale_schedule": r_scale_schedule,
                "QF_SCALE":        float(args.qf_scale),
                "have_B_tilde":    bool(have_B),
            },
            "perturbation": {
                "preset": args.preset, "base_seed": int(args.base_seed),
                "stress_test": stress_test.manifest(),
            },
            "rollout": {
                "suite": args.suite, "task_id": int(args.task_id),
                "task_desc_libero": task_desc_libero, "policy_prompt": args.prompt,
                "n_episodes": int(args.n_episodes), "max_env_steps": int(max_env_steps),
                "run_baseline": bool(args.run_baseline), "resolution": int(args.resolution),
                "save_activations": bool(args.save_activations),
                "video_fps": int(args.video_fps), "seed": int(seed),
                "save_inference_snapshots": bool(args.save_inference_snapshots),
                "num_steps_wait": int(args.num_steps_wait),
            },
            "parallel": {"world_size": world_size},
            "svd": {
                "svd_dir": str(svd_dir), "jac_dir_act": args.jac_dir_act,
                "jac_prompt": jac_prompt, "selected_timesteps": sel_t,
                "sampling_steps": sampling_steps, "L": L, "k_target": r,
                "partitions": partitions,
            },
            "model": {"ckpt_path": args.ckpt_path, "L_blocks": n_blocks},
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        print(f"[rank 0] wrote manifest -> {out_dir/'manifest.json'}", flush=True)

    n_succ = sum(r["success"] for r in results)
    print(f"[rank {rank}] DONE: {n_succ}/{len(results)} steered succeeded; "
          f"wall {time.time()-t_total:.1f}s")
    print(f"[rank {rank}] V cache: hits={vcache.stats['gpu_hits']} "
          f"swaps={vcache.stats['gpu_swaps']} "
          f"cpu->gpu={vcache.stats['cpu_to_gpu_s']:.1f}s")


def main():
    args = parse_args()
    if args.phase == "merge":
        run_merge_phase(args)
    else:
        run_rollout_phase(args)


if __name__ == "__main__":
    main()
