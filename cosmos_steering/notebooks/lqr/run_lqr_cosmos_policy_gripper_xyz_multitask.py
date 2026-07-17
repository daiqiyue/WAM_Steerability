#!/usr/bin/env python
"""Multi-task Activation-LQR (A-LQR) rollout of Cosmos-Policy under per-episode
RANDOM GRIPPER (x,y,z) PERTURBATION, spanning multiple LIBERO tasks.

Multi-task sibling of run_lqr_cosmos_policy_gripper_xyz.py (which handles a
single --task-id). Same closed-loop LQR steering machinery (SteeringRuntime +
install_lqr_hooks + chained_riccati_per_chunk + VCache) and same
exponentially-decaying R_SCALE schedule. Differences from the single-task
sibling:

  - --task-ids replaces --task-id; rollouts span every listed LIBERO task.
  - --n-episodes-per-task replaces --n-episodes. Per task, episode_idx
    restarts at 0, so (preset, base_seed, episode_idx) -> same xyz_delta as
    when collect_policy_inputs_gripper_xyz_perturbation_multitask.py was run
    with the same base_seed.
  - Each task uses its libero-default prompt unless --prompt overrides for
    all tasks. The LQR linearization is taken at one prompt + one scene (the
    jacobian under --svd-dir/--jac-dir-act); off-task rollouts reuse the same
    K matrices as a generalization.

Parallelization
---------------
Total jobs = sum(n_episodes_per_task for _ in task_ids). Workers are assigned
a contiguous block of (task_id, episode_idx) pairs:

    jobs = [(t, e) for t in task_ids for e in range(n_episodes_per_task)]
    start = rank * len(jobs) // world_size
    end   = (rank + 1) * len(jobs) // world_size
    my_jobs = jobs[start:end]

For ``world_size == len(task_ids)`` this assigns exactly one task per rank
(env built once per rank). For other world_sizes the rank may span multiple
tasks; envs are built lazily and closed when the rank moves on to the next
task. After all ranks finish, run with --phase merge (no GPU) to aggregate
per-rank results into results.json and patch the top-level manifest.json.

Outputs (under --out-dir)
-------------------------
    task{T:02d}/ep{i:02d}--{SUCCESS|FAILURE}.mp4              steered
    task{T:02d}/baseline/ep{i:02d}--{SUCCESS|FAILURE}__baseline.mp4
    results_rank{R}.json                                       per-rank
    results.json                                               merged
    manifest.json                                              full config
"""

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


def _setup_env():
    """Match notebooks/_setup.py -- run BEFORE any cosmos_policy import."""
    hf = os.environ.get("HF_HOME", "/work/nvme/bhde/jhong7/huggingface")
    hub = os.environ.get("HF_HUB_CACHE", str(Path(hf, "hub")))
    Path(hub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf)
    os.environ.setdefault("HF_HUB_CACHE", hub)
    os.environ.setdefault("TRANSFORMERS_CACHE", hub)

    if "HF_TOKEN" not in os.environ:
        for candidate in (
            Path(os.path.expanduser("~/.huggingface/token")),
            Path(os.path.expanduser("~/.cache/huggingface/token")),
        ):
            if candidate.exists():
                os.environ["HF_TOKEN"] = candidate.read_text().strip()
                break

    libero_cfg = os.environ.get("LIBERO_CONFIG_PATH",
                                "/work/nvme/bhde/jhong7/.libero")
    Path(libero_cfg).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", libero_cfg)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_setup_env()

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


# ====================================================================
# CLI
# ====================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ---- Mode --------------------------------------------------------
    ap.add_argument("--phase", choices=["rollout", "merge"], default="rollout",
                    help="'rollout' (default) runs the per-rank rollout. "
                         "'merge' reads results_rank*.json under --out-dir and "
                         "writes results.json + manifest.json. No GPU needed.")
    ap.add_argument("--world-size", type=int,
                    default=int(os.environ.get("WORLD_SIZE", 1)),
                    help="number of parallel ranks (workers) covering "
                         "--n-episodes. Each rank handles episodes "
                         "range(rank, n_episodes, world_size).")
    ap.add_argument("--rank", type=int,
                    default=int(os.environ.get("RANK", 0)),
                    help="this worker's rank in [0, world_size).")

    # ---- SVD / jacobian inputs --------------------------------------
    ap.add_argument("--svd-dir", type=Path, required=True,
                    help="SVD output dir (contains config.json, "
                         "svd_summary.pt, V_part*_t*.pt)")
    ap.add_argument("--jac-dir-act", type=str, default=None,
                    help="A_tilde output subdir under --svd-dir; must "
                         "contain A_tilde__full.pt. SINGLE-jacobian mode: "
                         "every task reuses these K matrices. Either this "
                         "or --jac-dir-act-per-task is required.")
    ap.add_argument("--jac-dir-act-per-task", type=Path, default=None,
                    help="PER-TASK jacobian mode. Path to a JSON file "
                         "mapping {\"<task_id>\": \"<jac_subdir_under_svd>\"} "
                         "for every task in --task-ids. Each subdir must "
                         "contain A_tilde__full.pt. When set, --jac-dir-act "
                         "is ignored and rollouts swap their LQR K matrices "
                         "as they move between tasks.")

    # ---- Gripper xyz perturbation -----------------------------------
    ap.add_argument("--preset", type=str, default="xyz_random_xlarge_3",
                    help="gripper perturbation preset name (see "
                         "11_gripper_xyz_perturbation.ipynb / build_presets); "
                         "default xyz_random_xlarge_3 (sigma=10cm isotropic, "
                         "RandomGripperXYZPerturbation).")
    ap.add_argument("--base-seed", type=int, default=42,
                    help="base seed for the per-episode RNG of random "
                         "presets; per-episode RNG = "
                         "SeedSequence([base_seed, episode_idx]).")
    ap.add_argument("--gripper-action", type=float, default=-1.0,
                    help="OSC gripper command held during the shift + pause "
                         "(-1.0 = open, +1.0 = close).")
    ap.add_argument("--pre-shift-settle-steps", type=int, default=10,
                    help="no-op env steps before the shift loop "
                         "(scene-settle; preset default = 10).")
    ap.add_argument("--shift-max-steps", type=int, default=30,
                    help="hard cap on env.step calls inside the shift loop.")
    ap.add_argument("--shift-tol-m", type=float, default=2e-3,
                    help="stop the shift loop once EE is within this many "
                         "meters of the target.")
    ap.add_argument("--post-shift-pause-steps", type=int, default=10,
                    help="zero-delta env steps after the shift loop "
                         "(pause / hold; preset default = 10).")

    # ---- LQR cost hyperparameters -----------------------------------
    ap.add_argument("--lambda-scale", type=float, default=1.0,
                    dest="lambda_scale", help="A-LQR setpoint scale (LAMBDA)")
    ap.add_argument("--q-scale",  type=float, default=10000.0, dest="q_scale")
    ap.add_argument("--r-scale",  type=float, default=75000.0, dest="r_scale",
                    help="initial LQR control cost at chunk 0; grows toward "
                         "--r-scale-final with time constant --r-scale-tau")
    ap.add_argument("--r-scale-tau", type=float, default=3.0,
                    dest="r_scale_tau",
                    help="exponential growth time constant of R_SCALE in chunks")
    ap.add_argument("--r-scale-final", type=float, default=1e9,
                    dest="r_scale_final",
                    help="upper clamp on R_SCALE (saturates to ~no steering)")
    ap.add_argument("--max-chunks", type=int, default=50, dest="max_chunks",
                    help="precompute per-chunk K matrices for c in "
                         "[0, MAX_CHUNKS); chunks beyond clamp to the last "
                         "(saturated) entry")
    ap.add_argument("--qf-scale", type=float, default=1.0, dest="qf_scale")

    # ---- Rollout ----------------------------------------------------
    ap.add_argument("--prompt", type=str, default=None,
                    help="optional override: when set, applied to ALL tasks. "
                         "Default uses each libero task's built-in "
                         "description (matches "
                         "collect_policy_inputs_gripper_xyz_perturbation_multitask.py).")
    ap.add_argument("--n-episodes-per-task", type=int, default=10,
                    dest="n_episodes_per_task",
                    help="number of rollouts PER libero task; total rollouts "
                         "= len(task_ids) * n_episodes_per_task.")
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--task-ids", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    dest="task_ids",
                    help="libero task IDs to roll out (default: all 10 in "
                         "libero_10).")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--num-steps-wait", type=int, default=10,
                    help="extra no-op env steps after the perturbation has "
                         "settled and before policy inference begins "
                         "(matches collect_policy_inputs_*.ipynb convention).")
    ap.add_argument("--max-env-steps", type=int, default=1000,
                    dest="max_env_steps",
                    help="cap on env steps per rollout (default 1000; "
                         "leaves headroom under heavy perturbations).")
    ap.add_argument("--run-baseline", dest="run_baseline",
                    action="store_true", default=False,
                    help="also run an unsteered baseline rollout per episode "
                         "from the same init_state + perturbation seed "
                         "(LQR hooks gated off); default off.")
    ap.add_argument("--no-baseline", dest="run_baseline",
                    action="store_false",
                    help="explicitly disable the per-episode baseline run")
    ap.add_argument("--baseline-only", dest="baseline_only",
                    action="store_true", default=False,
                    help="run only the unsteered baseline pass per episode "
                         "(skip the LQR-steered pass). Top-level fields in "
                         "results.json come from the baseline run so the "
                         "aggregator's success counter reflects baseline "
                         "success; the nested `baseline` field is null.")

    # ---- Model / seed -----------------------------------------------
    ap.add_argument("--seed", type=int, default=None,
                    help="rollout seed; default = cfg.json's 'seed' or 42")

    # ---- Output -----------------------------------------------------
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="output dir; manifest + results + mp4s land here")
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--no-save-video", dest="save_video",
                    action="store_false")
    ap.add_argument("--tag", type=str, default=None,
                    help="optional grouping tag recorded in manifest.json")

    return ap.parse_args()


# ====================================================================
# GripperXYZPerturbation -- copied from
# notebooks/stress_test/11_gripper_xyz_perturbation.ipynb
# (kept self-contained in this script, matching the repo's existing
# pattern of inlining stress-test classes per consumer).
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


def _shift_gripper_xyz_via_actions(env, target_pos, *, site_name=None,
                                    body_name=None, gripper_action=-1.0,
                                    max_steps=30, tol_m=2e-3,
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
        action = [float(action_xyz[0]), float(action_xyz[1]),
                  float(action_xyz[2]), 0.0, 0.0, 0.0,
                  float(gripper_action)]
        obs, _, _, _ = env.step(action)
        n_used = step + 1
        achieved = _ee_pos(sim, site_name=site_name, body_name=body_name)
    residual = float(np.linalg.norm(target_pos - achieved))
    return obs, achieved, residual, n_used


class _StressTest:
    slug: str = "stock"
    def transform_task(self, task, output_dir=None): return task
    def set_init_state(self, env, init_state, episode_idx=0):
        return env.set_init_state(init_state)
    def apply_to_env(self, env, episode_idx=0): return None
    def transform_task_desc(self, desc, env=None): return desc
    def manifest(self) -> dict:
        return {"kind": type(self).__name__, "slug": self.slug}


@dataclass
class GripperXYZPerturbation(_StressTest):
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
        h = hashlib.md5(
            f"xyz={xyz_str}|g={self.gripper_action:+.1f}".encode()
        ).hexdigest()[:6]
        return f"{self.name_hint}_{h}"

    def _resolve_xyz(self, episode_idx):
        return tuple(float(v) for v in self.xyz_delta)

    def set_init_state(self, env, init_state, episode_idx=0):
        problem = _resolve_problem(env)
        sim = problem.sim
        site_name, body_name = _resolve_ee_target(sim)

        from cosmos_policy.experiments.robot.libero.libero_utils import (
            get_libero_dummy_action,
        )
        baseline_obs = env.set_init_state(init_state)
        for _ in range(self.pre_shift_settle_steps):
            baseline_obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

        sim.forward()
        baseline_ee = _ee_pos(sim, site_name=site_name, body_name=body_name)

        dxyz = self._resolve_xyz(episode_idx)
        target_ee = baseline_ee + np.asarray(dxyz, dtype=np.float64)

        if np.allclose(dxyz, 0.0):
            achieved_ee = baseline_ee
            residual = 0.0
            n_used = 0
            final_obs = baseline_obs
        else:
            final_obs, achieved_ee, residual, n_used = (
                _shift_gripper_xyz_via_actions(
                    env, target_ee,
                    site_name=site_name, body_name=body_name,
                    gripper_action=self.gripper_action,
                    max_steps=self.shift_max_steps,
                    tol_m=self.shift_tol_m,
                    output_max_m=self.output_max_m,
                )
            )

        pause_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        float(self.gripper_action)]
        for _ in range(self.post_shift_pause_steps):
            final_obs, _, _, _ = env.step(pause_action)

        self._last_sample = {
            "episode_idx": int(episode_idx),
            "xyz_delta_m":       [float(v) for v in dxyz],
            "baseline_ee_pos":   [float(v) for v in baseline_ee],
            "target_ee_pos":     [float(v) for v in target_ee],
            "achieved_ee_pos":   [float(v) for v in achieved_ee],
            "achieved_xyz_delta_m": [float(a - b)
                                     for a, b in zip(achieved_ee, baseline_ee)],
            "shift_residual_m":  float(residual),
            "shift_steps_used":  int(n_used),
            "gripper_action":    float(self.gripper_action),
        }
        if residual > max(self.shift_tol_m * 5, 5e-3):
            print(f"  [shift warn] ep{episode_idx} residual={residual*1000:.2f} "
                  f"mm after {n_used} steps; target may be blocked or out of "
                  f"reach.", flush=True)
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
            "sampling": ("per-episode Gaussian, "
                         "seed=SeedSequence([base_seed, episode_idx])"),
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


def build_presets(args) -> dict:
    """Same preset names as 11_gripper_xyz_perturbation.ipynb."""
    common = dict(
        gripper_action=args.gripper_action,
        pre_shift_settle_steps=args.pre_shift_settle_steps,
        shift_max_steps=args.shift_max_steps,
        shift_tol_m=args.shift_tol_m,
        post_shift_pause_steps=args.post_shift_pause_steps,
    )
    rcommon = dict(common, base_seed=args.base_seed)
    return {
        "xyz_+x_2cm": GripperXYZPerturbation(xyz_delta=( 0.02,  0.00,  0.00), name_hint="xyz_+x_2cm", **common),
        "xyz_-x_2cm": GripperXYZPerturbation(xyz_delta=(-0.02,  0.00,  0.00), name_hint="xyz_-x_2cm", **common),
        "xyz_+y_2cm": GripperXYZPerturbation(xyz_delta=( 0.00,  0.02,  0.00), name_hint="xyz_+y_2cm", **common),
        "xyz_-y_2cm": GripperXYZPerturbation(xyz_delta=( 0.00, -0.02,  0.00), name_hint="xyz_-y_2cm", **common),
        "xyz_+z_2cm": GripperXYZPerturbation(xyz_delta=( 0.00,  0.00,  0.02), name_hint="xyz_+z_2cm", **common),
        "xyz_-z_2cm": GripperXYZPerturbation(xyz_delta=( 0.00,  0.00, -0.02), name_hint="xyz_-z_2cm", **common),
        "xyz_diag_+1cm": GripperXYZPerturbation(xyz_delta=( 0.01,  0.01,  0.01), name_hint="xyz_diag_+1cm", **common),
        "xyz_diag_+3cm": GripperXYZPerturbation(xyz_delta=( 0.03,  0.03,  0.03), name_hint="xyz_diag_+3cm", **common),
        "xyz_random_small":  RandomGripperXYZPerturbation(sigma_xyz_m=0.01, name_hint="xyz_random_small",  **rcommon),
        "xyz_random_medium": RandomGripperXYZPerturbation(sigma_xyz_m=0.02, name_hint="xyz_random_medium", **rcommon),
        "xyz_random_large":  RandomGripperXYZPerturbation(sigma_xyz_m=0.04, name_hint="xyz_random_large",  **rcommon),
        "xyz_random_xlarge": RandomGripperXYZPerturbation(sigma_xyz_m=0.06, name_hint="xyz_random_xlarge", **rcommon),
        "xyz_random_xlarge_2": RandomGripperXYZPerturbation(sigma_xyz_m=0.15, name_hint="xyz_random_xlarge_2", **rcommon),
        "xyz_random_xlarge_3": RandomGripperXYZPerturbation(sigma_xyz_m=0.10, name_hint="xyz_random_xlarge_3", **rcommon),
        "xyz_random_horizontal": RandomGripperXYZPerturbation(
            sigma_xyz_m=(0.03, 0.03, 0.005), name_hint="xyz_random_horizontal", **rcommon,
        ),
        "xyz_random": RandomGripperXYZPerturbation(
            sigma_xyz_m=(0.1, 0.1, 0.1), name_hint="xyz_random", **rcommon,
        ),
    }


# ====================================================================
# V-tile cache (per (partition, timestep))  -- identical to noised.py
# ====================================================================

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
        self.stats = {"cpu_loads": 0, "gpu_swaps": 0, "gpu_hits": 0,
                      "cpu_to_gpu_s": 0.0}

    def _vfile(self, p_idx, t_id):
        a, b = self.partitions[p_idx]
        pattern = f"V_part{p_idx}_layers{a}-{b}_t{t_id}_k*.pt"
        matches = sorted(self.svd_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"no V file matching {pattern} in {self.svd_dir}"
            )
        preferred = [m for m in matches
                     if m.name.endswith(f"_k{self.k_target}.pt")]
        return (preferred or matches)[0]

    def _load_cpu(self, p_idx, t_id):
        key = (p_idx, t_id)
        V = self._cpu.get(key)
        if V is not None:
            return V
        fp = self._vfile(p_idx, t_id)
        print(f"  disk-load {fp.name} "
              f"({fp.stat().st_size / 1e9:.2f} GB) -> CPU {self.dtype} ...",
              flush=True)
        t0 = time.time()
        raw = torch.load(fp, map_location="cpu", weights_only=False)["V"]
        V = raw.to(dtype=self.dtype).contiguous()
        del raw
        self._cpu[key] = V
        self.stats["cpu_loads"] += 1
        print(f"    CPU resident (p={p_idx}, t={t_id}): {tuple(V.shape)}  "
              f"~{V.element_size() * V.numel() / 1e9:.2f} GB  "
              f"({time.time() - t0:.1f}s)", flush=True)
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


# ====================================================================
# Chained Riccati  -- identical to noised.py
# ====================================================================

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

    K_intra_per_chunk = torch.zeros(
        n_chunks, T_diff, L - 1, r, r, dtype=torch.float32
    )
    K_step_per_chunk = torch.zeros(
        n_chunks, max(T_diff - 1, 0), r, r, dtype=torch.float32
    )

    for c, r_scale_c in enumerate(r_scale_schedule):
        R_chain = (r_scale_c * I_r).expand(K_total, r, r).contiguous()
        Tn = A_chain.shape[0]
        S = torch.zeros(Tn + 1, r, r, dtype=_dtype, device=device)
        K = torch.zeros(Tn, r, r, dtype=_dtype, device=device)
        S[Tn] = S_T
        for k in reversed(range(Tn)):
            Ak = A_chain[k]
            P = S[k + 1] + R_chain[k]
            F = S[k + 1] @ Ak
            G = Q_chain[k] + Ak.transpose(-2, -1) @ S[k + 1] @ Ak
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


# ====================================================================
# Steering runtime + hooks  -- identical to noised.py
# ====================================================================

class SteeringRuntime:
    def __init__(self, *, L, T_diff, sel_t, denoise_t_start, denoise_t_end,
                 T_p_denoise, H_p, W_p, D, sampling_steps,
                 lambda_scale, vcache, lqr):
        self.L = L
        self.T_diff = T_diff
        self.sel_t = list(sel_t)
        self.sel_idx_of = {t: i for i, t in enumerate(self.sel_t)}
        self.denoise_t_start = denoise_t_start
        self.denoise_t_end = denoise_t_end
        self.T_p_denoise = T_p_denoise
        self.H_p = H_p
        self.W_p = W_p
        self.D = D
        self.sampling_steps = sampling_steps

        self.lambda_scale = float(lambda_scale)
        self.vcache = vcache
        self.lqr = lqr

        self.pass_idx = -1
        self.u_step_pending = None
        self.in_ad = False
        self.u_norm_log = []
        self.steering_enabled = True

    def reset_chunk(self):
        self.pass_idx = -1
        self.u_step_pending = None

    def is_selected_step(self, pass_idx):
        if pass_idx < 0:
            return None, None
        step = pass_idx
        sel = self.sel_idx_of.get(step)
        if sel is None:
            return None, None
        return step, sel


def install_lqr_hooks(model, rt):
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
            z_full = (
                output[0, rt.denoise_t_start:rt.denoise_t_end, :, :, :]
                .detach().reshape(-1)
            )
            z_dt = output.dtype
            V_in = rt.vcache.for_layer(l_in, step)
            rt.in_ad = True
            try:
                x_proj = (z_full.to(V_in.dtype) @ V_in).float()
                v_fp = rt.lqr["v"][l_in, sel]
                mu_fp = rt.lqr["mu"][l_in, sel]
                K_fp = rt.lqr["K_intra"][sel, l_in]
                alpha = rt.lambda_scale * mu_fp - v_fp @ x_proj
                u_tilde = K_fp @ (alpha * v_fp)
                rt.u_norm_log.append(
                    (sel, l_in, float(u_tilde.norm()), float(alpha))
                )
            finally:
                rt.in_ad = False
            del V_in
            V_out = rt.vcache.for_layer(l_in + 1, step)
            rt.in_ad = True
            try:
                u_full = V_out @ u_tilde.to(V_out.dtype)
            finally:
                rt.in_ad = False
            u_add = u_full.to(z_dt).reshape(
                rt.T_p_denoise, rt.H_p, rt.W_p, rt.D
            )
            output[0, rt.denoise_t_start:rt.denoise_t_end, :, :, :] = (
                output[0, rt.denoise_t_start:rt.denoise_t_end, :, :, :] + u_add
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
            output[0, rt.denoise_t_start:rt.denoise_t_end, :, :, :]
            .detach().reshape(-1)
        )
        V_in = rt.vcache.for_layer(L - 1, step)
        rt.in_ad = True
        try:
            x_proj = (z_full.to(V_in.dtype) @ V_in).float()
            v_fp = rt.lqr["v"][L - 1, sel]
            mu_fp = rt.lqr["mu"][L - 1, sel]
            K_fp = rt.lqr["K_step"][sel]
            alpha = rt.lambda_scale * mu_fp - v_fp @ x_proj
            u_tilde = K_fp @ (alpha * v_fp)
            rt.u_norm_log.append(
                (sel, -1, float(u_tilde.norm()), float(alpha))
            )
            rt.u_step_pending = {"src_sel": sel, "u_tilde": u_tilde.detach()}
        finally:
            rt.in_ad = False
        return None

    def cross_step_apply(block, args, output):
        if rt.in_ad or not rt.steering_enabled:
            return None
        step, sel = rt.is_selected_step(rt.pass_idx)
        pending = rt.u_step_pending
        if step is None or pending is None or sel == 0 \
                or pending["src_sel"] != sel - 1:
            return None
        u_tilde = pending["u_tilde"]
        V_dest = rt.vcache.for_layer(0, step)
        rt.in_ad = True
        try:
            u_full = V_dest @ u_tilde.to(V_dest.dtype)
        finally:
            rt.in_ad = False
        u_add = u_full.to(output.dtype).reshape(
            rt.T_p_denoise, rt.H_p, rt.W_p, rt.D
        )
        output[0, rt.denoise_t_start:rt.denoise_t_end, :, :, :] = (
            output[0, rt.denoise_t_start:rt.denoise_t_end, :, :, :] + u_add
        )
        rt.u_step_pending = None
        return output

    handles.append(model.net.blocks[0].register_forward_pre_hook(_pass_tick))
    handles.append(model.net.blocks[0].register_forward_hook(cross_step_apply))
    for l_in in range(L - 1):
        handles.append(
            model.net.blocks[l_in + 1].register_forward_hook(
                make_intra_hook(l_in)
            )
        )
    handles.append(
        model.net.blocks[L - 1].register_forward_hook(cross_step_compute)
    )
    return handles


# ====================================================================
# Video helper
# ====================================================================

def save_video(frames, path, fps, flip_ud=True):
    writer = imageio.get_writer(str(path), fps=fps)
    for frame in frames:
        writer.append_data(np.flipud(frame) if flip_ud else frame)
    writer.close()


# ====================================================================
# Merge phase (no GPU; just aggregates results_rank*.json)
# ====================================================================

def run_merge_phase(args):
    out_dir = args.out_dir.resolve()
    pattern = sorted(out_dir.glob("results_rank*.json"))
    if not pattern:
        raise FileNotFoundError(
            f"no results_rank*.json under {out_dir}; "
            f"have the rollout workers finished?"
        )
    merged = []
    for fp in pattern:
        merged.extend(json.loads(fp.read_text()))
    # Multi-task: dedupe on (task_id, episode) and sort by the same key.
    merged.sort(key=lambda r: (int(r.get("task_id", -1)), int(r["episode"])))

    seen = set()
    dedup = []
    for r in merged:
        key = (int(r.get("task_id", -1)), int(r["episode"]))
        if key in seen:
            print(f"[merge] WARNING: duplicate (task={key[0]}, ep={key[1]}); "
                  f"keeping first occurrence", flush=True)
            continue
        seen.add(key)
        dedup.append(r)
    merged = dedup

    (out_dir / "results.json").write_text(json.dumps(merged, indent=2))

    n_total = len(merged)
    n_succ  = sum(r["success"] for r in merged)
    has_baseline = any(r.get("baseline") is not None for r in merged)
    n_base = sum((r.get("baseline") or {}).get("success", False)
                 for r in merged)

    # Per-task breakdown
    per_task = {}
    for r in merged:
        tid = int(r.get("task_id", -1))
        d = per_task.setdefault(tid, {"n": 0, "succ": 0, "base_succ": 0})
        d["n"] += 1
        if r["success"]:
            d["succ"] += 1
        if (r.get("baseline") or {}).get("success", False):
            d["base_succ"] += 1

    print(f"[merge] aggregated {n_total} rollouts from {len(pattern)} rank "
          f"files -> {out_dir / 'results.json'}", flush=True)
    print(f"[merge] steered overall:  {n_succ}/{n_total} succeeded "
          f"({100*n_succ/max(1,n_total):.0f}%)", flush=True)
    if has_baseline:
        print(f"[merge] baseline overall: {n_base}/{n_total} succeeded "
              f"({100*n_base/max(1,n_total):.0f}%)", flush=True)
    print(f"[merge] per-task:", flush=True)
    for tid in sorted(per_task):
        d = per_task[tid]
        line = (f"  task{tid:>2}: steered {d['succ']}/{d['n']} "
                f"({100*d['succ']/max(1,d['n']):.0f}%)")
        if has_baseline:
            line += (f"   baseline {d['base_succ']}/{d['n']} "
                     f"({100*d['base_succ']/max(1,d['n']):.0f}%)")
        print(line, flush=True)

    # Update manifest.json's totals if the file exists (written by rank 0).
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["totals"] = {
            "n_rollouts":        n_total,
            "n_success_steered": int(n_succ),
            "n_success_baseline": int(n_base) if has_baseline else None,
            "per_task": {str(tid): {
                "n":          int(per_task[tid]["n"]),
                "steered":    int(per_task[tid]["succ"]),
                "baseline":   (int(per_task[tid]["base_succ"])
                               if has_baseline else None),
            } for tid in sorted(per_task)},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"[merge] updated {manifest_path}", flush=True)


# ====================================================================
# Rollout phase
# ====================================================================

def run_rollout_phase(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.out_dir.resolve()
    rank = int(args.rank)
    world_size = int(args.world_size)
    assert 0 <= rank < world_size, (
        f"--rank {rank} not in [0, --world-size {world_size})"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_id = device.index if device.index is not None else 0

    # ---------------------------------------------------------------- SVD cfg
    svd_dir = args.svd_dir.resolve()
    cfg_path = svd_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"SVD config missing: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    sel_t           = list(cfg["selected_timesteps"])
    T_diff          = len(sel_t)
    sampling_steps  = int(cfg["sampling_steps"])
    L               = int(cfg["L"])
    r               = int(cfg["k_target"])
    partitions      = [tuple(p) for p in cfg["partitions"]]
    D               = int(cfg["D"])
    D_flat          = int(cfg["D_flat"])
    T_p_denoise     = int(cfg["T_p_denoise"])
    denoise_t_start = int(cfg["denoise_t_start"])
    denoise_t_end   = int(cfg["denoise_t_end"])
    H_p             = int(cfg["H_p"])
    W_p             = int(cfg["W_p"])
    ckpt_path = cfg.get("ckpt_path", "nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    config_name = cfg.get("config_name",
                          "cosmos_predict2_2b_480p_libero__inference_only")
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    print(f"[rank {rank}/{world_size}] svd: {svd_dir}", flush=True)
    print(f"[svd] sel_t={sel_t}  T_diff={T_diff}  L={L}  r={r}  "
          f"sampling_steps={sampling_steps}")
    print(f"[svd] partitions={partitions}  D={D}  D_flat={D_flat:,}")

    # ---------------------------------------------------------------- A_tilde
    if (args.jac_dir_act is None) == (args.jac_dir_act_per_task is None):
        raise SystemExit(
            "exactly one of --jac-dir-act or --jac-dir-act-per-task must be set"
        )

    per_task_jac_map: dict[int, str] = {}
    if args.jac_dir_act_per_task is not None:
        if not args.jac_dir_act_per_task.exists():
            raise FileNotFoundError(
                f"--jac-dir-act-per-task not found: "
                f"{args.jac_dir_act_per_task}"
            )
        per_task_jac_map = {
            int(k): str(v)
            for k, v in json.loads(args.jac_dir_act_per_task.read_text()).items()
        }
        missing = [int(t) for t in args.task_ids
                   if int(t) not in per_task_jac_map]
        if missing:
            raise SystemExit(
                f"per-task jacobian map missing entries for tasks {missing}; "
                f"available keys = {sorted(per_task_jac_map)}"
            )
        print(f"[jac] per-task mode: {len(per_task_jac_map)} task-jacobians, "
              f"map from {args.jac_dir_act_per_task}")
    else:
        for tid in args.task_ids:
            per_task_jac_map[int(tid)] = args.jac_dir_act
        print(f"[jac] single-jacobian mode: every task uses "
              f"{args.jac_dir_act!r}")

    sel_idx_of = {t: i for i, t in enumerate(sel_t)}

    def _load_A_B(jac_subdir: str):
        jac_dir = svd_dir / jac_subdir
        a_tilde_full = jac_dir / "A_tilde__full.pt"
        if not a_tilde_full.exists():
            raise FileNotFoundError(f"A_tilde missing: {a_tilde_full}")
        raw_ = torch.load(a_tilde_full, map_location="cpu",
                          weights_only=False)
        A_dict = raw_.get("A_tilde", {})
        B_dict = raw_.get("B_tilde", {})
        prompt = raw_.get("prompt", "<unknown>")

        A_tilde = torch.zeros(T_diff, L - 1, r, r, dtype=torch.float32)
        for (t, l_in), Atl in A_dict.items():
            if t in sel_idx_of:
                A_tilde[sel_idx_of[t], l_in] = Atl.float()

        have_B = len(B_dict) > 0
        if have_B:
            B_tilde = torch.zeros(T_diff - 1, r, r, dtype=torch.float32)
            for (t,), Bt in B_dict.items():
                if t in sel_idx_of and sel_idx_of[t] < T_diff - 1:
                    B_tilde[sel_idx_of[t]] = Bt.float()
        else:
            B_tilde = torch.zeros(max(T_diff - 1, 0), r, r,
                                   dtype=torch.float32)
        return A_tilde, B_tilde, prompt, have_B, len(A_dict), len(B_dict)

    # Reference task = first one in args.task_ids; used for the "primary"
    # K matrices that seed the LQR runtime and for B_tilde-presence logging.
    ref_tid = int(args.task_ids[0])
    A_tilde, B_tilde, jac_prompt, have_B, n_A, n_B = _load_A_B(
        per_task_jac_map[ref_tid]
    )
    print(f"[jac] reference task {ref_tid}: A entries={n_A} "
          f"(expected {T_diff*(L-1)})  B entries={n_B}  prompt={jac_prompt!r}")
    if not have_B:
        print("[jac] B_tilde empty (cosmos-policy default); chained Riccati "
              "degrades to per-step (zero cross-step A matrices).")

    # ---------------------------------------------------------------- LFS
    summary = torch.load(svd_dir / "svd_summary.pt", map_location="cpu",
                         weights_only=False)
    c_means = summary["c_means"].float()
    assert c_means.dim() == 3, (
        f"expected c_means shape (L, T_sel, k); got {tuple(c_means.shape)}"
    )
    tilde_mu = c_means.norm(dim=-1)
    tilde_v  = c_means / tilde_mu.unsqueeze(-1).clamp(min=1e-12)
    layer_to_part = list(summary["layer_to_part"])

    # ---------------------------------------------------------------- LQR schedule
    assert args.r_scale_tau > 0, "--r-scale-tau must be > 0"
    assert args.r_scale_final >= args.r_scale, (
        f"--r-scale-final ({args.r_scale_final:g}) must be >= "
        f"--r-scale ({args.r_scale:g})"
    )
    assert args.max_chunks >= 1, "--max-chunks must be >= 1"
    r_scale_schedule = [
        min(args.r_scale_final, args.r_scale * math.exp(c / args.r_scale_tau))
        for c in range(args.max_chunks)
    ]
    print(f"[lqr] λ={args.lambda_scale:g}  Q={args.q_scale:g}  "
          f"Qf={args.qf_scale:g}")
    print(f"[lqr] R_SCALE schedule: init={args.r_scale:g}  "
          f"tau={args.r_scale_tau:g}  final={args.r_scale_final:g}  "
          f"max_chunks={args.max_chunks}")
    _preview = ", ".join(f"{r_:.2e}" for r_ in r_scale_schedule[:8])
    print(f"[lqr]   R_SCALE(c=0..7): {_preview} ...")

    # Precompute K matrices for the reference task (always) plus every other
    # distinct task in per_task_jac_map. In single-jacobian mode there's only
    # one unique entry, so this collapses to the original single Riccati call.
    per_task_K: dict[int, dict[str, "torch.Tensor"]] = {}
    per_task_jac_prompt: dict[int, str] = {ref_tid: jac_prompt}
    seen_jac_subdirs: dict[str, int] = {per_task_jac_map[ref_tid]: ref_tid}

    t0 = time.time()
    K_intra_per_chunk, K_step_per_chunk = chained_riccati_per_chunk(
        A_tilde, B_tilde,
        args.q_scale, r_scale_schedule, args.qf_scale, device,
    )
    per_task_K[ref_tid] = {
        "K_intra_per_chunk": K_intra_per_chunk,
        "K_step_per_chunk":  K_step_per_chunk,
    }
    print(f"[lqr] chained Riccati (task {ref_tid}) over {args.max_chunks} "
          f"chunks in {time.time() - t0:.2f}s; "
          f"K_intra_per_chunk {tuple(K_intra_per_chunk.shape)}  "
          f"K_step_per_chunk {tuple(K_step_per_chunk.shape)}")

    for tid in sorted({int(t) for t in args.task_ids}):
        if tid == ref_tid:
            continue
        jac_subdir = per_task_jac_map[tid]
        # Share K matrices if two tasks point at the same jacobian subdir.
        if jac_subdir in seen_jac_subdirs:
            src_tid = seen_jac_subdirs[jac_subdir]
            per_task_K[tid] = per_task_K[src_tid]
            per_task_jac_prompt[tid] = per_task_jac_prompt[src_tid]
            print(f"[lqr] task {tid}: reusing K from task {src_tid} "
                  f"(same jacobian subdir)")
            continue
        At_t, Bt_t, jp_t, _, n_A_t, n_B_t = _load_A_B(jac_subdir)
        t1 = time.time()
        K_intra_t, K_step_t = chained_riccati_per_chunk(
            At_t, Bt_t,
            args.q_scale, r_scale_schedule, args.qf_scale, device,
        )
        per_task_K[tid] = {
            "K_intra_per_chunk": K_intra_t,
            "K_step_per_chunk":  K_step_t,
        }
        per_task_jac_prompt[tid] = jp_t
        seen_jac_subdirs[jac_subdir] = tid
        print(f"[lqr] chained Riccati (task {tid}) in "
              f"{time.time() - t1:.2f}s  jac_prompt={jp_t!r}  "
              f"A entries={n_A_t}  B entries={n_B_t}")

    # Seed K with the reference task's chunk-0 entries; will be swapped at
    # task boundaries (and per chunk within the rollout via _swap_K_for_chunk).
    K_intra = per_task_K[ref_tid]["K_intra_per_chunk"][0]
    K_step  = per_task_K[ref_tid]["K_step_per_chunk"][0]

    # ---------------------------------------------------------------- Stress test
    presets = build_presets(args)
    if args.preset not in presets:
        raise ValueError(
            f"unknown --preset={args.preset!r}; pick one of {sorted(presets)}"
        )
    stress_test = presets[args.preset]
    print(f"[stress] preset={args.preset}  ({type(stress_test).__name__})  "
          f"manifest={stress_test.manifest()}")

    # ---------------------------------------------------------------- Model
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        PolicyEvalConfig, prepare_observation, TASK_MAX_STEPS,
    )
    from cosmos_policy.experiments.robot.cosmos_utils import (
        COSMOS_IMAGE_SIZE, get_action, get_model,
        init_t5_text_embeddings_cache, load_dataset_stats,
    )

    eval_cfg = PolicyEvalConfig(
        config=config_name,
        ckpt_path=ckpt_path,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{ckpt_path}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{ckpt_path}/libero_t5_embeddings.pkl",
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        chunk_size=16,
        num_open_loop_steps=16,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=sampling_steps,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        task_suite_name=args.suite,
    )

    print(f"[model] loading dataset stats + T5 cache + weights on "
          f"cuda:{device_id} ...")
    dataset_stats = load_dataset_stats(eval_cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(eval_cfg.t5_text_embeddings_path,
                                   worker_id=rank)
    model, _ = get_model(eval_cfg)
    n_blocks = len(model.net.blocks)
    if n_blocks != L:
        raise RuntimeError(
            f"model has {n_blocks} DiT blocks; SVD config L={L}"
        )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free_gb, total_gb = (x / 1e9
                              for x in torch.cuda.mem_get_info(device_id))
        print(f"[model] after loader cleanup: GPU "
              f"{free_gb:.1f} / {total_gb:.1f} GB free")

    # ---------------------------------------------------------------- V cache
    vcache = VCache(svd_dir, partitions, layer_to_part, sel_t, r,
                    device=device, dtype=torch.bfloat16)
    print(f"[v] preloading all {len(partitions)} × {len(sel_t)} V tiles ...")
    vcache.preload([(p, t) for p in range(len(partitions)) for t in sel_t])
    vcache.for_layer(0, sel_t[0])

    # ---------------------------------------------------------------- LQR push
    lqr = {
        "K_intra": K_intra.to(device=device, dtype=torch.float32),
        "K_step":  K_step.to(device=device, dtype=torch.float32)
                   if K_step.numel() else K_step,
        "v":       tilde_v.to(device=device, dtype=torch.float32),
        "mu":      tilde_mu.to(device=device, dtype=torch.float32),
    }

    current_task_K = {"tid": ref_tid,
                       "K_intra": per_task_K[ref_tid]["K_intra_per_chunk"],
                       "K_step":  per_task_K[ref_tid]["K_step_per_chunk"]}

    def _set_active_task(tid: int) -> None:
        tid = int(tid)
        if current_task_K["tid"] == tid:
            return
        current_task_K["tid"]     = tid
        current_task_K["K_intra"] = per_task_K[tid]["K_intra_per_chunk"]
        current_task_K["K_step"]  = per_task_K[tid]["K_step_per_chunk"]
        # Force chunk-0 swap so the runtime picks up the new task's K.
        lqr["K_intra"].copy_(
            current_task_K["K_intra"][0].to(device=device,
                                              dtype=torch.float32)
        )
        if current_task_K["K_step"].numel():
            lqr["K_step"].copy_(
                current_task_K["K_step"][0].to(device=device,
                                                 dtype=torch.float32)
            )
        print(f"[lqr] active jacobian -> task {tid}", flush=True)

    def _swap_K_for_chunk(c: int) -> None:
        c_eff = min(c, args.max_chunks - 1)
        lqr["K_intra"].copy_(
            current_task_K["K_intra"][c_eff].to(device=device,
                                                  dtype=torch.float32)
        )
        if current_task_K["K_step"].numel():
            lqr["K_step"].copy_(
                current_task_K["K_step"][c_eff].to(device=device,
                                                     dtype=torch.float32)
            )

    # ---------------------------------------------------------------- Hooks
    rt = SteeringRuntime(
        L=L, T_diff=T_diff, sel_t=sel_t,
        denoise_t_start=denoise_t_start, denoise_t_end=denoise_t_end,
        T_p_denoise=T_p_denoise, H_p=H_p, W_p=W_p, D=D,
        sampling_steps=sampling_steps,
        lambda_scale=args.lambda_scale,
        vcache=vcache,
        lqr=lqr,
    )
    handles = install_lqr_hooks(model, rt)
    print(f"[hooks] registered {len(handles)} hooks "
          f"(1 pre_tick + 1 cross_apply + {L-1} intra + 1 cross_compute)")

    # ---------------------------------------------------------------- Env / tasks
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import (
        get_libero_env, get_libero_dummy_action,
    )

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    suite_max_steps = TASK_MAX_STEPS[args.suite]
    max_env_steps = int(args.max_env_steps)
    extra_for_shift = (args.pre_shift_settle_steps + args.shift_max_steps
                       + args.post_shift_pause_steps + 10)
    env_horizon = (args.num_steps_wait + max_env_steps + extra_for_shift)

    # Validate task ids and per-task init-state availability up front.
    task_meta = {}  # task_id -> {"task": task, "init_states": np.ndarray, "prompt": str}
    for tid in args.task_ids:
        task = task_suite.get_task(tid)
        init_states = task_suite.get_task_init_states(tid)
        if args.n_episodes_per_task > init_states.shape[0]:
            raise ValueError(
                f"--n-episodes-per-task {args.n_episodes_per_task} > available "
                f"init states for task {tid} ({init_states.shape[0]})"
            )
        # Libero exposes the canonical prompt on Task.language.
        libero_prompt = getattr(task, "language", None)
        task_meta[tid] = {
            "task": task,
            "init_states": init_states,
            "libero_prompt": libero_prompt,
        }

    # Build the flat (task_id, episode_idx) job list and assign a contiguous
    # block to this rank. With world_size == len(task_ids), each rank gets
    # exactly one task's worth of episodes (env built once per rank).
    jobs = [(int(tid), int(e))
            for tid in args.task_ids
            for e in range(args.n_episodes_per_task)]
    n_jobs = len(jobs)
    start = rank * n_jobs // world_size
    end   = (rank + 1) * n_jobs // world_size
    my_jobs = jobs[start:end]
    my_task_ids = sorted({t for t, _ in my_jobs})
    print(f"[rank {rank}/{world_size}] handling {len(my_jobs)} rollouts across "
          f"tasks {my_task_ids}: {my_jobs}", flush=True)

    # ---------------------------------------------------------------- Rollout
    def policy_fn(observation, desc):
        rt.reset_chunk()
        with torch.inference_mode():
            out = get_action(
                eval_cfg, model, dataset_stats, observation, desc,
                seed=seed,
                randomize_seed=False,
                num_denoising_steps_action=sampling_steps,
                generate_future_state_and_value_in_parallel=True,
            )
        return out["actions"]

    def rollout(env, ep_idx, init_state, task_desc, *, steered: bool):
        env.reset()
        # Apply gripper xyz perturbation. set_init_state internally:
        #   1. env.set_init_state(init_state)
        #   2. pre_shift_settle_steps dummy actions (scene settle)
        #   3. shift loop (OSC P-controller to xyz_delta target)
        #   4. post_shift_pause_steps zero-delta actions (hold)
        obs = stress_test.set_init_state(env, init_state,
                                          episode_idx=ep_idx)
        stress_test.apply_to_env(env, episode_idx=ep_idx)

        # Extra num_steps_wait dummy steps to match collect_policy_inputs_*
        # convention (the policy sees a fully settled scene at its first
        # inference).
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(
                get_libero_dummy_action(eval_cfg.model_family)
            )

        rt.steering_enabled = bool(steered)
        rt.reset_chunk()
        rt.u_norm_log.clear()

        queue = deque(maxlen=eval_cfg.num_open_loop_steps)
        frames = [obs["agentview_image"].copy()]
        chunk_idx = 0
        success = False
        t = 0
        while t < max_env_steps:
            if not queue:
                observation = prepare_observation(
                    obs, resize_size=COSMOS_IMAGE_SIZE,
                    flip_images=eval_cfg.flip_images,
                )
                if steered:
                    _swap_K_for_chunk(chunk_idx)
                    if (chunk_idx == 0 or chunk_idx == args.max_chunks - 1
                            or chunk_idx % 5 == 0):
                        r_now = r_scale_schedule[
                            min(chunk_idx, args.max_chunks - 1)
                        ]
                        saturated = (" (saturated)"
                                     if chunk_idx >= args.max_chunks - 1
                                     else "")
                        print(f"    chunk {chunk_idx:3d}: "
                              f"R_SCALE={r_now:.2e}{saturated}", flush=True)
                actions = policy_fn(observation, task_desc)
                for a in actions[:eval_cfg.num_open_loop_steps]:
                    queue.append(np.asarray(a, dtype=np.float32))
                chunk_idx += 1
            a = queue.popleft()
            obs, _, done, _ = env.step(a.tolist())
            frames.append(obs["agentview_image"].copy())
            if done:
                success = True
                break
            t += 1
        return success, t + args.num_steps_wait, frames, chunk_idx

    results = []
    t_total = time.time()

    # Per-task subdir layout: <out_dir>/task{T:02d}/...  baseline videos under
    # <out_dir>/task{T:02d}/baseline/.
    def _task_dir(tid):
        d = out_dir / f"task{tid:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _do_run(env, tid, ep, prompt_for_task, task_dir, label):
        steered = (label == "steered")
        suffix = "" if steered else "__baseline"
        t0 = time.time()
        success, env_steps, frames, n_chunks = rollout(
            env, ep, task_meta[tid]["init_states"][ep],
            prompt_for_task, steered=steered,
        )
        dt = time.time() - t0
        tag = "SUCCESS" if success else "FAILURE"
        video_path = None
        if args.save_video:
            video_dir = task_dir if steered else (task_dir / "baseline")
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"ep{ep:02d}--{tag}{suffix}.mp4"
            save_video(frames, video_path, fps=args.video_fps)
        sample = dict(getattr(stress_test, "_last_sample", {})) or None
        xyz_str = ("[" + ",".join(f"{v*1000:+5.1f}"
                                   for v in (sample or {}).get("xyz_delta_m",
                                                                [0, 0, 0]))
                   + "]" if sample else "[--]")
        print(f"[task{tid:>2} ep{ep:2d}] {label:8s} {tag:7s}  "
              f"steps={env_steps:4d}  chunks={n_chunks:3d}  "
              f"dxyz_mm={xyz_str}  {dt:6.1f}s  "
              f"-> {video_path.name if video_path else '(no video)'}",
              flush=True)
        return {
            "success":      bool(success),
            "env_steps":    int(env_steps),
            "n_chunks":     int(n_chunks),
            "wall_time_s":  float(dt),
            "video_path":   str(video_path) if video_path else None,
            "perturbation": sample,
        }

    # Group this rank's jobs by task so we build each env once. With
    # world_size == len(task_ids) and a block partition, each rank has exactly
    # one task in this loop.
    from itertools import groupby
    for tid, group in groupby(my_jobs, key=lambda j: j[0]):
        episodes_for_task = [e for _, e in group]
        task = task_meta[tid]["task"]
        libero_prompt = task_meta[tid]["libero_prompt"]
        _set_active_task(tid)
        env, task_desc_libero = get_libero_env(task, "cosmos",
                                                resolution=args.resolution,
                                                horizon=env_horizon)
        prompt_for_task = (args.prompt if args.prompt is not None
                           else task_desc_libero)
        print(f"[env] {args.suite} task{tid:02d}  "
              f"task_desc={task_desc_libero!r}  "
              f"prompt={prompt_for_task!r}  max_steps={max_env_steps} "
              f"(suite TASK_MAX_STEPS={suite_max_steps}; "
              f"env horizon={env_horizon})", flush=True)
        if args.prompt is not None and args.prompt != task_desc_libero:
            print(f"      note: --prompt differs from LIBERO task description; "
                  f"success is tracked against the LIBERO predicate.",
                  flush=True)
        task_dir = _task_dir(tid)
        try:
            for ep in episodes_for_task:
                if args.baseline_only:
                    baseline_rec = _do_run(env, tid, ep, prompt_for_task,
                                            task_dir, "baseline")
                    results.append({
                        "task_id":  int(tid),
                        "episode":  int(ep),
                        "rank":     rank,
                        **baseline_rec,
                        "baseline": None,
                    })
                else:
                    steered_rec = _do_run(env, tid, ep, prompt_for_task,
                                           task_dir, "steered")
                    baseline_rec = (
                        _do_run(env, tid, ep, prompt_for_task,
                                 task_dir, "baseline")
                        if args.run_baseline else None
                    )
                    results.append({
                        "task_id":  int(tid),
                        "episode":  int(ep),
                        "rank":     rank,
                        **steered_rec,
                        "baseline": baseline_rec,
                    })
        finally:
            env.close()

    for h in handles:
        h.remove()

    # ---------------------------------------------------------------- Save
    results_path = out_dir / f"results_rank{rank}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[rank {rank}] wrote {results_path}", flush=True)

    # Only rank 0 writes the manifest (config-only; final totals patched
    # in by the merge phase).
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
                "r_scale_schedule": [float(r_) for r_ in r_scale_schedule],
                "QF_SCALE":        float(args.qf_scale),
                "have_B_tilde":    bool(have_B),
            },
            "perturbation": {
                "preset": args.preset,
                "base_seed": int(args.base_seed),
                "stress_test": stress_test.manifest(),
                "applied_to": ("every episode start; per-episode RNG = "
                               "SeedSequence([base_seed, episode_idx])"),
            },
            "rollout": {
                "suite": args.suite,
                "task_ids": [int(t) for t in args.task_ids],
                "task_descs_libero": {str(tid): task_meta[tid]["libero_prompt"]
                                       for tid in args.task_ids},
                "policy_prompt_override": args.prompt,
                "n_episodes_per_task": int(args.n_episodes_per_task),
                "n_rollouts_total": int(len(jobs)),
                "max_env_steps": int(max_env_steps),
                "suite_default_max_steps": int(suite_max_steps),
                "run_baseline": bool(args.run_baseline),
                "baseline_only": bool(args.baseline_only),
                "resolution": int(args.resolution),
                "video_fps": int(args.video_fps),
                "seed": int(seed),
                "num_steps_wait": int(args.num_steps_wait),
            },
            "parallel": {
                "world_size": world_size,
                "ranks_emit": True,
                "assignment": ("contiguous block of (task_id, episode_idx) "
                               "pairs; jobs[rank*n//W : (rank+1)*n//W]"),
            },
            "svd": {
                "svd_dir": str(svd_dir),
                "jac_mode": ("per_task"
                              if args.jac_dir_act_per_task is not None
                              else "single"),
                "jac_dir_act": args.jac_dir_act,
                "jac_dir_act_per_task_file": (
                    str(args.jac_dir_act_per_task)
                    if args.jac_dir_act_per_task is not None else None
                ),
                "jac_dir_act_per_task": {str(t): per_task_jac_map[t]
                                          for t in sorted(per_task_jac_map)},
                "jac_prompt_per_task": {
                    str(t): per_task_jac_prompt.get(t, "<unknown>")
                    for t in sorted(per_task_jac_map)
                },
                "jac_prompt": jac_prompt,
                "selected_timesteps": sel_t,
                "sampling_steps": sampling_steps,
                "L": L,
                "k_target": r,
                "partitions": partitions,
            },
            "model": {
                "ckpt_path": ckpt_path,
                "config_name": config_name,
                "L_blocks": int(n_blocks),
            },
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        print(f"[rank 0] wrote manifest -> {out_dir / 'manifest.json'}",
              flush=True)

    n_succ = sum(r["success"] for r in results)
    print(f"[rank {rank}] DONE: {n_succ}/{len(results)} steered succeeded; "
          f"wall {time.time() - t_total:.1f}s")
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
