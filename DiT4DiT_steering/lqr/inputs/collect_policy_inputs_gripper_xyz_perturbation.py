#!/usr/bin/env python
"""Collect DiT4DiT positive/negative policy inputs by running N gripper-xyz-perturbed
rollouts of libero_10 and bucketing every captured inference by rollout outcome.

Analogue of the Cosmos-Policy version at:
  the repo root's notebooks/lqr/inputs/collect_policy_inputs_gripper_xyz_perturbation.py
Adapted for DiT4DiT's action-generation model (FlowmatchingActionHead / action DiT).

Scene / prompt / objects are unchanged — the only stress is a
RandomGripperXYZPerturbation (default preset: xyz_random_xlarge_3, sigma=10 cm
isotropic, per-episode Gaussian via SeedSequence([base_seed, episode_idx])).

For each of --n-episodes init_states:
  1. Perturb the gripper's initial Cartesian (x, y, z) per the preset.
  2. Run a DiT4DiT rollout under the original task prompt.
  3. Capture every observation dict the policy consumed.
  4. Label all captured rows by rollout outcome.

Successful rollouts -> positive.npz; failed rollouts -> negative.npz.

NOTE: this stage produces UNPAIRED npzs (positive and negative have different
row counts; rows are NOT 1-to-1 matched at identical states). Run
notebooks/lqr/svd/pair_inputs_by_similarity.sh on the resulting directory to
produce a paired, SVD-compatible version.

Output layout:
    <OUT_DIR>/
      positive.npz       # rows from all successful rollouts
      negative.npz       # rows from all failed rollouts
      manifest.json

NPZ schema:
    primary_images : (N, 256, 256, 3) uint8   (raw agentview, NOT resized, NOT concatenated)
    wrist_images   : (N, 256, 256, 3) uint8   (raw wrist view, NOT resized)
    proprios       : (N, 8) float32            (eef_pos[3] + axisangle(eef_quat)[3] + gripper_qpos[2])
    episode_idx    : (N,) int32
    inference_idx  : (N,) int32
    drive_source   : (N,) int32                (fixed to 0)
    success        : (N,) int32
    xyz_delta_m    : (N, 3) float32
    achieved_xyz_delta_m : (N, 3) float32

Images are stored RAW (flipped 180 deg to match training, but NOT resized/concatenated).
Preprocessing (resize to 224, concat primary+wrist widthwise, sin/cos state encoding)
happens at SVD / Jacobian / LQR time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# -----------------------------------------------------------------------
# Environment setup — must run BEFORE any DiT4DiT import.
# -----------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DIT4DIT_ROOT = _HERE.parent.parent.parent  # notebooks/lqr/inputs -> DiT4DiT root
if str(_DIT4DIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIT4DIT_ROOT))

LIBERO_HOME = os.environ.get("LIBERO_HOME", "/work/nvme/bhhv/jskifstad/LIBERO")
if LIBERO_HOME not in sys.path:
    sys.path.insert(0, LIBERO_HOME)

# Append FastWAM site-packages at the END so robosuite is found but
# dit4dit's own transformers/diffusers take priority over FastWAM's.
_FASTWAM_SITE = "/projects/bhhv/jskifstad/FastWAM/.conda/envs/fastwam/lib/python3.10/site-packages"
if _FASTWAM_SITE not in sys.path:
    sys.path.append(_FASTWAM_SITE)

os.environ.setdefault("LIBERO_HOME", LIBERO_HOME)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(LIBERO_HOME, "libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


# -----------------------------------------------------------------------
# Gripper XYZ perturbation (copied from the repo root's original pipeline)
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

        pause_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(self.gripper_action)]
        for _ in range(self.post_shift_pause_steps):
            final_obs, _, _, _ = env.step(pause_action)

        self._last_sample = {
            "episode_idx": int(episode_idx),
            "xyz_delta_m":       [float(v) for v in dxyz],
            "baseline_ee_pos":   [float(v) for v in baseline_ee],
            "target_ee_pos":     [float(v) for v in target_ee],
            "achieved_ee_pos":   [float(a - b) for a, b in zip(achieved_ee, baseline_ee)],
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
            "shift": {"max_steps": self.shift_max_steps, "tol_m": self.shift_tol_m,
                      "output_max_m": self.output_max_m},
        }


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
        return {
            "kind": "RandomGripperXYZPerturbation", "slug": self.slug,
            "sampling": "per-episode Gaussian, seed=SeedSequence([base_seed, episode_idx])",
            "sigma_xyz_m": [float(v) for v in sig],
            "base_seed": int(self.base_seed),
            "gripper_action": float(self.gripper_action),
            "pre_shift_settle_steps": int(self.pre_shift_settle_steps),
            "post_shift_pause_steps": int(self.post_shift_pause_steps),
            "shift": {"max_steps": self.shift_max_steps, "tol_m": self.shift_tol_m,
                      "output_max_m": self.output_max_m},
        }


def build_presets(base_seed: int):
    return {
        "xyz_random_small":    RandomGripperXYZPerturbation(sigma_xyz_m=0.01, base_seed=base_seed, name_hint="xyz_random_small"),
        "xyz_random_medium":   RandomGripperXYZPerturbation(sigma_xyz_m=0.02, base_seed=base_seed, name_hint="xyz_random_medium"),
        "xyz_random_large":    RandomGripperXYZPerturbation(sigma_xyz_m=0.04, base_seed=base_seed, name_hint="xyz_random_large"),
        "xyz_random_xlarge":   RandomGripperXYZPerturbation(sigma_xyz_m=0.06, base_seed=base_seed, name_hint="xyz_random_xlarge"),
        "xyz_random_xlarge_2": RandomGripperXYZPerturbation(sigma_xyz_m=0.15, base_seed=base_seed, name_hint="xyz_random_xlarge_2"),
        "xyz_random_xlarge_3": RandomGripperXYZPerturbation(sigma_xyz_m=0.10, base_seed=base_seed, name_hint="xyz_random_xlarge_3"),
        "xyz_random_horizontal": RandomGripperXYZPerturbation(sigma_xyz_m=(0.03, 0.03, 0.005), base_seed=base_seed, name_hint="xyz_random_horizontal"),
        "xyz_random":          RandomGripperXYZPerturbation(sigma_xyz_m=(0.1, 0.1, 0.1), base_seed=base_seed, name_hint="xyz_random"),
        "xyz_+x_2cm": GripperXYZPerturbation(xyz_delta=( 0.02,  0.00,  0.00), name_hint="xyz_+x_2cm"),
        "xyz_-x_2cm": GripperXYZPerturbation(xyz_delta=(-0.02,  0.00,  0.00), name_hint="xyz_-x_2cm"),
        "xyz_+y_2cm": GripperXYZPerturbation(xyz_delta=( 0.00,  0.02,  0.00), name_hint="xyz_+y_2cm"),
        "xyz_-y_2cm": GripperXYZPerturbation(xyz_delta=( 0.00, -0.02,  0.00), name_hint="xyz_-y_2cm"),
        "xyz_+z_2cm": GripperXYZPerturbation(xyz_delta=( 0.00,  0.00,  0.02), name_hint="xyz_+z_2cm"),
        "xyz_-z_2cm": GripperXYZPerturbation(xyz_delta=( 0.00,  0.00, -0.02), name_hint="xyz_-z_2cm"),
    }


# -----------------------------------------------------------------------
# Observation preprocessing for DiT4DiT
# -----------------------------------------------------------------------

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Robosuite quat (x,y,z,w) -> axis-angle (3-d)."""
    q = np.asarray(quat, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def prepare_dit4dit_observation(obs: dict, image_size: int = 224):
    """Convert a raw LIBERO env obs dict to a DiT4DiT example dict.

    Returns:
        example:  dict with keys 'image' (list[np.ndarray]), 'lang', 'state'
        primary_raw:  (256, 256, 3) uint8 — flipped, NOT resized
        wrist_raw:    (256, 256, 3) uint8 — flipped, NOT resized
        proprio_raw:  (8,) float32 — raw state before sin/cos encoding
    """
    # Flip 180 deg to match training preprocessing
    primary_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])        # (H, W, 3)
    wrist_raw   = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    # Resize to 224×224 and concat width-wise for the model
    primary_resized = cv2.resize(primary_raw, (image_size, image_size), interpolation=cv2.INTER_AREA)
    wrist_resized   = cv2.resize(wrist_raw,   (image_size, image_size), interpolation=cv2.INTER_AREA)
    concat_img = np.concatenate([primary_resized, wrist_resized], axis=1)  # (224, 448, 3)

    # Raw state vector: eef_pos(3) + axisangle(eef_quat)(3) + gripper_qpos(2) = 8
    proprio_raw = np.concatenate([
        obs["robot0_eef_pos"].astype(np.float32),
        _quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
        obs["robot0_gripper_qpos"].astype(np.float32),
    ])  # (8,)

    # Sin/cos encode: each dim d -> [sin(d), cos(d)], concatenated = 16-dim
    sin_s = np.sin(proprio_raw[None])   # (1, 8)
    cos_s = np.cos(proprio_raw[None])   # (1, 8)
    state_enc = np.stack([sin_s, cos_s], axis=-1).reshape(1, -1).astype(np.float32)  # (1, 16)

    example = {
        "image": [concat_img],  # list with single frame for build_cosmos_inputs
        "lang":  None,          # set by caller
        "state": state_enc,     # (1, 16)
    }
    return example, primary_raw, wrist_raw, proprio_raw


def binarize_gripper(open_val: float) -> float:
    """Binarize gripper: >0.5 -> open(1), else close(-1) in action space."""
    return 1.0 - 2.0 * float(open_val > 0.5)


def unnormalize_actions(normalized_actions: np.ndarray,
                         action_norm_stats: dict) -> np.ndarray:
    """Inverse of the min-max normalization applied at training time."""
    mask = action_norm_stats.get("mask", np.ones(normalized_actions.shape[-1], dtype=bool))
    action_high = np.array(action_norm_stats["max"], dtype=np.float32)
    action_low  = np.array(action_norm_stats["min"], dtype=np.float32)
    norm = np.clip(normalized_actions, -1.0, 1.0)
    return np.where(
        mask,
        0.5 * (norm + 1.0) * (action_high - action_low) + action_low,
        norm,
    )


# -----------------------------------------------------------------------
# DiT4DiT model helper
# -----------------------------------------------------------------------

def load_dit4dit_model(ckpt_path: str, device):
    import DiT4DiT.model.framework.DiT4DiT  # register framework before from_pretrained
    from DiT4DiT.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(ckpt_path)
    model = model.to(device).eval()
    return model


def load_norm_stats(ckpt_path: str) -> dict:
    """Load dataset_statistics.json from the checkpoint directory."""
    from DiT4DiT.model.framework.share_tools import read_mode_config
    _, norm_stats = read_mode_config(ckpt_path)
    return norm_stats


def get_unnorm_key(norm_stats: dict, unnorm_key: Optional[str] = None) -> str:
    if unnorm_key is None:
        assert len(norm_stats) == 1
        unnorm_key = next(iter(norm_stats))
    return unnorm_key


def predict_and_unnormalize(model, example: dict, norm_stats: dict,
                             unnorm_key: str, device, num_open_loop_steps: int = 8):
    """Run DiT4DiT inference and return unnormalized actions.

    Returns: list of (7,) float32 action arrays [world_vector(3) + rotation_delta(3) + gripper(1)].
    """
    example_with_device = dict(example)
    with torch.inference_mode():
        result = model.predict_action([example_with_device])
    norm_actions = result["normalized_actions"][0]  # (action_horizon, action_dim)

    action_stats = norm_stats[unnorm_key]["action"]
    # Unnormalize first 7 dims (8th is padding / unused in libero)
    raw = unnormalize_actions(norm_actions[:, :7], action_stats)  # (H, 7)

    # Binarize gripper (dim 6)
    raw[:, 6] = np.array([binarize_gripper(v) for v in norm_actions[:, 6]])
    return [raw[i] for i in range(min(num_open_loop_steps, len(raw)))]


# -----------------------------------------------------------------------
# Rollout that captures inputs at every inference call
# -----------------------------------------------------------------------

def rollout_collect_inputs(env, init_state, prompt, stress_test,
                            model, norm_stats, unnorm_key, device,
                            max_env_steps, num_open_loop_steps=8,
                            *, episode_idx=0, num_steps_wait=10):
    env.reset()
    obs = stress_test.set_init_state(env, init_state, episode_idx=episode_idx)
    stress_test.apply_to_env(env, episode_idx=episode_idx)

    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    queue = deque(maxlen=num_open_loop_steps)
    inputs = []
    success = False
    t = 0
    while t < max_env_steps:
        if not queue:
            example, primary_raw, wrist_raw, proprio_raw = prepare_dit4dit_observation(obs)
            example["lang"] = prompt
            inputs.append({
                "primary_image": primary_raw.copy(),
                "wrist_image":   wrist_raw.copy(),
                "proprio":       proprio_raw.copy(),
            })
            actions = predict_and_unnormalize(model, example, norm_stats, unnorm_key,
                                              device, num_open_loop_steps)
            for a in actions:
                queue.append(a)
        a = queue.popleft()
        obs, _, done, _ = env.step(a.tolist())
        if done:
            success = True
            break
        t += 1
    return success, t + num_steps_wait, inputs


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--resolution", type=int, default=256,
                    help="LIBERO environment rendering resolution.")
    ap.add_argument("--preset", type=str, default="xyz_random_xlarge_3")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--ckpt-path", type=str,
                    default="/projects/bhhv/jskifstad/DiT4DiT/checkpoint/dit4dit-model/"
                            "dit4dit_libero/final_model/pytorch_model.pt")
    ap.add_argument("--num-open-loop-steps", type=int, default=8,
                    help="How many actions to execute from each DiT4DiT chunk.")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir: {out_dir}")

    presets = build_presets(args.base_seed)
    if args.preset not in presets:
        raise ValueError(f"unknown --preset={args.preset!r}; choices: {sorted(presets)}")
    stress_test = presets[args.preset]
    _log(f"preset: {args.preset}  ({type(stress_test).__name__})")

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from pathlib import Path as _Path
    from libero.libero import get_libero_path

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    n_avail = int(init_states.shape[0])
    n_episodes = min(args.n_episodes, n_avail)
    if n_episodes < args.n_episodes:
        _log(f"WARNING: capping n_episodes {args.n_episodes} -> {n_episodes}")

    # LIBERO max steps per suite
    suite_max_steps = {
        "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
        "libero_10": 520, "libero_90": 400,
    }.get(args.suite, 520)

    task_bddl = (
        _Path(get_libero_path("bddl_files"))
        / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(42)

    prompt = args.prompt if args.prompt is not None else task.language
    _log(f"prompt: {prompt!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"loading DiT4DiT from {args.ckpt_path} on {device} ...")
    model = load_dit4dit_model(args.ckpt_path, device)
    norm_stats = load_norm_stats(args.ckpt_path)
    unnorm_key = get_unnorm_key(norm_stats)
    _log(f"model ready; unnorm_key={unnorm_key!r}")

    # ---- accumulators ----
    all_primary, all_wrist, all_proprio = [], [], []
    all_episode_idx, all_inference_idx = [], []
    all_success = []
    all_xyz_delta, all_achieved_xyz_delta = [], []
    rollout_summaries = []

    _log("=== run loop starting ===")
    for ep in range(n_episodes):
        t0 = time.time()
        try:
            success, env_steps, inputs = rollout_collect_inputs(
                env, init_states[ep], prompt, stress_test,
                model, norm_stats, unnorm_key, device,
                max_env_steps=suite_max_steps,
                num_open_loop_steps=args.num_open_loop_steps,
                episode_idx=ep,
            )
        except Exception as e:
            _log(f"ep{ep:02d} CRASHED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue
        dt = time.time() - t0
        sample = dict(getattr(stress_test, "_last_sample", {}) or {})
        xyz_d = sample.get("xyz_delta_m") or [0.0, 0.0, 0.0]
        ach_d = sample.get("achieved_ee_pos") or [0.0, 0.0, 0.0]

        for inf_idx, rec in enumerate(inputs):
            all_primary.append(rec["primary_image"])
            all_wrist.append(rec["wrist_image"])
            all_proprio.append(rec["proprio"])
            all_episode_idx.append(ep)
            all_inference_idx.append(inf_idx)
            all_success.append(int(success))
            all_xyz_delta.append(xyz_d)
            all_achieved_xyz_delta.append(ach_d)

        tag = "SUCCESS" if success else "FAILURE"
        xyz_str = ",".join(f"{v*1000:+5.1f}" for v in xyz_d)
        _log(f"ep{ep:02d} {tag} steps={env_steps:4d} inf={len(inputs):3d} "
             f"{dt:6.1f}s  dxyz_mm=[{xyz_str}]")
        rollout_summaries.append({
            "episode": ep, "success": bool(success),
            "env_steps": int(env_steps), "n_inferences": len(inputs),
            "wall_time_s": float(dt), "sample": sample or None,
        })
        torch.cuda.empty_cache()

    env.close()

    if not all_episode_idx:
        raise RuntimeError("no rollouts produced any inputs")

    primary_arr = np.stack(all_primary, axis=0)
    wrist_arr   = np.stack(all_wrist,   axis=0)
    proprio_arr = np.stack(all_proprio, axis=0)
    episode_arr   = np.asarray(all_episode_idx,   dtype=np.int32)
    inference_arr = np.asarray(all_inference_idx, dtype=np.int32)
    success_arr   = np.asarray(all_success,       dtype=np.int32)
    xyz_arr     = np.asarray(all_xyz_delta,        dtype=np.float32)
    ach_xyz_arr = np.asarray(all_achieved_xyz_delta, dtype=np.float32)

    pos_mask = success_arr == 1
    neg_mask = success_arr == 0
    n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())
    _log(f"total inferences: {primary_arr.shape[0]}  pos={n_pos}  neg={n_neg}")

    POSITIVE_NPZ  = out_dir / "positive.npz"
    NEGATIVE_NPZ  = out_dir / "negative.npz"
    MANIFEST_JSON = out_dir / "manifest.json"

    def _save(out_npz, mask):
        np.savez_compressed(
            out_npz,
            primary_images=primary_arr[mask],
            wrist_images=wrist_arr[mask],
            proprios=proprio_arr[mask],
            episode_idx=episode_arr[mask],
            inference_idx=inference_arr[mask],
            drive_source=np.zeros(int(mask.sum()), dtype=np.int32),
            success=success_arr[mask],
            xyz_delta_m=xyz_arr[mask],
            achieved_xyz_delta_m=ach_xyz_arr[mask],
        )
        _log(f"wrote {out_npz.name}  ({out_npz.stat().st_size/1e6:.1f} MB)  rows={int(mask.sum())}")

    _save(POSITIVE_NPZ, pos_mask)
    _save(NEGATIVE_NPZ, neg_mask)

    n_succ = sum(r["success"] for r in rollout_summaries)
    manifest = {
        "model": "DiT4DiT", "ckpt_path": args.ckpt_path,
        "suite": args.suite, "task_id": args.task_id, "n_episodes": n_episodes,
        "resolution": args.resolution, "prompt": prompt,
        "preset": args.preset, "base_seed": args.base_seed,
        "stress_test": stress_test.manifest(),
        "pairing": (
            "UNPAIRED: positive.npz has rows from SUCCESSFUL rollouts, "
            "negative.npz has rows from FAILED rollouts. "
            "Run notebooks/lqr/svd/pair_inputs_by_similarity.sh to produce "
            "a paired, SVD-compatible version."
        ),
        "image_layout": "HWC uint8, flipped 180 deg, NOT resized or concatenated",
        "proprio_layout": "eef_pos[3] + axisangle(eef_quat)[3] + gripper_qpos[2] = 8-dim float32",
        "sets": {
            "positive": {"out_npz": str(POSITIVE_NPZ), "role": "rows from successful rollouts"},
            "negative": {"out_npz": str(NEGATIVE_NPZ), "role": "rows from failed rollouts"},
        },
        "totals": {
            "rollouts": len(rollout_summaries), "rollout_successes": n_succ,
            "rollout_failures": len(rollout_summaries) - n_succ,
            "positive_rows": n_pos, "negative_rows": n_neg,
        },
        "rollouts": rollout_summaries,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    _log(f"wrote {MANIFEST_JSON.name}")
    _log(f"=== done: {n_succ}/{len(rollout_summaries)} success ===")


if __name__ == "__main__":
    main()
