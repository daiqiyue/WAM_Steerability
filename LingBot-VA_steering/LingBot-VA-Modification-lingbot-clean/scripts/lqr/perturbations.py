import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


AGENTVIEW_KEY = "agentview_image"
WRIST_KEY = "robot0_eye_in_hand_image"


def resolve_sim(env):
    cur, seen = env, set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "sim"):
            return cur.sim
        cur = getattr(cur, "env", None)
    raise RuntimeError("could not locate sim on env")


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"cannot normalize near-zero vector: {v.tolist()}")
    return v / n


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _euler_xyz_to_mat(angles: Tuple[float, float, float]) -> np.ndarray:
    ax, ay, az = angles
    sx, cx = np.sin(ax), np.cos(ax)
    sy, cy = np.sin(ay), np.cos(ay)
    sz, cz = np.sin(az), np.cos(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def project_world_points(points, cam_pos, cam_rot_mat, fovy_deg, image_size):
    p_w = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p_c = (p_w - cam_pos[None, :]) @ cam_rot_mat
    depth = -p_c[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    f = (image_size / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    u = image_size / 2.0 + f * (p_c[:, 0] / safe)
    v = image_size / 2.0 - f * (p_c[:, 1] / safe)
    return u, v, depth


class StressTest:
    slug = "stock"

    def set_init_state(self, env, init_state, episode_idx=0):
        return env.set_init_state(init_state)

    def apply_to_env(self, env, episode_idx=0):
        return None

    def transform_observation(self, obs: Dict[str, Any], episode_idx=0, inference_idx=0):
        return obs

    def manifest(self) -> Dict[str, Any]:
        return {"kind": type(self).__name__, "slug": self.slug}


@dataclass
class ImageGaussianNoise(StressTest):
    sigma: float = 90.0
    apply_to: Tuple[str, ...] = (AGENTVIEW_KEY, WRIST_KEY)
    per_episode_seed: bool = True
    name_hint: str = "noise"
    _rng: Optional[np.random.Generator] = field(default=None, init=False, repr=False)
    _last_episode: int = field(default=-1, init=False, repr=False)

    @property
    def slug(self) -> str:
        return f"{self.name_hint}_s{self.sigma:.1f}"

    def transform_observation(self, obs, episode_idx=0, inference_idx=0):
        del inference_idx
        if self.per_episode_seed and self._last_episode != episode_idx:
            self._rng = np.random.default_rng(seed=int(episode_idx))
            self._last_episode = int(episode_idx)
        elif self._rng is None:
            self._rng = np.random.default_rng()
        out = dict(obs)
        for key in self.apply_to:
            if key not in obs:
                continue
            img = obs[key].astype(np.float32)
            noise = self._rng.normal(0.0, float(self.sigma), size=img.shape).astype(np.float32)
            out[key] = np.clip(img + noise, 0, 255).astype(np.uint8)
        return out

    def manifest(self):
        return {
            "kind": "ImageGaussianNoise",
            "slug": self.slug,
            "sigma": float(self.sigma),
            "apply_to": list(self.apply_to),
            "per_episode_seed": bool(self.per_episode_seed),
        }


@dataclass
class RandomCameraViewPerturbation(StressTest):
    pos_sigma: float = 0.10
    rot_sigma_rad: float = float(np.radians(8.0))
    fov_sigma: float = 5.0
    base_seed: int = 42
    camera_name: str = "agentview"
    name_hint: str = "cam_random_large"
    enforce_visibility: bool = True
    workspace_table_z: float = 0.90
    workspace_visible_fraction: float = 0.55
    visibility_margin_px: int = 8
    image_size: int = 128
    max_rejection_attempts: int = 2000
    _baseline_pos: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _baseline_quat: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _baseline_fov: Optional[float] = field(default=None, init=False, repr=False)
    _workspace_corners: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _resolved_workspace_center: Optional[Tuple[float, float, float]] = field(default=None, init=False, repr=False)
    _resolved_workspace_half_extent: Optional[Tuple[float, float, float]] = field(default=None, init=False, repr=False)
    _last_sample: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _last_rejection_count: int = field(default=0, init=False, repr=False)

    @property
    def slug(self) -> str:
        payload = f"pos{self.pos_sigma:.4f}|rot{self.rot_sigma_rad:.4f}|fov{self.fov_sigma:.3f}|seed{self.base_seed}|vis{int(self.enforce_visibility)}"
        suffix = "_vis" if self.enforce_visibility else ""
        return f"{self.name_hint}{suffix}_seed{self.base_seed}_{hashlib.md5(payload.encode()).hexdigest()[:6]}"

    def _capture_baseline(self, env):
        sim = resolve_sim(env)
        cam_id = sim.model.camera_name2id(self.camera_name)
        self._baseline_pos = sim.model.cam_pos[cam_id].copy()
        self._baseline_quat = sim.model.cam_quat[cam_id].copy()
        self._baseline_fov = float(sim.model.cam_fovy[cam_id])
        if not self.enforce_visibility:
            return
        rot = _quat_wxyz_to_mat(self._baseline_quat)
        view_dir = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        if abs(view_dir[2]) >= 1e-6:
            t = (self.workspace_table_z - self._baseline_pos[2]) / view_dir[2]
            target = self._baseline_pos + t * view_dir
            distance = abs(float(t))
        else:
            target = np.array([0.0, 0.0, self.workspace_table_z], dtype=np.float64)
            distance = max(abs(float(self._baseline_pos[2] - self.workspace_table_z)), 0.5)
        visible_half = distance * np.tan(np.radians(self._baseline_fov) / 2.0)
        xy_half = float(visible_half * self.workspace_visible_fraction)
        center = (float(target[0]), float(target[1]), float(self.workspace_table_z + 0.05))
        half = (xy_half, xy_half, 0.08)
        self._resolved_workspace_center = center
        self._resolved_workspace_half_extent = half
        cx, cy, cz = center
        hx, hy, hz = half
        self._workspace_corners = np.array(
            [[cx + sx * hx, cy + sy * hy, cz + sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=np.float64,
        )

    def _candidate(self, dpos, drot_euler, dfov):
        cam_pos = self._baseline_pos + np.asarray(dpos, dtype=np.float64)
        base_rot = _quat_wxyz_to_mat(self._baseline_quat)
        new_rot = _euler_xyz_to_mat(drot_euler) @ base_rot
        return cam_pos, new_rot, float(self._baseline_fov + dfov)

    def _visibility_ok(self, dpos, drot_euler, dfov):
        if self._workspace_corners is None:
            return True
        cam_pos, rot, fovy = self._candidate(dpos, drot_euler, dfov)
        u, v, depth = project_world_points(self._workspace_corners, cam_pos, rot, fovy, self.image_size)
        m = float(self.visibility_margin_px)
        return bool(np.all(depth > 0.05) and np.all(u >= m) and np.all(u <= self.image_size - m) and np.all(v >= m) and np.all(v <= self.image_size - m))

    def _resolve_pose(self, episode_idx):
        ss = np.random.SeedSequence([int(self.base_seed), int(episode_idx)])
        rng = np.random.default_rng(ss)
        for attempt in range(self.max_rejection_attempts if self.enforce_visibility else 1):
            dpos = tuple(float(v) for v in rng.normal(0.0, self.pos_sigma, size=3))
            drot = tuple(float(v) for v in rng.normal(0.0, self.rot_sigma_rad, size=3))
            dfov = float(rng.normal(0.0, self.fov_sigma))
            if self._visibility_ok(dpos, drot, dfov):
                self._last_rejection_count = attempt
                return dpos, drot, dfov
        self._last_rejection_count = int(self.max_rejection_attempts)
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0

    def apply_to_env(self, env, episode_idx=0):
        if self._baseline_pos is None:
            self._capture_baseline(env)
        dpos, drot, dfov = self._resolve_pose(episode_idx)
        sim = resolve_sim(env)
        cam_id = sim.model.camera_name2id(self.camera_name)
        cam_pos, rot, fovy = self._candidate(dpos, drot, dfov)
        sim.model.cam_pos[cam_id] = cam_pos.astype(sim.model.cam_pos.dtype)
        sim.model.cam_quat[cam_id] = _mat_to_quat_wxyz(rot).astype(sim.model.cam_quat.dtype)
        sim.model.cam_fovy[cam_id] = fovy
        sim.forward()
        self._last_sample = {
            "episode_idx": int(episode_idx),
            "dpos": [float(v) for v in dpos],
            "drot_euler_rad": [float(v) for v in drot],
            "drot_euler_deg": [float(np.degrees(v)) for v in drot],
            "dfov_deg": float(dfov),
            "rejection_count": int(self._last_rejection_count),
        }

    def manifest(self):
        out = {
            "kind": "RandomCameraViewPerturbation",
            "slug": self.slug,
            "camera_name": self.camera_name,
            "sampling": "per-episode Gaussian, seed=SeedSequence([base_seed, episode_idx])",
            "pos_sigma_m": float(self.pos_sigma),
            "rot_sigma_rad": float(self.rot_sigma_rad),
            "rot_sigma_deg": float(np.degrees(self.rot_sigma_rad)),
            "fov_sigma_deg": float(self.fov_sigma),
            "base_seed": int(self.base_seed),
            "enforce_visibility": bool(self.enforce_visibility),
            "workspace_table_z": float(self.workspace_table_z),
            "workspace_visible_fraction": float(self.workspace_visible_fraction),
            "visibility_margin_px": int(self.visibility_margin_px),
            "image_size": int(self.image_size),
            "max_rejection_attempts": int(self.max_rejection_attempts),
        }
        if self._resolved_workspace_center is not None:
            out["resolved_workspace_center"] = [float(v) for v in self._resolved_workspace_center]
        if self._resolved_workspace_half_extent is not None:
            out["resolved_workspace_half_extent"] = [float(v) for v in self._resolved_workspace_half_extent]
        return out


@dataclass
class GripperXYZPerturbation(StressTest):
    xyz_delta: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_action: float = -1.0
    pre_shift_settle_steps: int = 10
    shift_max_steps: int = 30
    shift_tol_m: float = 2e-3
    output_max_m: float = 0.05
    post_shift_pause_steps: int = 10
    name_hint: str = "gripper_xyz"
    _last_sample: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @property
    def slug(self) -> str:
        xyz_str = ",".join(f"{a:+.3f}" for a in self.xyz_delta)
        return f"{self.name_hint}_{hashlib.md5(f'xyz={xyz_str}|g={self.gripper_action:+.1f}'.encode()).hexdigest()[:6]}"

    def _resolve_xyz(self, episode_idx):
        del episode_idx
        return tuple(float(v) for v in self.xyz_delta)

    def _resolve_ee_target(self, sim):
        site_candidates = ("gripper0_grip_site", "robot0_grip_site", "grip_site", "ee_site")
        body_candidates = ("robot0_right_hand", "right_hand", "gripper0_eef", "gripper0_hand")
        site_names = set(sim.model.site_names)
        for name in site_candidates:
            if name in site_names:
                return name, None
        body_names = set(sim.model.body_names)
        for name in body_candidates:
            if name in body_names:
                return None, name
        raise RuntimeError("no EE site/body found")

    def _ee_pos(self, sim, site_name=None, body_name=None):
        if site_name is not None:
            return sim.data.site_xpos[sim.model.site_name2id(site_name)].copy()
        return sim.data.body_xpos[sim.model.body_name2id(body_name)].copy()

    def _shift_gripper_xyz_via_actions(self, env, target_pos, *, site_name=None, body_name=None):
        sim = resolve_sim(env)
        obs = None
        n_used = 0
        achieved = self._ee_pos(sim, site_name=site_name, body_name=body_name)
        for step in range(int(self.shift_max_steps)):
            cur = self._ee_pos(sim, site_name=site_name, body_name=body_name)
            err = target_pos - cur
            if np.linalg.norm(err) < float(self.shift_tol_m):
                break
            action_xyz = np.clip(err / float(self.output_max_m), -1.0, 1.0)
            action = [
                float(action_xyz[0]),
                float(action_xyz[1]),
                float(action_xyz[2]),
                0.0,
                0.0,
                0.0,
                float(self.gripper_action),
            ]
            obs, _, _, _ = env.step(action)
            n_used = step + 1
            achieved = self._ee_pos(sim, site_name=site_name, body_name=body_name)
        residual = float(np.linalg.norm(target_pos - achieved))
        return obs, achieved, residual, n_used

    def set_init_state(self, env, init_state, episode_idx=0):
        sim = resolve_sim(env)
        site_name, body_name = self._resolve_ee_target(sim)
        baseline_obs = env.set_init_state(init_state)
        for _ in range(int(self.pre_shift_settle_steps)):
            baseline_obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(self.gripper_action)])
        sim.forward()
        baseline_ee = self._ee_pos(sim, site_name=site_name, body_name=body_name)
        dxyz = self._resolve_xyz(episode_idx)
        target_ee = baseline_ee + np.asarray(dxyz, dtype=np.float64)
        if np.allclose(dxyz, 0.0):
            achieved_ee = baseline_ee
            residual = 0.0
            n_used = 0
            final_obs = baseline_obs
        else:
            final_obs, achieved_ee, residual, n_used = self._shift_gripper_xyz_via_actions(
                env, target_ee, site_name=site_name, body_name=body_name
            )
        pause_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(self.gripper_action)]
        for _ in range(int(self.post_shift_pause_steps)):
            final_obs, _, _, _ = env.step(pause_action)
        self._last_sample = {
            "episode_idx": int(episode_idx),
            "xyz_delta_m": [float(v) for v in dxyz],
            "baseline_ee_pos": [float(v) for v in baseline_ee],
            "target_ee_pos": [float(v) for v in target_ee],
            "achieved_ee_pos": [float(v) for v in achieved_ee],
            "achieved_xyz_delta_m": [float(a - b) for a, b in zip(achieved_ee, baseline_ee)],
            "shift_residual_m": float(residual),
            "shift_steps_used": int(n_used),
            "gripper_action": float(self.gripper_action),
        }
        return final_obs

    def manifest(self):
        return {
            "kind": "GripperXYZPerturbation",
            "slug": self.slug,
            "xyz_delta_m": [float(v) for v in self.xyz_delta],
            "gripper_action": float(self.gripper_action),
            "pre_shift_settle_steps": int(self.pre_shift_settle_steps),
            "post_shift_pause_steps": int(self.post_shift_pause_steps),
            "shift": {
                "max_steps": int(self.shift_max_steps),
                "tol_m": float(self.shift_tol_m),
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
            return np.array([float(s)] * 3, dtype=np.float64)
        return np.array([float(v) for v in s], dtype=np.float64)

    @property
    def slug(self) -> str:
        sig = self._sigma_vec()
        sig_str = ",".join(f"{v:.4f}" for v in sig)
        return f"{self.name_hint}_seed{self.base_seed}_{hashlib.md5(f'sigma=[{sig_str}]|seed{self.base_seed}'.encode()).hexdigest()[:6]}"

    def _resolve_xyz(self, episode_idx):
        ss = np.random.SeedSequence([int(self.base_seed), int(episode_idx)])
        rng = np.random.default_rng(ss)
        return tuple(float(v) for v in rng.normal(0.0, self._sigma_vec()))

    def manifest(self):
        out = super().manifest()
        sig = self._sigma_vec()
        out.update(
            {
                "kind": "RandomGripperXYZPerturbation",
                "sampling": "per-episode Gaussian, seed=SeedSequence([base_seed, episode_idx])",
                "sigma_xyz_m": [float(v) for v in sig],
                "base_seed": int(self.base_seed),
            }
        )
        return out


def build_gripper_xyz_preset(name: str, base_seed: int = 42) -> GripperXYZPerturbation:
    presets = {
        "xyz_+x_2cm": GripperXYZPerturbation(xyz_delta=(0.02, 0.0, 0.0), name_hint="xyz_+x_2cm"),
        "xyz_-x_2cm": GripperXYZPerturbation(xyz_delta=(-0.02, 0.0, 0.0), name_hint="xyz_-x_2cm"),
        "xyz_+y_2cm": GripperXYZPerturbation(xyz_delta=(0.0, 0.02, 0.0), name_hint="xyz_+y_2cm"),
        "xyz_-y_2cm": GripperXYZPerturbation(xyz_delta=(0.0, -0.02, 0.0), name_hint="xyz_-y_2cm"),
        "xyz_+z_2cm": GripperXYZPerturbation(xyz_delta=(0.0, 0.0, 0.02), name_hint="xyz_+z_2cm"),
        "xyz_-z_2cm": GripperXYZPerturbation(xyz_delta=(0.0, 0.0, -0.02), name_hint="xyz_-z_2cm"),
        "xyz_diag_+1cm": GripperXYZPerturbation(xyz_delta=(0.01, 0.01, 0.01), name_hint="xyz_diag_+1cm"),
        "xyz_diag_+3cm": GripperXYZPerturbation(xyz_delta=(0.03, 0.03, 0.03), name_hint="xyz_diag_+3cm"),
        "xyz_random_small": RandomGripperXYZPerturbation(sigma_xyz_m=0.01, base_seed=base_seed, name_hint="xyz_random_small"),
        "xyz_random_medium": RandomGripperXYZPerturbation(sigma_xyz_m=0.02, base_seed=base_seed, name_hint="xyz_random_medium"),
        "xyz_random_large": RandomGripperXYZPerturbation(sigma_xyz_m=0.04, base_seed=base_seed, name_hint="xyz_random_large"),
        "xyz_random_xlarge": RandomGripperXYZPerturbation(sigma_xyz_m=0.06, base_seed=base_seed, name_hint="xyz_random_xlarge"),
        "xyz_random_xlarge_2": RandomGripperXYZPerturbation(sigma_xyz_m=0.15, base_seed=base_seed, name_hint="xyz_random_xlarge_2"),
        "xyz_random_xlarge_3": RandomGripperXYZPerturbation(sigma_xyz_m=0.10, base_seed=base_seed, name_hint="xyz_random_xlarge_3"),
        "xyz_random_horizontal": RandomGripperXYZPerturbation(
            sigma_xyz_m=(0.03, 0.03, 0.005), base_seed=base_seed, name_hint="xyz_random_horizontal"
        ),
        "xyz_random": RandomGripperXYZPerturbation(
            sigma_xyz_m=(0.10, 0.10, 0.10), base_seed=base_seed, name_hint="xyz_random"
        ),
    }
    if name not in presets:
        raise ValueError(f"unknown gripper xyz preset {name!r}; choose one of {sorted(presets)}")
    return presets[name]


def build_perturbation(spec: Dict[str, Any]) -> StressTest:
    kind = str(spec.get("kind", spec.get("type", ""))).lower()
    if kind in {"", "none", "nominal"}:
        return StressTest()
    if kind in {"gaussian", "image_gaussian_noise", "noise"}:
        return ImageGaussianNoise(
            sigma=float(spec.get("sigma", spec.get("image_noise_sigma", 90.0))),
            per_episode_seed=bool(spec.get("per_episode_seed", True)),
            name_hint=str(spec.get("name", "noise")),
        )
    if kind in {"camera", "camera_view", "random_camera"}:
        return RandomCameraViewPerturbation(
            pos_sigma=float(spec.get("pos_sigma", spec.get("pos_sigma_m", 0.10))),
            rot_sigma_rad=float(spec.get("rot_sigma_rad", np.radians(float(spec.get("rot_sigma_deg", 8.0))))),
            fov_sigma=float(spec.get("fov_sigma", spec.get("fov_sigma_deg", 5.0))),
            base_seed=int(spec.get("base_seed", 42)),
            camera_name=str(spec.get("camera_name", "agentview")),
            enforce_visibility=bool(spec.get("enforce_visibility", True)),
            workspace_table_z=float(spec.get("workspace_table_z", 0.90)),
            workspace_visible_fraction=float(spec.get("workspace_visible_fraction", 0.55)),
            visibility_margin_px=int(spec.get("visibility_margin_px", 8)),
            image_size=int(spec.get("image_size", 128)),
            max_rejection_attempts=int(spec.get("max_rejection_attempts", 2000)),
            name_hint=str(spec.get("name", "cam_random_large")),
        )
    if kind in {"init_position", "gripper_init", "init_pos", "gripper_xyz"}:
        if "preset" in spec:
            return build_gripper_xyz_preset(str(spec["preset"]), base_seed=int(spec.get("base_seed", 42)))
        if "sigma_xyz_m" in spec:
            return RandomGripperXYZPerturbation(
                sigma_xyz_m=spec["sigma_xyz_m"],
                base_seed=int(spec.get("base_seed", 42)),
                name_hint=str(spec.get("name", "xyz_random")),
            )
        return GripperXYZPerturbation(
            xyz_delta=tuple(float(v) for v in spec.get("xyz_delta_m", spec.get("xyz_delta", (0.0, 0.0, 0.0)))),
            name_hint=str(spec.get("name", "gripper_xyz")),
        )
    raise ValueError(f"unknown perturbation kind: {kind!r}")
