"""Small helpers for reproducible, inference-indexed LIBERO rollouts."""

from __future__ import annotations

import copy

import cv2
import numpy as np


def clone_observation(obs: dict) -> dict:
    """Deep-copy the exact observation consumed at a policy inference."""
    return {
        key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
        for key, value in obs.items()
    }


def flattened_sim_state(env) -> np.ndarray:
    """Capture MuJoCo qpos/qvel state through common LIBERO wrappers."""
    cur, seen = env, set()
    for _ in range(10):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        sim = getattr(cur, "sim", None)
        if sim is not None:
            state = sim.get_state()
            if hasattr(state, "flatten"):
                return np.asarray(state.flatten(), dtype=np.float64).copy()
            return np.asarray(state, dtype=np.float64).reshape(-1).copy()
        cur = getattr(cur, "env", None)
    raise RuntimeError("could not find MuJoCo sim through LIBERO env wrappers")


def annotate_inference(frame: np.ndarray, inference_idx: int) -> np.ndarray:
    """Add a high-contrast inference label to a displayed RGB video frame."""
    result = np.ascontiguousarray(frame).copy()
    label = f"inference #{int(inference_idx):03d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.62, 2
    (width, height), baseline = cv2.getTextSize(label, font, scale, thickness)
    x, y = 10, 12 + height
    cv2.rectangle(result, (x - 6, y - height - 6),
                  (x + width + 6, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(result, label, (x, y), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)
    return result
