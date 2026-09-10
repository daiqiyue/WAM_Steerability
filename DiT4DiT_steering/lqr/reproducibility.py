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


def compose_noisy_rollout_frame(
    clean_agentview: np.ndarray,
    noisy_model_image: np.ndarray | None,
    inference_idx: int,
    noise_sigma: float,
) -> np.ndarray:
    """Show the clean environment view beside the exact noisy model input.

    ``noisy_model_image`` is the post-noise, post-resize concatenation of the
    agent and wrist cameras that was actually consumed at ``inference_idx``.
    It is deliberately reused for every action executed from that inference
    chunk instead of drawing new visualization-only noise.
    """
    panel_size = 224
    clean = cv2.resize(
        np.ascontiguousarray(np.flipud(clean_agentview)),
        (panel_size, panel_size),
        interpolation=cv2.INTER_AREA,
    )
    clean = np.clip(clean, 0, 255).astype(np.uint8)

    if noisy_model_image is None:
        model = np.zeros((panel_size, panel_size * 2, 3), dtype=np.uint8)
    else:
        model = np.clip(noisy_model_image, 0, 255).astype(np.uint8)
        if model.ndim != 3 or model.shape[2] != 3 or model.shape[1] % 2:
            raise ValueError(f"expected concatenated RGB model image, got {model.shape}")
        midpoint = model.shape[1] // 2
        model = np.concatenate([
            cv2.resize(model[:, :midpoint], (panel_size, panel_size),
                       interpolation=cv2.INTER_AREA),
            cv2.resize(model[:, midpoint:], (panel_size, panel_size),
                       interpolation=cv2.INTER_AREA),
        ], axis=1)

    result = np.concatenate([clean, model], axis=1)
    labels = [
        "environment: clean agent",
        f"model: noisy agent  sigma={noise_sigma:g}",
        f"model: noisy wrist  sigma={noise_sigma:g}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for panel, label in enumerate(labels):
        x0 = panel * panel_size
        cv2.rectangle(result, (x0, panel_size - 27),
                      (x0 + panel_size - 1, panel_size - 1), (0, 0, 0), -1)
        cv2.putText(result, label, (x0 + 6, panel_size - 9), font, 0.39,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if panel:
            cv2.line(result, (x0, 0), (x0, panel_size - 1),
                     (255, 255, 255), 1)
    return annotate_inference(result, inference_idx)
