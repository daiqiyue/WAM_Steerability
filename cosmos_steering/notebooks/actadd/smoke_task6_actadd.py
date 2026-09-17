#!/usr/bin/env python
"""One-observation Cosmos task-6 activation-space ActAdd smoke test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_NOTEBOOKS_ROOT = _HERE.parent
if str(_NOTEBOOKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_ROOT))
from _setup import setup_env  # noqa: E402

setup_env()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from actadd.run_actadd_cosmos_policy import install_actadd_hooks  # noqa: E402
from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    COSMOS_IMAGE_SIZE,
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (  # noqa: E402
    PolicyEvalConfig,
    prepare_observation,
)
from libero.libero import benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()
    prompt = ("put the white mug on the plate and put the chocolate pudding "
              "to the right of the plate")
    cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=args.ckpt_path,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{args.ckpt_path}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{args.ckpt_path}/libero_t5_embeddings.pkl",
        use_wrist_image=True, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True, chunk_size=16, num_open_loop_steps=16,
        trained_with_image_aug=True, use_jpeg_compression=True,
        flip_images=True, num_denoising_steps_action=1,
        num_denoising_steps_future_state=1, num_denoising_steps_value=1,
        task_suite_name="libero_10", suite="libero",
    )
    stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    model, _ = get_model(cfg)

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(6)
    env, task_desc = get_libero_env(task, "cosmos", resolution=256, horizon=64)
    obs = env.set_init_state(suite.get_task_init_states(6)[0])
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
    prepared = prepare_observation(obs, resize_size=COSMOS_IMAGE_SIZE,
                                   flip_images=True)
    rng = np.random.default_rng(99)
    for key in ("primary_image", "wrist_image"):
        noise = rng.normal(0.0, 90.0, size=prepared[key].shape)
        prepared[key] = np.clip(prepared[key].astype(np.float32) + noise,
                                0, 255).astype(np.uint8)

    activation_shape = []
    handle = model.net.blocks[0].register_forward_hook(
        lambda _module, _inputs, output: activation_shape.append(tuple(output.shape))
    )
    baseline = get_action(
        cfg, model, stats, prepared, prompt, seed=1, randomize_seed=False,
        num_denoising_steps_action=1,
        generate_future_state_and_value_in_parallel=True,
    )["actions"]
    handle.remove()
    if not activation_shape:
        raise RuntimeError("block-0 activation hook did not fire")
    hidden_dim = int(activation_shape[0][-1])
    unit = torch.ones(hidden_dim, dtype=torch.float32) / hidden_dim ** 0.5
    handles = install_actadd_hooks(model, {0: unit}, 0.01, [True])
    try:
        steered = get_action(
            cfg, model, stats, prepared, prompt, seed=1, randomize_seed=False,
            num_denoising_steps_action=1,
            generate_future_state_and_value_in_parallel=True,
        )["actions"]
    finally:
        for item in handles:
            item.remove()
        env.close()
    baseline = np.asarray(baseline, dtype=np.float32)
    steered = np.asarray(steered, dtype=np.float32)
    result = {
        "task_id": 6,
        "task_description": task_desc,
        "gaussian_sigma_uint8": 90.0,
        "activation_shape_block0": list(activation_shape[0]),
        "native_hook": "install_actadd_hooks",
        "hook_alpha": 0.01,
        "action_shape": list(baseline.shape),
        "action_rms_change": float(np.sqrt(np.mean((steered - baseline) ** 2))),
        "action_max_abs_change": float(np.max(np.abs(steered - baseline))),
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
