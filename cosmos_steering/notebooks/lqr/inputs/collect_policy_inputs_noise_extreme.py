#!/usr/bin/env python
"""Collect Cosmos-Policy contrastive input pairs for libero_10 task 0 under
the `noise_extreme` preset (sigma=90 uint8 pixel units; matches
notebooks/stress_test/rollouts/libero_10__task00__img_noise_extreme/manifest.json).

Same structure as collect_policy_inputs_noise.ipynb but:
  - sigma = 90 (noise_extreme), not 75
  - shardable across GPUs via --episode-ids
  - reusable downstream (positive.npz / negative.npz / manifest.json), so the
    output plugs straight into svd/run_partition_svd_pairs_no_action.py.

Positive inputs: clean rollouts under the original task prompt
                 ("put both the alphabet soup and the tomato sauce in the basket").
Negative inputs: same rows, with i.i.d. Gaussian pixel noise (sigma=90 in
                 uint8 units, clipped to [0, 255]) added to the agentview
                 (primary) and wrist images. Per-row tags
                 (episode_idx, inference_idx, drive_source) preserve the
                 schema used by run_partition_svd_pairs.py.

When --episode-ids is passed, the script only runs the listed episodes;
out_dir is populated with rows tagged by *global* episode_idx so per-shard
NPZs can be concatenated (e.g. via merge_per_task_inputs.py) without
re-indexing.

Output layout:

    <OUT_DIR>/
      positive.npz   # clean renders captured at each inference
      negative.npz   # positive + Gaussian noise on wrist + primary
      manifest.json  # prompt, noise config, per-rollout summaries
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

# Match notebook _setup.py — must run BEFORE any cosmos_policy import.
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


def rollout_collect_inputs(env, init_state, prompt, eval_cfg, model,
                            dataset_stats, max_env_steps, *,
                            num_steps_wait=10):
    env.reset()
    obs = env.set_init_state(init_state)
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
                eval_cfg, model, dataset_stats, observation, prompt,
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="where to write positive.npz / negative.npz / manifest.json")
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=10,
                    help="number of init_states to roll out (capped by suite avail)")
    ap.add_argument("--episode-ids", type=int, nargs="+", default=None,
                    help="explicit list of episode indices to run (subset of "
                         "[0, n_episodes)). If omitted, runs range(n_episodes).")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--prompt", type=str,
                    default="put both the alphabet soup and the tomato sauce in the basket")
    ap.add_argument("--noise-sigma", type=float, default=90.0,
                    help="Gaussian noise sigma in uint8 pixel units (noise_extreme = 90)")
    ap.add_argument("--noise-per-episode-seed", action="store_true", default=True,
                    help="seed RNG per episode with seed=int(episode_idx) (default true)")
    ap.add_argument("--no-noise-per-episode-seed", dest="noise_per_episode_seed",
                    action="store_false")
    ap.add_argument("--ckpt-path", type=str,
                    default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir: {out_dir}")

    # ---------------- libero suite ----------------
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    n_avail = int(init_states.shape[0])
    n_episodes = min(args.n_episodes, n_avail)
    if n_episodes < args.n_episodes:
        _log(f"WARNING: --n-episodes={args.n_episodes} > available "
             f"init_states={n_avail}; using {n_episodes}")

    if args.episode_ids is None:
        episode_ids = list(range(n_episodes))
    else:
        episode_ids = sorted(set(int(e) for e in args.episode_ids))
        bad = [e for e in episode_ids if e < 0 or e >= n_avail]
        if bad:
            raise ValueError(f"--episode-ids out of range [0, {n_avail}): {bad}")

    max_env_steps = TASK_MAX_STEPS[args.suite]
    _log(f"libero suite={args.suite}  task={args.task_id}  "
         f"episode_ids={episode_ids}  max_env_steps={max_env_steps}")

    # ---------------- env + model ----------------
    env, base_task_desc = get_libero_env(task, "cosmos", resolution=args.resolution)
    prompt = args.prompt
    _log(f"prompt: {prompt!r}")
    _log(f"env-provided task_desc: {base_task_desc!r}")

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
    all_episode_idx, all_inference_idx = [], []
    rollout_summaries = []

    _log("=== run loop starting ===")
    for ep in episode_ids:
        t0 = time.time()
        try:
            success, env_steps, inputs = rollout_collect_inputs(
                env, init_states[ep], prompt,
                eval_cfg, model, dataset_stats, max_env_steps,
            )
        except Exception as e:
            _log(f"ep{ep:02d} CRASHED: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0

        for inf_idx, rec in enumerate(inputs):
            all_primary.append(rec["primary_image"])
            all_wrist.append(rec["wrist_image"])
            all_proprio.append(rec["proprio"])
            all_episode_idx.append(ep)
            all_inference_idx.append(inf_idx)

        tag = "SUCCESS" if success else "FAILURE"
        _log(f"ep{ep:02d} {tag} steps={env_steps:4d} inf={len(inputs):3d} {dt:6.1f}s")
        rollout_summaries.append({
            "episode": ep,
            "success": bool(success),
            "env_steps": int(env_steps),
            "n_inferences": len(inputs),
            "wall_time_s": dt,
        })
        _torch.cuda.empty_cache()

    env.close()

    if not all_episode_idx:
        raise RuntimeError("no rollouts produced any inputs")

    # ---------------- stack positives ----------------
    pos_primary = np.stack(all_primary, axis=0)
    pos_wrist   = np.stack(all_wrist,   axis=0)
    proprio_arr = np.stack(all_proprio, axis=0)
    episode_arr   = np.asarray(all_episode_idx,   dtype=np.int32)
    inference_arr = np.asarray(all_inference_idx, dtype=np.int32)
    drive_arr     = np.zeros_like(episode_arr, dtype=np.int32)

    _log(f"total inferences collected: {pos_primary.shape[0]}")
    _log(f"  primary_images: {pos_primary.shape}  dtype={pos_primary.dtype}")
    _log(f"  wrist_images:   {pos_wrist.shape}  dtype={pos_wrist.dtype}")
    _log(f"  proprios:       {proprio_arr.shape}  dtype={proprio_arr.dtype}")

    # ---------------- build negatives (Gaussian noise post-hoc) -------------
    # Per-episode RNG seeded with seed=int(episode_idx) (same recipe the
    # stress test's ImageGaussianNoise uses with per_episode_seed=True).
    # Within an episode, draws are made in inference order: primary noise
    # first, then wrist noise.
    neg_primary = np.empty_like(pos_primary)
    neg_wrist   = np.empty_like(pos_wrist)

    current_ep = None
    rng = None
    for row in range(pos_primary.shape[0]):
        ep = int(episode_arr[row])
        if args.noise_per_episode_seed and ep != current_ep:
            rng = np.random.default_rng(seed=ep)
            current_ep = ep
        elif rng is None:
            rng = np.random.default_rng()

        p_img = pos_primary[row].astype(np.float32)
        p_noise = rng.normal(loc=0.0, scale=args.noise_sigma, size=p_img.shape).astype(np.float32)
        neg_primary[row] = np.clip(p_img + p_noise, 0, 255).astype(np.uint8)

        w_img = pos_wrist[row].astype(np.float32)
        w_noise = rng.normal(loc=0.0, scale=args.noise_sigma, size=w_img.shape).astype(np.float32)
        neg_wrist[row] = np.clip(w_img + w_noise, 0, 255).astype(np.uint8)

    abs_diff_primary = np.abs(neg_primary.astype(np.float32) - pos_primary.astype(np.float32))
    abs_diff_wrist   = np.abs(neg_wrist.astype(np.float32)   - pos_wrist.astype(np.float32))
    _log(f"primary |delta| mean={abs_diff_primary.mean():.2f}  max={abs_diff_primary.max():.2f}")
    _log(f"wrist   |delta| mean={abs_diff_wrist.mean():.2f}  max={abs_diff_wrist.max():.2f}")

    # ---------------- save ----------------
    POSITIVE_NPZ  = out_dir / "positive.npz"
    NEGATIVE_NPZ  = out_dir / "negative.npz"
    MANIFEST_JSON = out_dir / "manifest.json"

    np.savez_compressed(
        POSITIVE_NPZ,
        primary_images=pos_primary,
        wrist_images=pos_wrist,
        proprios=proprio_arr,
        episode_idx=episode_arr,
        inference_idx=inference_arr,
        drive_source=drive_arr,
    )
    _log(f"wrote {POSITIVE_NPZ.name}  ({POSITIVE_NPZ.stat().st_size/1e6:.1f} MB)  "
         f"rows={pos_primary.shape[0]}")

    np.savez_compressed(
        NEGATIVE_NPZ,
        primary_images=neg_primary,
        wrist_images=neg_wrist,
        proprios=proprio_arr,
        episode_idx=episode_arr,
        inference_idx=inference_arr,
        drive_source=drive_arr,
    )
    _log(f"wrote {NEGATIVE_NPZ.name}  ({NEGATIVE_NPZ.stat().st_size/1e6:.1f} MB)  "
         f"rows={neg_primary.shape[0]}")

    manifest = {
        "suite": args.suite,
        "task_id": args.task_id,
        "n_episodes_requested": int(args.n_episodes),
        "episode_ids_run":      [int(e) for e in episode_ids],
        "resolution": args.resolution,
        "prompt": prompt,
        "pairing": (
            "row i in positive.npz and row i in negative.npz share the same "
            "MuJoCo state at capture time. negative is constructed post-hoc "
            "by adding Gaussian noise to positive's primary_image and "
            "wrist_image. drive_source is 0 for all rows (single clean drive "
            "campaign)."
        ),
        "image_layout": "HWC uint8, flip_images=True applied at capture time",
        "proprio_layout": ("concat(robot0_gripper_qpos[2], robot0_eef_pos[3], "
                          "robot0_eef_quat[4]) -> shape (9,) float32"),
        "noise": {
            "kind": "ImageGaussianNoise",
            "sigma": float(args.noise_sigma),
            "apply_to": ["primary_image", "wrist_image"],
            "per_episode_seed": bool(args.noise_per_episode_seed),
            "seed_recipe": ("np.random.default_rng(seed=int(episode_idx)); "
                            "within each episode, draw primary noise then "
                            "wrist noise per inference (in inference order)"),
            "clipped_to": [0, 255],
            "output_dtype": "uint8",
            "matches_rollout_preset": ("noise_extreme "
                "(notebooks/stress_test/rollouts/libero_10__task00__img_noise_extreme/manifest.json)"),
        },
        "drive_sources": [
            {"code": 0, "name": "clean_drive",
             "desc": "env steps cleanly under PROMPT; negative is post-hoc noised"},
        ],
        "sets": {
            "positive": {"out_npz": str(POSITIVE_NPZ),
                          "role": "clean renders at every captured pose"},
            "negative": {"out_npz": str(NEGATIVE_NPZ),
                          "role": "positive + Gaussian noise on both cameras"},
        },
        "totals": {
            "total_inferences": int(pos_primary.shape[0]),
            "rollouts": len(rollout_summaries),
        },
        "rollouts": rollout_summaries,
        "pixel_diff_stats": {
            "primary_abs_diff_mean": float(abs_diff_primary.mean()),
            "primary_abs_diff_max":  float(abs_diff_primary.max()),
            "wrist_abs_diff_mean":   float(abs_diff_wrist.mean()),
            "wrist_abs_diff_max":    float(abs_diff_wrist.max()),
        },
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    _log(f"wrote {MANIFEST_JSON.name}")

    n_succ = sum(r["success"] for r in rollout_summaries)
    _log("=== summary ===")
    _log(f"  rollouts             : {len(rollout_summaries)} "
         f"({n_succ} success / {len(rollout_summaries) - n_succ} failure)")
    _log(f"  total paired rows    : {pos_primary.shape[0]}")
    _log(f"  positive (clean)     : {POSITIVE_NPZ}")
    _log(f"  negative (noised)    : {NEGATIVE_NPZ}")
    _log(f"  manifest             : {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
