#!/usr/bin/env python
"""Activation-Addition (ActAdd) rollout of Cosmos-Policy-LIBERO-Predict2-2B
under noise_extreme image-noise conditions (Gaussian pixel noise on the
primary + wrist cameras, σ = --noise-sigma).

Steers the diffusion model by adding alpha * v to the output of each hooked
DiT block on every denoising step:

    output_steered = output + alpha * v_layer

v_layer (shape D, the per-token hidden dim) is loaded from --v-path. The
supported formats are:
  - .pt dict keyed by integer layer index -> 1-D tensor (d,)  (per-layer dict)
  - bare .pt or .npy tensor of shape (d,)                     (same vec all layers)

Noise pipeline (matches notebooks/lqr/run_lqr_cosmos_policy_noised.py and
notebooks/lqr/inputs/collect_policy_inputs_noise_extreme.*):
  - Per-episode RNG: np.random.default_rng(seed = noise_seed_base + episode_idx)
    when --noise-per-episode-seed (the default); otherwise a single RNG is
    re-used across episodes.
  - Within each episode, Gaussian noise is drawn fresh per chunk-inference,
    first on `primary_image` then on `wrist_image` (matches the recipe in
    collect_policy_inputs_noise_extreme).
  - Noise is added to the post-prepare_observation policy-seen view, clipped
    to [0,255], cast back to uint8.

Parallelization
---------------
Pass --world-size W and --rank R (0 <= R < W) to split the N episodes across W
independent processes / GPUs. Each rank handles range(rank, N, world_size).
Per-rank outputs:

    <out_dir>/ep{i:02d}--{SUCCESS|FAILURE}.mp4
    <out_dir>/results_rank{R}.json

After all ranks finish, run with --phase merge to aggregate into results.json
and manifest.json.

Outputs (under --out-dir)
-------------------------
    ep{i:02d}--{SUCCESS|FAILURE}.mp4                    steered agentview
    baseline/ep{i:02d}--{SUCCESS|FAILURE}__baseline.mp4 (when --run-baseline)
    results_rank{R}.json                                per-rank rollouts
    results.json                                        merged (after merge)
    manifest.json                                       full config (after merge)
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import deque
from pathlib import Path


def _setup_env():
    """Match notebooks/_setup.py -- run BEFORE any cosmos_policy import."""
    hf = os.environ.get("HF_HOME", "/usr/scratch/jhong392/huggingface")
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

    libero_cfg = os.environ.get(
        "LIBERO_CONFIG_PATH", "/usr/scratch/jhong392/.libero"
    )
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
    ap.add_argument("--phase", choices=["rollout", "merge"], default="rollout")
    ap.add_argument("--world-size", type=int,
                    default=int(os.environ.get("WORLD_SIZE", 1)),
                    help="number of parallel ranks covering --n-episodes.")
    ap.add_argument("--rank", type=int,
                    default=int(os.environ.get("RANK", 0)))

    # ---- ActAdd steering --------------------------------------------
    ap.add_argument("--v-path", type=Path, required=True,
                    help="steering vectors; .pt dict {int layer: Tensor(d,)} "
                         "or bare .pt/.npy Tensor(d,) (applied to all layers)")
    ap.add_argument("--alpha", type=float, required=True,
                    help="scalar multiplier: output += alpha * v[layer]")

    # ---- Model -------------------------------------------------------
    ap.add_argument("--sampling-steps", type=int, default=10,
                    help="number of denoising steps for the action head")
    ap.add_argument("--ckpt-path", type=str,
                    default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")

    # ---- Image noise -------------------------------------------------
    ap.add_argument("--noise-sigma", type=float, default=90.0,
                    help="Gaussian noise σ in uint8 units (default 90.0 = "
                         "noise_extreme preset). Matches "
                         "run_lqr_cosmos_policy_noised.py.")
    ap.add_argument("--noise-seed-base", type=int, default=99,
                    help="per-episode RNG seed base; seed = "
                         "noise_seed_base + episode_idx when "
                         "--noise-per-episode-seed.")
    ap.add_argument("--noise-per-episode-seed",
                    dest="noise_per_episode_seed",
                    action="store_true", default=True,
                    help="reseed noise RNG per episode (default).")
    ap.add_argument("--no-noise-per-episode-seed",
                    dest="noise_per_episode_seed", action="store_false",
                    help="single RNG across all episodes")

    # ---- Rollout ----------------------------------------------------
    ap.add_argument("--prompt", type=str, required=True,
                    help="task description sent to the policy")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--num-steps-wait", type=int, default=10,
                    help="no-op env steps after env.reset() and before policy "
                         "inference begins.")
    ap.add_argument("--max-env-steps", type=int, default=520,
                    dest="max_env_steps",
                    help="cap on env steps per rollout (default 520, libero_10 "
                         "TASK_MAX_STEPS).")
    ap.add_argument("--run-baseline", dest="run_baseline",
                    action="store_true", default=False,
                    help="also run an unsteered baseline rollout per episode "
                         "from the same init_state + noise seed (hooks gated "
                         "off); default off.")
    ap.add_argument("--no-baseline", dest="run_baseline",
                    action="store_false",
                    help="explicitly disable the per-episode baseline run")
    ap.add_argument("--baseline-only", dest="baseline_only",
                    action="store_true", default=False,
                    help="run only the unsteered baseline pass per episode "
                         "(skip the steered pass).")

    # ---- Seed -------------------------------------------------------
    ap.add_argument("--seed", type=int, default=1,
                    help="rollout seed (default 1).")

    # ---- Output -----------------------------------------------------
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--no-save-video", dest="save_video",
                    action="store_false")
    ap.add_argument("--tag", type=str, default=None)

    return ap.parse_args()


# ====================================================================
# Image-noise helper (matches run_lqr_cosmos_policy_noised.py)
# ====================================================================

class EpisodeNoise:
    """Per-episode-seeded Gaussian pixel noise. Matches
    `ImageGaussianNoise(sigma, per_episode_seed=True)` and the noise applied
    in `collect_policy_inputs_noise_extreme.py`."""

    def __init__(self, sigma, per_episode_seed=True, seed_base=99):
        self.sigma = float(sigma)
        self.per_episode_seed = bool(per_episode_seed)
        self.seed_base = int(seed_base)
        self.rng = None
        self.current_ep = None

    def reset_episode(self, ep):
        if self.per_episode_seed:
            self.rng = np.random.default_rng(seed=self.seed_base + int(ep))
        elif self.rng is None:
            self.rng = np.random.default_rng()
        self.current_ep = int(ep)

    def apply(self, img):
        if self.rng is None:
            self.rng = np.random.default_rng()
        noise = self.rng.normal(
            loc=0.0, scale=self.sigma, size=img.shape
        ).astype(np.float32)
        return np.clip(
            img.astype(np.float32) + noise, 0, 255
        ).astype(np.uint8)


# ====================================================================
# ActAdd hook
# ====================================================================

def load_steering_vectors(v_path: Path) -> dict:
    """Load per-layer steering vectors; returns {layer_idx (int): tensor}.

    Each layer's tensor is either:
      - shape (D,)      — per-layer mode (broadcast over [B, T, H, W, D]),
      - shape (T, D)    — per-timestep mode (broadcast as (1, T, 1, 1, D)).

    Supported source formats:
      - .npy bare 1-D array shape (D,)         -- same vector for all layers
      - .pt bare 1-D tensor shape (D,)         -- same vector for all layers
      - .pt dict keyed by int layer index, each value shape (D,) or (T, D)
      - .pt dict with key 'v' or single-entry dict: bare-tensor fallback (1-D)
    Bare-tensor cases return {-1: v} as a sentinel; the caller expands it to
    all layer indices after the model is loaded.
    """
    v_path = Path(v_path)
    if not v_path.exists():
        raise FileNotFoundError(f"--v-path not found: {v_path}")
    suffix = v_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(v_path)
        v = torch.from_numpy(arr).float()
        if v.dim() != 1:
            raise ValueError(
                f"bare .npy must be 1-D (shape d,); got {tuple(v.shape)}"
            )
        return {-1: v}
    raw = torch.load(v_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        int_keys = [k for k in raw if isinstance(k, int) and k >= 0]
        if int_keys:
            result = {}
            for k in int_keys:
                v = raw[k].float()
                if v.dim() not in (1, 2):
                    raise ValueError(
                        f"v[{k}] must be 1-D (D,) or 2-D (T, D); "
                        f"got shape {tuple(v.shape)}"
                    )
                result[k] = v
            return result
        elif "v" in raw:
            v = raw["v"].float()
        elif len(raw) == 1:
            v = next(iter(raw.values())).float()
        else:
            raise ValueError(
                f"--v-path .pt dict has keys {list(raw.keys())}; expected "
                f"integer layer keys, a key named 'v', or a single-entry dict."
            )
    else:
        v = raw.float()
    # Bare-tensor fallback path: only 1-D supported (no per-T broadcast for
    # bare tensors — use a dict-keyed file for per-timestep).
    if v.dim() != 1:
        raise ValueError(
            f"bare steering vector must be 1-D (shape D,); got shape "
            f"{tuple(v.shape)}. For per-timestep mode use a dict-keyed .pt "
            f"with shape (T, D) per layer."
        )
    return {-1: v}


def install_actadd_hooks(model, v_per_layer: dict, alpha: float,
                         steering_enabled: list):
    """Register forward hooks on every DiT block present in v_per_layer.

    v_per_layer: {layer_idx: float32 tensor} with shape
        (D,)    -- per-layer mode; broadcasts over [B, T, H, W, D].
        (T, D)  -- per-timestep mode; pre-reshaped to (1, T, 1, 1, D) and
                   broadcasts over [B, T, H, W, D] (B/H/W expand, T matches).
    Each block l gets: output += alpha * v_per_layer[l]
    Vectors are pre-cast to bfloat16 on the model device; the hook re-casts
    to match output.dtype at call time. steering_enabled is a mutable
    1-element list for cheap baseline gating.
    """
    device = next(model.parameters()).device
    v_gpu = {}
    for l, v in v_per_layer.items():
        if v.dim() == 1:
            v_r = v                                # (D,)
        elif v.dim() == 2:
            v_r = v[None, :, None, None, :]        # (T, D) -> (1, T, 1, 1, D)
        else:
            raise ValueError(
                f"v[{l}] must be 1-D (D,) or 2-D (T, D); got shape "
                f"{tuple(v.shape)}"
            )
        v_gpu[l] = v_r.to(dtype=torch.bfloat16, device=device).contiguous()

    def make_hook(v_l):
        def hook(_block, _args, output):
            if not steering_enabled[0]:
                return None
            return output + alpha * v_l.to(dtype=output.dtype)
        return hook

    handles = []
    for l in sorted(v_gpu):
        handles.append(
            model.net.blocks[l].register_forward_hook(make_hook(v_gpu[l]))
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
    merged.sort(key=lambda r: r["episode"])

    seen = set()
    dedup = []
    for r in merged:
        if r["episode"] in seen:
            print(f"[merge] WARNING: duplicate episode {r['episode']}; "
                  f"keeping first occurrence", flush=True)
            continue
        seen.add(r["episode"])
        dedup.append(r)
    merged = dedup

    (out_dir / "results.json").write_text(json.dumps(merged, indent=2))
    n_succ = sum(r["success"] for r in merged)
    n_base = sum((r.get("baseline") or {}).get("success", False)
                 for r in merged)
    print(f"[merge] aggregated {len(merged)} episodes from "
          f"{len(pattern)} rank files -> {out_dir / 'results.json'}",
          flush=True)
    print(f"[merge] steered:  {n_succ}/{len(merged)} succeeded", flush=True)
    if any(r.get("baseline") is not None for r in merged):
        print(f"[merge] baseline: {n_base}/{len(merged)} succeeded",
              flush=True)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["totals"] = {
            "n_episodes": len(merged),
            "n_success_steered": int(n_succ),
            "n_success_baseline": int(n_base) if any(
                r.get("baseline") is not None for r in merged
            ) else None,
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

    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    device_id = device.index if device.index is not None else 0
    seed = int(args.seed)

    # ---------------------------------------------------------------- v vectors
    v_per_layer = load_steering_vectors(args.v_path)
    _v_sample = next(iter(v_per_layer.values()))
    _v_mode = "per-timestep" if _v_sample.dim() == 2 else "per-layer"
    print(f"[actadd] loaded v from {args.v_path}  "
          f"{len(v_per_layer)} layer(s) in file  "
          f"mode={_v_mode}  shape={tuple(_v_sample.shape)}  "
          f"alpha={args.alpha:g}", flush=True)

    # ---------------------------------------------------------------- Noise
    noise = EpisodeNoise(
        sigma=args.noise_sigma,
        per_episode_seed=args.noise_per_episode_seed,
        seed_base=args.noise_seed_base,
    )
    print(f"[noise] sigma={args.noise_sigma}  "
          f"per_episode_seed={args.noise_per_episode_seed}  "
          f"seed_base={args.noise_seed_base}", flush=True)

    # ---------------------------------------------------------------- Model
    config_name = "cosmos_predict2_2b_480p_libero__inference_only"
    ckpt_path = args.ckpt_path
    sampling_steps = args.sampling_steps

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
    L = len(model.net.blocks)
    # Expand bare-tensor sentinel to all layers.
    if list(v_per_layer) == [-1]:
        v_bare = v_per_layer[-1]
        v_per_layer = {l: v_bare for l in range(L)}
    else:
        invalid = [l for l in v_per_layer if not (0 <= l < L)]
        if invalid:
            raise ValueError(
                f"v dict has layer indices {invalid} outside [0, {L - 1}]"
            )
    hooked_layers = sorted(v_per_layer)
    print(f"[actadd] hooking {len(hooked_layers)} blocks: {hooked_layers}",
          flush=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free_gb, total_gb = (x / 1e9
                             for x in torch.cuda.mem_get_info(device_id))
        print(f"[model] after loader cleanup: GPU "
              f"{free_gb:.1f} / {total_gb:.1f} GB free")

    # ---------------------------------------------------------------- Hook
    steering_enabled = [True]
    handles = install_actadd_hooks(model, v_per_layer, args.alpha,
                                   steering_enabled)
    print(f"[hooks] registered {len(handles)} actadd hooks "
          f"(alpha={args.alpha:g})", flush=True)

    # ---------------------------------------------------------------- Env
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import (
        get_libero_env, get_libero_dummy_action,
    )

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    if args.n_episodes > init_states.shape[0]:
        raise ValueError(
            f"--n-episodes {args.n_episodes} > available init states "
            f"{init_states.shape[0]}"
        )
    suite_max_steps = TASK_MAX_STEPS[args.suite]
    max_env_steps = int(args.max_env_steps)
    env_horizon = (args.num_steps_wait + max_env_steps + 10)
    env, task_desc_libero = get_libero_env(task, "cosmos",
                                           resolution=args.resolution,
                                           horizon=env_horizon)
    print(f"[env] {args.suite} task {args.task_id:02d}  "
          f"task_desc={task_desc_libero!r}  max_steps={max_env_steps} "
          f"(suite TASK_MAX_STEPS={suite_max_steps}; "
          f"env horizon={env_horizon})")
    if args.prompt != task_desc_libero:
        print(f"      note: --prompt differs from LIBERO task description; "
              f"success is tracked against the LIBERO predicate, "
              f"not --prompt.")

    # ---------------------------------------------------------------- Rollout
    def policy_fn(observation, desc):
        with torch.inference_mode():
            out = get_action(
                eval_cfg, model, dataset_stats, observation, desc,
                seed=seed,
                randomize_seed=False,
                num_denoising_steps_action=sampling_steps,
                generate_future_state_and_value_in_parallel=True,
            )
        return out["actions"]

    def rollout(ep_idx, init_state, task_desc, *, steered: bool):
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(
                get_libero_dummy_action(eval_cfg.model_family)
            )

        # Reseed noise so steered and baseline runs of episode `ep_idx` see
        # the same noise draws on every chunk.
        noise.reset_episode(ep_idx)
        steering_enabled[0] = bool(steered)
        # Sanity-print pre/post noise std once per rank's first episode, so
        # the worker log proves the image-noise path is live (a near-zero
        # delta = noise silently bypassed).
        log_noise_once = (ep_idx == my_episodes[0])

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
                if log_noise_once and chunk_idx == 0:
                    pre_prim = float(
                        observation["primary_image"].astype(np.float32).std()
                    )
                    pre_wrst = float(
                        observation["wrist_image"].astype(np.float32).std()
                    )
                observation["primary_image"] = noise.apply(
                    observation["primary_image"]
                )
                observation["wrist_image"] = noise.apply(
                    observation["wrist_image"]
                )
                if log_noise_once and chunk_idx == 0:
                    post_prim = float(
                        observation["primary_image"].astype(np.float32).std()
                    )
                    post_wrst = float(
                        observation["wrist_image"].astype(np.float32).std()
                    )
                    print(f"[noise-check rank={rank} ep={ep_idx} "
                          f"{'steered' if steered else 'baseline'}] "
                          f"sigma={noise.sigma:g}  primary std "
                          f"{pre_prim:.1f}->{post_prim:.1f}  "
                          f"wrist std {pre_wrst:.1f}->{post_wrst:.1f}  "
                          f"(if pre==post here, noise is being skipped)",
                          flush=True)
                    log_noise_once = False
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

    # Per-rank slice: every world_size-th episode starting at rank.
    my_episodes = list(range(rank, args.n_episodes, world_size))
    print(f"[rank {rank}/{world_size}] handling {len(my_episodes)} episodes: "
          f"{my_episodes}", flush=True)

    results = []
    t_total = time.time()

    baseline_dir = out_dir / "baseline"
    if (args.run_baseline or args.baseline_only) and args.save_video:
        baseline_dir.mkdir(parents=True, exist_ok=True)

    def _do_run(ep, label):
        steered = (label == "steered")
        suffix = "" if steered else "__baseline"
        t0 = time.time()
        success, env_steps, frames, n_chunks = rollout(
            ep, init_states[ep], args.prompt, steered=steered,
        )
        dt = time.time() - t0
        tag = "SUCCESS" if success else "FAILURE"
        video_path = None
        if args.save_video:
            video_dir = baseline_dir if not steered else out_dir
            video_path = video_dir / f"ep{ep:02d}--{tag}{suffix}.mp4"
            save_video(frames, video_path, fps=args.video_fps)
        print(f"[ep {ep:2d}] {label:8s} {tag:7s}  steps={env_steps:4d}  "
              f"chunks={n_chunks:3d}  {dt:6.1f}s  "
              f"-> {video_path.name if video_path else '(no video)'}",
              flush=True)
        return {
            "success":      bool(success),
            "env_steps":    int(env_steps),
            "n_chunks":     int(n_chunks),
            "wall_time_s":  float(dt),
            "video_path":   str(video_path) if video_path else None,
        }

    for ep in my_episodes:
        if args.baseline_only:
            baseline_rec = _do_run(ep, "baseline")
            results.append({
                "episode":  ep,
                "rank":     rank,
                **baseline_rec,
                "baseline": None,
            })
        else:
            steered_rec = _do_run(ep, "steered")
            baseline_rec = _do_run(ep, "baseline") if args.run_baseline else None
            results.append({
                "episode":  ep,
                "rank":     rank,
                **steered_rec,
                "baseline": baseline_rec,
            })

    for h in handles:
        h.remove()
    env.close()

    # ---------------------------------------------------------------- Save
    results_path = out_dir / f"results_rank{rank}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[rank {rank}] wrote {results_path}", flush=True)

    if rank == 0:
        manifest = {
            "tag": args.tag,
            "actadd": {
                "v_path":         str(args.v_path.resolve()),
                "alpha":          float(args.alpha),
                "hooked_layers":  hooked_layers,
                "n_hooked_layers": len(hooked_layers),
                "v_mode":         _v_mode,
                "v_shape_per_layer": list(_v_sample.shape),
                "sampling_steps": int(sampling_steps),
            },
            "noise": {
                "kind": "EpisodeNoise (Gaussian pixel)",
                "sigma": float(args.noise_sigma),
                "seed_base": int(args.noise_seed_base),
                "per_episode_seed": bool(args.noise_per_episode_seed),
                "apply_to": ["primary_image", "wrist_image"],
                "matches": ("run_lqr_cosmos_policy_noised.py + "
                            "collect_policy_inputs_noise_extreme.py"),
            },
            "rollout": {
                "suite": args.suite,
                "task_id": int(args.task_id),
                "task_desc_libero": task_desc_libero,
                "policy_prompt": args.prompt,
                "n_episodes": int(args.n_episodes),
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
                "assignment": "episodes range(rank, n_episodes, world_size)",
            },
            "model": {
                "ckpt_path": ckpt_path,
                "config_name": config_name,
                "L_blocks": int(L),
            },
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        print(f"[rank 0] wrote manifest -> {out_dir / 'manifest.json'}",
              flush=True)

    n_succ = sum(r["success"] for r in results)
    print(f"[rank {rank}] DONE: {n_succ}/{len(results)} succeeded; "
          f"wall {time.time() - t_total:.1f}s")


def main():
    args = parse_args()
    if args.phase == "merge":
        run_merge_phase(args)
    else:
        run_rollout_phase(args)


if __name__ == "__main__":
    main()
