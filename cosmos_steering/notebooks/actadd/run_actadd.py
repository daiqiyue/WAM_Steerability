#!/usr/bin/env python
"""Activation addition (ActAdd) for Cosmos-Policy-LIBERO-Predict2-2B.

At every DiT block forward and at every denoising step, the script adds a
scaled contrastive direction to the block activation:

    output[0, denoise_t_start:denoise_t_end, :, :, :]
        += alpha * v_{l,t}

where v_{l,t} is the rank-r reconstruction of the mean contrastive
activation direction for layer l, denoising step t, obtained from the SVD
factorization produced by

    notebooks/lqr/svd/run_partition_svd_pairs.sh

That script writes, per (partition, timestep) tile of layers,

    V_part{p}_layers{a}-{b}_t{t}_k{k}.pt   ── {V, c_per_layer, ...}

with V: (D_flat, k) bfloat16 the rank-k orthonormal basis, and
c_per_layer: (L_p, k) float32 the projection of the per-layer mean
contrast onto V. The full-dimensional steering direction for layer
l_local in the partition at timestep t is then

    v_flat = V[:, :r] @ c_per_layer[l_local, :r]    ──  (D_flat,)
    v_grid = v_flat.view(T_p_denoise, H_p, W_p, D)

which is exactly the layout of the DiT block's denoising slots
(state_t[denoise_t_start:denoise_t_end]).

Rollout setting
---------------
First N episodes of `libero_10` task 0 with the **prompt swapped**:

    "put both the alphabet soup and the tomato sauce in the basket"
    ─►  "put both the milk and the tomato sauce in the basket"

via a single ('alphabet soup' -> 'milk') substring substitution on the
task language. The MuJoCo scene is the stock 8-object task scene (no
SceneRemoveObjects), so the rollout is exactly the off-distribution
condition the V-basis was learned to steer toward
(distractors-absent / milk-only).

Outputs (under --out-dir)
-------------------------
    ep{i:02d}--{SUCCESS|FAILURE}.mp4   agentview video per episode
    results.json                       per-episode steps / success / wall
    manifest.json                      full config + SVD dir + hyperparams

Memory: precomputes v_{l,t} for every applied (layer, step) tile on GPU
in bf16. For L=28, T_sel=5, D_flat≈2.0e6 that is ~560 MB.
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
    """Match notebooks/_setup.py — run BEFORE any cosmos_policy import."""
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

import imageio
import numpy as np
import torch


DEFAULT_PROMPT_OLD = "alphabet soup"
DEFAULT_PROMPT_NEW = "milk"
DEFAULT_SUITE = "libero_10"
DEFAULT_TASK_ID = 0


# ====================================================================
# CLI
# ====================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ── SVD source ───────────────────────────────────────────────────
    ap.add_argument("--svd-dir", type=Path, required=True,
                    help="directory produced by run_partition_svd_pairs.sh "
                         "(contains config.json + V_part*_t*.pt + "
                         "svd_summary.pt)")

    # ── ActAdd hyperparameters ───────────────────────────────────────
    ap.add_argument("--alpha", type=float, required=True,
                    help="steering scale. v_{l,t} points from "
                         "distractors-present mean activation toward "
                         "distractors-absent, so positive alpha pushes the "
                         "model 'as if distractors were absent'.")
    ap.add_argument("--rank", type=int, default=-1,
                    help="number of top SVD components to use for the "
                         "rank-r reconstruction v = V[:,:r] @ c[l,:r]. "
                         "-1 = all available k_target components.")
    ap.add_argument("--layers", type=str, default="all",
                    help="which DiT layers to add steering to: 'all', a "
                         "comma-separated list ('0,5,12'), or inclusive "
                         "ranges ('10-18' or '0-4,20-27').")
    ap.add_argument("--timesteps", type=str, default="all",
                    help="which denoising steps to steer at: 'all' or a "
                         "comma-separated subset of the steps the SVD "
                         "factorized (see config.json:selected_timesteps).")

    # ── Rollout ──────────────────────────────────────────────────────
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--suite", type=str, default=DEFAULT_SUITE)
    ap.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    ap.add_argument("--num-denoising-steps-action", type=int, default=5,
                    help="denoising steps used at inference. MUST match "
                         "config.json:sampling_steps from --svd-dir; "
                         "otherwise the (layer, step) → steering map is "
                         "ill-defined.")
    ap.add_argument("--prompt-old", type=str, default=DEFAULT_PROMPT_OLD,
                    help=f"substring in the task language to replace "
                         f"(default: '{DEFAULT_PROMPT_OLD}')")
    ap.add_argument("--prompt-new", type=str, default=DEFAULT_PROMPT_NEW,
                    help=f"replacement (default: '{DEFAULT_PROMPT_NEW}')")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--num-steps-wait", type=int, default=10,
                    help="no-op env steps at the start of each rollout so "
                         "the cached agentview camera observable refreshes")

    # ── Model ────────────────────────────────────────────────────────
    ap.add_argument("--ckpt-path", type=str,
                    default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    ap.add_argument("--config-name", type=str,
                    default="cosmos_predict2_2b_480p_libero__inference_only")
    ap.add_argument("--config-file", type=str,
                    default="cosmos_policy/config/config.py")
    ap.add_argument("--dataset-stats-path", type=str, default=None,
                    help="default: <ckpt-path>/libero_dataset_statistics.json")
    ap.add_argument("--t5-cache-path", type=str, default=None,
                    help="default: <ckpt-path>/libero_t5_embeddings.pkl")
    ap.add_argument("--seed", type=int, default=1)

    # ── Output ───────────────────────────────────────────────────────
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")

    return ap.parse_args()


# ====================================================================
# Layer / timestep parsing
# ====================================================================

def parse_layer_spec(spec, L):
    if spec.strip().lower() == "all":
        return set(range(L))
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    bad = [l for l in out if l < 0 or l >= L]
    if bad:
        raise ValueError(f"--layers entries {bad} out of range [0, {L - 1}]")
    return out


def parse_timestep_spec(spec, available):
    avail_set = set(available)
    if spec.strip().lower() == "all":
        return set(available)
    out = {int(x) for x in spec.split(",") if x.strip()}
    bad = sorted(out - avail_set)
    if bad:
        raise ValueError(
            f"--timesteps {bad} not in SVD selected_timesteps {sorted(avail_set)}"
        )
    return out


# ====================================================================
# Load V matrices, build per-(layer, step) steering tensors
# ====================================================================

def load_steering_tensors(svd_dir: Path, rank: int,
                          layer_set, timestep_set,
                          device, dtype=torch.bfloat16):
    """Build {(layer, denoise_step): bf16 tensor (T_p_denoise, H_p, W_p, D)}.

    Reconstruction
    --------------
        v_{l, t} = V_p[:, :r] @ c_p[l_local, :r]

    where p indexes the partition containing layer l. The result is laid out
    so it can be added directly into output[0, denoise_t_start:denoise_t_end]
    inside the forward hook.

    Returns
    -------
    steering : dict {(layer, t_id) -> tensor on `device`, dtype `dtype`}
    meta     : dict with shape + SVD config + per-(l,t) reconstruction norms
    """
    cfg_path = svd_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"SVD config missing: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    T_p_denoise = cfg["T_p_denoise"]
    H_p = cfg["H_p"]
    W_p = cfg["W_p"]
    D = cfg["D"]
    D_flat = cfg["D_flat"]
    expected_D_flat = T_p_denoise * H_p * W_p * D
    if D_flat != expected_D_flat:
        raise RuntimeError(
            f"SVD config inconsistent: D_flat={D_flat} != "
            f"T_p_denoise*H_p*W_p*D = {expected_D_flat}"
        )

    # Find every V file in the directory matching the expected naming.
    V_paths = sorted(svd_dir.glob("V_part*_t*.pt"))
    if not V_paths:
        raise FileNotFoundError(
            f"no V_part*_t*.pt files under {svd_dir}. Did the SVD finalize?"
        )

    steering = {}
    recon_norms = {}  # (l, t) -> ||v||
    used_partitions = set()
    used_timesteps = set()

    for V_path in V_paths:
        data = torch.load(V_path, map_location="cpu", weights_only=False)
        l_start = int(data["l_start"])
        l_end = int(data["l_end"])
        t_id = int(data["timestep"])
        L_p = int(data["L_p"])
        if t_id not in timestep_set:
            continue
        if not any((l in layer_set) for l in range(l_start, l_end + 1)):
            continue

        V_bf = data["V"]                       # (D_flat, k_eff) bf16
        c = data["c_per_layer"].float()        # (L_p, k_eff) f32
        k_eff = int(data["k_eff"])
        r = k_eff if rank < 0 else max(0, min(int(rank), k_eff))
        if r == 0:
            continue

        V_f = V_bf[:, :r].to(device=device, dtype=torch.float32,
                              non_blocking=True)
        c_dev = c[:, :r].to(device=device, non_blocking=True)

        # v_layers : (L_p, D_flat). Rows are layer-local indices.
        v_layers = c_dev @ V_f.T

        for l_local in range(L_p):
            l = l_start + l_local
            if l not in layer_set:
                continue
            v_flat = v_layers[l_local]
            recon_norms[(l, t_id)] = float(v_flat.float().norm().item())
            steering[(l, t_id)] = (
                v_flat.to(dtype)
                .view(T_p_denoise, H_p, W_p, D)
                .contiguous()
            )

        used_partitions.add((l_start, l_end))
        used_timesteps.add(t_id)
        del V_f, c_dev, v_layers, V_bf, c, data
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    meta = {
        "L": int(cfg["L"]),
        "T_p_denoise": T_p_denoise,
        "H_p": H_p, "W_p": W_p, "D": D, "D_flat": D_flat,
        "denoise_t_start": int(cfg["denoise_t_start"]),
        "denoise_t_end": int(cfg["denoise_t_end"]),
        "sampling_steps": int(cfg["sampling_steps"]),
        "selected_timesteps": list(cfg["selected_timesteps"]),
        "partitions": [tuple(p) for p in cfg["partitions"]],
        "svd_prompt": cfg.get("prompt"),
        "svd_input_mode": cfg.get("input_mode"),
        "used_partitions": sorted(used_partitions),
        "used_timesteps": sorted(used_timesteps),
        "reconstruction_norms": recon_norms,
        "rank_requested": rank,
    }
    return steering, meta


# ====================================================================
# ActAdd hooks
# ====================================================================

def install_actadd_hooks(model, steering, meta, alpha):
    """Register a forward hook on every DiT block. Returns
    (handles, reset_step_counter).

    The hook adds `alpha * v_{l,t}` (where v lives in `steering`) to
    output[0, denoise_t_start:denoise_t_end, :, :, :]. The denoising step
    counter is incremented every time block 0 fires its hook — this matches
    the SVD-side convention (PASSES_PER_STEP=1, no CFG).
    """
    denoise_t_start = meta["denoise_t_start"]
    denoise_t_end = meta["denoise_t_end"]
    sampling_steps = meta["sampling_steps"]

    state = {"pass_idx": -1, "fire_counts": {}}

    def make_hook(layer_idx):
        is_block0 = (layer_idx == 0)

        def hook(module, inputs, output):
            if is_block0:
                state["pass_idx"] += 1
            step = state["pass_idx"]
            v = steering.get((layer_idx, step))
            if v is None:
                return output
            # output: (1, state_t, H_p, W_p, D), bf16. We mutate the slice
            # in place and return the (still-the-same) output tensor.
            slot = output[0, denoise_t_start:denoise_t_end, :, :, :]
            slot.add_(v, alpha=float(alpha))
            state["fire_counts"][(layer_idx, step)] = (
                state["fire_counts"].get((layer_idx, step), 0) + 1
            )
            return output

        return hook

    handles = [
        b.register_forward_hook(make_hook(i))
        for i, b in enumerate(model.net.blocks)
    ]

    def reset():
        # Call before every model.generate_samples_from_batch invocation —
        # i.e. before every get_action() call inside the rollout.
        state["pass_idx"] = -1

    def get_state():
        return state

    return handles, reset, get_state


# ====================================================================
# Rollout
# ====================================================================

def save_video(frames, path, fps):
    writer = imageio.get_writer(str(path), fps=fps)
    for frame in frames:
        # LIBERO's agentview_image is rendered upside-down; flip for viewing.
        writer.append_data(np.flipud(frame))
    writer.close()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] svd_dir       : {args.svd_dir}")
    print(f"[setup] out_dir       : {args.out_dir}")
    print(f"[setup] alpha         : {args.alpha}")
    print(f"[setup] rank          : {args.rank} (-1=all)")
    print(f"[setup] layers        : {args.layers}")
    print(f"[setup] timesteps     : {args.timesteps}")
    print(f"[setup] n_episodes    : {args.n_episodes}")
    print(f"[setup] suite/task    : {args.suite} / {args.task_id}")
    print(f"[setup] denoise steps : {args.num_denoising_steps_action}")
    print(f"[setup] prompt swap   : "
          f"'{args.prompt_old}' -> '{args.prompt_new}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Parse SVD config + load steering vectors ----------
    cfg_svd = json.loads((args.svd_dir / "config.json").read_text())
    L = int(cfg_svd["L"])
    if int(cfg_svd["sampling_steps"]) != args.num_denoising_steps_action:
        raise ValueError(
            f"sampling_steps mismatch: SVD was run with "
            f"sampling_steps={cfg_svd['sampling_steps']} but "
            f"--num-denoising-steps-action={args.num_denoising_steps_action}. "
            f"The (layer, step) -> steering map is only valid when these "
            f"match."
        )
    layer_set = parse_layer_spec(args.layers, L)
    timestep_set = parse_timestep_spec(args.timesteps,
                                       cfg_svd["selected_timesteps"])
    print(f"[setup] |layer_set|={len(layer_set)}, "
          f"|timestep_set|={len(timestep_set)}")

    steering, meta = load_steering_tensors(
        args.svd_dir, args.rank, layer_set, timestep_set, device,
        dtype=torch.bfloat16,
    )
    print(f"[svd]   loaded {len(steering)} (layer, step) steering tensors")
    if meta["reconstruction_norms"]:
        norms = list(meta["reconstruction_norms"].values())
        print(f"[svd]   ||v_{{l,t}}|| min/median/max = "
              f"{min(norms):.3e} / "
              f"{sorted(norms)[len(norms)//2]:.3e} / "
              f"{max(norms):.3e}")

    # ---------- Build env (no scene modification) ----------
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import (
        get_libero_env, get_libero_dummy_action,
    )
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        PolicyEvalConfig, prepare_observation, TASK_MAX_STEPS,
    )
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action, get_model, init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    max_env_steps = TASK_MAX_STEPS[args.suite]
    base_desc = task.language
    effective_prompt = base_desc.replace(args.prompt_old, args.prompt_new)
    if effective_prompt == base_desc and args.prompt_old:
        print(f"[warn]  prompt substring '{args.prompt_old}' not present in "
              f"task.language={base_desc!r} — using base description verbatim")
    if args.n_episodes > init_states.shape[0]:
        raise ValueError(
            f"--n-episodes {args.n_episodes} > available init states "
            f"{init_states.shape[0]}"
        )

    print(f"[env]   base_desc      : {base_desc!r}")
    print(f"[env]   effective prompt: {effective_prompt!r}")
    print(f"[env]   max_env_steps  : {max_env_steps}")

    env, env_base_desc = get_libero_env(task, "cosmos",
                                         resolution=args.resolution)
    print(f"[env]   built {type(env).__name__}  base_task_desc="
          f"{env_base_desc!r}")

    # ---------- Build PolicyEvalConfig + load model ----------
    cfg = PolicyEvalConfig(
        config=args.config_name,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        dataset_stats_path=(
            args.dataset_stats_path
            or f"{args.ckpt_path}/libero_dataset_statistics.json"
        ),
        t5_text_embeddings_path=(
            args.t5_cache_path
            or f"{args.ckpt_path}/libero_t5_embeddings.pkl"
        ),
        use_wrist_image=True, use_proprio=True,
        normalize_proprio=True, unnormalize_actions=True,
        chunk_size=16, num_open_loop_steps=16,
        trained_with_image_aug=True,
        use_jpeg_compression=True, flip_images=True,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        task_suite_name=args.suite,
    )

    print(f"[model] loading dataset stats + T5 cache + weights ...")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    model, _ = get_model(cfg)
    L_runtime = len(model.net.blocks)
    if L_runtime != L:
        raise RuntimeError(
            f"model has {L_runtime} DiT blocks but SVD config L={L}"
        )

    # ---------- Install ActAdd hooks ----------
    handles, reset_step, get_state = install_actadd_hooks(
        model, steering, meta, args.alpha,
    )
    print(f"[hooks] installed forward hook on {len(handles)} blocks")

    # ---------- Rollout loop ----------
    def policy_fn(obs, desc):
        # Reset the denoising-step counter so the hook's pass_idx starts at
        # 0 for the first DiT block call of this get_action invocation.
        reset_step()
        out = get_action(
            cfg, model, dataset_stats, obs, desc,
            seed=args.seed,
            randomize_seed=False,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=True,
        )
        return out["actions"]

    results = []
    fire_summary_first_call = None

    for ep in range(args.n_episodes):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        # Settle the scene + refresh cached agentview observable.
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

        queue = deque(maxlen=cfg.num_open_loop_steps)
        frames = [obs["agentview_image"].copy()]
        success = False
        t = 0
        t0 = time.time()
        n_get_action = 0

        while t < max_env_steps:
            if not queue:
                observation = prepare_observation(
                    obs, resize_size=224, flip_images=cfg.flip_images,
                )
                actions = policy_fn(observation, effective_prompt)
                n_get_action += 1
                if fire_summary_first_call is None:
                    fc = get_state()["fire_counts"]
                    fire_summary_first_call = {
                        "n_layer_step_pairs_fired": len(fc),
                        "max_fires_per_pair": max(fc.values()) if fc else 0,
                        "min_fires_per_pair": min(fc.values()) if fc else 0,
                        "expected_steering_pairs": len(steering),
                    }
                    print(f"[fire]  first get_action: "
                          f"{fire_summary_first_call}")
                for a in actions[:cfg.num_open_loop_steps]:
                    queue.append(np.asarray(a, dtype=np.float32))
            a = queue.popleft()
            obs, _, done, _ = env.step(a.tolist())
            frames.append(obs["agentview_image"].copy())
            if done:
                success = True
                break
            t += 1

        dt = time.time() - t0
        tag = "SUCCESS" if success else "FAILURE"
        env_steps = t + args.num_steps_wait

        video_path = None
        if args.save_video:
            video_path = args.out_dir / f"ep{ep:02d}--{tag}.mp4"
            save_video(frames, video_path, fps=args.video_fps)

        print(f"[ep {ep:2d}] {tag:7s}  steps={env_steps:4d}  "
              f"get_action_calls={n_get_action:3d}  {dt:6.1f}s  "
              f"-> {video_path.name if video_path else '(no video)'}")
        results.append({
            "episode": ep,
            "success": bool(success),
            "env_steps": int(env_steps),
            "wall_time_s": float(dt),
            "n_get_action_calls": int(n_get_action),
            "video_path": str(video_path) if video_path else None,
        })

    for h in handles:
        h.remove()
    env.close()

    # ---------- Save manifest / results ----------
    manifest = {
        "actadd": {
            "alpha": float(args.alpha),
            "rank_requested": int(args.rank),
            "layers": sorted(layer_set),
            "timesteps": sorted(timestep_set),
            "svd_dir": str(args.svd_dir),
            "svd_prompt": meta["svd_prompt"],
            "svd_input_mode": meta["svd_input_mode"],
            "n_steering_pairs": len(steering),
            "fire_summary_first_call": fire_summary_first_call,
            "reconstruction_norms_stats": (
                {
                    "min": min(meta["reconstruction_norms"].values()),
                    "median": sorted(
                        meta["reconstruction_norms"].values()
                    )[len(meta["reconstruction_norms"]) // 2],
                    "max": max(meta["reconstruction_norms"].values()),
                    "n": len(meta["reconstruction_norms"]),
                } if meta["reconstruction_norms"] else None
            ),
        },
        "rollout": {
            "suite": args.suite,
            "task_id": int(args.task_id),
            "n_episodes": int(args.n_episodes),
            "num_denoising_steps_action": args.num_denoising_steps_action,
            "resolution": int(args.resolution),
            "video_fps": int(args.video_fps),
            "seed": int(args.seed),
            "num_steps_wait": int(args.num_steps_wait),
            "task_language_base": base_desc,
            "task_language_effective": effective_prompt,
            "prompt_old": args.prompt_old,
            "prompt_new": args.prompt_new,
            "env_base_task_desc": env_base_desc,
        },
        "model": {
            "ckpt_path": args.ckpt_path,
            "config_name": args.config_name,
            "L_blocks": int(L_runtime),
            "model_id": "Cosmos-Policy-LIBERO-Predict2-2B",
        },
        "svd_meta": {
            k: v for k, v in meta.items()
            if k not in ("reconstruction_norms",)
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))

    n_succ = sum(r["success"] for r in results)
    print()
    print(f"[done]  {n_succ}/{len(results)} succeeded "
          f"({100 * n_succ / max(1, len(results)):.0f}%)")
    print(f"[done]  manifest: {args.out_dir / 'manifest.json'}")
    print(f"[done]  results : {args.out_dir / 'results.json'}")
    if args.save_video:
        print(f"[done]  videos  : {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
