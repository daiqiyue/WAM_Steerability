#!/usr/bin/env python
"""Activation-LQR (A-LQR) rollout of DiT4DiT under Gaussian-noised camera input.

Analogue of the repo root's notebooks/lqr/run_lqr_cosmos_policy_noised.py adapted for
DiT4DiT's FlowmatchingActionHead / BasicTransformerBlock.

Perturbation: at every policy inference, Gaussian pixel noise (σ = --noise-sigma)
is added to both the primary and wrist images before feeding to the model.
The proprio/state is left unchanged (matching how negatives are generated in
collect_policy_inputs_noise.py).

Per-episode noise seeding: RNG seeded with (--noise-seed-base + episode_idx),
matching the seeding convention used in collect_policy_inputs_noise.py.

LQR steering uses a fixed R (no exponential schedule), appropriate for a
perturbation that is persistent every chunk rather than a one-time displacement.
VCache, SteeringRuntime, and install_lqr_hooks are identical to the gripper_xyz
version — only the observation pipeline and Riccati solver differ.

Parallelization: --world-size W, --rank R (striped episodes), same as gripper_xyz.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DIT4DIT_ROOT = _HERE.parent.parent
if str(_DIT4DIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIT4DIT_ROOT))

LIBERO_HOME = os.environ.get("LIBERO_HOME", "/work/nvme/bhde/jhong7/LIBERO_pkg")
if LIBERO_HOME not in sys.path:
    sys.path.insert(0, LIBERO_HOME)

# Append FastWAM site-packages at the END so robosuite is found but
# dit4dit's own transformers/diffusers take priority over FastWAM's.
_FASTWAM_SITE = "/projects/bhde/jhong7/dit4dit-env/libero-sim/lib/python3.10/site-packages"
if _FASTWAM_SITE not in sys.path:
    sys.path.append(_FASTWAM_SITE)
os.environ.setdefault("LIBERO_HOME", LIBERO_HOME)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(LIBERO_HOME, "libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DIT4DIT_ROOT = _DIT4DIT_ROOT
CKPT_DEFAULT = str(DIT4DIT_ROOT / "checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt")


def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--phase", choices=["rollout", "merge"], default="rollout")
    ap.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    ap.add_argument("--rank",       type=int, default=int(os.environ.get("RANK", 0)))

    ap.add_argument("--svd-dir",     type=Path, required=True)
    ap.add_argument("--jac-dir-act", type=str,  required=True)

    # Noise knobs
    ap.add_argument("--noise-sigma",      type=float, default=75.0,
                    help="Gaussian pixel noise σ in uint8 units (default 75).")
    ap.add_argument("--noise-seed-base",  type=int,   default=0,
                    help="Base for per-episode RNG seed: seed = base + episode_idx.")

    # LQR cost hyperparameters (fixed R, no schedule)
    ap.add_argument("--lambda-scale", type=float, default=1.0,  dest="lambda_scale")
    ap.add_argument("--q-scale",      type=float, default=10000.0, dest="q_scale")
    ap.add_argument("--r-scale",      type=float, default=75000.0, dest="r_scale")
    ap.add_argument("--qf-scale",     type=float, default=1.0,  dest="qf_scale")

    # Rollout
    ap.add_argument("--prompt",         type=str,  required=True)
    ap.add_argument("--n-episodes",     type=int,  default=10)
    ap.add_argument("--suite",          type=str,  default="libero_10")
    ap.add_argument("--task-id",        type=int,  default=0)
    ap.add_argument("--resolution",     type=int,  default=256)
    ap.add_argument("--video-fps",      type=int,  default=30)
    ap.add_argument("--num-steps-wait", type=int,  default=10)
    ap.add_argument("--max-env-steps",  type=int,  default=1000, dest="max_env_steps")
    ap.add_argument("--run-baseline",   dest="run_baseline", action="store_true",  default=False)
    ap.add_argument("--no-baseline",    dest="run_baseline", action="store_false")

    ap.add_argument("--ckpt-path", type=str, default=CKPT_DEFAULT)
    ap.add_argument("--seed",      type=int, default=None)

    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    ap.add_argument("--tag", type=str, default=None)
    return ap.parse_args()


# -----------------------------------------------------------------------
# Re-use VCache, SteeringRuntime, install_lqr_hooks from gripper_xyz module
# -----------------------------------------------------------------------

def _import_shared():
    sys.path.insert(0, str(DIT4DIT_ROOT / "notebooks/lqr"))
    from run_lqr_dit4dit_gripper_xyz import (
        VCache, SteeringRuntime, install_lqr_hooks,
    )
    sys.path.insert(0, str(DIT4DIT_ROOT / "notebooks/lqr/svd"))
    from run_partition_svd_pairs_no_action import run_denoising_loop
    return VCache, SteeringRuntime, install_lqr_hooks, run_denoising_loop


# -----------------------------------------------------------------------
# Chained Riccati — fixed R (no per-chunk schedule)
# -----------------------------------------------------------------------

def chained_riccati(A_tilde, B_tilde, q_scale, r_scale, qf_scale, device):
    T_diff, L_minus1, r, _ = A_tilde.shape
    L = L_minus1 + 1
    K_total = T_diff * L - 1 if T_diff > 0 else 0

    _dt = torch.float64
    I_r     = torch.eye(r, dtype=_dt, device=device)
    Q_chain = (q_scale  * I_r).expand(K_total, r, r).contiguous()
    R_chain = (r_scale  * I_r).expand(K_total, r, r).contiguous()
    S_T     = (qf_scale * I_r).contiguous()

    A_dev = A_tilde.to(device=device, dtype=_dt)
    B_dev = B_tilde.to(device=device, dtype=_dt)
    A_chain = torch.zeros(K_total, r, r, dtype=_dt, device=device)
    for t in range(T_diff):
        for l in range(L - 1):
            A_chain[t * L + l] = A_dev[t, l]
        if t < T_diff - 1 and B_dev.numel():
            A_chain[t * L + (L - 1)] = B_dev[t]

    Tn = A_chain.shape[0]
    S = torch.zeros(Tn + 1, r, r, dtype=_dt, device=device)
    K = torch.zeros(Tn,     r, r, dtype=_dt, device=device)
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
    K_intra = torch.zeros(T_diff, L - 1, r, r, dtype=torch.float32)
    K_step  = torch.zeros(max(T_diff - 1, 0), r, r, dtype=torch.float32)
    for t in range(T_diff):
        for l in range(L - 1):
            K_intra[t, l] = K_chain[t * L + l]
        if t < T_diff - 1:
            K_step[t] = K_chain[t * L + (L - 1)]
    return K_intra, K_step


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
    (out_dir / "results.json").write_text(json.dumps(dedup, indent=2))
    n_succ = sum(r["success"] for r in dedup)
    n_base = sum((r.get("baseline") or {}).get("success", False) for r in dedup)
    print(f"[merge] {len(dedup)} episodes -> results.json  steered={n_succ}/{len(dedup)}", flush=True)
    if any(r.get("baseline") is not None for r in dedup):
        print(f"[merge] baseline: {n_base}/{len(dedup)}", flush=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["totals"] = {"n_episodes": len(dedup),
                               "n_success_steered": int(n_succ),
                               "n_success_baseline": int(n_base) if any(r.get("baseline") is not None for r in dedup) else None}
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


# -----------------------------------------------------------------------
# Rollout phase
# -----------------------------------------------------------------------

def run_rollout_phase(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.out_dir.resolve()
    rank, world_size = int(args.rank), int(args.world_size)
    assert 0 <= rank < world_size

    VCache, SteeringRuntime, install_lqr_hooks, run_denoising_loop = _import_shared()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ---- SVD config ----
    svd_dir = args.svd_dir.resolve()
    cfg = json.loads((svd_dir / "config.json").read_text())
    sel_t           = list(cfg["selected_timesteps"])
    T_diff          = len(sel_t)
    sampling_steps  = int(cfg["sampling_steps"])
    L               = int(cfg["L"])
    r               = int(cfg["k_target"])
    partitions      = [tuple(p) for p in cfg["partitions"]]
    T_p_denoise     = int(cfg["T_p_denoise"])
    denoise_t_start = int(cfg["denoise_t_start"])
    denoise_t_end   = int(cfg["denoise_t_end"])
    action_horizon  = cfg.get("action_horizon", T_p_denoise)
    inner_dim       = cfg.get("inner_dim", cfg["D_flat"] // action_horizon)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    # ---- A_tilde ----
    jac_dir = svd_dir / args.jac_dir_act
    raw = torch.load(jac_dir / "A_tilde__full.pt", map_location="cpu", weights_only=False)
    A_dict, B_dict = raw.get("A_tilde", {}), raw.get("B_tilde", {})
    jac_prompt = raw.get("prompt", "<unknown>")
    sel_idx_of = {t: i for i, t in enumerate(sel_t)}
    A_tilde = torch.zeros(T_diff, L - 1, r, r, dtype=torch.float32)
    for (t, l_in), Atl in A_dict.items():
        if t in sel_idx_of:
            A_tilde[sel_idx_of[t], l_in] = Atl.float()
    have_B = len(B_dict) > 0
    B_tilde = torch.zeros(max(T_diff - 1, 0), r, r, dtype=torch.float32)
    if have_B:
        for (t,), Bt in B_dict.items():
            if t in sel_idx_of and sel_idx_of[t] < T_diff - 1:
                B_tilde[sel_idx_of[t]] = Bt.float()

    # ---- LQR setpoint ----
    summary = torch.load(svd_dir / "svd_summary.pt", map_location="cpu", weights_only=False)
    c_means  = summary["c_means"].float()
    tilde_mu = c_means.norm(dim=-1)
    tilde_v  = c_means / tilde_mu.unsqueeze(-1).clamp(min=1e-12)
    layer_to_part = list(summary["layer_to_part"])

    # ---- Riccati (fixed R) ----
    t0 = time.time()
    K_intra, K_step = chained_riccati(
        A_tilde, B_tilde, args.q_scale, args.r_scale, args.qf_scale, device,
    )
    print(f"[lqr] λ={args.lambda_scale:g}  Q={args.q_scale:g}  R={args.r_scale:g}  "
          f"Qf={args.qf_scale:g}  Riccati: {time.time()-t0:.2f}s")

    # ---- Load model ----
    import DiT4DiT.model.framework.DiT4DiT  # register framework before from_pretrained
    from DiT4DiT.model.framework.base_framework import baseframework
    from DiT4DiT.model.framework.share_tools import read_mode_config
    print(f"[model] loading DiT4DiT from {args.ckpt_path} ...")
    model        = baseframework.from_pretrained(args.ckpt_path).to(device).eval()
    action_model = model.action_model
    action_dit   = action_model.model
    n_blocks = len(action_dit.transformer_blocks)
    assert n_blocks == L, f"block count mismatch: model={n_blocks} SVD L={L}"
    _, norm_stats = read_mode_config(args.ckpt_path)
    unnorm_key   = next(iter(norm_stats))
    action_stats = norm_stats[unnorm_key]["action"]
    action_high  = np.array(action_stats["max"], dtype=np.float32)
    action_low   = np.array(action_stats["min"], dtype=np.float32)
    action_mask  = np.array(action_stats.get("mask", np.ones(len(action_high), dtype=bool)))
    max_state_dim = action_model.config.state_dim
    gc.collect()
    torch.cuda.empty_cache()

    # ---- V cache + LQR push ----
    vcache = VCache(svd_dir, partitions, layer_to_part, sel_t, r,
                    device=device, dtype=torch.bfloat16)
    vcache.preload([(p, t) for p in range(len(partitions)) for t in sel_t])
    lqr = {
        "K_intra": K_intra.to(device=device, dtype=torch.float32),
        "K_step":  K_step.to(device=device, dtype=torch.float32) if K_step.numel() else K_step,
        "v":       tilde_v.to(device=device, dtype=torch.float32),
        "mu":      tilde_mu.to(device=device, dtype=torch.float32),
    }

    # ---- Hooks ----
    rt = SteeringRuntime(
        L=L, T_diff=T_diff, sel_t=sel_t,
        denoise_t_start=denoise_t_start, denoise_t_end=denoise_t_end,
        T_p_denoise=T_p_denoise, inner_dim=inner_dim,
        sampling_steps=sampling_steps,
        lambda_scale=args.lambda_scale, vcache=vcache, lqr=lqr,
    )
    handles = install_lqr_hooks(action_dit, rt)
    print(f"[hooks] {len(handles)} hooks registered")

    # ---- Env ----
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from pathlib import Path as _Path

    task_suite  = benchmark.get_benchmark_dict()[args.suite]()
    task        = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    if args.n_episodes > init_states.shape[0]:
        raise ValueError(f"--n-episodes {args.n_episodes} > {init_states.shape[0]}")
    suite_max_steps = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
                       "libero_10": 520, "libero_90": 400}.get(args.suite, 520)
    max_env_steps = int(args.max_env_steps)
    task_bddl = _Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(task_bddl),
                              camera_heights=args.resolution, camera_widths=args.resolution)
    env.seed(42)
    task_desc_libero = task.language
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    NUM_OPEN_LOOP = action_model.action_horizon

    def _add_noise(img_f32: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.clip(img_f32 + rng.normal(0.0, args.noise_sigma, img_f32.shape), 0, 255).astype(np.uint8)

    def policy_fn(obs: dict, rng: np.random.Generator | None, steered: bool):
        IMAGE_SIZE = 224
        primary_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.float32)
        wrist_raw   = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.float32)
        if rng is not None:
            primary_raw = _add_noise(primary_raw, rng).astype(np.float32)
            wrist_raw   = _add_noise(wrist_raw,   rng).astype(np.float32)
        primary = cv2.resize(primary_raw.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        wrist   = cv2.resize(wrist_raw.astype(np.uint8),   (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        concat_img = np.concatenate([primary, wrist], axis=1)

        eef_pos  = obs["robot0_eef_pos"].astype(np.float32)
        eef_quat = obs["robot0_eef_quat"].astype(np.float32)
        q = eef_quat.copy(); q[3] = np.clip(q[3], -1.0, 1.0)
        den = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
        axisangle = (np.zeros(3) if math.isclose(den, 0.0)
                     else (q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)
        gripper   = obs["robot0_gripper_qpos"].astype(np.float32)
        proprio   = np.concatenate([eef_pos, axisangle, gripper])
        sin_s = np.sin(proprio[None]); cos_s = np.cos(proprio[None])
        state_enc = np.stack([sin_s, cos_s], axis=-1).reshape(1, -1).astype(np.float32)
        pad = max_state_dim - state_enc.shape[-1]
        if pad > 0:
            state_enc = np.pad(state_enc, ((0, 0), (0, pad)), "constant")

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bi = model.backbone_interface.build_cosmos_inputs(
                    images=[[concat_img]], instructions=[args.prompt])
                bout = model.backbone_interface(
                    **bi, output_hidden_states=True, output_attentions=False, return_dict=True)
                vl_embs = bout.hidden_states[-1]

            state_t = torch.from_numpy(state_enc).unsqueeze(0).to(device=device, dtype=vl_embs.dtype)
            with torch.autocast("cuda", dtype=torch.float32):
                rt.reset_chunk()
                norm_actions = run_denoising_loop(
                    action_model, vl_embs, state_t, seed=seed, num_steps=sampling_steps)

        norm_np = norm_actions[0].cpu().numpy()
        norm7 = np.clip(norm_np[:, :7], -1.0, 1.0)
        raw = np.where(action_mask, 0.5 * (norm7 + 1.0) * (action_high - action_low) + action_low, norm7)
        raw[:, 6] = np.where(norm_np[:, 6] < 0.5, -1.0, 1.0)
        return [raw[i] for i in range(NUM_OPEN_LOOP)]

    def save_video(frames, path, fps):
        writer = imageio.get_writer(str(path), fps=fps)
        for frame in frames:
            writer.append_data(np.flipud(frame))
        writer.close()

    def rollout(ep_idx, init_state, *, steered: bool):
        env.reset()
        obs = env.set_init_state(init_state)
        terminated_early = False
        for _ in range(args.num_steps_wait):
            obs, _, terminated_early, _ = env.step(LIBERO_DUMMY_ACTION)
            if terminated_early:
                break

        rng = np.random.default_rng(seed=args.noise_seed_base + ep_idx) if args.noise_sigma > 0 else None
        rt.steering_enabled = bool(steered)
        rt.reset_chunk()
        rt.u_norm_log.clear()

        queue: deque = deque(maxlen=NUM_OPEN_LOOP)
        frames = [obs["agentview_image"].copy()]
        success = False
        t = 0
        while not terminated_early and t < max_env_steps:
            if not queue:
                actions = policy_fn(obs, rng if steered else None, steered)
                for a in actions:
                    queue.append(np.asarray(a, dtype=np.float32))
            a = queue.popleft()
            obs, _, done, _ = env.step(a.tolist())
            frames.append(obs["agentview_image"].copy())
            if done:
                success = True
                break
            t += 1
        return success, t + args.num_steps_wait, frames

    my_episodes = list(range(rank, args.n_episodes, world_size))
    print(f"[rank {rank}/{world_size}] handling {len(my_episodes)} episodes: {my_episodes}")

    results = []
    baseline_dir = out_dir / "baseline"
    if args.run_baseline and args.save_video:
        baseline_dir.mkdir(parents=True, exist_ok=True)

    def _do_run(ep, label):
        steered = (label == "steered")
        suffix  = "" if steered else "__baseline"
        t0 = time.time()
        success, env_steps, frames = rollout(ep, init_states[ep], steered=steered)
        dt  = time.time() - t0
        tag = "SUCCESS" if success else "FAILURE"
        video_path = None
        if args.save_video:
            vdir = baseline_dir if not steered else out_dir
            video_path = vdir / f"ep{ep:02d}--{tag}{suffix}.mp4"
            save_video(frames, video_path, fps=args.video_fps)
        print(f"[ep {ep:2d}] {label:8s} {tag:7s}  steps={env_steps:4d}  {dt:6.1f}s", flush=True)
        return {"success": bool(success), "env_steps": int(env_steps),
                "wall_time_s": float(dt), "video_path": str(video_path) if video_path else None}

    t_total = time.time()
    for ep in my_episodes:
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
            "lqr": {"lambda": float(args.lambda_scale), "Q_SCALE": float(args.q_scale),
                    "R_SCALE": float(args.r_scale), "QF_SCALE": float(args.qf_scale)},
            "noise": {"sigma": float(args.noise_sigma), "seed_base": int(args.noise_seed_base),
                      "apply_to": ["primary_image", "wrist_image"],
                      "note": "noise applied at every inference step (persistent, not one-time)"},
            "rollout": {"suite": args.suite, "task_id": int(args.task_id),
                        "task_desc_libero": task_desc_libero, "policy_prompt": args.prompt,
                        "n_episodes": int(args.n_episodes), "max_env_steps": int(max_env_steps),
                        "run_baseline": bool(args.run_baseline),
                        "resolution": int(args.resolution), "seed": int(seed)},
            "svd": {"svd_dir": str(svd_dir), "jac_dir_act": args.jac_dir_act,
                    "jac_prompt": jac_prompt, "selected_timesteps": sel_t, "L": L, "k_target": r},
            "model": {"ckpt_path": args.ckpt_path, "L_blocks": n_blocks},
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    n_succ = sum(r["success"] for r in results)
    print(f"[rank {rank}] DONE: {n_succ}/{len(results)} steered; wall {time.time()-t_total:.1f}s")


def main():
    args = parse_args()
    if args.phase == "merge":
        run_merge_phase(args)
    else:
        run_rollout_phase(args)


if __name__ == "__main__":
    main()
