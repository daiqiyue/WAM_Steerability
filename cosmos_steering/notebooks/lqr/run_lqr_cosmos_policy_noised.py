#!/usr/bin/env python
"""Activation-LQR (A-LQR) on Cosmos-Policy-LIBERO-Predict2-2B under
Gaussian-noised camera input.

Mirrors notebooks/lqr/run_lqr_cosmos_policy.ipynb (full steering, no ablation
gates) with one change to the observation pipeline:

1. The first inference of episode 0 uses the *negative* observation from
   `<pair-dir>/negative.npz` at row `--obs-index` (default 0). This is
   identical to the input fed to `compute_jacobians_full.py` /
   `compute_jacobians_text.py` when the jacobians under `--jac-dir-act` were
   collected -- so the LQR linearization is taken at exactly this point.

2. Every subsequent policy inference uses `prepare_observation(env_obs, ...)`
   followed by additive Gaussian pixel noise on `primary_image` and
   `wrist_image` (σ = `--noise-sigma`, clip to [0,255], uint8). The proprio
   from the env is left untouched -- noise was only applied to images in the
   data-collection notebook, and we match that.

Within each episode the noise RNG is seeded with
`--noise-seed-base + episode_idx` (when `--noise-per-episode-seed`), matching
the per-episode seeding convention used by
`notebooks/lqr/inputs/collect_policy_inputs_noise.ipynb` and the
`noise_extreme` rollouts under `notebooks/stress_test/rollouts/`.

N_EPISODES > 1
--------------
Only episode 0's first chunk uses the negative-npz override (that's the only
chunk for which the LQR linearization is exact). Episodes > 0 step from their
own init_states and noise every chunk's env obs from chunk 0 onward.

Parallelism (--world-size / --rank / --phase)
---------------------------------------------
Pass --world-size W and --rank R (0 <= R < W) to split the N episodes across
W independent workers (one GPU each). Each rank handles the episode slice
`range(rank, n_episodes, world_size)` -- same striped scheme as the
gripper-xyz sweep. Per-rank outputs:
    <out_dir>/results_rank{R}.json
After all ranks finish, run with --phase merge (no GPU) to aggregate the
per-rank results into results.json and patch the top-level manifest.json.

Outputs (under --out-dir)
-------------------------
    ep{i:02d}--{SUCCESS|FAILURE}.mp4                    steered agentview
    ep{i:02d}--{SUCCESS|FAILURE}__noised.mp4            steered policy-seen view
                                                        (primary | wrist, post-
                                                        noise), held across each
                                                        chunk's open-loop steps
                                                        so it lines up frame-
                                                        for-frame with the
                                                        agentview video
    ep{i:02d}--{SUCCESS|FAILURE}__baseline.mp4          unsteered agentview
                                                        (same init_state + noise
                                                        seed as steered run;
                                                        only present when
                                                        --run-baseline)
    ep{i:02d}--{SUCCESS|FAILURE}__baseline__noised.mp4  unsteered policy-seen
                                                        view
    results_rank{R}.json                                per-rank rollouts
    results.json                                        merged (after --phase merge)
    manifest.json                                       full config + noise spec
                                                        + svd_dir
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path


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

import imageio
import numpy as np
import torch


# ====================================================================
# CLI
# ====================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ---- SVD / jacobian inputs --------------------------------------
    ap.add_argument("--svd-dir", type=Path, required=True,
                    help="SVD output dir (contains config.json, "
                         "svd_summary.pt, V_part*_t*.pt)")
    ap.add_argument("--jac-dir-act", type=str, required=True,
                    help="A_tilde output subdir under --svd-dir; must "
                         "contain A_tilde__full.pt")

    # ---- Negative-input override (matches jacobian collection) ------
    ap.add_argument("--pair-dir", type=Path, required=True,
                    help="dir containing positive.npz / negative.npz produced "
                         "by collect_policy_inputs_noise.ipynb. negative.npz "
                         "is the source of the override observation for "
                         "ep0/chunk0.")
    ap.add_argument("--obs-index", type=int, default=0,
                    help="row in negative.npz to use for ep0/chunk0 override "
                         "(must match the jacobian collection's --obs-index)")
    ap.add_argument("--no-override-first-chunk", action="store_true",
                    help="disable the ep0/chunk0 override and noise from the "
                         "very first inference. Default: override enabled.")

    # ---- Noise on subsequent env observations -----------------------
    ap.add_argument("--noise-sigma", type=float, default=90.0,
                    help="Gaussian noise σ in uint8 units (default 90.0 = "
                         "noise_extreme preset)")
    ap.add_argument("--noise-per-episode-seed", dest="noise_per_episode_seed",
                    action="store_true", default=True,
                    help="seed noise RNG with (--noise-seed-base + episode) "
                         "per episode (default true; matches data collection)")
    ap.add_argument("--no-noise-per-episode-seed",
                    dest="noise_per_episode_seed", action="store_false")
    ap.add_argument("--noise-seed-base", type=int, default=0)

    # ---- LQR cost hyperparameters -----------------------------------
    ap.add_argument("--lambda-scale", type=float, default=1.0,
                    dest="lambda_scale", help="A-LQR setpoint scale (LAMBDA)")
    ap.add_argument("--q-scale",  type=float, default=10000.0, dest="q_scale")
    # --r-scale is the *initial* R_SCALE at chunk 0. It grows exponentially with
    # chunk index toward --r-scale-final with time constant --r-scale-tau (in
    # chunks). Matches run_lqr_decay_cosmos_policy.ipynb's schedule:
    #   R(c) = min(R_FINAL, R * exp(c / R_TAU))
    ap.add_argument("--r-scale",  type=float, default=75000.0, dest="r_scale",
                    help="initial LQR control cost at chunk 0; grows toward "
                         "--r-scale-final with time constant --r-scale-tau")
    ap.add_argument("--r-scale-tau", type=float, default=3.0, dest="r_scale_tau",
                    help="exponential growth time constant of R_SCALE in chunks")
    ap.add_argument("--r-scale-final", type=float, default=1e9,
                    dest="r_scale_final",
                    help="upper clamp on R_SCALE (saturates to ~no steering)")
    ap.add_argument("--max-chunks", type=int, default=50, dest="max_chunks",
                    help="precompute per-chunk K matrices for c in "
                         "[0, MAX_CHUNKS); chunks beyond clamp to the last "
                         "(saturated) entry")
    ap.add_argument("--qf-scale", type=float, default=1.0,     dest="qf_scale")

    # ---- Rollout ----------------------------------------------------
    ap.add_argument("--prompt", type=str, required=True,
                    help="task description sent to the policy")
    ap.add_argument("--n-episodes", type=int, default=1,
                    help="number of rollouts. Only ep0/chunk0 takes the "
                         "negative-npz override; episodes > 0 noise every "
                         "chunk from the start.")
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--num-steps-wait", type=int, default=10,
                    help="no-op env steps at episode start so the cached "
                         "agentview observable refreshes (matches "
                         "collect_policy_inputs_noise.ipynb)")
    ap.add_argument("--max-env-steps", type=int, default=1000,
                    dest="max_env_steps",
                    help="cap on env steps per rollout. Overrides the suite's "
                         "TASK_MAX_STEPS (libero_10=520) so the policy has "
                         "headroom under heavy noise + steering. (default 1000)")
    ap.add_argument("--run-baseline", dest="run_baseline",
                    action="store_true", default=True,
                    help="also run an unsteered baseline rollout per episode "
                         "from the same init_state, with the same noise seed "
                         "and override obs, with the LQR hooks gated off "
                         "(default true)")
    ap.add_argument("--no-baseline", dest="run_baseline",
                    action="store_false",
                    help="disable the per-episode baseline run")
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
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    ap.add_argument("--tag", type=str, default=None,
                    help="optional grouping tag recorded in manifest.json")

    # ---- Parallelism (sbatch array) ---------------------------------
    ap.add_argument("--phase", choices=["rollout", "merge"], default="rollout",
                    help="'rollout' (default) runs the per-rank rollout. "
                         "'merge' reads results_rank*.json under --out-dir and "
                         "writes results.json + patches manifest.json. No GPU "
                         "needed.")
    ap.add_argument("--world-size", type=int, default=1, dest="world_size",
                    help="number of parallel ranks (workers) covering "
                         "--n-episodes. Each rank handles episodes "
                         "range(rank, n_episodes, world_size).")
    ap.add_argument("--rank", type=int, default=0,
                    help="this worker's rank in [0, world_size).")
    ap.add_argument("--vcache-max-gpu-tiles", type=int, default=None,
                    dest="vcache_max_gpu_tiles",
                    help="cap on the number of V tiles held GPU-resident at "
                         "once (LRU evicts beyond it). Default (None) keeps "
                         "all P*T tiles on GPU, which needs ~30 GB on top of "
                         "the model — fits 80 GB H100/GH200 but OOMs on "
                         "smaller GPUs. Set to ~num_layers (e.g. 28) on a "
                         "47 GB RTX 6000 Ada.")

    return ap.parse_args()


# ====================================================================
# V-tile cache (per (partition, timestep))
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
# Chained Riccati
# ====================================================================

def chained_riccati_per_chunk(A_tilde, B_tilde, q_scale, r_scale_schedule,
                              qf_scale, device):
    """Solve the chained Riccati once per `r_scale_schedule` entry, reusing the
    shared `A_chain` and `Q_chain`. Returns per-chunk K stacks:
        K_intra_per_chunk : (n_chunks, T_diff, L-1, r, r)  fp32, CPU
        K_step_per_chunk  : (n_chunks, max(T_diff-1, 0), r, r)  fp32, CPU
    Mirrors the schedule logic in run_lqr_decay_cosmos_policy.ipynb (Section
    2.1)."""
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
# Steering runtime + hooks (full steering, no ablation gates)
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
        # Set False to run the policy unsteered while keeping the hooks
        # registered. The apply hooks early-return; `_pass_tick` still bumps
        # pass_idx so the runtime stays internally consistent if you flip
        # this back on mid-run.
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
# Noise helper
# ====================================================================

class EpisodeNoise:
    """Per-episode-seeded Gaussian pixel noise. Matches
    `ImageGaussianNoise(sigma, per_episode_seed=True)` and the noise applied
    in `collect_policy_inputs_noise.ipynb`."""

    def __init__(self, sigma, per_episode_seed=True, seed_base=0):
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
# Rollout helpers
# ====================================================================

def save_video(frames, path, fps, flip_ud=True):
    """Write `frames` (HWC uint8) to `path` at `fps`. `flip_ud` controls the
    LIBERO agentview upside-down compensation; set False for frames that are
    already right-side-up (e.g. the policy-seen view post-`prepare_observation`,
    which is already flipped by `flip_images=True`)."""
    writer = imageio.get_writer(str(path), fps=fps)
    for frame in frames:
        writer.append_data(np.flipud(frame) if flip_ud else frame)
    writer.close()


def stack_policy_view(observation):
    """Horizontally concatenate the policy's primary + wrist images for video
    recording. Both come out of `prepare_observation` (or the negative.npz
    override) as HWC uint8 at COSMOS_IMAGE_SIZE, so a plain `np.hstack` is
    enough. Returns a contiguous HWC uint8 array."""
    prim = np.ascontiguousarray(observation["primary_image"])
    wrst = np.ascontiguousarray(observation["wrist_image"])
    return np.hstack([prim, wrst])


def load_override_obs(pair_dir: Path, obs_index: int) -> dict:
    """Read the (obs-index)th row of <pair-dir>/negative.npz and return it as
    a `prepare_observation`-compatible dict (the same dict shape that
    `get_action` consumes)."""
    neg_npz = pair_dir / "negative.npz"
    if not neg_npz.exists():
        raise FileNotFoundError(f"negative.npz not found at {neg_npz}")
    z = np.load(neg_npz)
    n = int(z["proprios"].shape[0])
    if not (0 <= obs_index < n):
        raise ValueError(
            f"--obs-index {obs_index} out of range for negative.npz "
            f"({n} rows)"
        )
    return {
        "primary_image": z["primary_images"][obs_index].copy(),
        "wrist_image":   z["wrist_images"][obs_index].copy(),
        "proprio":       z["proprios"][obs_index].astype(np.float32).copy(),
    }


# ====================================================================
# Main
# ====================================================================

# ====================================================================
# Merge phase (no GPU; aggregates results_rank*.json)
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

    rank = int(args.rank)
    world_size = int(args.world_size)
    assert world_size >= 1, f"--world-size must be >= 1; got {world_size}"
    assert 0 <= rank < world_size, (
        f"--rank {rank} not in [0, --world-size {world_size})"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_id = device.index if device.index is not None else 0
    print(f"[rank {rank}/{world_size}] starting", flush=True)

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

    print(f"[svd] {svd_dir}")
    print(f"[svd] sel_t={sel_t}  T_diff={T_diff}  L={L}  r={r}  "
          f"sampling_steps={sampling_steps}")
    print(f"[svd] partitions={partitions}  D={D}  D_flat={D_flat:,}")

    # ---------------------------------------------------------------- A_tilde
    jac_dir = svd_dir / args.jac_dir_act
    a_tilde_full = jac_dir / "A_tilde__full.pt"
    if not a_tilde_full.exists():
        raise FileNotFoundError(f"A_tilde missing: {a_tilde_full}")
    raw = torch.load(a_tilde_full, map_location="cpu", weights_only=False)
    A_dict = raw.get("A_tilde", {})
    B_dict = raw.get("B_tilde", {})
    jac_prompt = raw.get("prompt", "<unknown>")
    print(f"[jac] A_tilde dict: {len(A_dict)} entries (expected {T_diff*(L-1)})  "
          f"B_tilde dict: {len(B_dict)} entries  prompt={jac_prompt!r}")

    sel_idx_of = {t: i for i, t in enumerate(sel_t)}
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
        print("[jac] B_tilde empty (cosmos-policy default); chained Riccati "
              "degrades to per-step (zero cross-step A matrices).")
        B_tilde = torch.zeros(max(T_diff - 1, 0), r, r, dtype=torch.float32)

    # ---------------------------------------------------------------- LFS
    summary = torch.load(svd_dir / "svd_summary.pt", map_location="cpu",
                         weights_only=False)
    c_means = summary["c_means"].float()      # (L, T_sel, k)
    assert c_means.dim() == 3, (
        f"expected c_means shape (L, T_sel, k); got {tuple(c_means.shape)}"
    )
    tilde_mu = c_means.norm(dim=-1)                                # (L, T_sel)
    tilde_v  = c_means / tilde_mu.unsqueeze(-1).clamp(min=1e-12)   # (L, T_sel, k)
    layer_to_part = list(summary["layer_to_part"])

    # ---------------------------------------------------------------- LQR schedule
    # R_SCALE(c) = min(R_FINAL, R_INIT * exp(c / R_TAU)). Matches
    # run_lqr_decay_cosmos_policy.ipynb Section 2.1.
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
    _saturate_c = next(
        (c for c, r_ in enumerate(r_scale_schedule)
         if r_ >= args.r_scale_final),
        None,
    )
    print(f"[lqr] λ={args.lambda_scale:g}  Q={args.q_scale:g}  Qf={args.qf_scale:g}")
    print(f"[lqr] R_SCALE schedule: init={args.r_scale:g}  tau={args.r_scale_tau:g}  "
          f"final={args.r_scale_final:g}  max_chunks={args.max_chunks}")
    _preview = ", ".join(f"{r_:.2e}" for r_ in r_scale_schedule[:8])
    print(f"[lqr]   R_SCALE(c=0..7): {_preview} ...")
    if _saturate_c is not None:
        print(f"[lqr]   saturates at R_SCALE_FINAL starting at chunk c={_saturate_c}")
    else:
        print(f"[lqr]   does not reach R_SCALE_FINAL within max_chunks="
              f"{args.max_chunks}; last entry = {r_scale_schedule[-1]:.2e}")

    t0 = time.time()
    K_intra_per_chunk, K_step_per_chunk = chained_riccati_per_chunk(
        A_tilde, B_tilde,
        args.q_scale, r_scale_schedule, args.qf_scale, device,
    )
    print(f"[lqr] chained Riccati over {args.max_chunks} chunks done in "
          f"{time.time() - t0:.2f}s; "
          f"K_intra_per_chunk {tuple(K_intra_per_chunk.shape)}  "
          f"K_step_per_chunk {tuple(K_step_per_chunk.shape)}")
    # Chunk-0 view used to seed the GPU-resident _lqr pointers; the per-chunk
    # swap below overwrites these in place each chunk.
    K_intra = K_intra_per_chunk[0]
    K_step  = K_step_per_chunk[0]

    # ---------------------------------------------------------------- Override obs
    if args.no_override_first_chunk:
        override_obs = None
        print(f"[override] disabled; ep0/chunk0 will use a noised env obs")
    else:
        override_obs = load_override_obs(args.pair_dir, args.obs_index)
        print(f"[override] ep0/chunk0 will use "
              f"{args.pair_dir / 'negative.npz'} row {args.obs_index} "
              f"(primary {override_obs['primary_image'].shape} uint8, "
              f"wrist {override_obs['wrist_image'].shape} uint8, "
              f"proprio {override_obs['proprio'].shape} float32)")

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

    print(f"[model] loading dataset stats + T5 cache + weights on cuda:{device_id} ...")
    dataset_stats = load_dataset_stats(eval_cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(eval_cfg.t5_text_embeddings_path, worker_id=rank)
    model, _ = get_model(eval_cfg)
    n_blocks = len(model.net.blocks)
    if n_blocks != L:
        raise RuntimeError(f"model has {n_blocks} DiT blocks; SVD config L={L}")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free_gb, total_gb = (x / 1e9 for x in torch.cuda.mem_get_info(device_id))
        print(f"[model] after loader cleanup: GPU {free_gb:.1f} / {total_gb:.1f} GB free")

    # ---------------------------------------------------------------- V cache
    vcache = VCache(svd_dir, partitions, layer_to_part, sel_t, r,
                    device=device, dtype=torch.bfloat16,
                    max_gpu_tiles=getattr(args, "vcache_max_gpu_tiles", None))
    print(f"[v] preloading all {len(partitions)} × {len(sel_t)} V tiles ...")
    vcache.preload([(p, t) for p in range(len(partitions)) for t in sel_t])
    vcache.for_layer(0, sel_t[0])

    # ---------------------------------------------------------------- LQR push
    # GPU-resident K_intra/K_step start as the chunk-0 slice; `_swap_K_for_chunk`
    # overwrites them in place at chunk boundaries during rollout.
    lqr = {
        "K_intra": K_intra.to(device=device, dtype=torch.float32),
        "K_step":  K_step.to(device=device, dtype=torch.float32) if K_step.numel() else K_step,
        "v":       tilde_v.to(device=device, dtype=torch.float32),
        "mu":      tilde_mu.to(device=device, dtype=torch.float32),
    }
    total_mb = sum(t.element_size() * t.numel() for t in lqr.values()) / 1e6
    print(f"[lqr] pushed to {device} ({total_mb:.1f} MB, chunk-0 K slice; "
          f"per-chunk swap on CPU is "
          f"{(K_intra_per_chunk.element_size() * K_intra_per_chunk.numel() + K_step_per_chunk.element_size() * K_step_per_chunk.numel()) / 1e6:.1f} MB)")

    def _swap_K_for_chunk(c: int) -> None:
        """Swap `lqr["K_intra"]` / `lqr["K_step"]` to the precomputed slice for
        chunk index `c`. Chunks beyond `max_chunks-1` clamp to the last
        (saturated) entry, which already has R_SCALE at R_SCALE_FINAL so the
        gains are effectively zero -- the policy runs ~unsteered."""
        c_eff = min(c, args.max_chunks - 1)
        lqr["K_intra"].copy_(
            K_intra_per_chunk[c_eff].to(device=device, dtype=torch.float32)
        )
        if K_step_per_chunk.numel():
            lqr["K_step"].copy_(
                K_step_per_chunk[c_eff].to(device=device, dtype=torch.float32)
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

    # ---------------------------------------------------------------- Noise
    noise = EpisodeNoise(
        sigma=args.noise_sigma,
        per_episode_seed=args.noise_per_episode_seed,
        seed_base=args.noise_seed_base,
    )
    print(f"[noise] σ={args.noise_sigma}  per_episode_seed="
          f"{args.noise_per_episode_seed}  seed_base={args.noise_seed_base}")

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
    # robosuite raises "executing action in terminated episode" once it has
    # taken `horizon` steps. We call env.step up to `num_steps_wait` (no-op
    # warm-up) + `max_env_steps` (rollout loop) times per episode, so bump the
    # env's horizon past that with a small buffer.
    env_horizon = args.num_steps_wait + max_env_steps + 10
    env, task_desc_libero = get_libero_env(task, "cosmos",
                                            resolution=args.resolution,
                                            horizon=env_horizon)
    print(f"[env] {args.suite} task {args.task_id:02d}  "
          f"task_desc={task_desc_libero!r}  max_steps={max_env_steps} "
          f"(--max-env-steps; suite TASK_MAX_STEPS={suite_max_steps}; "
          f"env horizon={env_horizon})")
    if args.prompt != task_desc_libero:
        print(f"      note: --prompt differs from LIBERO task description; "
              f"success is tracked against the LIBERO predicate, not --prompt.")

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

    def rollout(ep_idx, init_state, task_desc, *, steered: bool):
        """Run one episode. When `steered=False`, the LQR hooks early-return
        and the policy runs unsteered while still consuming the exact same
        noised observations (and ep0/chunk0 override) as the steered run --
        so steered vs. baseline is a direct comparison."""
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(
                get_libero_dummy_action(eval_cfg.model_family)
            )

        # Reseed noise so the steered and baseline runs of episode `ep_idx`
        # see the same noise draws on every chunk (`per_episode_seed` keys
        # on ep_idx).
        noise.reset_episode(ep_idx)
        rt.steering_enabled = bool(steered)
        rt.reset_chunk()
        rt.u_norm_log.clear()

        queue = deque(maxlen=eval_cfg.num_open_loop_steps)
        frames = [obs["agentview_image"].copy()]
        # `noised_frames` holds the policy-seen view (primary | wrist) at each
        # env step. We refresh `last_noised_view` at every chunk boundary and
        # reuse it for every env step until the next chunk, so the noised
        # video lines up frame-for-frame with the clean agentview video.
        noised_frames = []
        last_noised_view = None
        chunk_idx = 0
        success = False
        t = 0
        used_override = False
        while t < max_env_steps:
            if not queue:
                if ep_idx == 0 and chunk_idx == 0 and override_obs is not None:
                    # First inference of episode 0: use the negative-npz
                    # observation that the jacobians were taken at. The env's
                    # actual state at this point matches the *positive* obs
                    # (clean) -- we deliberately substitute the noised view to
                    # match the linearization point.
                    observation = {
                        "primary_image": override_obs["primary_image"].copy(),
                        "wrist_image":   override_obs["wrist_image"].copy(),
                        "proprio":       override_obs["proprio"].copy(),
                    }
                    used_override = True
                else:
                    observation = prepare_observation(
                        obs, resize_size=COSMOS_IMAGE_SIZE,
                        flip_images=eval_cfg.flip_images,
                    )
                    observation["primary_image"] = noise.apply(
                        observation["primary_image"]
                    )
                    observation["wrist_image"] = noise.apply(
                        observation["wrist_image"]
                    )
                last_noised_view = stack_policy_view(observation)
                if not noised_frames:
                    # Seed the noised video with the first chunk's view so it
                    # has the same initial frame count as `frames` (which
                    # already includes obs["agentview_image"] from before the
                    # first env.step).
                    noised_frames.append(last_noised_view.copy())
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
            noised_frames.append(last_noised_view.copy())
            if done:
                success = True
                break
            t += 1
        return (success, t + args.num_steps_wait, frames, noised_frames,
                chunk_idx, used_override)

    results = []
    t_total = time.time()
    def _do_run(ep, label):
        """Run one rollout pass (steered or baseline) and write its videos.
        Returns a dict matching the per-pass schema in `results.json`."""
        steered = (label == "steered")
        suffix = "" if steered else "__baseline"
        t0 = time.time()
        (success, env_steps, frames, noised_frames, n_chunks,
         used_override) = rollout(ep, init_states[ep], args.prompt,
                                  steered=steered)
        dt = time.time() - t0
        tag = "SUCCESS" if success else "FAILURE"
        video_path = None
        noised_video_path = None
        if args.save_video:
            video_path = args.out_dir / f"ep{ep:02d}--{tag}{suffix}.mp4"
            save_video(frames, video_path, fps=args.video_fps)
            if noised_frames:
                noised_video_path = (
                    args.out_dir / f"ep{ep:02d}--{tag}{suffix}__noised.mp4"
                )
                # Policy-seen frames are already flipped by
                # `prepare_observation`, so don't re-flip them.
                save_video(noised_frames, noised_video_path,
                           fps=args.video_fps, flip_ud=False)
        noised_name = (noised_video_path.name
                       if noised_video_path else "(no noised video)")
        print(f"[ep {ep:2d}] {label:8s} {tag:7s}  steps={env_steps:4d}  "
              f"chunks={n_chunks:3d}  override={'Y' if used_override else 'N'}  "
              f"{dt:6.1f}s  "
              f"-> {video_path.name if video_path else '(no video)'} "
              f"+ {noised_name}", flush=True)
        return {
            "success":            bool(success),
            "env_steps":          int(env_steps),
            "n_chunks":           int(n_chunks),
            "used_override":      bool(used_override),
            "wall_time_s":        float(dt),
            "video_path":         str(video_path) if video_path else None,
            "noised_video_path":  (str(noised_video_path)
                                   if noised_video_path else None),
        }

    # Per-rank slice: every world_size-th episode starting at rank. With the
    # default world_size=1 this collapses to the original `range(n_episodes)`.
    my_episodes = list(range(rank, args.n_episodes, world_size))
    print(f"[rank {rank}/{world_size}] handling {len(my_episodes)} episodes: "
          f"{my_episodes}", flush=True)

    for ep in my_episodes:
        if args.baseline_only:
            # Unsteered-only: put baseline fields at the top level so the
            # aggregator's success counter (which reads r["success"]) counts
            # baseline successes. The nested baseline slot stays null.
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
            # Keep the steered-run fields at the top level for back-compat with
            # downstream consumers; nest the baseline pass under `baseline`.
            results.append({
                "episode":            ep,
                "rank":               rank,
                **steered_rec,
                "baseline":           baseline_rec,
            })

    for h in handles:
        h.remove()
    env.close()

    # ---------------------------------------------------------------- Per-rank results
    results_path = args.out_dir / f"results_rank{rank}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[rank {rank}] wrote {results_path}", flush=True)

    # ---------------------------------------------------------------- Manifest (rank 0)
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
            "override": {
                "enabled":   not args.no_override_first_chunk,
                "pair_dir":  str(args.pair_dir),
                "obs_index": int(args.obs_index),
                "applied_to": ("ep0/chunk0 only"
                               if not args.no_override_first_chunk else None),
            },
            "noise": {
                "sigma": float(args.noise_sigma),
                "per_episode_seed": bool(args.noise_per_episode_seed),
                "seed_base": int(args.noise_seed_base),
                "applied_to": ["primary_image", "wrist_image"],
                "applied_when": ("every env-derived obs (every chunk after the "
                                 "ep0/chunk0 override; every chunk in episodes>0)"),
                "clip_range": [0, 255],
                "output_dtype": "uint8",
                "matches_contrastive_seed": (
                    bool(args.noise_per_episode_seed)
                    and int(args.noise_seed_base) == 0
                ),
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
            "svd": {
                "svd_dir": str(svd_dir),
                "jac_dir_act": args.jac_dir_act,
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
            "v_cache_stats": dict(vcache.stats),
        }
        (args.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        print(f"[rank 0] wrote manifest -> {args.out_dir / 'manifest.json'}",
              flush=True)

    n_succ = sum(r["success"] for r in results)
    print()
    print(f"[rank {rank}] DONE: steered {n_succ}/{len(results)} succeeded "
          f"({100 * n_succ / max(1, len(results)):.0f}%)  "
          f"total {time.time() - t_total:.1f}s")
    if args.run_baseline:
        n_base = sum(
            (r["baseline"] or {}).get("success", False) for r in results
        )
        print(f"[rank {rank}] baseline: {n_base}/{len(results)} succeeded "
              f"({100 * n_base / max(1, len(results)):.0f}%)")
    print(f"[rank {rank}] V cache: hits={vcache.stats['gpu_hits']} "
          f"swaps={vcache.stats['gpu_swaps']} "
          f"cpu->gpu={vcache.stats['cpu_to_gpu_s']:.1f}s")
    if args.save_video:
        print(f"[rank {rank}] videos: {args.out_dir.resolve()}")


def main():
    args = parse_args()
    if args.phase == "merge":
        run_merge_phase(args)
    else:
        run_rollout_phase(args)


if __name__ == "__main__":
    main()
