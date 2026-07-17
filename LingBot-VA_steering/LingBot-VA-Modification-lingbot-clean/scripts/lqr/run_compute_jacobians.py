"""Projected Jacobian collection for LingBot LQR (repo-root-aligned VJP path).

Computes within-step projected Jacobians on a real LIBERO observation:

  A_tilde[(t, l_in)] = V_{l_in+1, t}^T  J_{block_{l_in+1}}(z_{t, l_in})  V_{l_in, t}

by running the matching LingBot denoising loop once and applying autograd VJPs
at selected transformer blocks. Action-mode SVDs use the repo-root-aligned action
loop; video-mode SVDs use the video denoising loop.

B_tilde (cross-step) is skipped by default, consistent with Cosmos-Policy.

Usage (single GPU)::

    python scripts/lqr/run_compute_jacobians.py \\
        --svd-dir outputs/lqr/svd_init_pos \\
        --inputs-npz outputs/lqr/pairs_init_pos_paired/negative.npz \\
        --out-subdir A_tilde_lingbot

Sharded worker / merge (repo-root-style)::

    python ... --phase worker --num-shards 4 --rank 0
    python ... --phase merge  --num-shards 4

Legacy ridge fit on SVD projected diffs (deprecated)::

    python ... --method ridge --ridge 1e-3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch


def _default_master_port() -> str:
    return str(12355 + (os.getpid() % 1000))


def _ensure_dist_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _default_master_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def _build_server(config_name: str):
    import torch.distributed as dist
    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.wan_va_server import VA_Server

    _ensure_dist_env()
    if not dist.is_initialized():
        init_distributed(world_size=1, local_rank=0, rank=0)
    cfg = copy.deepcopy(VA_CONFIGS[config_name])
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    # VJP autograd through blocks is incompatible with FSDP DTensor mixing.
    cfg.lqr_disable_fsdp = True
    return VA_Server(cfg)


def _obs_from_npz(npz_obj, idx: int, cam_keys: List[str]) -> Dict:
    cam0 = npz_obj["primary_images"][idx]
    cam1 = npz_obj["wrist_images"][idx]
    return {"obs": [{cam_keys[0]: np.ascontiguousarray(cam0), cam_keys[1]: np.ascontiguousarray(cam1)}]}


def _v_filename(svd_dir: Path, p_idx: int, t_id: int, partitions, k_target: int) -> Path:
    a, b = partitions[p_idx]
    pattern = f"V_part{p_idx}_layers{a}-{b}_t{t_id}_k*.pt"
    matches = sorted(svd_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no V file matching {pattern} in {svd_dir}")
    preferred = [m for m in matches if m.name.endswith(f"_k{k_target}.pt")]
    return (preferred or matches)[0]


def make_layer_shards(num_l_in: int, num_shards: int) -> List[List[int]]:
    chunks: List[List[int]] = []
    base, extra = divmod(num_l_in, num_shards)
    start = 0
    for i in range(num_shards):
        size = base + (1 if i < extra else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


def _to_local_tensor(t: torch.Tensor) -> torch.Tensor:
    if hasattr(t, "to_local"):
        try:
            t = t.to_local()
        except Exception:
            pass
    return t.detach().clone()


def _to_normal(t):
    if torch.is_tensor(t):
        return _to_local_tensor(t)
    if isinstance(t, (tuple, list)):
        return type(t)(_to_normal(x) for x in t)
    return t


def projected_jacobian(
    block_fn,
    z: torch.Tensor,
    V_in: torch.Tensor,
    V_out: torch.Tensor,
    mode: str = "vjp_no_retain",
    verbose: bool = True,
    log_every: int = 25,
) -> torch.Tensor:
    z = z.detach()
    z_dev, z_dt = z.device, z.dtype
    r_in, r_out = V_in.shape[1], V_out.shape[1]
    d_in, d_out = V_in.shape[0], V_out.shape[0]
    out_dtype = torch.float32

    def col(V, j):
        return V[:, j].contiguous().to(device=z_dev, dtype=z_dt)

    def to_out(t, device):
        return t.to(device=device, dtype=out_dtype)

    def _sync():
        if z_dev.type == "cuda":
            torch.cuda.synchronize(z_dev)

    def _log_iter(i, total, t0):
        if not verbose:
            return
        if (i + 1) % log_every != 0 and (i + 1) != total:
            return
        _sync()
        el = time.time() - t0
        rate = (i + 1) / el if el > 0 else 0.0
        eta = (total - i - 1) / rate if rate > 0 else float("inf")
        print(f"      [{i + 1:3d}/{total}]  {el:6.1f}s  {rate:5.2f} it/s  ETA {eta:6.1f}s", flush=True)

    if mode == "jvp":
        with torch.enable_grad():
            M = torch.empty(r_in, d_out, dtype=out_dtype, device=V_out.device)
            t0 = time.time()
            for j in range(r_in):
                _, jvp = torch.func.jvp(block_fn, (z,), (col(V_in, j),))
                M[j].copy_(to_out(jvp, V_out.device))
                _log_iter(j, r_in, t0)
        return (M @ V_out.to(out_dtype)).T.contiguous()

    if mode == "vjp":
        with torch.enable_grad():
            z_g = z.clone().requires_grad_(True)
            out = block_fn(z_g)
            G = torch.empty(r_out, d_in, dtype=out_dtype, device=V_in.device)
            t0 = time.time()
            for i in range(r_out):
                (g,) = torch.autograd.grad(
                    out,
                    z_g,
                    col(V_out, i),
                    retain_graph=(i < r_out - 1),
                )
                G[i].copy_(to_out(g, V_in.device))
                _log_iter(i, r_out, t0)
        return G @ V_in.to(out_dtype)

    if mode == "vjp_no_retain":
        with torch.enable_grad():
            G = torch.empty(r_out, d_in, dtype=out_dtype, device=V_in.device)
            t0 = time.time()
            for i in range(r_out):
                z_g = z.clone().requires_grad_(True)
                out = block_fn(z_g)
                (g,) = torch.autograd.grad(out, z_g, col(V_out, i))
                G[i].copy_(to_out(g, V_in.device))
                del out, z_g, g
                _log_iter(i, r_out, t0)
        return G @ V_in.to(out_dtype)

    raise ValueError(f"unknown mode {mode!r}")


def _run_denoising_with_hooks(server, obs: Dict, hooks_state: Dict) -> None:
    """Run video + action loops from VA_Server._infer with Jacobian hooks active."""
    import torch.nn.functional as F
    from einops import rearrange
    from tqdm import tqdm
    from wan_va.utils import data_seq_to_patch
    frame_st_id = 0
    init_latent = server._encode_obs(obs)
    server.init_latent = init_latent

    latents = torch.randn(
        1,
        48,
        server.job_config.frame_chunk_size,
        server.latent_height,
        server.latent_width,
        device=server.device,
        dtype=server.dtype,
    )
    actions = torch.randn(
        1,
        server.job_config.action_dim,
        server.job_config.frame_chunk_size,
        server.action_per_frame,
        1,
        device=server.device,
        dtype=server.dtype,
    )

    video_inference_step = server.job_config.num_inference_steps
    action_inference_step = server.job_config.action_num_inference_steps
    video_step = server.job_config.video_exec_step

    server.scheduler.set_timesteps(video_inference_step)
    server.action_scheduler.set_timesteps(action_inference_step)
    timesteps = server.scheduler.timesteps
    action_timesteps = server.action_scheduler.timesteps
    timesteps = F.pad(timesteps, (0, 1), mode="constant", value=0)
    if video_step != -1:
        timesteps = timesteps[:video_step]
    action_timesteps = F.pad(action_timesteps, (0, 1), mode="constant", value=0)

    hooks_state["phase"] = "video"
    for i, t in enumerate(timesteps):
        last_step = i == len(timesteps) - 1
        latent_cond = init_latent[:, :, 0:1].to(server.dtype) if frame_st_id == 0 else None
        input_dict = server._prepare_latent_input(
            latents,
            None,
            t,
            t,
            latent_cond,
            None,
            frame_st_id=frame_st_id,
        )
        video_noise_pred = server.transformer(
            server._repeat_input_for_cfg(input_dict["latent_res_lst"]),
            update_cache=1 if last_step else 0,
            cache_name=server.cache_name,
            action_mode=False,
        )
        if not last_step or video_step != -1:
            video_noise_pred = data_seq_to_patch(
                server.job_config.patch_size,
                video_noise_pred,
                server.job_config.frame_chunk_size,
                server.latent_height,
                server.latent_width,
                batch_size=2 if server.use_cfg else 1,
            )
            if server.job_config.guidance_scale > 1:
                video_noise_pred = video_noise_pred[1:] + server.job_config.guidance_scale * (
                    video_noise_pred[:1] - video_noise_pred[1:]
                )
            else:
                video_noise_pred = video_noise_pred[:1]
            latents = server.scheduler.step(video_noise_pred, t, latents, return_dict=False)
        latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

    hooks_state["pass_idx"] = -1
    hooks_state["phase"] = "action"
    for i, t in enumerate(tqdm(action_timesteps, desc="action denoise")):
        last_step = i == len(action_timesteps) - 1
        action_cond = (
            torch.zeros(
                [1, server.job_config.action_dim, 1, server.action_per_frame, 1],
                device=server.device,
                dtype=server.dtype,
            )
            if frame_st_id == 0
            else None
        )
        input_dict = server._prepare_latent_input(
            None,
            actions,
            t,
            t,
            None,
            action_cond,
            frame_st_id=frame_st_id,
        )
        hooks_state["phase"] = "action"
        try:
            action_noise_pred = server.transformer(
                server._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=server.cache_name,
                action_mode=True,
            )
        finally:
            hooks_state["phase"] = None

        if not last_step:
            action_noise_pred = rearrange(
                action_noise_pred,
                "b (f n) c -> b c f n 1",
                f=server.job_config.frame_chunk_size,
            )
            if server.job_config.action_guidance_scale > 1:
                action_noise_pred = action_noise_pred[1:] + server.job_config.action_guidance_scale * (
                    action_noise_pred[:1] - action_noise_pred[1:]
                )
            else:
                action_noise_pred = action_noise_pred[:1]
            actions = server.action_scheduler.step(action_noise_pred, t, actions, return_dict=False)
        actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]


class _AllTargetsRecorded(Exception):
    pass


def _save_shard(
    out_path: Path,
    jac_store: Dict,
    args: argparse.Namespace,
    partitions,
    layer_to_part,
    cfg: Dict,
    L: int,
    target_tl_full: List[Tuple[int, int]],
) -> None:
    torch.save(
        {
            "A_tilde": dict(jac_store),
            "B_tilde": {},
            "rank": args.rank,
            "num_shards": args.num_shards,
            "prompt": args.prompt,
            "mode": args.mode,
            "method": "vjp",
            "V_dtype": args.v_dtype,
            "V_device": args.v_device,
            "partitions": partitions,
            "layer_to_part": layer_to_part,
            "k": cfg["k_target"],
            "L": L,
            "selected_timesteps": cfg["selected_timesteps"],
            "targets": target_tl_full,
            "inputs_npz": str(args.inputs_npz),
            "obs_index": int(args.obs_index),
        },
        out_path,
    )


def run_vjp_worker(args: argparse.Namespace) -> int:
    rank = args.rank
    num_shards = args.num_shards
    assert 0 <= rank < num_shards

    log = lambda msg: print(f"[rank {rank}/{num_shards}] {msg}", flush=True)

    cfg = json.loads((args.svd_dir / "config.json").read_text(encoding="utf-8"))
    summary = torch.load(args.svd_dir / "svd_summary.pt", map_location="cpu", weights_only=False)
    L = int(summary["c_means"].shape[0])
    sel_t = list(cfg["selected_timesteps"])
    partitions = [tuple(p) for p in cfg["partitions"]]
    layer_to_part = list(summary["layer_to_part"])
    k_target = int(cfg["k_target"])
    svd_mode = str(cfg.get("mode", "action"))
    if svd_mode not in {"action", "video", "both"}:
        raise ValueError(
            f"VJP Jacobians require SVD mode action/video/both, got {svd_mode!r}."
        )
    jac_phase = "video" if svd_mode == "video" else "action"

    num_l_in = L - 1
    layer_shards = make_layer_shards(num_l_in, num_shards)
    my_layers = layer_shards[rank]
    target_tl: Set[Tuple[int, int]] = {(int(t), int(l)) for t in sel_t for l in my_layers}
    log(f"L={L} sel_t={sel_t} my_layers={my_layers} -> {len(target_tl)} (t, l_in) pairs")
    if not target_tl:
        return 0

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_rank{rank}.pt"

    existing: Dict[Tuple[int, int], torch.Tensor] = {}
    if args.resume and out_path.exists():
        d = torch.load(out_path, map_location="cpu", weights_only=False)
        existing = dict(d.get("A_tilde", {}))
        before = len(target_tl)
        target_tl = {tl for tl in target_tl if tl not in existing}
        log(f"resume: skipped {before - len(target_tl)}; {len(target_tl)} remaining")
        if not target_tl:
            return 0

    if not args.inputs_npz.exists():
        raise FileNotFoundError(f"--inputs-npz not found: {args.inputs_npz}")
    npz = np.load(args.inputs_npz)
    n_avail = int(npz["primary_images"].shape[0])
    if not (0 <= args.obs_index < n_avail):
        raise ValueError(f"--obs-index {args.obs_index} out of range [0, {n_avail})")

    config_name = args.config_name or str(cfg.get("config_name", "libero"))
    prompt = args.prompt or str(cfg.get("prompt", ""))
    if not prompt:
        raise ValueError("Prompt required (--prompt or svd config.json prompt).")

    server = _build_server(config_name)
    log("loaded VA_Server with lqr_disable_fsdp=True (local transformer for VJP)")
    server.vae.eval().requires_grad_(False)
    server.text_encoder.eval().requires_grad_(False)
    if getattr(server, "streaming_vae_half", None) is not None:
        server.streaming_vae_half.vae.eval().requires_grad_(False)

    transformer = server.transformer
    n_blocks = len(transformer.blocks)
    if n_blocks != L:
        raise RuntimeError(f"model block count {n_blocks} != SVD L={L}")

    cam_keys = list(server.job_config.obs_cam_keys)
    obs = _obs_from_npz(npz, args.obs_index, cam_keys)
    log(
        f"obs from {args.inputs_npz.name}[{args.obs_index}]: "
        f"primary={obs['obs'][0][cam_keys[0]].shape} wrist={obs['obs'][0][cam_keys[1]].shape}"
    )

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    v_device = torch.device(args.v_device)
    v_dtype = dtype_map[args.v_dtype]
    v_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def v_for(layer_idx: int, t_id: int) -> torch.Tensor:
        p_idx = layer_to_part[layer_idx]
        key = (p_idx, t_id)
        cur = v_cache.get(key)
        if cur is not None:
            return cur
        fp = _v_filename(args.svd_dir, p_idx, t_id, partitions, k_target)
        log(f"lazy-load {fp.name}")
        vt = torch.load(fp, map_location="cpu", weights_only=False)["V"]
        vt = vt.to(device=v_device, dtype=v_dtype).contiguous()
        v_cache[key] = vt
        return vt

    hooks_state: Dict = {"pass_idx": -1, "phase": None, "target_phase": jac_phase}
    in_ad = {"flag": False}
    jac_store: Dict[Tuple[int, int], torch.Tensor] = dict(existing)
    n_target_total = len(target_tl) + len(existing)
    run_t0 = [0.0]
    checkpoint_every = 25
    target_tl_full = sorted(target_tl | set(existing.keys()))

    by_lin: Dict[int, Set[int]] = {}
    for t, l_in in target_tl:
        by_lin.setdefault(l_in, set()).add(t)

    def _pass_tick(_block, _args):
        if in_ad["flag"] or hooks_state.get("phase") != hooks_state.get("target_phase"):
            return None
        hooks_state["pass_idx"] += 1
        return None

    def make_pre_hook(l_in: int, target_steps: Set[int]):
        l_out = l_in + 1

        def hook(block_next, h_args, kwargs):
            if in_ad["flag"] or hooks_state.get("phase") != hooks_state.get("target_phase"):
                return None
            step = hooks_state["pass_idx"]
            if step not in target_steps:
                return None
            if (step, l_in) in jac_store:
                return None

            hidden_states = h_args[0]
            encoder_hidden_states = h_args[1]
            temb = h_args[2]
            rotary_emb = h_args[3]
            update_cache = int(kwargs.get("update_cache", 0))
            cache_name = str(kwargs.get("cache_name", "pos"))

            print(
                f"  [t=+{time.time() - run_t0[0]:.1f}s] hook t={step}, l_in={l_in} -> projected_jacobian",
                flush=True,
            )
            in_ad["flag"] = True
            try:
                with torch.inference_mode(False):
                    h_template = _to_normal(hidden_states)
                    enc = _to_normal(encoder_hidden_states)
                    temb_c = _to_normal(temb)
                    rope = _to_normal(rotary_emb)
                    h_shape = h_template[0].shape
                    z_full = h_template[0].reshape(-1)

                    def block_fn(z_flat):
                        hs = h_template.clone()
                        hs[0] = z_flat.reshape(h_shape)
                        out = block_next(
                            hs,
                            enc,
                            temb_c,
                            rope,
                            update_cache=update_cache,
                            cache_name=cache_name,
                        )
                        return out[0].reshape(-1)

                    v_in = v_for(l_in, step)
                    v_out = v_for(l_out, step)
                    if z_full.numel() != v_in.shape[0]:
                        raise RuntimeError(
                            f"D_flat mismatch at t={step} l_in={l_in}: "
                            f"z={z_full.numel()} vs V_in={v_in.shape[0]}"
                        )
                    j_tilde = projected_jacobian(
                        block_fn,
                        z_full,
                        v_in,
                        v_out,
                        mode=args.mode,
                        verbose=not args.quiet,
                    )
            finally:
                in_ad["flag"] = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            jac_store[(step, l_in)] = j_tilde.detach().to(torch.float32).cpu()
            done = len(jac_store)
            print(
                f"  recorded A[t={step}, l_in={l_in}] shape={tuple(j_tilde.shape)} ({done}/{n_target_total})",
                flush=True,
            )
            if done % checkpoint_every == 0:
                _save_shard(out_path, jac_store, args, partitions, layer_to_part, cfg, L, target_tl_full)
                print(f"  checkpoint -> {out_path.name}", flush=True)
            if done >= n_target_total:
                raise _AllTargetsRecorded
            return None

        return hook

    handles = [transformer.blocks[0].register_forward_pre_hook(_pass_tick)]
    for l_in, steps in by_lin.items():
        handles.append(
            transformer.blocks[l_in + 1].register_forward_pre_hook(
                make_pre_hook(l_in, steps),
                with_kwargs=True,
            )
        )

    server._reset(prompt=prompt)
    run_t0[0] = time.time()
    log(f"starting {jac_phase} denoising VJP pass (prompt={prompt!r}) ...")
    try:
        with torch.no_grad():
            _run_denoising_with_hooks(server, obs, hooks_state)
    except _AllTargetsRecorded:
        log(f"early-stop: all targets recorded after {time.time() - run_t0[0]:.1f}s")
    finally:
        for h in handles:
            h.remove()

    _save_shard(out_path, jac_store, args, partitions, layer_to_part, cfg, L, target_tl_full)
    log(f"saved -> {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return 0 if len(jac_store) == n_target_total else 3


def run_vjp_merge(args: argparse.Namespace) -> int:
    cfg = json.loads((args.svd_dir / "config.json").read_text(encoding="utf-8"))
    summary = torch.load(args.svd_dir / "svd_summary.pt", map_location="cpu", weights_only=False)
    L = int(summary["c_means"].shape[0])
    sel_t = list(cfg["selected_timesteps"])
    partitions = [tuple(p) for p in cfg["partitions"]]
    layer_to_part = list(summary["layer_to_part"])
    num_l_in = L - 1
    want_a = len(sel_t) * num_l_in

    merged_a: Dict[Tuple[int, int], torch.Tensor] = {}
    seen_ranks: List[int] = []
    for rank in range(args.num_shards):
        shard_path = args.out_dir / f"shard_rank{rank}.pt"
        if not shard_path.exists():
            print(f"  [warn] missing {shard_path}", flush=True)
            continue
        d = torch.load(shard_path, map_location="cpu", weights_only=False)
        merged_a.update(d["A_tilde"])
        seen_ranks.append(rank)
        print(f"  rank {rank}: {len(d['A_tilde'])} A entries", flush=True)

    final_path = args.out_dir / "A_tilde__full.pt"
    torch.save(
        {
            "A_tilde": merged_a,
            "B_tilde": {},
            "prompt": args.prompt or cfg.get("prompt"),
            "mode": args.mode,
            "method": "vjp",
            "V_dtype": args.v_dtype,
            "V_device": args.v_device,
            "partitions": partitions,
            "layer_to_part": layer_to_part,
            "k": cfg["k_target"],
            "L": L,
            "selected_timesteps": sel_t,
            "num_shards": args.num_shards,
            "ranks_seen": seen_ranks,
            "inputs_npz": str(args.inputs_npz),
            "obs_index": int(args.obs_index),
            "note": (
                "A_tilde[(t, l_in)] = V_{l_in+1,t}^T J_{block_{l_in+1}}(z_{t,l_in}) V_{l_in,t}; "
                f"autograd VJP on LingBot {str(cfg.get('mode', 'action'))} denoising. "
                "B_tilde skipped (Cosmos-Policy default)."
            ),
        },
        final_path,
    )
    have_a = len(merged_a)
    print(f"[merge] merged {have_a}/{want_a} A_tilde -> {final_path}", flush=True)
    if have_a < want_a:
        missing = [(t, l) for t in sel_t for l in range(num_l_in) if (t, l) not in merged_a]
        print(f"  [warn] {len(missing)} pairs missing, e.g. {missing[:5]}", flush=True)
        return 2
    return 0


def run_ridge(args: argparse.Namespace) -> int:
    """Legacy ridge fit on SVD projected diffs (pre-alignment method)."""
    proj_raw = torch.load(args.svd_dir / "projected_diffs.pt", map_location="cpu", weights_only=False)
    config_json = json.loads((args.svd_dir / "config.json").read_text(encoding="utf-8"))
    projected_diffs: Dict[Tuple[int, int, int], torch.Tensor] = proj_raw["projected_diffs"]
    selected_timesteps = list(config_json["selected_timesteps"])
    n_layers = int(config_json["L"])
    sample_ids = sorted({k[0] for k in projected_diffs.keys()})

    def _fit(xs, ys, ridge):
        X = torch.stack(xs, dim=1).float()
        Y = torch.stack(ys, dim=1).float()
        k = X.shape[0]
        xxt = X @ X.transpose(0, 1)
        reg = ridge * torch.eye(k, dtype=xxt.dtype)
        return ((Y @ X.transpose(0, 1)) @ torch.linalg.inv(xxt + reg)).float()

    a_tilde = {}
    b_tilde = {}
    for t in selected_timesteps:
        for l_in in range(n_layers - 1):
            xs, ys = [], []
            for s in sample_ids:
                k_x = (s, l_in, t)
                k_y = (s, l_in + 1, t)
                if k_x in projected_diffs and k_y in projected_diffs:
                    xs.append(projected_diffs[k_x])
                    ys.append(projected_diffs[k_y])
            if len(xs) >= 2:
                a_tilde[(t, l_in)] = _fit(xs, ys, args.ridge)

    for idx in range(len(selected_timesteps) - 1):
        t = selected_timesteps[idx]
        t_next = selected_timesteps[idx + 1]
        xs, ys = [], []
        for s in sample_ids:
            k_x = (s, n_layers - 1, t)
            k_y = (s, 0, t_next)
            if k_x in projected_diffs and k_y in projected_diffs:
                xs.append(projected_diffs[k_x])
                ys.append(projected_diffs[k_y])
        if len(xs) >= 2:
            b_tilde[(t,)] = _fit(xs, ys, args.ridge)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "A_tilde__full.pt"
    torch.save(
        {
            "A_tilde": a_tilde,
            "B_tilde": b_tilde,
            "prompt": config_json.get("prompt"),
            "mode": config_json.get("mode"),
            "selected_timesteps": selected_timesteps,
            "ridge": float(args.ridge),
            "method": "ridge",
            "num_samples": len(sample_ids),
        },
        out_path,
    )
    print(f"[jac/ridge] wrote {out_path}")
    print(f"[jac/ridge] A_tilde entries: {len(a_tilde)}  B_tilde entries: {len(b_tilde)}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["vjp", "ridge"], default="vjp")
    p.add_argument(
        "--phase",
        choices=["all", "worker", "merge"],
        default="all",
        help="all = worker+merge on this node (default for single-GPU pipeline).",
    )
    p.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--svd-dir", type=Path, required=True)
    p.add_argument(
        "--out-subdir",
        type=str,
        default="A_tilde_lingbot",
        help="Subdirectory under --svd-dir for Jacobian artifacts.",
    )
    p.add_argument("--inputs-npz", type=Path, default=None, help="Real obs NPZ (default: negative.npz).")
    p.add_argument("--obs-index", type=int, default=0)
    p.add_argument("--config-name", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--mode", type=str, default="vjp_no_retain", choices=["jvp", "vjp", "vjp_no_retain"])
    p.add_argument("--v-device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--v-dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--ridge", type=float, default=1e-3, help="Ridge coef (legacy --method ridge only).")
    p.add_argument("--skip-step-jacobian", action="store_true", default=True, help="Skip B_tilde (repo root's default).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir = args.svd_dir / args.out_subdir

    if args.method == "ridge":
        proj_path = args.svd_dir / "projected_diffs.pt"
        if not proj_path.exists():
            raise FileNotFoundError(
                f"Ridge method needs {proj_path}; use --method vjp (default) for repo-root-aligned Jacobians."
            )
        return run_ridge(args)

    if args.inputs_npz is None:
        for candidate in (
            args.svd_dir.parent / "negative.npz",
            args.svd_dir / "negative.npz",
        ):
            if candidate.exists():
                args.inputs_npz = candidate
                break
        if args.inputs_npz is None:
            raise ValueError(
                "VJP Jacobians need a real observation. Pass --inputs-npz "
                "(typically the paired negative.npz from collection)."
            )

    if args.phase == "worker":
        return run_vjp_worker(args)
    if args.phase == "merge":
        return run_vjp_merge(args)

    rc = run_vjp_worker(args)
    if rc not in (0, 3):
        return rc
    merge_rc = run_vjp_merge(args)
    return merge_rc if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
