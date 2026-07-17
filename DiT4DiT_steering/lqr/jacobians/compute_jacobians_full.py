"""Sbatch-array projected Jacobian collection on the DiT4DiT action DiT.

Analogue of the repo root's notebooks/lqr/jacobians/compute_jacobians_full.py
but adapted for DiT4DiT's FlowmatchingActionHead / BasicTransformerBlock.

Architecture change vs Cosmos-Policy
-------------------------------------
Cosmos:   hooks on model.net.blocks[i], which takes 5-D latent (B, T_p, H_p, W_p, D).
DiT4DiT:  hooks on model.action_model.model.transformer_blocks[i], which takes
          hidden_states (B, T_seq, inner_dim) where T_seq = n_state_tokens + action_horizon.

We extract the action token slice [denoise_t_start:denoise_t_end] and reshape it to
z_flat (D_flat,) for the SVD projection — identical math to Cosmos, different tensor
shapes.

Key invariants:
  D_flat          = action_horizon * inner_dim  (e.g. 8 * 768 = 6144)
  denoise_t_start = n_state_tokens             (from config.json, e.g. 1)
  denoise_t_end   = n_state_tokens + action_horizon  (e.g. 9)
  A_tilde[(t, l_in)] = V_{p(l_in+1), t}^T  J_{block_{l_in+1}}(z_{t, l_in}) V_{p(l_in), t}

Pipeline (two-phase like the Cosmos version):
  --phase worker : one sbatch array task per rank
  --phase merge  : single login-node task that sums shards -> A_tilde__full.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DIT4DIT_ROOT = _HERE.parent.parent.parent
if str(_DIT4DIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIT4DIT_ROOT))

LIBERO_HOME = os.environ.get("LIBERO_HOME", "/work/nvme/bhhv/jskifstad/LIBERO")
if LIBERO_HOME not in sys.path:
    sys.path.insert(0, LIBERO_HOME)

# Append FastWAM site-packages at the END so robosuite is found but
# dit4dit's own transformers/diffusers take priority over FastWAM's.
_FASTWAM_SITE = "/projects/bhhv/jskifstad/FastWAM/.conda/envs/fastwam/lib/python3.10/site-packages"
if _FASTWAM_SITE not in sys.path:
    sys.path.append(_FASTWAM_SITE)
os.environ.setdefault("LIBERO_HOME", LIBERO_HOME)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(LIBERO_HOME, "libero"))

DIT4DIT_ROOT = _DIT4DIT_ROOT
CKPT_DEFAULT = str(DIT4DIT_ROOT / "checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["worker", "merge"], required=True)
    p.add_argument("--rank", type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards", type=int, required=True)

    p.add_argument("--svd-dir",  type=Path, required=True)
    p.add_argument("--out-dir",  type=Path, required=True)

    p.add_argument("--prompt",   type=str,
                   default="put both the cream cheese box and the butter in the basket")
    p.add_argument("--inputs-npz", type=Path,
                   default=Path(DIT4DIT_ROOT / "notebooks/lqr/inputs/policy_inputs/"
                                "libero_10__task01__xyz_random_xlarge_3__seed42__pos_neg__paired/"
                                "negative.npz"),
                   help="NPZ with primary_images, wrist_images, proprios.")
    p.add_argument("--obs-index", type=int, default=0)
    p.add_argument("--mode",  type=str, default="vjp_no_retain",
                   choices=["jvp", "vjp", "vjp_no_retain"])
    p.add_argument("--v-device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--v-dtype",  type=str, default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--verify",  action="store_true")
    p.add_argument("--resume",  action="store_true")
    p.add_argument("--ckpt-path", type=str, default=CKPT_DEFAULT)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


# -----------------------------------------------------------------------
# Sharding (identical to Cosmos version)
# -----------------------------------------------------------------------

def make_layer_shards(num_l_in: int, num_shards: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    base, extra = divmod(num_l_in, num_shards)
    start = 0
    for i in range(num_shards):
        size = base + (1 if i < extra else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


# -----------------------------------------------------------------------
# V file resolver (identical to Cosmos version)
# -----------------------------------------------------------------------

def _v_filename(svd_dir, p_idx, t_id, partitions, k_target):
    a, b = partitions[p_idx]
    pattern = f"V_part{p_idx}_layers{a}-{b}_t{t_id}_k*.pt"
    matches = sorted(svd_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no V file matching {pattern} in {svd_dir}")
    preferred = [m for m in matches if m.name.endswith(f"_k{k_target}.pt")]
    return (preferred or matches)[0]


# -----------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np
    import torch

    rank = args.rank
    num_shards = args.num_shards
    assert 0 <= rank < num_shards

    log = lambda msg: print(f"[rank {rank}/{num_shards}] {msg}", flush=True)

    log(f"mode={args.mode}  v_device={args.v_device}  v_dtype={args.v_dtype}  "
        f"verify={args.verify}")

    # ---- Load SVD config ----
    cfg = json.loads((args.svd_dir / "config.json").read_text())
    summary = torch.load(args.svd_dir / "svd_summary.pt",
                         map_location="cpu", weights_only=False)
    L = int(summary["c_means"].shape[0])
    sel_t         = list(cfg["selected_timesteps"])
    sampling_steps = cfg["sampling_steps"]
    partitions    = [tuple(p) for p in cfg["partitions"]]
    layer_to_part = list(summary["layer_to_part"])
    k_target      = cfg["k_target"]
    seed          = args.seed if args.seed is not None else int(cfg.get("seed", 42))

    D_flat          = cfg["D_flat"]
    T_p_denoise     = cfg["T_p_denoise"]
    denoise_t_start = cfg["denoise_t_start"]
    denoise_t_end   = cfg["denoise_t_end"]
    action_horizon  = cfg.get("action_horizon", T_p_denoise)

    num_l_in = L - 1
    layer_shards = make_layer_shards(num_l_in, num_shards)
    my_layers = layer_shards[rank]
    target_tl: set[tuple[int, int]] = {
        (int(t), int(l)) for t in sel_t for l in my_layers
    }
    log(f"L={L}  num_l_in={num_l_in}  sel_t={sel_t}  "
        f"my_layers={my_layers}  -> {len(target_tl)} (t, l_in) pairs")
    if not target_tl:
        log("no targets; exiting.")
        return 0

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_rank{rank}.pt"

    existing: dict[tuple[int, int], torch.Tensor] = {}
    if args.resume and out_path.exists():
        d = torch.load(out_path, map_location="cpu", weights_only=False)
        existing = dict(d.get("A_tilde", {}))
        before = len(target_tl)
        target_tl = {tl for tl in target_tl if tl not in existing}
        log(f"resume: {len(existing)} done; {before - len(target_tl)} skipped; "
            f"{len(target_tl)} remaining")
        if not target_tl:
            log("all targets already present; exiting.")
            return 0

    # ---- Load DiT4DiT model ----
    import DiT4DiT.model.framework.DiT4DiT  # register framework before from_pretrained
    from DiT4DiT.model.framework.base_framework import baseframework
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"loading DiT4DiT from {args.ckpt_path} ...")
    model = baseframework.from_pretrained(args.ckpt_path).to(device).eval()
    action_model = model.action_model
    action_dit   = action_model.model  # DiT with transformer_blocks

    n_blocks = len(action_dit.transformer_blocks)
    assert n_blocks == L, f"L mismatch: model={n_blocks} blocks, summary L={L}"
    inner_dim = action_dit.inner_dim
    log(f"loaded; {n_blocks} transformer_blocks  inner_dim={inner_dim}  "
        f"action_horizon={action_horizon}")

    # ---- V loader ----
    _DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    V_DEVICE = torch.device(args.v_device)
    V_DTYPE  = _DTYPE_MAP[args.v_dtype]
    _V_CACHE: dict[tuple[int, int], torch.Tensor] = {}

    def V_for(layer_idx: int, t_id: int) -> torch.Tensor:
        p_idx = layer_to_part[layer_idx]
        key = (p_idx, t_id)
        V = _V_CACHE.get(key)
        if V is not None:
            return V
        fp = _v_filename(args.svd_dir, p_idx, t_id, partitions, k_target)
        log(f"lazy-load {fp.name} ({fp.stat().st_size/1e9:.2f} GB) -> "
            f"{V_DEVICE} as {V_DTYPE} ...")
        Vt = torch.load(fp, map_location="cpu", weights_only=False)["V"]
        Vt = Vt.to(device=V_DEVICE, dtype=V_DTYPE).contiguous()
        _V_CACHE[key] = Vt
        log(f"  p={p_idx} t={t_id}: {tuple(Vt.shape)}  "
            f"~{Vt.element_size()*Vt.numel()/1e9:.2f} GB")
        return Vt

    # ---- Load observation ----
    IMAGE_SIZE = 224
    if not args.inputs_npz.exists():
        raise FileNotFoundError(f"--inputs-npz not found: {args.inputs_npz}")
    npz = np.load(args.inputs_npz)
    n_avail = int(npz["proprios"].shape[0])
    if not (0 <= args.obs_index < n_avail):
        raise ValueError(f"--obs-index {args.obs_index} out of range [0, {n_avail})")
    obs_dict = {
        "wrist_image":   npz["wrist_images"][args.obs_index].copy(),
        "primary_image": npz["primary_images"][args.obs_index].copy(),
        "proprio":       npz["proprios"][args.obs_index].astype(np.float32).copy(),
    }
    log(f"obs from {args.inputs_npz.name}[idx={args.obs_index}]: "
        f"primary={obs_dict['primary_image'].shape}  proprio={obs_dict['proprio'].shape}")

    # Pre-process observation into model inputs
    primary = cv2.resize(obs_dict["primary_image"], (IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=cv2.INTER_AREA)
    wrist   = cv2.resize(obs_dict["wrist_image"],   (IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=cv2.INTER_AREA)
    concat_img = np.concatenate([primary, wrist], axis=1)

    proprio_raw = obs_dict["proprio"]
    sin_s = np.sin(proprio_raw[None])
    cos_s = np.cos(proprio_raw[None])
    state_enc = np.stack([sin_s, cos_s], axis=-1).reshape(1, -1).astype(np.float32)
    max_state_dim = action_model.config.state_dim
    pad = max_state_dim - state_enc.shape[-1]
    if pad > 0:
        state_enc = np.pad(state_enc, ((0, 0), (0, pad)), "constant")

    example = {"image": [concat_img], "lang": args.prompt, "state": state_enc}
    batch_images = [example["image"]]
    instructions = [example["lang"]]

    # ---- Pre-compute VLM embeddings (fixed for all (t, l) pairs) ----
    log(f"pre-computing VLM embeddings for prompt={args.prompt!r} ...")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bi = model.backbone_interface.build_cosmos_inputs(
                images=batch_images, instructions=instructions,
            )
            bout = model.backbone_interface(
                **bi, output_hidden_states=True, output_attentions=False, return_dict=True,
            )
            vl_embs = bout.hidden_states[-1].detach().clone()  # (B, seq_len, H)

    state_t = None
    if action_model.state_encoder is not None:
        st_np = state_enc
        model_dtype = next(action_model.state_encoder.parameters()).dtype
        state_t = torch.from_numpy(st_np).unsqueeze(0).to(
            device=device, dtype=model_dtype
        )  # (B, 1, state_dim)

    # Pre-compute static state features
    state_features = None
    if state_t is not None:
        with torch.no_grad():
            state_features = action_model.state_encoder(
                state_t.to(device=device, dtype=model_dtype)
            ).detach().clone()  # (B, 1, inner_dim)

    log(f"VLM embeddings ready: {tuple(vl_embs.shape)}")

    # ---- Projected Jacobian helper (identical math to Cosmos version) ----
    def projected_jacobian(block_fn, z, V_in, V_out, mode="vjp_no_retain",
                           verbose=True, log_every=25):
        z = z.detach()
        z_dev, z_dt = z.device, z.dtype
        r_in, r_out = V_in.shape[1], V_out.shape[1]
        OUT_DTYPE = torch.float32

        def col(V, j):
            return V[:, j].contiguous().to(device=z_dev, dtype=z_dt)

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
            eta  = (total - i - 1) / rate if rate > 0 else float("inf")
            print(f"      [{i+1:3d}/{total}]  {el:6.1f}s  {rate:5.2f} it/s  "
                  f"ETA {eta:6.1f}s", flush=True)

        if mode == "jvp":
            with torch.enable_grad():
                M = torch.empty(r_in, r_out, dtype=OUT_DTYPE, device=V_out.device)
                t0 = time.time()
                for j in range(r_in):
                    _, jvp = torch.func.jvp(block_fn, (z,), (col(V_in, j),))
                    M[j].copy_(jvp.to(device=V_out.device, dtype=OUT_DTYPE) @ V_out.to(OUT_DTYPE))
                    _log_iter(j, r_in, t0)
            return M.T.contiguous()

        if mode == "vjp":
            with torch.enable_grad():
                z_g = z.clone().requires_grad_(True)
                out = block_fn(z_g)
                G = torch.empty(r_out, r_in, dtype=OUT_DTYPE, device=V_in.device)
                V_in_mat = V_in.reshape(-1, r_in).to(dtype=OUT_DTYPE)
                t0 = time.time()
                for i in range(r_out):
                    (g,) = torch.autograd.grad(out, z_g, col(V_out, i),
                                               retain_graph=(i < r_out - 1))
                    G[i].copy_(g.reshape(-1).to(device=V_in.device, dtype=OUT_DTYPE) @ V_in_mat)
                    _log_iter(i, r_out, t0)
            return G

        if mode == "vjp_no_retain":
            with torch.enable_grad():
                G = torch.empty(r_out, r_in, dtype=OUT_DTYPE, device=V_in.device)
                V_in_mat = V_in.reshape(-1, r_in).to(dtype=OUT_DTYPE)
                t0 = time.time()
                for i in range(r_out):
                    z_g = z.clone().requires_grad_(True)
                    out = block_fn(z_g)
                    (g,) = torch.autograd.grad(out, z_g, col(V_out, i))
                    G[i].copy_(g.reshape(-1).to(device=V_in.device, dtype=OUT_DTYPE) @ V_in_mat)
                    del out, z_g, g
                    _log_iter(i, r_out, t0)
            return G

        raise ValueError(f"unknown mode {mode!r}")

    # ---- Hook plumbing ----
    PASSES_PER_STEP = 1
    state_ctr  = {"pass_idx": -1}
    in_ad      = {"flag": False}
    jac_store: dict[tuple[int, int], torch.Tensor] = dict(existing)
    n_target_total = len(target_tl) + len(existing)
    _RUN_T0 = [0.0]
    sel_t_set = set(sel_t)

    class _AllTargetsRecorded(Exception):
        pass

    by_lin: dict[int, set[int]] = {}
    for (t, l_in) in target_tl:
        by_lin.setdefault(l_in, set()).add(t)

    checkpoint_every = 25
    target_tl_full = sorted(target_tl | set(existing.keys()))

    def make_pre_hook(l_in: int, target_steps: set[int]):
        l_out = l_in + 1

        def hook(block_next, h_args, kwargs):
            if in_ad["flag"]:
                return None
            pass_idx = state_ctr["pass_idx"]
            step = pass_idx  # PASSES_PER_STEP = 1
            if step not in target_steps:
                return None
            if (step, l_in) in jac_store:
                return None

            # h_args[0] is hidden_states: (B, T_seq, inner_dim)
            hs = h_args[0]
            temb_clone = kwargs.get("temb")
            enc_hs     = kwargs.get("encoder_hidden_states")
            enc_mask   = kwargs.get("encoder_attention_mask")
            attn_mask  = kwargs.get("attention_mask")

            print(f"  [t=+{time.time()-_RUN_T0[0]:.1f}s] hook t={step}, "
                  f"l_in={l_in} -> projected_jacobian", flush=True)

            in_ad["flag"] = True
            try:
                with torch.inference_mode(False):
                    # Clone all tensors to escape inference-tensor pool
                    def _clone(x):
                        if torch.is_tensor(x):
                            return x.detach().clone()
                        return x
                    hs_tmpl   = _clone(hs)
                    temb_c    = _clone(temb_clone)
                    enc_hs_c  = _clone(enc_hs)
                    enc_mask_c = _clone(enc_mask)
                    attn_mask_c = _clone(attn_mask)

                    # z_full: action token slice flattened
                    z_full = hs_tmpl[0, denoise_t_start:denoise_t_end, :].reshape(-1)
                    assert z_full.numel() == D_flat, (
                        f"D_flat mismatch: got {z_full.numel()}, expected {D_flat}"
                    )

                    def block_fn(z_flat):
                        x_in = hs_tmpl.clone()
                        x_in[0, denoise_t_start:denoise_t_end, :] = (
                            z_flat.reshape(action_horizon, inner_dim)
                        )
                        out = block_next(
                            x_in,
                            attention_mask=attn_mask_c,
                            encoder_hidden_states=enc_hs_c,
                            encoder_attention_mask=enc_mask_c,
                            temb=temb_c,
                        )
                        return out[0, denoise_t_start:denoise_t_end, :].reshape(-1)

                    t_v = time.time()
                    V_in_  = V_for(l_in, step)
                    V_out_ = V_for(l_out, step)
                    log(f"  V ready ({time.time()-t_v:.1f}s); entering projected_jacobian")

                    t_pj = time.time()
                    J_tilde = projected_jacobian(
                        block_fn, z_full, V_in_, V_out_, mode=args.mode,
                    )
                    print(f"  [t=+{time.time()-_RUN_T0[0]:.1f}s] projected_jacobian "
                          f"done in {time.time()-t_pj:.1f}s", flush=True)
            finally:
                in_ad["flag"] = False

            jac_store[(step, l_in)] = J_tilde.detach().to(torch.float32).cpu()
            done, want = len(jac_store), n_target_total
            print(f"  recorded Ã[t={step}, l_in={l_in}]  shape={tuple(J_tilde.shape)}  "
                  f"({done}/{want})", flush=True)

            if done % checkpoint_every == 0:
                _save_shard(out_path, jac_store, args, partitions, layer_to_part,
                            cfg, len(sel_t), L, target_tl_full)
                print(f"  checkpoint -> {out_path.name}", flush=True)

            if done >= want:
                raise _AllTargetsRecorded
            return None

        return hook

    def _pass_tick(_block, _args):
        if not in_ad["flag"]:
            state_ctr["pass_idx"] += 1

    handles = [action_dit.transformer_blocks[0].register_forward_pre_hook(_pass_tick)]
    for l_in, steps in by_lin.items():
        handles.append(
            action_dit.transformer_blocks[l_in + 1].register_forward_pre_hook(
                make_pre_hook(l_in, steps), with_kwargs=True,
            )
        )
    log(f"registered {len(handles)} hook(s); {len(target_tl)} pending pairs")

    # ---- Run denoising loop (no inference_mode — hooks need enable_grad locally) ----
    from DiT4DiT.model.modules.action_model.ActionDiT import FlowmatchingActionHead
    # Import the loop helper from the SVD script (same repo)
    sys.path.insert(0, str(DIT4DIT_ROOT / "notebooks/lqr/svd"))
    from run_partition_svd_pairs_no_action import run_denoising_loop

    _RUN_T0[0] = time.time()
    log(f"entering denoising loop (prompt={args.prompt!r}, seed={seed}, "
        f"steps={sampling_steps}) ...")
    try:
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float32):
                _ = run_denoising_loop(
                    action_model, vl_embs, state_t,
                    seed=seed, num_steps=sampling_steps,
                )
    except _AllTargetsRecorded:
        log(f"early-stop: all targets recorded in {time.time()-_RUN_T0[0]:.1f}s")
    finally:
        for h in handles:
            h.remove()

    elapsed = time.time() - _RUN_T0[0]
    log(f"inference elapsed {elapsed:.1f}s; recorded {len(jac_store)}/{n_target_total} A_tilde")

    _save_shard(out_path, jac_store, args, partitions, layer_to_part,
                cfg, len(sel_t), L, target_tl_full)
    log(f"saved -> {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")
    return 0 if len(jac_store) == n_target_total else 3


def _save_shard(out_path, jac_store, args, partitions, layer_to_part,
                cfg, T, L, target_tl_full):
    import torch
    torch.save({
        "A_tilde":            dict(jac_store),
        "B_tilde":            {},
        "rank":               args.rank,
        "num_shards":         args.num_shards,
        "prompt":             args.prompt,
        "mode":               args.mode,
        "V_dtype":            args.v_dtype,
        "V_device":           args.v_device,
        "partitions":         partitions,
        "layer_to_part":      layer_to_part,
        "k":                  cfg["k_target"],
        "T":                  T,
        "L":                  L,
        "selected_timesteps": cfg["selected_timesteps"],
        "targets":            target_tl_full,
        "T_p_denoise":        cfg["T_p_denoise"],
        "denoise_t_start":    cfg["denoise_t_start"],
        "denoise_t_end":      cfg["denoise_t_end"],
        "D_flat":             cfg["D_flat"],
    }, out_path)


# -----------------------------------------------------------------------
# Merge (identical to Cosmos version; no model required)
# -----------------------------------------------------------------------

def run_merge(args: argparse.Namespace) -> int:
    import torch

    cfg     = json.loads((args.svd_dir / "config.json").read_text())
    summary = torch.load(args.svd_dir / "svd_summary.pt",
                         map_location="cpu", weights_only=False)
    L       = int(summary["c_means"].shape[0])
    sel_t   = list(cfg["selected_timesteps"])
    T       = len(sel_t)
    partitions    = [tuple(p) for p in cfg["partitions"]]
    layer_to_part = list(summary["layer_to_part"])
    num_l_in      = L - 1
    want_a        = T * num_l_in

    print(f"[merge] num_shards={args.num_shards}  T={T}  L={L}  "
          f"expected total A_tilde = {want_a}", flush=True)

    merged_a: dict[tuple[int, int], torch.Tensor] = {}
    seen_ranks: list[int] = []
    for rank in range(args.num_shards):
        shard_path = args.out_dir / f"shard_rank{rank}.pt"
        if not shard_path.exists():
            print(f"  [warn] missing {shard_path}")
            continue
        d = torch.load(shard_path, map_location="cpu", weights_only=False)
        merged_a.update(d["A_tilde"])
        seen_ranks.append(rank)
        print(f"  rank {rank}: {len(d['A_tilde'])} A entries")

    final_path = args.out_dir / "A_tilde__full.pt"
    torch.save({
        "A_tilde":            merged_a,
        "B_tilde":            {},
        "prompt":             args.prompt,
        "mode":               args.mode,
        "V_dtype":            args.v_dtype,
        "V_device":           args.v_device,
        "partitions":         partitions,
        "layer_to_part":      layer_to_part,
        "k":                  cfg["k_target"],
        "T":                  T,
        "L":                  L,
        "selected_timesteps": sel_t,
        "num_shards":         args.num_shards,
        "ranks_seen":         seen_ranks,
        "T_p_denoise":        cfg["T_p_denoise"],
        "denoise_t_start":    cfg["denoise_t_start"],
        "denoise_t_end":      cfg["denoise_t_end"],
        "D_flat":             cfg["D_flat"],
        "note": (
            "A_tilde[(t, l_in)] = V_{p(l_in+1),t}^T J_{block_{l_in+1}}(z_{t,l_in}) V_{p(l_in),t}; "
            "DiT4DiT action-DiT transformer_blocks; action token subspace only."
        ),
    }, final_path)

    have_a = len(merged_a)
    print(f"[merge] merged {have_a}/{want_a} A_tilde -> "
          f"{final_path}  ({final_path.stat().st_size/1e6:.1f} MB)")
    if have_a < want_a:
        missing = [(t, l) for t in sel_t for l in range(num_l_in)
                   if (t, l) not in merged_a]
        print(f"  [warn] {len(missing)} pairs missing, e.g. {missing[:5]}")
        return 2
    # Clean up shard files now that the merged output is complete
    for rank in seen_ranks:
        (args.out_dir / f"shard_rank{rank}.pt").unlink(missing_ok=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.phase == "worker":
        return run_worker(args)
    if args.phase == "merge":
        return run_merge(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
