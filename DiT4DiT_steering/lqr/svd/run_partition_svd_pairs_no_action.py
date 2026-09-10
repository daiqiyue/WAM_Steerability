#!/usr/bin/env python
"""Per-(partition, timestep) randomized SVD of DiT4DiT action-DiT activations
with a *paired observation* contrast.

Analogue of:
  the repo root's notebooks/lqr/svd/run_partition_svd_pairs_no_action.py
Adapted for DiT4DiT's FlowmatchingActionHead / action DiT (BasicTransformerBlock).

Architecture differences from Cosmos-Policy
-------------------------------------------
Cosmos-Policy operates on a spatiotemporal latent (B, T_p, H_p, W_p, D) and
hooks intercept denoising slots [denoise_t_start:denoise_t_end, :, :, :].

DiT4DiT's action DiT operates on a sequence of action tokens:
  transformer_blocks input/output: (B, T_seq, inner_dim)
  where T_seq = n_state_tokens + action_horizon = 1 + 8 = 9 (for libero)

We extract only the action-token portion:
  denoise_t_start = n_state_tokens  (= 1, skipping the 1 state token prepended)
  denoise_t_end   = n_state_tokens + action_horizon  (= 9)
  T_p_denoise     = action_horizon  (= 8)
  D_flat          = action_horizon * inner_dim  (= 8 * 768 = 6144 for DiT-B)

These match the Cosmos config.json key names so downstream Jacobian and LQR
scripts can read them transparently.

Modes
-----
--mode sketch : one Python invocation per rank; accumulates per-partition-
                per-timestep Y / W / μ using forward hooks.
--mode svd    : single-rank finalize; reads per-rank sketches, runs Halko
                one-pass SVD, and writes V_part*_t*.pt + svd_summary.pt.
"""

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_LOCAL_DIT4DIT_ROOT = _HERE.parent.parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from runtime_paths import configure_runtime  # noqa: E402

_DIT4DIT_ROOT, LIBERO_HOME = configure_runtime(_LOCAL_DIT4DIT_ROOT)
os.environ.setdefault("LIBERO_HOME", LIBERO_HOME)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DIT4DIT_ROOT = _DIT4DIT_ROOT
CKPT_DEFAULT = os.environ.get(
    "CKPT_PATH",
    str(DIT4DIT_ROOT / "checkpoint/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt"),
)

DEFAULT_PROMPT = "put both the cream cheese box and the butter in the basket"
DEFAULT_POS_NPZ = str(DIT4DIT_ROOT / "notebooks/lqr/inputs/policy_inputs/libero_10__task01__xyz_random_xlarge_3__seed42__pos_neg/positive.npz")
DEFAULT_NEG_NPZ = str(DIT4DIT_ROOT / "notebooks/lqr/inputs/policy_inputs/libero_10__task01__xyz_random_xlarge_3__seed42__pos_neg/negative.npz")

IMAGE_SIZE = 224  # DiT4DiT training image size per view


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sketch", "svd"], required=True)
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    ap.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))

    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    ap.add_argument("--pos-npz", type=Path, default=Path(DEFAULT_POS_NPZ))
    ap.add_argument("--neg-npz", type=Path, default=Path(DEFAULT_NEG_NPZ))
    ap.add_argument("--drive-source", type=str, default="all")
    ap.add_argument("--N", type=int, default=-1)

    ap.add_argument("--k-target", type=int, default=64)
    ap.add_argument("--p-over", type=int, default=10)
    ap.add_argument("--partitions", type=str, default="0-5,6-10,11-15",
                    help="Layer partition ranges for the 16-block action DiT.")
    ap.add_argument("--num-layers", type=int, default=16,
                    help="Number of transformer_blocks in the action DiT (DiT-B libero = 16).")

    ap.add_argument("--sampling-steps", type=int, default=4,
                    help="DiT4DiT num_inference_timesteps (default 4).")
    ap.add_argument("--timesteps", type=str, default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sketch-seed", type=int, default=0xC057)

    ap.add_argument("--ckpt-path", type=str, default=CKPT_DEFAULT)
    ap.add_argument("--vl-embs-path", type=Path, default=None,
                    help="Path to precomputed vl_embs .pt file (from precompute_vl_embs.py). "
                         "If given, sketch ranks load only the small action DiT instead of the "
                         "full 20 GB model. Strongly recommended for multi-rank sketches.")

    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scratch-dir", type=Path, default=None)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--svd-device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--svd-chunk", type=int, default=500_000)
    return ap.parse_args()


def parse_partitions(spec, L):
    parts = []
    for tok in spec.split(","):
        a, b = tok.split("-")
        parts.append((int(a), int(b)))
    covered = [l for s, e in parts for l in range(s, e + 1)]
    if sorted(covered) != list(range(L)):
        raise ValueError(f"partitions {parts} must tile [0, {L-1}] exactly")
    return parts


def parse_timesteps(spec, T_total):
    if spec.strip().lower() == "all":
        return list(range(T_total))
    return sorted({int(x) for x in spec.split(",")})


def my_indices(N, rank, world_size):
    return list(range(rank, N, world_size))


def scratch_root(args):
    return args.scratch_dir or (args.out_dir / "scratch")


# -----------------------------------------------------------------------
# Disk I/O helpers (identical to Cosmos version)
# -----------------------------------------------------------------------

def write_raw(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dtype == torch.bfloat16:
        arr = tensor.detach().contiguous().view(torch.uint16).cpu().numpy()
    else:
        arr = tensor.detach().contiguous().cpu().numpy()
    arr.tofile(path)


def read_bf16_raw(path, shape):
    arr = np.fromfile(path, dtype=np.uint16).reshape(shape)
    return torch.from_numpy(arr).view(torch.bfloat16)


# -----------------------------------------------------------------------
# Observation loading — paired (identical key names as Cosmos version)
# -----------------------------------------------------------------------

def _filter_indices(drive_source_arr, drive_filter):
    if drive_filter == "all":
        return np.arange(drive_source_arr.shape[0], dtype=np.int64)
    code = int(drive_filter)
    return np.nonzero(drive_source_arr == code)[0]


def load_paired_observations(pos_npz, neg_npz, drive_filter, N_request):
    pos = np.load(pos_npz)
    neg = np.load(neg_npz)

    for key in ("primary_images", "wrist_images", "proprios"):
        if pos[key].shape != neg[key].shape:
            raise ValueError(f"{key} shape mismatch: pos={pos[key].shape} neg={neg[key].shape}")
    if not np.array_equal(pos["episode_idx"], neg["episode_idx"]):
        raise ValueError("episode_idx differs between pos and neg npz — pairing broken")
    if not np.array_equal(pos["inference_idx"], neg["inference_idx"]):
        raise ValueError("inference_idx differs between pos and neg npz — pairing broken")

    n_total = int(pos["proprios"].shape[0])
    kept = _filter_indices(pos["drive_source"], drive_filter)
    n_avail = int(kept.shape[0])
    N_used = n_avail if N_request < 0 else N_request
    selected = kept[:N_used]

    # Pre-load each array once — npzfile["key"] re-decompresses on every call.
    pos_primary = pos["primary_images"]
    pos_wrist   = pos["wrist_images"]
    pos_proprios = pos["proprios"]
    neg_primary = neg["primary_images"]
    neg_wrist   = neg["wrist_images"]
    neg_proprios = neg["proprios"]

    obs_pos_list, obs_neg_list = [], []
    for i in selected:
        obs_pos_list.append({
            "wrist_image":   pos_wrist[i].copy(),
            "primary_image": pos_primary[i].copy(),
            "proprio":       pos_proprios[i].astype(np.float32).copy(),
        })
        obs_neg_list.append({
            "wrist_image":   neg_wrist[i].copy(),
            "primary_image": neg_primary[i].copy(),
            "proprio":       neg_proprios[i].astype(np.float32).copy(),
        })

    return obs_pos_list, obs_neg_list, N_used, n_avail, selected, pos["drive_source"][selected].astype(np.int32)


# -----------------------------------------------------------------------
# DiT4DiT observation -> model input
# -----------------------------------------------------------------------

def obs_to_example(obs_dict: dict, prompt: str, image_size: int = IMAGE_SIZE,
                    max_state_dim: int = 16) -> dict:
    """Convert stored NPZ observation to DiT4DiT model input dict."""
    primary_raw = obs_dict["primary_image"]  # (H, W, 3) uint8
    wrist_raw   = obs_dict["wrist_image"]    # (H, W, 3) uint8
    proprio_raw = obs_dict["proprio"]        # (8,) float32

    primary = cv2.resize(primary_raw, (image_size, image_size), interpolation=cv2.INTER_AREA)
    wrist   = cv2.resize(wrist_raw,   (image_size, image_size), interpolation=cv2.INTER_AREA)
    concat_img = np.concatenate([primary, wrist], axis=1)  # (H, 2W, 3)

    sin_s = np.sin(proprio_raw[None])   # (1, D)
    cos_s = np.cos(proprio_raw[None])   # (1, D)
    state_enc = np.stack([sin_s, cos_s], axis=-1).reshape(1, -1).astype(np.float32)  # (1, 2D)
    pad = max_state_dim - state_enc.shape[-1]
    if pad > 0:
        state_enc = np.pad(state_enc, ((0, 0), (0, pad)), "constant")

    return {
        "image": [concat_img],
        "lang":  prompt,
        "state": state_enc,
    }


# -----------------------------------------------------------------------
# DiT4DiT denoising loop (manual, to allow hooks + gradient control)
# -----------------------------------------------------------------------

def run_denoising_loop(action_model, vl_embs, state_t, seed=42,
                        num_steps=None, context_manager=None):
    """Run the FlowmatchingActionHead denoising loop without @torch.no_grad().

    This extracts the loop from predict_action so we can install hooks on
    transformer_blocks and (for Jacobians) call torch.enable_grad() locally.

    Args:
        action_model: FlowmatchingActionHead instance.
        vl_embs: (B, seq_len, H) backbone output.
        state_t:  (B, 1, state_dim) state tensor, or None.
        seed: random seed for initial noise.
        num_steps: override num_inference_timesteps; None = use model config.
        context_manager: optional context to wrap the whole loop (e.g., torch.no_grad()).
    """
    import torch
    device = vl_embs.device
    dtype  = vl_embs.dtype
    batch_size = vl_embs.shape[0]
    action_horizon = action_model.config.action_horizon
    action_dim     = action_model.config.action_dim
    if num_steps is None:
        num_steps = action_model.num_inference_timesteps
    dt = 1.0 / num_steps

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None
    actions = torch.randn(
        batch_size, action_horizon, action_dim, dtype=dtype, device=device,
        generator=gen,
    )

    state_features = None
    if state_t is not None:
        state_features = action_model.state_encoder(state_t.to(device=device, dtype=dtype))

    for t in range(num_steps):
        t_cont = 1.0 - t / float(num_steps)
        t_disc = int(t_cont * action_model.num_timestep_buckets)
        ts = torch.full((batch_size,), fill_value=t_disc, device=device)

        action_features = action_model.action_encoder(actions, ts)
        if action_model.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + action_model.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = (torch.cat((state_features, action_features), dim=1)
                   if state_features is not None else action_features)

        # model.action_model.model is the DiT; transformer_blocks are hooked.
        model_output = action_model.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=ts,
        )
        pred = action_model.action_decoder(model_output)
        pred_velocity = pred[:, -action_horizon:]
        actions = actions - dt * pred_velocity

    return actions


# -----------------------------------------------------------------------
# Sketch mode
# -----------------------------------------------------------------------

def run_sketch(args):
    sel_t = parse_timesteps(args.timesteps, args.sampling_steps)
    T_sel = len(sel_t)
    sel_t_set = set(sel_t)
    sel_t_pos = {t: i for i, t in enumerate(sel_t)}
    partitions = parse_partitions(args.partitions, args.num_layers)
    P_count = len(partitions)
    k_plus_p = args.k_target + args.p_over
    L = args.num_layers
    rank = args.rank
    world_size = args.world_size
    assert 0 <= rank < world_size

    print(f"[rank {rank}/{world_size}] sketch: partitions={partitions}  "
          f"selected_steps={sel_t}  k+p={k_plus_p}", flush=True)

    # ---- Load pre-computed vl_embs (or full observations) ----
    # With --vl-embs-path: load the rank's NPZ slice (small) + pre-computed embeddings.
    # Without it: load all observations and run VLM inline (slow, not recommended).
    print(f"[rank {rank}] loading observations ...", flush=True)
    t_load = time.time()
    (obs_pos_list, obs_neg_list, N_used, n_avail,
     selected_rows, drive_source_used) = load_paired_observations(
        args.pos_npz, args.neg_npz, args.drive_source, args.N,
    )
    args.N = N_used
    idx = my_indices(N_used, rank, world_size)
    print(f"[rank {rank}] obs loaded in {time.time()-t_load:.1f}s  "
          f"N={N_used}  my_pairs={len(idx)}", flush=True)

    # ---- Load model ----
    import DiT4DiT.model.framework.DiT4DiT  # register framework before from_pretrained
    from DiT4DiT.model.framework.base_framework import baseframework
    device = torch.device("cuda:0")

    if args.vl_embs_path is not None:
        # Fast path: load pre-computed embeddings; only need the action DiT from the checkpoint.
        print(f"[rank {rank}] loading pre-computed vl_embs from {args.vl_embs_path} ...", flush=True)
        t_vl = time.time()
        vl_cache = torch.load(args.vl_embs_path, map_location="cpu", weights_only=False)
        vl_embs_pos_all = vl_cache["pos"]  # (N, seq_len, H) float16
        vl_embs_neg_all = vl_cache["neg"]
        assert vl_embs_pos_all.shape[0] == N_used, (
            f"vl_embs_pos rows {vl_embs_pos_all.shape[0]} != N_used {N_used}"
        )
        print(f"[rank {rank}] vl_embs loaded in {time.time()-t_vl:.1f}s  "
              f"shape={tuple(vl_embs_pos_all.shape)}", flush=True)

        print(f"[rank {rank}] loading DiT4DiT from {args.ckpt_path} ...", flush=True)
        t_model = time.time()
        model = baseframework.from_pretrained(args.ckpt_path).to(device).eval()
        action_model = model.action_model
        # Free the VLM backbone — we don't need it since vl_embs are pre-computed.
        if hasattr(model, "backbone_interface"):
            del model.backbone_interface
            import gc; gc.collect()
            torch.cuda.empty_cache()
        print(f"[rank {rank}] action DiT ready in {time.time()-t_model:.1f}s", flush=True)
    else:
        print(f"[rank {rank}] WARNING: --vl-embs-path not set; loading full 20 GB model "
              f"(slow with multiple ranks). Consider running precompute_vl_embs.py first.",
              flush=True)
        print(f"[rank {rank}] loading DiT4DiT from {args.ckpt_path} ...", flush=True)
        t_model = time.time()
        model = baseframework.from_pretrained(args.ckpt_path).to(device).eval()
        action_model = model.action_model
        vl_embs_pos_all = None
        vl_embs_neg_all = None
        print(f"[rank {rank}] model loaded in {time.time()-t_model:.1f}s", flush=True)

    action_dit = action_model.model  # the DiT

    # ---- Derive geometry ----
    L_runtime = len(action_dit.transformer_blocks)
    assert L_runtime == L, f"L mismatch: model has {L_runtime} blocks, --num-layers={L}"
    action_horizon = action_model.action_horizon
    inner_dim      = action_dit.inner_dim
    num_steps      = action_model.num_inference_timesteps
    assert num_steps == args.sampling_steps, (
        f"model num_inference_timesteps={num_steps} != --sampling-steps={args.sampling_steps}"
    )
    # State token: 1 token prepended (state_dim > 0)
    has_state = action_model.state_encoder is not None
    n_state_tokens = 1 if has_state else 0
    denoise_t_start = n_state_tokens
    denoise_t_end   = n_state_tokens + action_horizon
    T_p_denoise     = action_horizon
    D               = inner_dim
    D_flat          = T_p_denoise * D

    print(f"  L={L}  action_horizon={action_horizon}  inner_dim={inner_dim}  "
          f"num_steps={num_steps}  n_state_tokens={n_state_tokens}", flush=True)
    print(f"  denoise=[{denoise_t_start}..{denoise_t_end-1}]  "
          f"T_p_denoise={T_p_denoise}  D_flat={D_flat:,}", flush=True)

    # ---- Sketch buffers ----
    BUF_SIZE = 50
    part_state = []
    gen_cpu = torch.Generator(device="cpu").manual_seed(args.sketch_seed + 1)
    for p_idx, (l_start, l_end) in enumerate(partitions):
        L_p = l_end - l_start + 1
        R_p = N_used * L_p
        OR = torch.randn(R_p, k_plus_p, generator=gen_cpu, dtype=torch.float32)
        Y_list  = [torch.zeros(R_p, k_plus_p, dtype=torch.float32) for _ in range(T_sel)]
        W_list  = [torch.zeros(k_plus_p, D_flat, dtype=torch.bfloat16) for _ in range(T_sel)]
        mu_list = [torch.zeros(L_p, D_flat, dtype=torch.float32) for _ in range(T_sel)]
        e_buf   = torch.zeros(BUF_SIZE, D_flat, dtype=torch.bfloat16)
        part_state.append({
            "idx": p_idx, "l_start": l_start, "l_end": l_end, "L_p": L_p,
            "R_p": R_p, "OR": OR,
            "Y_list": Y_list, "W_list": W_list, "mu_list": mu_list,
            "e_buf": e_buf, "meta_buf": [None] * BUF_SIZE, "buf_count": 0,
        })

    layer_to_part = [0] * L
    for ps in part_state:
        for l in range(ps["l_start"], ps["l_end"] + 1):
            layer_to_part[l] = ps["idx"]

    gen_gpu = torch.Generator(device=device).manual_seed(args.sketch_seed)
    omega_left = torch.randn(D_flat, k_plus_p, generator=gen_gpu,
                              dtype=torch.bfloat16, device=device)

    state = {"pass_idx": -1, "n": -1, "sign": 0.0}
    mu_denom = float(N_used)

    def flush_partition(p_idx):
        pbuf = part_state[p_idx]
        n_in_buf = pbuf["buf_count"]
        if n_in_buf == 0:
            return
        sign = state["sign"]
        E_b = pbuf["e_buf"][:n_in_buf]
        metas = pbuf["meta_buf"][:n_in_buf]
        per_t = {}
        for j, (row, l_local, t_pos) in enumerate(metas):
            per_t.setdefault(t_pos, []).append((j, row, l_local))
        for t_pos, items in per_t.items():
            idxs_j = [it[0] for it in items]
            rows   = [it[1] for it in items]
            E_sub  = E_b[idxs_j]
            OR_sub_bf = pbuf["OR"][rows].T.to(torch.bfloat16)
            pbuf["W_list"][t_pos].addmm_(OR_sub_bf, E_sub, alpha=sign)
            for j, _, l_local in items:
                pbuf["mu_list"][t_pos][l_local].add_(E_b[j].float(), alpha=sign / mu_denom)
        pbuf["buf_count"] = 0

    def flush_all():
        for p in range(len(part_state)):
            flush_partition(p)

    # Hook on BasicTransformerBlock output: shape (B, T_seq, inner_dim)
    def make_hook(layer_idx):
        p_idx = layer_to_part[layer_idx]
        pbuf  = part_state[p_idx]
        l_start = pbuf["l_start"]
        L_p = pbuf["L_p"]

        def hook(module, inputs, output):
            if layer_idx == 0:
                state["pass_idx"] += 1
            pass_idx = state["pass_idx"]
            step = pass_idx  # PASSES_PER_STEP = 1 (no CFG)
            if step not in sel_t_set:
                return output

            # Extract action tokens: output shape (B, T_seq, inner_dim)
            act_bf = (
                output.detach()[0, denoise_t_start:denoise_t_end, :]
                .reshape(-1)
            )  # (D_flat,)
            assert act_bf.numel() == D_flat, (
                f"D_flat mismatch at layer {layer_idx} step={step}: "
                f"got {act_bf.numel()}, expected {D_flat}"
            )

            t_pos  = sel_t_pos[step]
            l_local = layer_idx - l_start
            row = state["n"] * L_p + l_local

            y_chunk = (act_bf @ omega_left).float().cpu()
            pbuf["Y_list"][t_pos][row].add_(y_chunk, alpha=state["sign"])

            bi = pbuf["buf_count"]
            pbuf["e_buf"][bi].copy_(act_bf.cpu())
            pbuf["meta_buf"][bi] = (row, l_local, t_pos)
            pbuf["buf_count"] = bi + 1
            if pbuf["buf_count"] >= BUF_SIZE:
                flush_partition(p_idx)
            return output
        return hook

    handles = [
        b.register_forward_hook(make_hook(i))
        for i, b in enumerate(action_dit.transformer_blocks)
    ]

    # ---- Run pairs ----
    def run_one(prompt, obs_dict, n, sign, precomputed_vl_embs=None):
        state["pass_idx"] = -1
        state["n"] = int(n)
        state["sign"] = float(sign)

        if precomputed_vl_embs is not None:
            # Fast path: use cached vl_embs; skip VLM backbone entirely.
            vl_embs = precomputed_vl_embs.to(device=device, dtype=torch.bfloat16).unsqueeze(0)
            # Still need state for action encoder
            example = obs_to_example(obs_dict, prompt)
            state_raw = example["state"]
        else:
            example = obs_to_example(obs_dict, prompt)
            batch_images  = [example["image"]]
            instructions  = [example["lang"]]
            state_raw = example["state"]

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    backbone_inputs = model.backbone_interface.build_cosmos_inputs(
                        images=batch_images, instructions=instructions,
                    )
                    backbone_out = model.backbone_interface(
                        **backbone_inputs,
                        output_hidden_states=True, output_attentions=False, return_dict=True,
                    )
                    vl_embs = backbone_out.hidden_states[-1]  # (B, seq_len, H)

        with torch.no_grad():
            state_t = torch.from_numpy(state_raw).unsqueeze(0).to(
                device=device, dtype=vl_embs.dtype
            ) if state_raw is not None else None  # (B, 1, state_dim)

            with torch.autocast("cuda", dtype=torch.float32):
                _ = run_denoising_loop(
                    action_model, vl_embs, state_t,
                    seed=args.seed, num_steps=num_steps,
                )
        flush_all()
        assert state["pass_idx"] + 1 == num_steps, (
            f"got {state['pass_idx']+1} block-0 fires, expected {num_steps}"
        )

    t0 = time.time()
    for j_local, n in enumerate(idx):
        tp = time.time()
        pos_vl = vl_embs_pos_all[n] if vl_embs_pos_all is not None else None
        neg_vl = vl_embs_neg_all[n] if vl_embs_neg_all is not None else None
        run_one(args.prompt, obs_pos_list[n], n=n, sign=+1.0, precomputed_vl_embs=pos_vl)
        run_one(args.prompt, obs_neg_list[n], n=n, sign=-1.0, precomputed_vl_embs=neg_vl)
        print(f"[rank {rank}] ({j_local+1}/{len(idx)}) pair {n} "
              f"(ds={int(drive_source_used[n])}): {time.time()-tp:.1f}s", flush=True)
    print(f"[rank {rank}] sketch inference done: {time.time()-t0:.1f}s", flush=True)

    for h in handles:
        h.remove()
    del omega_left
    torch.cuda.empty_cache()

    # ---- Dump sketch state ----
    rank_dir = scratch_root(args) / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rank {rank}] dumping sketch state to {rank_dir} ...", flush=True)
    t_dump = time.time()
    for ps in part_state:
        p_idx = ps["idx"]
        for t_pos in range(T_sel):
            t_id = sel_t[t_pos]
            write_raw(rank_dir / f"part{p_idx}_t{t_id}_W.bin", ps["W_list"][t_pos])
            np.save(rank_dir / f"part{p_idx}_t{t_id}_Y.npy", ps["Y_list"][t_pos].numpy())
            np.save(rank_dir / f"part{p_idx}_t{t_id}_mu.npy", ps["mu_list"][t_pos].numpy())
    print(f"[rank {rank}] dump: {time.time()-t_dump:.1f}s", flush=True)

    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        cfg_out = {
            "N": N_used, "k_target": args.k_target, "p_over": args.p_over,
            "partitions": partitions, "selected_timesteps": sel_t, "T_sel": T_sel,
            "sampling_steps": num_steps, "seed": args.seed, "sketch_seed": args.sketch_seed,
            "world_size": world_size, "prompt": args.prompt,
            "pos_npz": str(args.pos_npz), "neg_npz": str(args.neg_npz),
            "drive_source_filter": args.drive_source,
            "n_available_obs": int(n_avail),
            "selected_rows_first_8": [int(x) for x in selected_rows[:8]],
            "selected_rows_last_8":  [int(x) for x in selected_rows[-8:]],
            "drive_source_distribution": {
                "0_count": int((drive_source_used == 0).sum()),
                "1_count": int((drive_source_used == 1).sum()),
            },
            # Geometry (compatible with Cosmos config.json key names for downstream scripts)
            "P_tok": T_p_denoise,           # = action_horizon = 8
            "T_p_denoise": T_p_denoise,     # = action_horizon = 8
            "H_p": 1, "W_p": 1,             # no spatial dims; set to 1
            "D": D,                          # = inner_dim = 768
            "D_flat": D_flat,               # = action_horizon * inner_dim = 6144
            "L": L,
            "denoise_t_start": denoise_t_start,
            "denoise_t_end":   denoise_t_end,
            "n_state_tokens":  n_state_tokens,
            "action_horizon":  action_horizon,
            "inner_dim":       inner_dim,
            # DiT4DiT-specific metadata
            "model_id": "DiT4DiT",
            "ckpt_path": args.ckpt_path,
            "action_model_type": str(getattr(action_model.config, "action_model_type", "DiT-B")),
            "input_mode": "paired_obs_fixed_prompt_action_tokens_only",
            "W_dtype": "bfloat16", "mu_dtype": "float32",
        }
        (args.out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))

    (rank_dir / "DONE").write_text(
        f"rank {rank} sketch complete at {time.time()}\npairs={len(idx)}\n"
    )
    print(f"[rank {rank}] sketch: done", flush=True)


# -----------------------------------------------------------------------
# SVD mode  (identical algorithm to Cosmos version; just reads our config)
# -----------------------------------------------------------------------

def run_svd(args):
    cfg = json.loads((args.out_dir / "config.json").read_text())
    N          = cfg["N"]
    L          = cfg["L"]
    T_sel      = cfg["T_sel"]
    sel_t      = cfg["selected_timesteps"]
    D_flat     = cfg["D_flat"]
    k_target   = cfg["k_target"]
    p_over     = cfg["p_over"]
    k_plus_p   = k_target + p_over
    partitions = [(int(a), int(b)) for a, b in cfg["partitions"]]
    world_size = cfg.get("world_size", 1)

    if args.svd_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.svd_device)
    chunk = max(1, int(args.svd_chunk))
    print(f"finalize device: {device}  D-chunk: {chunk:,}  "
          f"D_flat: {D_flat:,}  T_sel: {T_sel}  world_size: {world_size}  "
          f"partitions: {partitions}", flush=True)
    print(f"denoise=[{cfg['denoise_t_start']}..{cfg['denoise_t_end']-1}]  "
          f"action_horizon={cfg.get('action_horizon')}  inner_dim={cfg.get('inner_dim')}",
          flush=True)

    scratch = scratch_root(args)
    for r in range(world_size):
        done = scratch / f"rank{r}" / "DONE"
        if not done.exists():
            raise RuntimeError(f"rank {r} sketch incomplete (missing {done})")

    layer_to_part = [0] * L
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for l in range(l_start, l_end + 1):
            layer_to_part[l] = p_idx

    Vs = [[None] * T_sel for _ in range(len(partitions))]
    sigmas = [[None] * T_sel for _ in range(len(partitions))]
    k_safes = [[0] * T_sel for _ in range(len(partitions))]
    c_per_part_t   = [[None] * T_sel for _ in range(len(partitions))]
    mu_norms_ppt   = [[None] * T_sel for _ in range(len(partitions))]
    c_norms_ppt    = [[None] * T_sel for _ in range(len(partitions))]

    for p_idx, (l_start, l_end) in enumerate(partitions):
        L_p = l_end - l_start + 1
        R_p = N * L_p
        OR = _part_OR(cfg, p_idx, R_p, k_plus_p)
        print(f"=== partition {p_idx} layers={l_start}-{l_end}: "
              f"R_p={R_p}, k+p={k_plus_p}, T_sel={T_sel}", flush=True)

        for t_pos, t_id in enumerate(sel_t):
            print(f"--- partition {p_idx}  t={t_id} ({t_pos+1}/{T_sel})", flush=True)

            if R_p < k_plus_p:
                print(f"  WARNING: R_p={R_p} < k+p={k_plus_p}; emitting zero V.", flush=True)
                Vs[p_idx][t_pos]       = torch.zeros(D_flat, k_target, dtype=torch.float32)
                sigmas[p_idx][t_pos]   = torch.zeros(k_target, dtype=torch.float32)
                c_per_part_t[p_idx][t_pos] = torch.zeros(L_p, k_target, dtype=torch.float32)
                mu_norms_ppt[p_idx][t_pos] = torch.zeros(L_p, dtype=torch.float32)
                c_norms_ppt[p_idx][t_pos]  = torch.zeros(L_p, dtype=torch.float32)
                continue

            Y_np = np.zeros((R_p, k_plus_p), dtype=np.float32)
            for r in range(world_size):
                Y_np += np.load(scratch / f"rank{r}" / f"part{p_idx}_t{t_id}_Y.npy")
            Y_t = torch.from_numpy(Y_np)

            print(f"  summing per-rank W ({k_plus_p} x {D_flat:,}) bf16 -> f32  "
                  f"world_size={world_size}", flush=True)
            W = torch.zeros(k_plus_p, D_flat, dtype=torch.float32)
            for r in range(world_size):
                Wr_bf = read_bf16_raw(scratch / f"rank{r}" / f"part{p_idx}_t{t_id}_W.bin",
                                       (k_plus_p, D_flat))
                for s in range(0, D_flat, chunk):
                    e = min(s + chunk, D_flat)
                    W[:, s:e].add_(Wr_bf[:, s:e].float())
                del Wr_bf

            Q_L, _ = torch.linalg.qr(Y_t)
            OR_Q = OR.T @ Q_L
            cond = torch.linalg.cond(OR_Q).item()
            print(f"  cond(Ω_R^T Q_L) = {cond:.3e}")
            del Q_L, Y_np, Y_t

            LU, pivots = torch.linalg.lu_factor(OR_Q.to(device))
            G_dev = torch.zeros(k_plus_p, k_plus_p, dtype=torch.float32, device=device)
            for s in range(0, D_flat, chunk):
                e = min(s + chunk, D_flat)
                W_chunk = W[:, s:e].to(device, non_blocking=True)
                B_chunk = torch.linalg.lu_solve(LU, pivots, W_chunk)
                G_dev.addmm_(B_chunk, B_chunk.T)
                del W_chunk, B_chunk
            if device.type == "cuda":
                torch.cuda.synchronize()
            G = G_dev.to("cpu")
            del G_dev

            eigvals, U = torch.linalg.eigh(G)
            eigvals = eigvals.flip(0)
            U = U.flip(1)
            sigma = eigvals.clamp(min=0).sqrt()

            TAU_REL = 1e-6
            TAU_VNORM = 1.05
            sigma_max = sigma[0].item()
            k_rank = int((sigma > TAU_REL * sigma_max).sum().item())
            k_pre  = min(k_target, R_p - p_over, k_rank)
            print(f"  σ_max={sigma_max:.3e}  k_rank={k_rank}  k_pre={k_pre}/k_target={k_target}", flush=True)

            U_k_dev = U[:, :k_pre].to(device).contiguous()
            sigma_pre_dev = sigma[:k_pre].to(device)
            V_norm_sq_dev = torch.zeros(k_pre, dtype=torch.float64, device=device)
            V_k = torch.zeros(D_flat, k_target, dtype=torch.float32)
            for s in range(0, D_flat, chunk):
                e = min(s + chunk, D_flat)
                W_chunk = W[:, s:e].to(device, non_blocking=True)
                B_chunk = torch.linalg.lu_solve(LU, pivots, W_chunk)
                V_chunk = B_chunk.T @ U_k_dev
                V_chunk.div_(sigma_pre_dev)
                V_norm_sq_dev.add_(V_chunk.double().pow(2).sum(dim=0))
                V_k[s:e, :k_pre] = V_chunk.to("cpu")
                del W_chunk, B_chunk, V_chunk

            V_col_norms = V_norm_sq_dev.sqrt().to("cpu").float()
            bad_norm = V_col_norms > TAU_VNORM
            n_bad = int(bad_norm.sum().item())
            if n_bad > 0:
                bad_idx = bad_norm.nonzero(as_tuple=True)[0]
                print(f"  zeroing {n_bad} cols with ‖V‖ > {TAU_VNORM}", flush=True)
                V_k[:, bad_idx] = 0
            else:
                print(f"  ‖V‖ max = {V_col_norms.max():.3e}", flush=True)
            k_safe = k_pre - n_bad

            sigma_k = torch.zeros(k_target, dtype=torch.float32)
            sigma_k[:k_pre] = sigma[:k_pre]
            if n_bad > 0:
                sigma_k[bad_norm.nonzero(as_tuple=True)[0]] = 0

            if device.type == "cuda":
                torch.cuda.synchronize()
                del LU, pivots, U_k_dev, sigma_pre_dev, V_norm_sq_dev
                torch.cuda.empty_cache()

            Vs[p_idx][t_pos]       = V_k
            sigmas[p_idx][t_pos]   = sigma_k
            k_safes[p_idx][t_pos]  = k_safe

            mu_np = np.zeros((L_p, D_flat), dtype=np.float32)
            for r in range(world_size):
                mu_np += np.load(scratch / f"rank{r}" / f"part{p_idx}_t{t_id}_mu.npy")
            mu_t = torch.from_numpy(mu_np)

            c_acc      = torch.zeros(L_p, k_target, dtype=torch.float32, device=device)
            mu_sq_acc  = torch.zeros(L_p, dtype=torch.float32, device=device)
            for s in range(0, D_flat, chunk):
                e = min(s + chunk, D_flat)
                V_chunk  = V_k[s:e].to(device, non_blocking=True)
                mu_chunk = mu_t[:, s:e].to(device, non_blocking=True)
                c_acc.add_(mu_chunk @ V_chunk)
                mu_sq_acc.add_(mu_chunk.pow(2).sum(dim=1))
                del V_chunk, mu_chunk
            if device.type == "cuda":
                torch.cuda.synchronize()
            c_per_part_t[p_idx][t_pos]  = c_acc.to("cpu")
            mu_norms_ppt[p_idx][t_pos]  = mu_sq_acc.sqrt().to("cpu")
            c_norms_ppt[p_idx][t_pos]   = c_acc.norm(dim=1).to("cpu")

            del W, G, U, mu_t
            gc.collect()
        del OR
        gc.collect()

    # Save per-partition V files
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for t_pos, t_id in enumerate(sel_t):
            V_k   = Vs[p_idx][t_pos]
            V_bf  = V_k.to(torch.bfloat16).contiguous()
            V_path = args.out_dir / (
                f"V_part{p_idx}_layers{l_start}-{l_end}_t{t_id}_k{V_bf.shape[1]}.pt"
            )
            torch.save({
                "V": V_bf,
                "c_per_layer": c_per_part_t[p_idx][t_pos],
                "c_layer_indices": list(range(l_start, l_end + 1)),
                "singular_values": sigmas[p_idx][t_pos],
                "l_start": l_start, "l_end": l_end, "L_p": l_end - l_start + 1,
                "R_p": N * (l_end - l_start + 1),
                "timestep": int(t_id), "k_eff": V_bf.shape[1],
                "k_safe": k_safes[p_idx][t_pos],
                "P_tok": cfg["P_tok"], "D": cfg["D"], "D_flat": D_flat,
                "T_p_denoise": cfg["T_p_denoise"],
                "denoise_t_start": cfg["denoise_t_start"],
                "denoise_t_end": cfg["denoise_t_end"],
                "mu_norms_per_layer": mu_norms_ppt[p_idx][t_pos],
                "c_norms_per_layer": c_norms_ppt[p_idx][t_pos],
                "V_dtype": "bfloat16",
            }, V_path)
            print(f"saved {V_path} ({V_path.stat().st_size/1e9:.2f} GB)", flush=True)

    # Build global summary
    c_means_global  = torch.zeros(L, T_sel, k_target, dtype=torch.float32)
    mu_norms_global = torch.zeros(L, T_sel, dtype=torch.float32)
    c_norms_global  = torch.zeros(L, T_sel, dtype=torch.float32)
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for t_pos in range(T_sel):
            c_means_global[l_start:l_end+1, t_pos]  = c_per_part_t[p_idx][t_pos]
            mu_norms_global[l_start:l_end+1, t_pos]  = mu_norms_ppt[p_idx][t_pos]
            c_norms_global[l_start:l_end+1, t_pos]   = c_norms_ppt[p_idx][t_pos]
    captured = (c_norms_global / mu_norms_global.clamp(min=1e-12)) ** 2

    summary_path = args.out_dir / "svd_summary.pt"
    torch.save({
        "c_means": c_means_global,
        "mu_norms": mu_norms_global,
        "c_norms": c_norms_global,
        "captured": captured,
        "singular_values": sigmas,
        "k_safes": k_safes,
        "layer_partitions": partitions,
        "layer_to_part": layer_to_part,
        "selected_timesteps": sel_t,
        "T_sel": T_sel,
        "per_partition_V": True,
        "per_layer_c": True,
        "per_timestep_V": True,
        "per_timestep_c": True,
        **cfg,
    }, summary_path)
    print(f"saved {summary_path} ({summary_path.stat().st_size/1e6:.2f} MB)", flush=True)

    if not args.keep_scratch:
        shutil.rmtree(scratch)
        print(f"cleaned scratch: {scratch}", flush=True)


def _part_OR(cfg, p_idx, R_p, k_plus_p):
    gen_cpu = torch.Generator(device="cpu").manual_seed(cfg["sketch_seed"] + 1)
    OR_self = None
    for i, (a, b) in enumerate(cfg["partitions"]):
        L_p_i = int(b) - int(a) + 1
        R_p_i = cfg["N"] * L_p_i
        OR_i = torch.randn(R_p_i, k_plus_p, generator=gen_cpu, dtype=torch.float32)
        if i == p_idx:
            OR_self = OR_i
    assert OR_self is not None and OR_self.shape == (R_p, k_plus_p)
    return OR_self


def main():
    args = parse_args()
    if args.mode == "sketch":
        run_sketch(args)
    else:
        run_svd(args)


if __name__ == "__main__":
    main()
