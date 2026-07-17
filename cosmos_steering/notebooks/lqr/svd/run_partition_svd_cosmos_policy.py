#!/usr/bin/env python
"""Per-(partition, timestep) randomized SVD of Cosmos-Policy-LIBERO-Predict2-2B
DiT block activations, restricted to the denoising slots, with a fixed
contrastive prompt pair and per-pair observation inputs.

Differences from the original run_partition_svd_cosmos_policy.py:

  * Activations are restricted to the 5 denoising slots (T_idx 4..8 for
    LIBERO state_t=9). After patch-embed each slot contributes 14*14=196
    tokens × D=2048 channels, so D_flat = 5 * 196 * 2048 = 2,007,040
    (vs. 9 * 196 * 2048 = 3,612,672 for the full sequence).
  * Pooling for the right-singular projection is across (pairs, layers in
    partition) only — NOT across timesteps. One V is produced per
    (partition, timestep). Each (partition, timestep) sketch has
    R_p = N * L_p rows.
  * Contrastive vectors are per (layer, timestep). μ_{l,t} is averaged
    over pairs only; c_{l,t} = μ_{l,t} @ V_{p(l), t}.
  * Contrastive pairs are fixed: all N pairs use the SAME pair of prompts
    (--orig-prompt / --disturbed-prompt). Per-pair variation comes from
    per-pair observation inputs (proprio, wrist image, primary image)
    loaded from --inputs-npz. Both the positive and negative pass for
    pair n use obs[n], so the (pos − neg) contrast isolates prompt-induced
    activation differences.
  * Slurm-ready: the shell driver wraps each rank in its own sbatch job.

Modes
-----
--mode sketch : one Python invocation **per rank**. Pin one GPU via
                CUDA_VISIBLE_DEVICES, pass --rank R --world-size W, and the
                process iterates pairs `range(R, N, W)` with forward hooks
                accumulating per-partition-per-timestep Y / W / μ.

--mode svd    : single-rank finalize. Reads per-rank sketch state for each
                (partition, timestep), sums across ranks, runs Halko
                one-pass, projects μ_{l,t} → c_{l,t}, and saves
                V_part{p}_t{t}*.pt and svd_summary.pt.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path


def _setup_env():
    """Match notebooks/_setup.py — must run BEFORE any cosmos_policy import."""
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

    libero_cfg = os.environ.get("LIBERO_CONFIG_PATH", "/work/nvme/bhde/jhong7/.libero")
    Path(libero_cfg).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", libero_cfg)

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_setup_env()

import numpy as np
import torch


DEFAULT_ORIG_PROMPT = (
    "put both the alphabet soup and the tomato sauce in the basket"
)
DEFAULT_DISTURBED_PROMPT = (
    "put both the alphabet soup and the tomato sauce in the basket. "
    "the cream cheese, ketchup, orange juice, milk, and butter are also on the table."
)
DEFAULT_INPUTS_NPZ = (
    "/u/jhong7/Workspace/cosmos-policy/notebooks/lqr/inputs/"
    "policy_inputs/libero_10__task00/inputs.npz"
)


# ====================================================================
# CLI
# ====================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sketch", "svd"], required=True)
    ap.add_argument("--rank", type=int,
                    default=int(os.environ.get("RANK", 0)),
                    help="this process's rank in [0, world_size). Sketch "
                         "mode only; svd reads all ranks.")
    ap.add_argument("--world-size", type=int,
                    default=int(os.environ.get("WORLD_SIZE", 1)),
                    help="total number of parallel sketch processes")

    ap.add_argument("--orig-prompt", type=str, default=DEFAULT_ORIG_PROMPT)
    ap.add_argument("--disturbed-prompt", type=str, default=DEFAULT_DISTURBED_PROMPT)
    ap.add_argument("--inputs-npz", type=Path, default=Path(DEFAULT_INPUTS_NPZ),
                    help="npz with keys primary_images, wrist_images, proprios")
    ap.add_argument("--N", type=int, default=64,
                    help="number of observations (pairs) to use. The script "
                         "will use the first N rows of --inputs-npz.")

    ap.add_argument("--k-target", type=int, default=64)
    ap.add_argument("--p-over", type=int, default=10)
    ap.add_argument("--partitions", type=str, default="0-9,10-18,19-27",
                    help="inclusive layer ranges tiling [0, num_layers)")
    ap.add_argument("--num-layers", type=int, default=28,
                    help="number of DiT blocks (Cosmos-Policy 2B = 28)")

    ap.add_argument("--sampling-steps", type=int, default=5,
                    help="total denoising steps fed to generate_samples_from_batch")
    ap.add_argument("--timesteps", type=str, default="all",
                    help="comma-sep step indices to extract from "
                         "(0..sampling_steps-1), or 'all'. e.g., '0,2,4'")
    ap.add_argument("--guide-scale", type=float, default=1.0,
                    help="CFG scale used at inference (1.0 = no CFG)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sketch-seed", type=int, default=0xC057)

    ap.add_argument("--ckpt-path", type=str,
                    default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    ap.add_argument("--config-name", type=str,
                    default="cosmos_predict2_2b_480p_libero__inference_only")
    ap.add_argument("--config-file", type=str,
                    default="cosmos_policy/config/config.py")
    ap.add_argument("--dataset-stats-path", type=str, default=None,
                    help="defaults to <ckpt-path>/libero_dataset_statistics.json")
    ap.add_argument("--t5-cache-path", type=str, default=None,
                    help="defaults to <ckpt-path>/libero_t5_embeddings.pkl")

    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scratch-dir", type=Path, default=None)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--svd-device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--svd-chunk", type=int, default=1_000_000,
                    help="row chunk size along D_flat for GPU streaming")
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
    sel = sorted({int(x) for x in spec.split(",")})
    if any(t < 0 or t >= T_total for t in sel):
        raise ValueError(f"timesteps {sel} out of range [0, {T_total - 1}]")
    return sel


def my_indices(N, rank, world_size):
    return list(range(rank, N, world_size))


def scratch_root(args):
    return args.scratch_dir or (args.out_dir / "scratch")


# ====================================================================
# Disk I/O helpers — raw byte dumps
# ====================================================================

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


# ====================================================================
# Observation loading
# ====================================================================

def load_observations(npz_path: Path, N: int):
    """Load the first N (proprio, wrist, primary) tuples from npz.

    Returns a list of obs dicts in the format get_action expects.
    """
    data = np.load(npz_path)
    avail = int(data["proprios"].shape[0])
    if N > avail:
        raise ValueError(
            f"requested N={N} but {npz_path} only has {avail} entries"
        )
    obs_list = []
    for i in range(N):
        obs_list.append({
            "wrist_image": data["wrist_images"][i].copy(),
            "primary_image": data["primary_images"][i].copy(),
            "proprio": data["proprios"][i].astype(np.float32).copy(),
        })
    return obs_list, avail


# ====================================================================
# Sketch mode
# ====================================================================

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
    assert 0 <= rank < world_size, (
        f"--rank {rank} not in [0, --world-size {world_size})"
    )

    print(f"[rank {rank}/{world_size}] sketch: partitions={partitions}  "
          f"selected_steps={sel_t}  k+p={k_plus_p}", flush=True)

    # ---- Load observations ----
    obs_list, n_avail = load_observations(args.inputs_npz, args.N)
    idx = my_indices(args.N, rank, world_size)
    print(f"[rank {rank}] inputs: {args.inputs_npz} (avail={n_avail}, "
          f"using N={args.N}); {len(idx)} pairs assigned: "
          f"{idx[:5]}{'...' if len(idx) > 5 else ''}", flush=True)

    # ---- Build PolicyEvalConfig and load model ----
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        PolicyEvalConfig,
    )
    from cosmos_policy.experiments.robot.cosmos_utils import (
        COSMOS_IMAGE_SIZE,
        get_action,
        get_model,
        get_t5_embedding_from_cache,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    ckpt_path = args.ckpt_path
    dataset_stats_path = (
        args.dataset_stats_path
        or f"{ckpt_path}/libero_dataset_statistics.json"
    )
    t5_cache_path = (
        args.t5_cache_path
        or f"{ckpt_path}/libero_t5_embeddings.pkl"
    )

    cfg = PolicyEvalConfig(
        config=args.config_name,
        ckpt_path=ckpt_path,
        config_file=args.config_file,
        dataset_stats_path=dataset_stats_path,
        t5_text_embeddings_path=t5_cache_path,
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=False,
        unnormalize_actions=False,
        chunk_size=16,
        num_open_loop_steps=16,
        trained_with_image_aug=False,
        use_jpeg_compression=False,
        flip_images=False,
        num_denoising_steps_action=args.sampling_steps,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )

    print(f"[rank {rank}] loading dataset stats + T5 cache + model ...",
          flush=True)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path,
                                    worker_id=rank)
    model, _ = get_model(cfg)

    # Precompute T5 for both prompts so the in-loop call hits cache.
    # (avoids any race on save_t5_text_embeddings_cache across ranks)
    _ = get_t5_embedding_from_cache(args.orig_prompt)
    _ = get_t5_embedding_from_cache(args.disturbed_prompt)

    L_runtime = len(model.net.blocks)
    assert L_runtime == L, (
        f"L mismatch: model has {L_runtime} blocks, --num-layers={L}"
    )
    state_t = model.config.state_t
    min_cond = model.config.min_num_conditional_frames
    # Denoising slots are T_idx in [min_cond, state_t). For LIBERO:
    # state_t=9, min_cond=4 -> slots 4..8 (action, future_proprio,
    # future_wrist, future_third_person, value).
    denoise_t_start = min_cond
    denoise_t_end = state_t

    spatial_compression = 8
    H_lat = W_lat = COSMOS_IMAGE_SIZE // spatial_compression
    p_t = model.net.patch_temporal
    p_s = model.net.patch_spatial
    assert p_t == 1, (
        f"patch_temporal={p_t}; slot slicing only correct for p_t=1"
    )
    T_p_full = state_t // p_t
    H_p = H_lat // p_s
    W_p = W_lat // p_s
    T_p_denoise = denoise_t_end - denoise_t_start  # since p_t == 1
    P_tok = T_p_denoise * H_p * W_p
    D = model.net.model_channels
    D_flat = P_tok * D

    print(f"  state_t={state_t}  min_cond={min_cond}  "
          f"denoise_slots=[{denoise_t_start}..{denoise_t_end - 1}]  "
          f"H_lat={H_lat}  W_lat={W_lat}  p_t={p_t}  p_s={p_s}  "
          f"T_p_denoise={T_p_denoise}  H_p={H_p}  W_p={W_p}  P_tok={P_tok}  "
          f"D={D}  D_flat={D_flat:,}", flush=True)

    # ---- Allocate per-partition / per-timestep sketch state ----
    # R_p (per timestep) = N * L_p — no T_sel factor.
    # For each partition, Y, W, mu are lists of length T_sel.
    BUF_SIZE = 50
    part_state = []
    gen_cpu = torch.Generator(device="cpu").manual_seed(args.sketch_seed + 1)
    for p_idx, (l_start, l_end) in enumerate(partitions):
        L_p = l_end - l_start + 1
        R_p = args.N * L_p
        # OR is shared across timesteps within the partition.
        OR = torch.randn(R_p, k_plus_p, generator=gen_cpu, dtype=torch.float32)
        # Per-timestep state.
        Y_list = [torch.zeros(R_p, k_plus_p, dtype=torch.float32)
                  for _ in range(T_sel)]
        W_list = [torch.zeros(k_plus_p, D_flat, dtype=torch.bfloat16)
                  for _ in range(T_sel)]
        mu_list = [torch.zeros(L_p, D_flat, dtype=torch.float32)
                   for _ in range(T_sel)]
        e_buf = torch.zeros(BUF_SIZE, D_flat, dtype=torch.bfloat16)
        part_state.append({
            "idx": p_idx, "l_start": l_start, "l_end": l_end, "L_p": L_p,
            "R_p": R_p, "OR": OR,
            "Y_list": Y_list, "W_list": W_list, "mu_list": mu_list,
            "e_buf": e_buf,
            # meta is (row, l_local, t_pos)
            "meta_buf": [None] * BUF_SIZE, "buf_count": 0,
        })

    layer_to_part = [0] * L
    for ps in part_state:
        for l in range(ps["l_start"], ps["l_end"] + 1):
            layer_to_part[l] = ps["idx"]

    # Cosmos-Policy inference path (via get_action) runs ONE forward per
    # denoising step. Empirically: 5 sampling steps -> 5 layer-0 fires per
    # run_one call. (The CFG cond+uncond split happens only on a different
    # codepath.) If you ever switch to that path, set PASSES_PER_STEP=2 and
    # the hook will skip every other pass.
    PASSES_PER_STEP = 1

    # ---- Omega_left for the Y sketch (D_flat, k+p) on GPU ----
    device = torch.device("cuda:0")
    gen_gpu = torch.Generator(device=device).manual_seed(args.sketch_seed)
    omega_left = torch.randn(D_flat, k_plus_p, generator=gen_gpu,
                              dtype=torch.bfloat16, device=device)

    # ---- Driver state shared between hooks and the per-pair loop ----
    state = {"pass_idx": -1, "n": -1, "sign": 0.0}

    # μ_{l,t} denominator: each (layer, timestep) is hit once per pair within
    # a single signed pass, so we average across N pairs only.
    mu_denom = float(args.N)

    def flush_partition(p_idx):
        """Drain one partition's CPU buffer into per-timestep W and μ."""
        pbuf = part_state[p_idx]
        n_in_buf = pbuf["buf_count"]
        if n_in_buf == 0:
            return
        sign = state["sign"]
        E_b = pbuf["e_buf"][:n_in_buf]            # bf16 (n_in_buf, D_flat)
        metas = pbuf["meta_buf"][:n_in_buf]       # list of (row, l_local, t_pos)
        # Group by timestep position.
        per_t = {}
        for j, (row, l_local, t_pos) in enumerate(metas):
            per_t.setdefault(t_pos, []).append((j, row, l_local))
        for t_pos, items in per_t.items():
            idxs_j = [it[0] for it in items]
            rows = [it[1] for it in items]
            E_sub = E_b[idxs_j]                                  # bf16 (m, D)
            OR_sub_bf = pbuf["OR"][rows].T.to(torch.bfloat16)    # bf16 (k+p, m)
            pbuf["W_list"][t_pos].addmm_(OR_sub_bf, E_sub, alpha=sign)
            for j, _, l_local in items:
                pbuf["mu_list"][t_pos][l_local].add_(
                    E_b[j].float(), alpha=sign / mu_denom
                )
        pbuf["buf_count"] = 0

    def flush_all():
        for p in range(len(part_state)):
            flush_partition(p)

    def make_hook(layer_idx):
        p_idx = layer_to_part[layer_idx]
        pbuf = part_state[p_idx]
        l_start = pbuf["l_start"]
        L_p = pbuf["L_p"]

        def hook(module, inputs, output):
            if layer_idx == 0:
                state["pass_idx"] += 1
            pass_idx = state["pass_idx"]
            # Skip the unconditional pass (every other forward).
            if pass_idx % PASSES_PER_STEP != 0:
                return output
            step = pass_idx // PASSES_PER_STEP
            if step not in sel_t_set:
                return output

            # Block output is (B, T_p, H_p, W_p, D). Take batch 0 and slice the
            # denoising slots along the T_p axis.
            act_bf = (
                output.detach()[0, denoise_t_start:denoise_t_end, :, :, :]
                .reshape(-1)
            )
            if act_bf.numel() != D_flat:
                raise RuntimeError(
                    f"activation size mismatch: got {act_bf.numel()}, "
                    f"expected D_flat={D_flat} at layer {layer_idx} step={step}"
                )

            t_pos = sel_t_pos[step]
            l_local = layer_idx - l_start
            row = state["n"] * L_p + l_local  # R_p = N * L_p

            # Y_list[t_pos] += sign · (act @ omega_left)
            y_chunk = (act_bf @ omega_left).float().cpu()
            pbuf["Y_list"][t_pos][row].add_(y_chunk, alpha=state["sign"])

            # Stage activation for W and μ flush.
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
        for i, b in enumerate(model.net.blocks)
    ]

    def run_one(prompt, obs, n, sign):
        state["pass_idx"] = -1
        state["n"] = int(n)
        state["sign"] = float(sign)
        with torch.inference_mode():
            _ = get_action(
                cfg, model, dataset_stats, obs, prompt,
                seed=args.seed,
                randomize_seed=False,
                num_denoising_steps_action=args.sampling_steps,
                generate_future_state_and_value_in_parallel=True,
            )
        flush_all()
        expected = PASSES_PER_STEP * args.sampling_steps
        assert state["pass_idx"] + 1 == expected, (
            f"got {state['pass_idx']+1} layer-0 fires, expected "
            f"{expected} ({args.sampling_steps} steps × {PASSES_PER_STEP} "
            f"passes/step). If Cosmos-Policy was patched to skip the "
            f"unconditional forward, set PASSES_PER_STEP=1."
        )

    t0 = time.time()
    for j_local, n in enumerate(idx):
        tp = time.time()
        obs_n = obs_list[n]
        run_one(args.orig_prompt, obs_n, n=n, sign=+1.0)
        run_one(args.disturbed_prompt, obs_n, n=n, sign=-1.0)
        print(f"[rank {rank}] ({j_local+1}/{len(idx)}) pair {n}: "
              f"{time.time()-tp:.1f}s", flush=True)
    print(f"[rank {rank}] sketch inference done: {time.time()-t0:.1f}s",
          flush=True)

    for h in handles:
        h.remove()
    del omega_left
    torch.cuda.empty_cache()

    # ---- Per-rank dump ----
    rank_dir = scratch_root(args) / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rank {rank}] dumping sketch state to {rank_dir} ...", flush=True)
    t_dump = time.time()
    for ps in part_state:
        p_idx = ps["idx"]
        for t_pos in range(T_sel):
            t_id = sel_t[t_pos]
            write_raw(rank_dir / f"part{p_idx}_t{t_id}_W.bin",
                       ps["W_list"][t_pos])
            np.save(rank_dir / f"part{p_idx}_t{t_id}_Y.npy",
                     ps["Y_list"][t_pos].numpy())
            np.save(rank_dir / f"part{p_idx}_t{t_id}_mu.npy",
                     ps["mu_list"][t_pos].numpy())
    print(f"[rank {rank}] dump: {time.time()-t_dump:.1f}s", flush=True)

    # ---- Persist config (rank 0 only; other ranks would race) ----
    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        cfg_out = {
            "N": args.N, "k_target": args.k_target, "p_over": args.p_over,
            "partitions": partitions,
            "selected_timesteps": sel_t,
            "T_sel": T_sel,
            "sampling_steps": args.sampling_steps,
            "guide_scale": args.guide_scale,
            "seed": args.seed,
            "sketch_seed": args.sketch_seed,
            "world_size": world_size,
            "orig_prompt": args.orig_prompt,
            "disturbed_prompt": args.disturbed_prompt,
            "inputs_npz": str(args.inputs_npz),
            "n_available_obs": int(n_avail),
            "P_tok": P_tok, "T_p_denoise": T_p_denoise,
            "H_p": H_p, "W_p": W_p, "D": D,
            "D_flat": D_flat, "L": L, "state_t": state_t,
            "min_num_conditional_frames": min_cond,
            "denoise_t_start": denoise_t_start,
            "denoise_t_end": denoise_t_end,
            "ckpt_path": ckpt_path, "config_name": args.config_name,
            "model_id": "Cosmos-Policy-LIBERO-Predict2-2B",
            "input_mode": "fixed_prompt_pair_per_obs",
            "W_dtype": "bfloat16",
            "mu_dtype": "float32",
        }
        (args.out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))

    (rank_dir / "DONE").write_text(
        f"rank {rank} sketch complete at {time.time()}\n"
        f"pairs={len(idx)}\n"
    )
    print(f"[rank {rank}] sketch: done", flush=True)


# ====================================================================
# SVD mode — one V per (partition, timestep), c per (layer, timestep)
# ====================================================================

def run_svd(args):
    cfg = json.loads((args.out_dir / "config.json").read_text())
    N = cfg["N"]
    L = cfg["L"]
    T_sel = cfg["T_sel"]
    sel_t = cfg["selected_timesteps"]
    D_flat = cfg["D_flat"]
    k_target = cfg["k_target"]
    p_over = cfg["p_over"]
    k_plus_p = k_target + p_over
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

    scratch = scratch_root(args)
    for r in range(world_size):
        done = scratch / f"rank{r}" / "DONE"
        if not done.exists():
            raise RuntimeError(f"rank {r} sketch incomplete (missing {done})")

    layer_to_part = [0] * L
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for l in range(l_start, l_end + 1):
            layer_to_part[l] = p_idx

    # Results indexed by (partition, t_pos).
    Vs = [[None] * T_sel for _ in range(len(partitions))]
    sigmas = [[None] * T_sel for _ in range(len(partitions))]
    k_safes = [[0] * T_sel for _ in range(len(partitions))]
    # c_per_part_t[p][t_pos]: (L_p, k_target)
    c_per_part_t = [[None] * T_sel for _ in range(len(partitions))]
    mu_norms_per_part_t = [[None] * T_sel for _ in range(len(partitions))]
    c_norms_per_part_t = [[None] * T_sel for _ in range(len(partitions))]

    for p_idx, (l_start, l_end) in enumerate(partitions):
        L_p = l_end - l_start + 1
        R_p = N * L_p
        OR = part_OR(cfg, p_idx, R_p, k_plus_p)
        print(f"=== partition {p_idx} layers={l_start}-{l_end}: "
              f"R_p={R_p}, k+p={k_plus_p}, T_sel={T_sel}", flush=True)

        for t_pos, t_id in enumerate(sel_t):
            print(f"--- partition {p_idx}  t={t_id} ({t_pos+1}/{T_sel})",
                  flush=True)

            if R_p < k_plus_p:
                print(f"  WARNING: R_p={R_p} < k+p={k_plus_p}; emitting zero V.",
                      flush=True)
                Vs[p_idx][t_pos] = torch.zeros(D_flat, k_target,
                                                dtype=torch.float32)
                sigmas[p_idx][t_pos] = torch.zeros(k_target, dtype=torch.float32)
                c_per_part_t[p_idx][t_pos] = torch.zeros(L_p, k_target,
                                                          dtype=torch.float32)
                mu_norms_per_part_t[p_idx][t_pos] = torch.zeros(
                    L_p, dtype=torch.float32)
                c_norms_per_part_t[p_idx][t_pos] = torch.zeros(
                    L_p, dtype=torch.float32)
                continue

            # Sum Y across ranks.
            Y_np = np.zeros((R_p, k_plus_p), dtype=np.float32)
            for r in range(world_size):
                Y_np += np.load(
                    scratch / f"rank{r}" / f"part{p_idx}_t{t_id}_Y.npy"
                )
            Y_t = torch.from_numpy(Y_np)

            # Sum W across ranks (bf16 -> f32 accumulate).
            print(f"  summing per-rank W ({k_plus_p} x {D_flat:,}) bf16 -> "
                  f"f32 ({k_plus_p*D_flat*4/1e9:.2f} GB)  across world_size="
                  f"{world_size}", flush=True)
            W = torch.zeros(k_plus_p, D_flat, dtype=torch.float32)
            for r in range(world_size):
                Wr_bf = read_bf16_raw(
                    scratch / f"rank{r}" / f"part{p_idx}_t{t_id}_W.bin",
                    (k_plus_p, D_flat),
                )
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
            G_dev = torch.zeros(k_plus_p, k_plus_p, dtype=torch.float32,
                                device=device)
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
            k_pre = min(k_target, R_p - p_over, k_rank)
            print(f"  σ_max={sigma_max:.3e}  "
                  f"k_rank(τ_rel={TAU_REL:.0e})={k_rank}  "
                  f"k_pre={k_pre}/k_target={k_target}", flush=True)

            U_k_dev = U[:, :k_pre].to(device).contiguous()
            sigma_pre_dev = sigma[:k_pre].to(device)
            V_norm_sq_dev = torch.zeros(k_pre, dtype=torch.float64,
                                          device=device)
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
                print(f"  zeroing {n_bad} cols with ‖V‖ > {TAU_VNORM} "
                      f"(max ‖V‖ before zeroing: "
                      f"{V_col_norms[bad_idx].max():.3e})", flush=True)
                V_k[:, bad_idx] = 0
            else:
                print(f"  ‖V‖ max = {V_col_norms.max():.3e}  "
                      f"(all ≤ {TAU_VNORM})", flush=True)
            k_safe = k_pre - n_bad

            sigma_k = torch.zeros(k_target, dtype=torch.float32)
            sigma_k[:k_pre] = sigma[:k_pre]
            if n_bad > 0:
                sigma_k[bad_idx] = 0

            if device.type == "cuda":
                torch.cuda.synchronize()
                del LU, pivots, U_k_dev, sigma_pre_dev, V_norm_sq_dev
                torch.cuda.empty_cache()

            Vs[p_idx][t_pos] = V_k
            sigmas[p_idx][t_pos] = sigma_k
            k_safes[p_idx][t_pos] = k_safe
            print(f"  top-10 σ: {[f'{s:.3g}' for s in sigma[:10].tolist()]}  "
                  f"k_safe={k_safe} (padded to k_target={k_target})", flush=True)

            # ---- Project μ_{l,t} → c_{l,t} for each layer l in this partition. ----
            mu_np = np.zeros((L_p, D_flat), dtype=np.float32)
            for r in range(world_size):
                mu_r = np.load(
                    scratch / f"rank{r}" / f"part{p_idx}_t{t_id}_mu.npy"
                )
                assert mu_r.shape == (L_p, D_flat), (
                    f"rank {r} part {p_idx} t {t_id} μ shape {mu_r.shape}, "
                    f"expected ({L_p}, {D_flat})"
                )
                mu_np += mu_r
            mu_t = torch.from_numpy(mu_np)  # f32 (L_p, D_flat)

            c_acc = torch.zeros(L_p, k_target, dtype=torch.float32,
                                 device=device)
            mu_sq_acc = torch.zeros(L_p, dtype=torch.float32, device=device)
            for s in range(0, D_flat, chunk):
                e = min(s + chunk, D_flat)
                V_chunk = V_k[s:e].to(device, non_blocking=True)
                mu_chunk = mu_t[:, s:e].to(device, non_blocking=True)
                c_acc.add_(mu_chunk @ V_chunk)
                mu_sq_acc.add_(mu_chunk.pow(2).sum(dim=1))
                del V_chunk, mu_chunk
            if device.type == "cuda":
                torch.cuda.synchronize()
            c_per_part_t[p_idx][t_pos] = c_acc.to("cpu")
            mu_norms_per_part_t[p_idx][t_pos] = mu_sq_acc.sqrt().to("cpu")
            c_norms_per_part_t[p_idx][t_pos] = c_acc.norm(dim=1).to("cpu")
            mu_mean = mu_norms_per_part_t[p_idx][t_pos].mean()
            c_mean = c_norms_per_part_t[p_idx][t_pos].mean()
            cap_mean = ((c_norms_per_part_t[p_idx][t_pos] /
                         mu_norms_per_part_t[p_idx][t_pos].clamp(min=1e-12))
                         ** 2).mean()
            print(f"  per-layer ‖μ_l‖ mean={mu_mean:.3e}  "
                  f"‖c_l‖ mean={c_mean:.3e}  captured mean={cap_mean:.3f}",
                  flush=True)

            del W, G, U, mu_t
            gc.collect()

        del OR
        gc.collect()

    # ---- Save per-(partition, timestep) V + per-layer contrastive vectors. ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for t_pos, t_id in enumerate(sel_t):
            V_k = Vs[p_idx][t_pos]
            sigma_k = sigmas[p_idx][t_pos]
            c_layers = c_per_part_t[p_idx][t_pos]                # (L_p, k_target)
            mu_norms_p = mu_norms_per_part_t[p_idx][t_pos]       # (L_p,)
            c_norms_p = c_norms_per_part_t[p_idx][t_pos]         # (L_p,)
            V_bf = V_k.to(torch.bfloat16).contiguous()
            V_path = args.out_dir / (
                f"V_part{p_idx}_layers{l_start}-{l_end}"
                f"_t{t_id}_k{V_bf.shape[1]}.pt"
            )
            torch.save({
                "V": V_bf,
                "c_per_layer": c_layers,                # (L_p, k_target)
                "c_layer_indices": list(range(l_start, l_end + 1)),
                "singular_values": sigma_k,
                "l_start": l_start, "l_end": l_end, "L_p": l_end - l_start + 1,
                "R_p": N * (l_end - l_start + 1),
                "timestep": int(t_id),
                "k_eff": V_bf.shape[1],
                "k_safe": k_safes[p_idx][t_pos],
                "P_tok": cfg["P_tok"], "D": cfg["D"], "D_flat": D_flat,
                "T_p_denoise": cfg["T_p_denoise"],
                "denoise_t_start": cfg["denoise_t_start"],
                "denoise_t_end": cfg["denoise_t_end"],
                "mu_norms_per_layer": mu_norms_p,       # (L_p,)
                "c_norms_per_layer": c_norms_p,         # (L_p,)
                "V_dtype": "bfloat16",
            }, V_path)
            print(f"saved {V_path} ({V_path.stat().st_size / 1e9:.2f} GB)",
                  flush=True)

    # ---- Build global per-(layer, timestep) summary tensors. ----
    c_means_global = torch.zeros(L, T_sel, k_target, dtype=torch.float32)
    mu_norms_global = torch.zeros(L, T_sel, dtype=torch.float32)
    c_norms_global = torch.zeros(L, T_sel, dtype=torch.float32)
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for t_pos in range(T_sel):
            c_means_global[l_start:l_end + 1, t_pos] = c_per_part_t[p_idx][t_pos]
            mu_norms_global[l_start:l_end + 1, t_pos] = mu_norms_per_part_t[p_idx][t_pos]
            c_norms_global[l_start:l_end + 1, t_pos] = c_norms_per_part_t[p_idx][t_pos]
    captured = (c_norms_global / mu_norms_global.clamp(min=1e-12)) ** 2

    summary_path = args.out_dir / "svd_summary.pt"
    torch.save({
        "c_means": c_means_global,          # (L, T_sel, k_target)
        "mu_norms": mu_norms_global,        # (L, T_sel)
        "c_norms": c_norms_global,          # (L, T_sel)
        "captured": captured,               # (L, T_sel)
        "singular_values": sigmas,          # nested list [P][T_sel] of (k_target,)
        "k_safes": k_safes,                 # nested list [P][T_sel]
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
    print(f"saved {summary_path} "
          f"({summary_path.stat().st_size / 1e6:.2f} MB)", flush=True)

    print("\ncaptured ‖c_{l,t}‖² / ‖μ_{l,t}‖² per (layer, timestep):")
    header = f"{'layer':>5s} {'part':>4s}  " + "  ".join(
        [f"t={t:>2d}" for t in sel_t]
    )
    print(header)
    for l in range(L):
        row = f"{l:>5d} {layer_to_part[l]:>4d}  " + "  ".join(
            [f"{captured[l, t_pos].item():>5.3f}" for t_pos in range(T_sel)]
        )
        print(row)

    if not args.keep_scratch:
        shutil.rmtree(scratch)
        print(f"cleaned scratch: {scratch}", flush=True)


def part_OR(cfg, p_idx, R_p, k_plus_p):
    """Regenerate the random sketch matrix used in the sketch pass.

    Same seed (sketch_seed+1) and same RNG-consumption order as the sketch
    worker, so each partition's OR matches what was used to compute Y.
    """
    gen_cpu = torch.Generator(device="cpu").manual_seed(cfg["sketch_seed"] + 1)
    OR_self = None
    for i, (a, b) in enumerate(cfg["partitions"]):
        L_p_i = int(b) - int(a) + 1
        R_p_i = cfg["N"] * L_p_i  # no T_sel factor
        OR_i = torch.randn(R_p_i, k_plus_p, generator=gen_cpu,
                            dtype=torch.float32)
        if i == p_idx:
            OR_self = OR_i
    assert OR_self is not None
    assert OR_self.shape == (R_p, k_plus_p), (
        f"OR shape mismatch: got {OR_self.shape}, expected ({R_p}, {k_plus_p})"
    )
    return OR_self


def main():
    args = parse_args()
    if args.mode == "sketch":
        run_sketch(args)
    else:
        run_svd(args)


if __name__ == "__main__":
    main()
