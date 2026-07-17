"""Sbatch-array projected Jacobian collection on Cosmos-Policy-LIBERO-Predict2-2B.

Computes the within-step (A_tilde) projected Jacobians along a single trajectory
driven by `get_action(...)` with a real LIBERO observation (proprio + wrist +
primary images) loaded from --inputs-npz and a configurable prompt. Defaults to
the first observation of libero_10 task 0 (original scene: "put both the
alphabet soup and the tomato sauce in the basket"). Zeroed inputs were avoided
because the model is not trained on them, which made earlier Jacobians
unreliable.

  A_tilde[(t, l_in)] = V_{p(l_in+1), t}^T  J_{block_{l_in+1}}(z_{t, l_in}) V_{p(l_in), t}

for every (t in selected_timesteps, l_in in [0, L-2]), where p(l) is the
partition containing layer l. Each V is the right-singular projection from the
modified run_partition_svd_cosmos_policy.py:

  * one V per (partition, timestep);
  * restricted to the denoising slots (T_p_denoise = state_t -
    min_num_conditional_frames; D_flat = T_p_denoise * H_p * W_p * D).

Pipeline (matches the SVD shell's two-stage pattern):

  --phase worker  : One sbatch array task per rank. Pins a single GPU via
                    Slurm allocation, derives its (t, l_in) slice from
                    (--rank, --num-shards, sel_t, L), runs get_action(...)
                    once, and saves its `jac_store` to
                    <out-dir>/shard_rank<R>.pt. `--resume` supported.

  --phase merge   : Single dependent sbatch task. Reads all per-rank shard
                    files and writes <out-dir>/A_tilde__full.pt.

CFG-pass structure
------------------
Cosmos-Policy `get_action` runs ONE forward per denoising step (no CFG --
is_negative_prompt=False in generate_samples_from_batch). PASSES_PER_STEP = 1.

B_tilde (across-step) not implemented; --skip-step-jacobian defaults to True.

Typical use:
    bash run_jacobians_full.sh
or directly:
    python compute_jacobians_full.py --phase worker --rank 0 --num-shards 8 \
        --svd-dir /work/nvme/.../<RUN_TAG> --out-dir <dest> \
        --prompt "open the drawer"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Env setup (must run BEFORE any cosmos_policy import in the worker)
# ---------------------------------------------------------------------------

def _setup_env() -> None:
    """Match run_partition_svd_cosmos_policy.py's environment."""
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
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase', choices=['worker', 'merge'], required=True,
                   help='worker = one shard worker (sbatch array task); '
                        'merge = single dependent finalize that sums shards.')

    p.add_argument('--rank', type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
                   help='this process rank in [0, num-shards). Defaults to '
                        '$SLURM_ARRAY_TASK_ID.')
    p.add_argument('--num-shards', type=int, required=True,
                   help='total number of parallel worker ranks')

    p.add_argument('--svd-dir', type=Path, required=True,
                   help='SVD output dir (config.json, svd_summary.pt, V_part*_t*.pt).')
    p.add_argument('--out-dir', type=Path, required=True,
                   help='Output dir for per-rank shards and final A_tilde__full.pt.')

    p.add_argument('--prompt', type=str,
                   default='put both the alphabet soup and the tomato sauce in the basket',
                   help='LIBERO task prompt to drive the trajectory. Default '
                        'matches the original libero_10 task 0 prompt; override '
                        'to study counterfactual prompts on the same scene.')
    p.add_argument('--inputs-npz', type=Path,
                   default=Path('/u/jhong7/Workspace/cosmos-policy/notebooks/'
                                'lqr/inputs/policy_inputs/libero_10__task00/'
                                'inputs.npz'),
                   help='npz with keys primary_images, wrist_images, proprios. '
                        'Default is the libero_10 task 0 capture used by the '
                        'SVD pipeline.')
    p.add_argument('--obs-index', type=int, default=0,
                   help='Row of --inputs-npz to use as the observation. 0 is '
                        'the very first frame of episode 0 (original scene).')
    p.add_argument('--mode', type=str, default='vjp',
                   choices=['jvp', 'vjp', 'vjp_no_retain'],
                   help='Jacobian compute mode (jvp / vjp / vjp_no_retain).')
    p.add_argument('--v-device', type=str, default='cpu', choices=['cpu', 'cuda'],
                   help='Where V partitions live. cpu keeps GPU peak low.')
    p.add_argument('--v-dtype', type=str, default='fp32',
                   choices=['fp32', 'bf16', 'fp16'],
                   help='V storage dtype. Accumulator and final V projection '
                        'are always fp32 regardless of this setting.')
    p.add_argument('--verify', action='store_true',
                   help='Run FD verification per (t, l_in). Adds ~10s per Jacobian.')
    p.add_argument('--resume', action='store_true',
                   help='Skip (t, l_in) already present in the per-rank shard file.')
    p.add_argument('--num-cpu-threads', type=int, default=None)
    p.add_argument('--skip-step-jacobian', action='store_true', default=True,
                   help='Skip B_tilde (not implemented for Cosmos-Policy).')

    # Cosmos-Policy model loading args.
    p.add_argument('--ckpt-path', type=str,
                   default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    p.add_argument('--config-name', type=str,
                   default="cosmos_predict2_2b_480p_libero__inference_only")
    p.add_argument('--config-file', type=str,
                   default="cosmos_policy/config/config.py")
    p.add_argument('--dataset-stats-path', type=str, default=None)
    p.add_argument('--t5-cache-path', type=str, default=None)
    p.add_argument('--seed', type=int, default=None,
                   help='Override sampling seed. Default: read from svd-dir config.')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

def make_layer_shards(num_l_in: int, num_shards: int) -> list[list[int]]:
    """Split range(num_l_in) into `num_shards` contiguous chunks (sizes differ by ≤1).

    Contiguous-by-l_in keeps each shard within at most two V partitions per timestep.
    """
    chunks: list[list[int]] = []
    base, extra = divmod(num_l_in, num_shards)
    start = 0
    for i in range(num_shards):
        size = base + (1 if i < extra else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


# ---------------------------------------------------------------------------
# V file resolver (one V per (partition, timestep))
# ---------------------------------------------------------------------------

def _v_filename(svd_dir: Path, p_idx: int, t_id: int,
                partitions, k_target) -> Path:
    """Resolve V_part{p}_layers{a}-{b}_t{t_id}_k*.pt."""
    a, b = partitions[p_idx]
    pattern = f"V_part{p_idx}_layers{a}-{b}_t{t_id}_k*.pt"
    matches = sorted(svd_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no V file matching {pattern} in {svd_dir}")
    if len(matches) > 1:
        preferred = [m for m in matches if m.name.endswith(f"_k{k_target}.pt")]
        if preferred:
            return preferred[0]
    return matches[0]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> int:
    _setup_env()

    import numpy as np
    import torch

    rank = args.rank
    num_shards = args.num_shards
    assert 0 <= rank < num_shards, f"--rank {rank} not in [0, --num-shards {num_shards})"

    log = lambda msg: print(f'[rank {rank}/{num_shards}] {msg}', flush=True)

    if args.skip_step_jacobian:
        log('B_tilde (inter-step) computation: SKIPPED (not implemented).')

    log(f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}  '
        f'mode={args.mode}  v_device={args.v_device} v_dtype={args.v_dtype}  '
        f'verify={args.verify}')

    n_threads = args.num_cpu_threads
    if n_threads is None:
        n_threads = max(1, (os.cpu_count() or 1) // max(1, num_shards))
    torch.set_num_threads(n_threads)
    log(f'torch.set_num_threads({n_threads})')

    # ---- Load SVD config + summary to derive this rank's slice. ----
    cfg = json.loads((args.svd_dir / 'config.json').read_text())
    summary = torch.load(args.svd_dir / 'svd_summary.pt',
                         map_location='cpu', weights_only=False)
    L = int(summary['c_means'].shape[0])
    sel_t = list(cfg['selected_timesteps'])
    sampling_steps = cfg['sampling_steps']
    partitions = [tuple(p) for p in cfg['partitions']]
    layer_to_part = list(summary['layer_to_part'])
    k_target = cfg['k_target']
    seed = args.seed if args.seed is not None else int(cfg.get('seed', 42))

    P_tok = cfg['P_tok']
    D = cfg['D']
    D_flat = cfg['D_flat']
    T_p_denoise = cfg['T_p_denoise']
    denoise_t_start = cfg['denoise_t_start']
    denoise_t_end = cfg['denoise_t_end']

    num_l_in = L - 1
    layer_shards = make_layer_shards(num_l_in, num_shards)
    my_layers = layer_shards[rank]
    target_tl: set[tuple[int, int]] = {
        (int(t), int(l)) for t in sel_t for l in my_layers
    }
    log(f'L={L}  num_l_in={num_l_in}  sel_t={sel_t}  '
        f'my_layers={my_layers}  -> {len(target_tl)} (t, l_in) pairs')
    if not target_tl:
        log('no targets; exiting.')
        return 0

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'shard_rank{rank}.pt'

    existing: dict[tuple[int, int], "torch.Tensor"] = {}
    if args.resume and out_path.exists():
        d = torch.load(out_path, map_location='cpu', weights_only=False)
        existing = dict(d.get('A_tilde', {}))
        before = len(target_tl)
        target_tl = {tl for tl in target_tl if tl not in existing}
        log(f'resume: {len(existing)} A already done; {before - len(target_tl)} '
            f'A skipped; {len(target_tl)} A remaining')
        if not target_tl:
            log('all A targets already present; exiting.')
            return 0

    log(f'P_tok={P_tok}  D={D}  D_flat={D_flat:,}  '
        f'denoise=[{denoise_t_start}..{denoise_t_end - 1}]  '
        f'T_p_denoise={T_p_denoise}  sampling_steps={sampling_steps}')

    # ---- Cosmos-Policy model load. ----
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        PolicyEvalConfig,
    )
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
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

    eval_cfg = PolicyEvalConfig(
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
        num_denoising_steps_action=sampling_steps,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )

    log('loading dataset stats + T5 cache + model ...')
    dataset_stats = load_dataset_stats(eval_cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(eval_cfg.t5_text_embeddings_path,
                                  worker_id=rank)
    model, _ = get_model(eval_cfg)
    n_blocks = len(model.net.blocks)
    log(f'loaded; {n_blocks} DiT blocks')
    assert n_blocks == L, f'L mismatch: model has {n_blocks} blocks, summary says {L}'

    # ---- V loader: one V per (partition, timestep). ----
    _DTYPE_MAP = {
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
    }
    V_DEVICE = torch.device(args.v_device)
    V_DTYPE  = _DTYPE_MAP[args.v_dtype]
    _V_CACHE: dict[tuple[int, int], "torch.Tensor"] = {}

    def V_for(layer_idx: int, t_id: int) -> "torch.Tensor":
        p_idx = layer_to_part[layer_idx]
        key = (p_idx, t_id)
        V = _V_CACHE.get(key)
        if V is not None:
            return V
        fp = _v_filename(args.svd_dir, p_idx, t_id, partitions, k_target)
        log(f'lazy-load {fp.name} ({fp.stat().st_size/1e9:.2f} GB) -> '
            f'{V_DEVICE} as {V_DTYPE} ...')
        Vt = torch.load(fp, map_location='cpu', weights_only=False)['V']
        Vt = Vt.to(device=V_DEVICE, dtype=V_DTYPE).contiguous()
        _V_CACHE[key] = Vt
        gb = Vt.element_size() * Vt.numel() / 1e9
        log(f'  partition {p_idx} t={t_id}: {tuple(Vt.shape)}  ~{gb:.2f} GB resident')
        return Vt

    needed: set[tuple[int, int]] = set()
    for (t, l_in) in target_tl:
        needed.add((layer_to_part[l_in], int(t)))
        needed.add((layer_to_part[l_in + 1], int(t)))
    log(f'V (partition, t) tiles this rank will touch: {sorted(needed)}')

    # ---- Hook plumbing. PASSES_PER_STEP=1 (no CFG on get_action codepath). ----
    PASSES_PER_STEP = 1
    state   = {'pass_idx': -1}
    in_ad   = {'flag': False}
    jac_store: dict[tuple[int, int], "torch.Tensor"] = dict(existing)
    n_target_total = len(target_tl) + len(existing)
    _RUN_T0 = [0.0]

    sel_t_set = set(sel_t)

    class _AllTargetsRecorded(Exception):
        pass

    def projected_jacobian(block_fn, z, V_in, V_out, mode='vjp',
                           verbose=True, log_every=25):
        import torch.func  # noqa: F401
        z = z.detach()
        z_dev, z_dt = z.device, z.dtype
        r_in, r_out = V_in.shape[1], V_out.shape[1]
        d_in, d_out = V_in.shape[0], V_out.shape[0]
        OUT_DTYPE = torch.float32

        def col(V, j):
            return V[:, j].contiguous().to(device=z_dev, dtype=z_dt)

        def to_out(t, device):
            return t.to(device=device, dtype=OUT_DTYPE)

        def _sync():
            if z_dev.type == 'cuda':
                torch.cuda.synchronize(z_dev)

        def _log_iter(i, total, t0):
            if not verbose:
                return
            if (i + 1) % log_every != 0 and (i + 1) != total:
                return
            _sync()
            el = time.time() - t0
            rate = (i + 1) / el if el > 0 else 0.0
            eta = (total - i - 1) / rate if rate > 0 else float('inf')
            print(f'      [{i+1:3d}/{total}]  {el:6.1f}s  {rate:5.2f} it/s  '
                  f'ETA {eta:6.1f}s', flush=True)

        if verbose:
            scratch_elems = (r_in * d_out) if mode == 'jvp' else (r_out * d_in)
            scratch_gb = scratch_elems * torch.tensor([], dtype=OUT_DTYPE).element_size() / 1e9
            print(f"    projected_jacobian(mode={mode!r}): r_in={r_in}, r_out={r_out},"
                  f" d_in={d_in}, d_out={d_out};"
                  f" z on {z_dev} {z_dt}; V on {V_in.device} {V_in.dtype};"
                  f" accumulator dtype = {OUT_DTYPE};"
                  f" scratch ~{scratch_gb:.2f} GB on {V_in.device}",
                  flush=True)

        if mode == 'jvp':
            with torch.enable_grad():
                M = torch.empty(r_in, d_out, dtype=OUT_DTYPE, device=V_out.device)
                t0 = time.time()
                for j in range(r_in):
                    _, jvp = torch.func.jvp(block_fn, (z,), (col(V_in, j),))
                    M[j].copy_(to_out(jvp, V_out.device))
                    _log_iter(j, r_in, t0)
            return (M @ V_out.to(OUT_DTYPE)).T.contiguous()

        if mode == 'vjp':
            with torch.enable_grad():
                z_g = z.clone().requires_grad_(True)
                out = block_fn(z_g)
                G = torch.empty(r_out, d_in, dtype=OUT_DTYPE, device=V_in.device)
                t0 = time.time()
                for i in range(r_out):
                    (g,) = torch.autograd.grad(
                        out, z_g, col(V_out, i),
                        retain_graph=(i < r_out - 1),
                    )
                    G[i].copy_(to_out(g, V_in.device))
                    _log_iter(i, r_out, t0)
            return G @ V_in.to(OUT_DTYPE)

        if mode == 'vjp_no_retain':
            with torch.enable_grad():
                G = torch.empty(r_out, d_in, dtype=OUT_DTYPE, device=V_in.device)
                t0 = time.time()
                for i in range(r_out):
                    z_g = z.clone().requires_grad_(True)
                    out = block_fn(z_g)
                    (g,) = torch.autograd.grad(out, z_g, col(V_out, i))
                    G[i].copy_(to_out(g, V_in.device))
                    del out, z_g, g
                    _log_iter(i, r_out, t0)
            return G @ V_in.to(OUT_DTYPE)

        raise ValueError(f"unknown mode {mode!r}")

    def verify_projected_jacobian(block_fn, z, V_in, V_out, J_tilde,
                                  eps_list=(1.0, 1e-1, 1e-2, 1e-3),
                                  seed=0):
        z_dev, z_dt = z.device, z.dtype
        r_in = V_in.shape[1]
        with torch.no_grad():
            out0   = block_fn(z).detach()
            out0_b = block_fn(z).detach()
        floor = ((out0_b - out0).to(device=V_out.device, dtype=V_out.dtype) @ V_out).norm().item()
        gen = torch.Generator(device='cpu').manual_seed(seed)
        dx = torch.randn(r_in, generator=gen)
        dx = dx / dx.norm()
        dx_v = dx.to(device=V_in.device, dtype=V_in.dtype)
        dz = (V_in @ dx_v).to(device=z_dev, dtype=z_dt)
        with torch.no_grad():
            deltas = [(block_fn(z + eps * dz).detach() - out0).to(
                          device=V_out.device, dtype=V_out.dtype)
                      for eps in eps_list]
        deltas = torch.stack(deltas, dim=0)
        dx_meas_all = deltas @ V_out
        J_dx = J_tilde @ dx.to(device=J_tilde.device, dtype=J_tilde.dtype)
        for k, eps in enumerate(eps_list):
            dx_meas = dx_meas_all[k].to(device=J_dx.device, dtype=J_dx.dtype)
            dx_pred = eps * J_dx
            m, p = dx_meas.norm().item(), dx_pred.norm().item()
            cos = ((dx_meas @ dx_pred).item() / (m * p)) if (m * p) > 0 else float('nan')
            ratio = m / p if p > 0 else float('nan')
            snr = m / floor if floor > 0 else float('inf')
            print(f'      verify  eps={eps:.1e}  cos={cos:.5f}  ratio={ratio:.4f}  '
                  f'snr={snr:.2f}', flush=True)

    def _pass_tick(_block, _args):
        if not in_ad['flag']:
            state['pass_idx'] += 1
            if state['pass_idx'] == 0:
                print(f'  [t=+{time.time() - _RUN_T0[0]:.1f}s] pass 0 (step 0) '
                      f'started — block 0 about to run', flush=True)
        return None

    by_lin: dict[int, set[int]] = {}
    for (t, l_in) in target_tl:
        by_lin.setdefault(l_in, set()).add(t)

    def _to_normal(t):
        """Clone a tensor (inside torch.inference_mode(False)) so it leaves the
        inference-tensor pool. Pass-through for non-tensors and tuples.
        """
        if torch.is_tensor(t):
            return t.detach().clone()
        if isinstance(t, tuple):
            return tuple(_to_normal(x) for x in t)
        return t

    def make_pre_hook(l_in: int, target_steps: set[int]):
        l_out = l_in + 1

        def hook(block_next, h_args, kwargs):
            if in_ad['flag']:
                return None
            pass_idx = state['pass_idx']
            if pass_idx % PASSES_PER_STEP != 0:
                return None
            step = pass_idx // PASSES_PER_STEP
            if step not in target_steps:
                return None
            if (step, l_in) in jac_store:
                return None

            x_B_T_H_W_D = h_args[0]
            emb_B_T_D = h_args[1] if len(h_args) > 1 else kwargs.get('emb_B_T_D')
            context_input = h_args[2] if len(h_args) > 2 else kwargs.get('crossattn_emb')
            rope_emb = kwargs.get('rope_emb_L_1_1_D')
            adaln_lora = kwargs.get('adaln_lora_B_T_3D')
            extra_pos_emb = kwargs.get('extra_per_block_pos_emb')
            kv_cache_cfg = kwargs.get('kv_cache_cfg')

            print(f'  [t=+{time.time() - _RUN_T0[0]:.1f}s] hook t={step}, '
                  f'l_in={l_in} -> loading V (lazy) and starting '
                  f'projected_jacobian', flush=True)

            in_ad['flag'] = True
            try:
                # Cosmos-Policy's get_action wraps the whole sampler in
                # torch.inference_mode(), so the tensors we capture here are
                # "inference tensors" that cannot participate in autograd.
                # We exit inference mode and clone every tensor that flows into
                # the block forward so PyTorch will let us call autograd.grad.
                with torch.inference_mode(False):
                    x_template     = _to_normal(x_B_T_H_W_D)
                    emb_clone      = _to_normal(emb_B_T_D)
                    ctx_clone      = _to_normal(context_input)
                    rope_clone     = _to_normal(rope_emb)
                    adaln_clone    = _to_normal(adaln_lora)
                    extra_clone    = _to_normal(extra_pos_emb)
                    # kv_cache_cfg is typically None or a config object; leave as-is.

                    x_shape = x_template.shape  # (B, T_p, H_p, W_p, D)
                    denoise_shape = (T_p_denoise, x_shape[2], x_shape[3], x_shape[4])
                    z_full = x_template[0, denoise_t_start:denoise_t_end, :, :, :].reshape(-1)
                    assert z_full.numel() == D_flat, (
                        f'D_flat mismatch: z_full.numel()={z_full.numel()} vs '
                        f'cfg D_flat={D_flat}'
                    )

                    def block_fn(z_flat):
                        x_in = x_template.clone()
                        x_in[0, denoise_t_start:denoise_t_end, :, :, :] = (
                            z_flat.reshape(*denoise_shape)
                        )
                        out = block_next(
                            x_in, emb_clone, ctx_clone,
                            rope_emb_L_1_1_D=rope_clone,
                            adaln_lora_B_T_3D=adaln_clone,
                            extra_per_block_pos_emb=extra_clone,
                            kv_cache_cfg=kv_cache_cfg,
                        )
                        return out[0, denoise_t_start:denoise_t_end, :, :, :].reshape(-1)

                    t_v = time.time()
                    V_in_  = V_for(l_in, step)
                    V_out_ = V_for(l_out, step)
                    print(f'  [t=+{time.time() - _RUN_T0[0]:.1f}s] V ready '
                          f'({time.time()-t_v:.1f}s); entering projected_jacobian',
                          flush=True)

                    t_pj = time.time()
                    J_tilde = projected_jacobian(
                        block_fn, z_full, V_in_, V_out_, mode=args_mode,
                    )
                    print(f'  [t=+{time.time() - _RUN_T0[0]:.1f}s] projected_jacobian '
                          f'done in {time.time()-t_pj:.1f}s', flush=True)
                    if do_verify:
                        verify_projected_jacobian(block_fn, z_full, V_in_, V_out_, J_tilde)
            finally:
                in_ad['flag'] = False

            jac_store[(step, l_in)] = J_tilde.detach().to(torch.float32).cpu()
            done, want = len(jac_store), n_target_total
            print(f'  recorded Ã[t={step}, l_in={l_in}]  '
                  f'shape={tuple(J_tilde.shape)}  ({done}/{want})', flush=True)

            if done % checkpoint_every == 0:
                _save_shard(out_path, jac_store, args, partitions,
                            layer_to_part, cfg, len(sel_t), L, target_tl_full)
                print(f'  checkpoint -> {out_path.name}', flush=True)

            if done >= want:
                raise _AllTargetsRecorded
            return None

        return hook

    args_mode = args.mode
    do_verify = bool(args.verify)
    checkpoint_every = 25
    target_tl_full = sorted(target_tl | set(existing.keys()))

    handles = [model.net.blocks[0].register_forward_pre_hook(_pass_tick)]
    for l_in, steps in by_lin.items():
        handles.append(
            model.net.blocks[l_in + 1].register_forward_pre_hook(
                make_pre_hook(l_in, steps), with_kwargs=True,
            )
        )

    log(f'registered 1 pass-tick + {len(by_lin)} Jacobian hook(s); '
        f'{len(target_tl)} pending pairs to record')

    # Load a real LIBERO observation rather than zeros: the model has not been
    # trained on all-zero inputs, so Jacobians collected on zeroed obs are
    # unreliable. Row args.obs_index of args.inputs_npz is the captured
    # (proprio, wrist, primary) tuple from the libero_10 task 0 rollouts;
    # the default (idx=0) is the very first frame of episode 0's original-prompt
    # rollout — i.e., the canonical initial scene of "put both the alphabet
    # soup and the tomato sauce in the basket".
    if not args.inputs_npz.exists():
        raise FileNotFoundError(f'--inputs-npz not found: {args.inputs_npz}')
    npz = np.load(args.inputs_npz)
    n_avail = int(npz['proprios'].shape[0])
    if not (0 <= args.obs_index < n_avail):
        raise ValueError(f'--obs-index {args.obs_index} out of range '
                         f'[0, {n_avail}) for {args.inputs_npz}')
    obs = {
        'wrist_image':   npz['wrist_images'][args.obs_index].copy(),
        'primary_image': npz['primary_images'][args.obs_index].copy(),
        'proprio':       npz['proprios'][args.obs_index].astype(np.float32).copy(),
    }
    log(f"obs from {args.inputs_npz.name}[idx={args.obs_index}]: "
        f"wrist={obs['wrist_image'].shape}{obs['wrist_image'].dtype}, "
        f"primary={obs['primary_image'].shape}{obs['primary_image'].dtype}, "
        f"proprio={obs['proprio'].shape}{obs['proprio'].dtype}")

    _RUN_T0[0] = time.time()
    log(f'entering get_action(prompt={args.prompt!r}, seed={seed}, '
        f'steps={sampling_steps}) ...')
    # NOTE: torch.no_grad() rather than torch.inference_mode(): the Jacobian
    # hook needs to call torch.autograd.grad on block activations, and tensors
    # created inside inference_mode are "inference tensors" that cannot
    # participate in autograd even after clone(). no_grad lets us re-enable
    # gradients locally inside projected_jacobian's `with torch.enable_grad()`.
    try:
        with torch.no_grad():
            _ = get_action(
                eval_cfg, model, dataset_stats, obs, args.prompt,
                seed=seed,
                randomize_seed=False,
                num_denoising_steps_action=sampling_steps,
                generate_future_state_and_value_in_parallel=True,
            )
    except _AllTargetsRecorded:
        log(f'early-stop: all targets recorded after {time.time() - _RUN_T0[0]:.1f}s')
    finally:
        for h in handles:
            h.remove()
    elapsed = time.time() - _RUN_T0[0]
    log(f'inference elapsed {elapsed:.1f}s; recorded '
        f'{len(jac_store)}/{n_target_total} A_tilde')

    _save_shard(out_path, jac_store, args, partitions, layer_to_part,
                cfg, len(sel_t), L, target_tl_full)
    log(f'saved -> {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)')
    return 0 if (len(jac_store) == n_target_total) else 3


def _save_shard(out_path, jac_store, args, partitions, layer_to_part,
                cfg, T, L, target_tl_full):
    import torch
    torch.save({
        'A_tilde':            dict(jac_store),
        'B_tilde':            {},
        'rank':               args.rank,
        'num_shards':         args.num_shards,
        'prompt':             args.prompt,
        'mode':               args.mode,
        'V_dtype':            args.v_dtype,
        'V_device':           args.v_device,
        'partitions':         partitions,
        'layer_to_part':      layer_to_part,
        'k':                  cfg['k_target'],
        'T':                  T,
        'L':                  L,
        'selected_timesteps': cfg['selected_timesteps'],
        'targets':            target_tl_full,
        'T_p_denoise':        cfg['T_p_denoise'],
        'denoise_t_start':    cfg['denoise_t_start'],
        'denoise_t_end':      cfg['denoise_t_end'],
        'D_flat':             cfg['D_flat'],
    }, out_path)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def run_merge(args: argparse.Namespace) -> int:
    _setup_env()
    import torch

    cfg = json.loads((args.svd_dir / 'config.json').read_text())
    summary = torch.load(args.svd_dir / 'svd_summary.pt',
                         map_location='cpu', weights_only=False)
    L = int(summary['c_means'].shape[0])
    sel_t = list(cfg['selected_timesteps'])
    T = len(sel_t)
    partitions = [tuple(p) for p in cfg['partitions']]
    layer_to_part = list(summary['layer_to_part'])
    num_l_in = L - 1
    want_a = T * num_l_in

    print(f'[merge] num_shards={args.num_shards}  T={T}  L={L}  '
          f'expected total A_tilde entries = {want_a}', flush=True)
    print(f'[merge] out_dir = {args.out_dir}')

    merged_a: dict[tuple[int, int], "torch.Tensor"] = {}
    seen_ranks: list[int] = []
    for rank in range(args.num_shards):
        shard_path = args.out_dir / f'shard_rank{rank}.pt'
        if not shard_path.exists():
            print(f'  [warn] missing {shard_path}')
            continue
        d = torch.load(shard_path, map_location='cpu', weights_only=False)
        merged_a.update(d['A_tilde'])
        seen_ranks.append(rank)
        print(f'  rank {rank}: {len(d["A_tilde"])} A entries  ({shard_path.name})')

    final_path = args.out_dir / 'A_tilde__full.pt'
    torch.save({
        'A_tilde':           merged_a,
        'B_tilde':           {},
        'prompt':            args.prompt,
        'mode':              args.mode,
        'V_dtype':           args.v_dtype,
        'V_device':          args.v_device,
        'partitions':        partitions,
        'layer_to_part':     layer_to_part,
        'k':                 cfg['k_target'],
        'T':                 T,
        'L':                 L,
        'selected_timesteps': sel_t,
        'num_shards':        args.num_shards,
        'ranks_seen':        seen_ranks,
        'verify':            bool(args.verify),
        'T_p_denoise':       cfg['T_p_denoise'],
        'denoise_t_start':   cfg['denoise_t_start'],
        'denoise_t_end':     cfg['denoise_t_end'],
        'D_flat':            cfg['D_flat'],
        'note': ('A_tilde[(t, l_in)] = V_{p(l_in+1), t}^T J_{block_{l_in+1}}(z_{t, l_in}) V_{p(l_in), t}; '
                 'one V per (partition, timestep); '
                 'block input/output restricted to denoising slots only. '
                 'B_tilde not implemented for Cosmos-Policy.'),
    }, final_path)

    have_a = len(merged_a)
    print(f'[merge] merged {have_a}/{want_a} A_tilde -> '
          f'{final_path}  ({final_path.stat().st_size / 1e6:.1f} MB)')
    if have_a < want_a:
        missing = [(t, l) for t in sel_t for l in range(num_l_in)
                   if (t, l) not in merged_a]
        print(f'  [warn] {len(missing)} pairs missing, e.g. {missing[:5]}')
        return 2
    return 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.phase == 'worker':
        return run_worker(args)
    if args.phase == 'merge':
        return run_merge(args)
    print(f'unknown --phase {args.phase!r}', file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
