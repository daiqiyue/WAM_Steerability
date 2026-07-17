"""Sbatch-array projected text Jacobian collection on Cosmos-Policy-LIBERO-Predict2-2B.

Driven by `get_action(...)` with a real LIBERO observation (proprio + wrist +
primary images) loaded from --inputs-npz and a configurable prompt. Defaults to
the first observation of libero_10 task 0 (original scene: "put both the
alphabet soup and the tomato sauce in the basket"). Zeroed inputs were avoided
because the model is not trained on them, which made earlier Jacobians
unreliable.

For every (denoising step t in selected_timesteps, DiT block l in [0, L)),
computes the projected Jacobian of the block's output w.r.t. a broadcast
text-context shift u in R^{Q_text}:

    delta_c[pos, :] = u     for pos in [0, n_active)
                      0     otherwise
    J_text[(t, l)]  = V_{p(l), t}^T  J_block_l(c -> output_denoise_slots)  at u = 0

V is per (partition, timestep), restricted to denoising slots, matching the
modified run_partition_svd_cosmos_policy.py.

Pipeline (matches the SVD shell's two-stage pattern):

  --phase worker  : one sbatch array task per rank; derives its (t, l) slice
                    from (--rank, --num-shards, sel_t, L); writes
                    <out-dir>/shard_rank<R>.pt. `--resume` supported.

  --phase merge   : single dependent task; reads all per-rank shards, writes
                    <out-dir>/B_text__full.pt.

CFG-pass structure
------------------
PASSES_PER_STEP = 1 (`get_action` runs one forward per denoising step).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


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
    p.add_argument('--phase', choices=['worker', 'merge'], required=True)

    p.add_argument('--rank', type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument('--num-shards', type=int, required=True)

    p.add_argument('--svd-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)

    p.add_argument('--prompt', type=str,
                   default='put both the alphabet soup and the tomato sauce in the basket',
                   help='LIBERO task prompt. Default matches the original '
                        'libero_10 task 0 prompt; override to study '
                        'counterfactual prompts on the same scene.')
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
                   choices=['vjp', 'vjp_no_retain'])
    p.add_argument('--v-device', type=str, default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--v-dtype', type=str, default='fp32',
                   choices=['fp32', 'bf16', 'fp16'])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--num-cpu-threads', type=int, default=None)

    p.add_argument('--ckpt-path', type=str,
                   default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    p.add_argument('--config-name', type=str,
                   default="cosmos_predict2_2b_480p_libero__inference_only")
    p.add_argument('--config-file', type=str,
                   default="cosmos_policy/config/config.py")
    p.add_argument('--dataset-stats-path', type=str, default=None)
    p.add_argument('--t5-cache-path', type=str, default=None)
    p.add_argument('--seed', type=int, default=None)

    return p.parse_args()


def make_layer_shards(num_l: int, num_shards: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    base, extra = divmod(num_l, num_shards)
    start = 0
    for i in range(num_shards):
        size = base + (1 if i < extra else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


def _v_filename(svd_dir: Path, p_idx: int, t_id: int,
                partitions, k_target) -> Path:
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

    log(f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}  '
        f'mode={args.mode}  v_device={args.v_device} v_dtype={args.v_dtype}')

    n_threads = args.num_cpu_threads
    if n_threads is None:
        n_threads = max(1, (os.cpu_count() or 1) // max(1, num_shards))
    torch.set_num_threads(n_threads)
    log(f'torch.set_num_threads({n_threads})')

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

    num_l = L
    layer_shards = make_layer_shards(num_l, num_shards)
    my_layers = layer_shards[rank]
    target_tl: set[tuple[int, int]] = {
        (int(t), int(l)) for t in sel_t for l in my_layers
    }
    log(f'L={L}  num_l={num_l}  sel_t={sel_t}  '
        f'my_layers={my_layers}  -> {len(target_tl)} (t, l) pairs')
    if not target_tl:
        log('no targets; exiting.')
        return 0

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'shard_rank{rank}.pt'

    existing: dict[tuple[int, int], "torch.Tensor"] = {}
    if args.resume and out_path.exists():
        d = torch.load(out_path, map_location='cpu', weights_only=False)
        existing = dict(d.get('B_text', {}))
        before = len(target_tl)
        target_tl = {tl for tl in target_tl if tl not in existing}
        log(f'resume: {len(existing)} already done; {before - len(target_tl)} skipped; '
            f'{len(target_tl)} remaining')
        if not target_tl:
            log('all targets already present; exiting.')
            return 0

    log(f'denoise=[{denoise_t_start}..{denoise_t_end - 1}]  '
        f'T_p_denoise={T_p_denoise}  D_flat={D_flat:,}')

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

    needed = sorted({(layer_to_part[l], int(t)) for (t, l) in target_tl})
    log(f'V (partition, t) tiles this rank will touch: {needed}')

    U_DTYPE = torch.float32
    OUT_DTYPE = torch.float32
    U_DEVICE = torch.device('cuda', 0)
    state_qtext = {'r_text': None}

    PASSES_PER_STEP = 1
    state   = {'pass_idx': -1, 'n_active': None}
    in_ad   = {'flag': False}
    jac_store: dict[tuple[int, int], "torch.Tensor"] = dict(existing)
    n_target_total = len(target_tl) + len(existing)
    _RUN_T0 = [0.0]

    class _AllTargetsRecorded(Exception):
        pass

    def projected_jacobian_text(block_fn_u, V_out, r_text, mode='vjp',
                                verbose=True, log_every=25):
        r_out = V_out.shape[1]
        d_out = V_out.shape[0]
        u_zero = torch.zeros(r_text, dtype=U_DTYPE, device=U_DEVICE)

        def _sync():
            if U_DEVICE.type == 'cuda':
                torch.cuda.synchronize(U_DEVICE)

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
            print(f"    projected_jacobian_text(mode={mode!r}): r_text={r_text}, "
                  f"r_out={r_out}, d_out={d_out}; u on {u_zero.device} {u_zero.dtype}; "
                  f"V_out on {V_out.device} {V_out.dtype}; "
                  f"G accumulator dtype = {OUT_DTYPE}", flush=True)

        if mode == 'vjp':
            with torch.enable_grad():
                u_g = u_zero.clone().requires_grad_(True)
                out = block_fn_u(u_g)
                G = torch.empty(r_out, r_text,
                                dtype=OUT_DTYPE, device=V_out.device)
                t0 = time.time()
                for i in range(r_out):
                    cot = V_out[:, i].contiguous().to(device=out.device, dtype=out.dtype)
                    (g,) = torch.autograd.grad(
                        out, u_g, cot,
                        retain_graph=(i < r_out - 1),
                    )
                    G[i].copy_(g.to(device=V_out.device, dtype=OUT_DTYPE))
                    _log_iter(i, r_out, t0)
            return G

        if mode == 'vjp_no_retain':
            with torch.enable_grad():
                G = torch.empty(r_out, r_text,
                                dtype=OUT_DTYPE, device=V_out.device)
                t0 = time.time()
                for i in range(r_out):
                    u_g = u_zero.clone().requires_grad_(True)
                    out = block_fn_u(u_g)
                    cot = V_out[:, i].contiguous().to(device=out.device, dtype=out.dtype)
                    (g,) = torch.autograd.grad(out, u_g, cot)
                    G[i].copy_(g.to(device=V_out.device, dtype=OUT_DTYPE))
                    del out, u_g, g
                    _log_iter(i, r_out, t0)
            return G

        raise ValueError(f"unknown mode {mode!r}")

    def _pass_tick(_block, _args):
        if not in_ad['flag']:
            state['pass_idx'] += 1
            if state['pass_idx'] == 0:
                print(f'  [t=+{time.time() - _RUN_T0[0]:.1f}s] pass 0 (step 0) '
                      f'started — block 0 about to run', flush=True)
        return None

    by_l: dict[int, set[int]] = {}
    for (t, l) in target_tl:
        by_l.setdefault(l, set()).add(t)

    def _split_context(context_input):
        if isinstance(context_input, tuple):
            return context_input[0], context_input[1:]
        return context_input, None

    def _rebuild_context(text_emb_new, img_emb_rest):
        if img_emb_rest is None:
            return text_emb_new
        return (text_emb_new, *img_emb_rest)

    def _to_normal(t):
        """Clone a tensor (inside torch.inference_mode(False)) so it leaves the
        inference-tensor pool. Pass-through for non-tensors and tuples.
        """
        if torch.is_tensor(t):
            return t.detach().clone()
        if isinstance(t, tuple):
            return tuple(_to_normal(x) for x in t)
        return t

    def make_pre_hook(l: int, target_steps: set[int]):

        def hook(block, h_args, kwargs):
            if in_ad['flag']:
                return None
            pass_idx = state['pass_idx']
            if pass_idx % PASSES_PER_STEP != 0:
                return None
            step = pass_idx // PASSES_PER_STEP
            if step not in target_steps:
                return None
            if (step, l) in jac_store:
                return None

            x_B_T_H_W_D = h_args[0]
            emb_B_T_D = h_args[1] if len(h_args) > 1 else kwargs.get('emb_B_T_D')
            context_input = h_args[2] if len(h_args) > 2 else kwargs.get('crossattn_emb')
            rope_emb = kwargs.get('rope_emb_L_1_1_D')
            adaln_lora = kwargs.get('adaln_lora_B_T_3D')
            extra_pos_emb = kwargs.get('extra_per_block_pos_emb')
            kv_cache_cfg = kwargs.get('kv_cache_cfg')

            text_emb_for_inspect, _ = _split_context(context_input)
            assert isinstance(text_emb_for_inspect, torch.Tensor) and text_emb_for_inspect.dim() == 3
            B, N_tok, Q_text = text_emb_for_inspect.shape
            r_text = Q_text

            if state_qtext['r_text'] is None:
                state_qtext['r_text'] = r_text
                log(f'  latched r_text = Q_text = {r_text}')
            else:
                assert state_qtext['r_text'] == r_text

            if state['n_active'] is None:
                with torch.no_grad():
                    row_amax = text_emb_for_inspect[0].abs().amax(dim=-1)
                    nz_mask = row_amax > 0
                    if nz_mask.all():
                        n_active = int(N_tok)
                    else:
                        last_nz = int(torch.nonzero(nz_mask, as_tuple=False).max().item())
                        n_active = last_nz + 1
                state['n_active'] = n_active
                log(f'  [t=+{time.time() - _RUN_T0[0]:.1f}s] '
                    f'captured n_active={n_active} (N_tok={N_tok})')
            n_active_local = state['n_active']

            print(f'  [t=+{time.time() - _RUN_T0[0]:.1f}s] hook t={step}, l={l} '
                  f'(n_active={n_active_local}, r_text={r_text}) -> loading V_{{{l},t={step}}}',
                  flush=True)

            in_ad['flag'] = True
            try:
                # Cosmos-Policy's get_action wraps the sampler in
                # torch.inference_mode(), so the tensors we capture here are
                # inference tensors. Exit inference mode and clone every
                # tensor that flows into the block forward so PyTorch will let
                # us call autograd.grad.
                with torch.inference_mode(False):
                    x_template     = _to_normal(x_B_T_H_W_D)
                    emb_clone      = _to_normal(emb_B_T_D)
                    ctx_clone      = _to_normal(context_input)
                    rope_clone     = _to_normal(rope_emb)
                    adaln_clone    = _to_normal(adaln_lora)
                    extra_clone    = _to_normal(extra_pos_emb)

                    text_clone, img_emb_rest = _split_context(ctx_clone)
                    text_baseline = text_clone
                    ctx_dtype = text_baseline.dtype

                    def block_fn_u(u_flat):
                        per_token = u_flat.to(ctx_dtype)
                        delta = torch.zeros_like(text_baseline)
                        delta[0, :n_active_local, :] = per_token
                        text_perturbed = text_baseline + delta
                        context_perturbed = _rebuild_context(text_perturbed, img_emb_rest)
                        out = block(
                            x_template, emb_clone, context_perturbed,
                            rope_emb_L_1_1_D=rope_clone,
                            adaln_lora_B_T_3D=adaln_clone,
                            extra_per_block_pos_emb=extra_clone,
                            kv_cache_cfg=kv_cache_cfg,
                        )
                        return out[0, denoise_t_start:denoise_t_end, :, :, :].reshape(-1)

                    t_v = time.time()
                    V_out_ = V_for(l, step)
                    print(f'  [t=+{time.time() - _RUN_T0[0]:.1f}s] V ready '
                          f'({time.time()-t_v:.1f}s); entering projected_jacobian_text',
                          flush=True)

                    t_pj = time.time()
                    J_text = projected_jacobian_text(
                        block_fn_u, V_out_, r_text, mode=args_mode,
                    )
                    print(f'  [t=+{time.time() - _RUN_T0[0]:.1f}s] '
                          f'projected_jacobian_text done in {time.time()-t_pj:.1f}s',
                          flush=True)
            finally:
                in_ad['flag'] = False

            jac_store[(step, l)] = J_text.detach().to(torch.float32).cpu()
            done, want = len(jac_store), n_target_total
            print(f'  recorded J_text[t={step}, l={l}]  '
                  f'shape={tuple(J_text.shape)}  ({done}/{want})', flush=True)

            if done % checkpoint_every == 0:
                _save_shard(out_path, jac_store, args, partitions, layer_to_part,
                            cfg, len(sel_t), L, target_tl_full,
                            state['n_active'], state_qtext['r_text'])
                print(f'  checkpoint -> {out_path.name}', flush=True)

            if done >= want:
                raise _AllTargetsRecorded
            return None

        return hook

    args_mode = args.mode
    checkpoint_every = 25
    target_tl_full = sorted(target_tl | set(existing.keys()))

    handles = [model.net.blocks[0].register_forward_pre_hook(_pass_tick)]
    for l, steps in by_l.items():
        handles.append(
            model.net.blocks[l].register_forward_pre_hook(
                make_pre_hook(l, steps), with_kwargs=True,
            )
        )

    log(f'registered 1 pass-tick + {len(by_l)} Jacobian hook(s); '
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
    # gradients locally inside projected_jacobian_text's `with torch.enable_grad()`.
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
    log(f'inference elapsed {elapsed:.1f}s; recorded {len(jac_store)}/{n_target_total}')

    _save_shard(out_path, jac_store, args, partitions, layer_to_part,
                cfg, len(sel_t), L, target_tl_full,
                state['n_active'], state_qtext['r_text'])
    log(f'saved -> {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)')
    return 0 if (len(jac_store) == n_target_total) else 3


def _save_shard(out_path, jac_store, args, partitions, layer_to_part,
                cfg, T, L, target_tl_full, n_active, r_text):
    import torch
    torch.save({
        'B_text':             dict(jac_store),
        'rank':               args.rank,
        'num_shards':         args.num_shards,
        'prompt':             args.prompt,
        'mode':               args.mode,
        'V_dtype':            args.v_dtype,
        'V_device':           args.v_device,
        'partitions':         partitions,
        'layer_to_part':      layer_to_part,
        'k':                  cfg['k_target'],
        'r_text':             r_text,
        'n_active':           n_active,
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
    num_l = L
    want = T * num_l

    print(f'[merge] num_shards={args.num_shards}  T={T}  L={L}  '
          f'expected total B_text entries = {want}', flush=True)
    print(f'[merge] out_dir = {args.out_dir}')

    merged: dict[tuple[int, int], "torch.Tensor"] = {}
    n_active_seen: int | None = None
    r_text_seen: int | None = None
    seen_ranks: list[int] = []

    for rank in range(args.num_shards):
        shard_path = args.out_dir / f'shard_rank{rank}.pt'
        if not shard_path.exists():
            print(f'  [warn] missing {shard_path}')
            continue
        d = torch.load(shard_path, map_location='cpu', weights_only=False)
        merged.update(d['B_text'])
        seen_ranks.append(rank)
        if n_active_seen is None:
            n_active_seen = d.get('n_active')
        elif d.get('n_active') is not None and d.get('n_active') != n_active_seen:
            print(f'  [warn] n_active mismatch across shards: '
                  f'{n_active_seen} vs {d.get("n_active")}')
        if r_text_seen is None:
            r_text_seen = d.get('r_text')
        print(f'  rank {rank}: {len(d["B_text"])} entries  ({shard_path.name})')

    final_path = args.out_dir / 'B_text__full.pt'
    final_r_text = (int(next(iter(merged.values())).shape[1])
                    if merged else r_text_seen)
    torch.save({
        'B_text':             merged,
        'prompt':             args.prompt,
        'mode':               args.mode,
        'V_dtype':            args.v_dtype,
        'V_device':           args.v_device,
        'partitions':         partitions,
        'layer_to_part':      layer_to_part,
        'k':                  cfg['k_target'],
        'r_text':             final_r_text,
        'n_active':           n_active_seen,
        'T':                  T,
        'L':                  L,
        'selected_timesteps': sel_t,
        'num_shards':         args.num_shards,
        'ranks_seen':         seen_ranks,
        'T_p_denoise':        cfg['T_p_denoise'],
        'denoise_t_start':    cfg['denoise_t_start'],
        'denoise_t_end':      cfg['denoise_t_end'],
        'D_flat':             cfg['D_flat'],
        'note': ('B_text[(t, l)] = V_{p(l), t}^T J_block_l(c -> output_denoise_slots) '
                 'at u=0; delta_c[pos, :] = u for pos in [0, n_active); '
                 'V is per (partition, timestep); block output restricted to '
                 'denoising slots.'),
    }, final_path)
    have = len(merged)
    print(f'[merge] merged {have}/{want} entries  '
          f'-> {final_path}  ({final_path.stat().st_size / 1e6:.1f} MB)')
    if have < want:
        missing = [(t, l) for t in sel_t for l in range(num_l)
                   if (t, l) not in merged]
        print(f'  [warn] {len(missing)} entries missing, e.g. {missing[:5]}')
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
