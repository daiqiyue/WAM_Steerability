#!/usr/bin/env python
"""Postprocess: per-inference-index contrastive vectors for Cosmos-Policy.

Reuses the V projection matrices written by run_partition_svd_pairs.py and
re-runs the paired forward passes from positive.npz / negative.npz, but this
time accumulates the (pos - neg) activation difference *separately for each
inference index j* (= inference_idx column in the paired npz files written
by collect_policy_inputs_milk_first_chunks.ipynb).

For each pair n with inference_idx j_n:

    mu_{l, t, j_n} += (act_pos[n, l, t] - act_neg[n, l, t]) / count(j_n)

then project once at the end:

    c_{l, t, j_n} = mu_{l, t, j_n} @ V_{p(l), t}

Output: <svd-dir>/c_means_per_j.pt  with keys

    c_means_per_j      : float32 (L, J, T_sel, k_target)
    c_norms_per_j      : float32 (L, J, T_sel)        -- ‖c_{l,t,j}‖
    j_counts           : int64   (J,)                 -- # pairs at each j
    inference_idx_set  : list[int]                    -- sorted unique j's
    selected_timesteps : list[int]
    layer_partitions   : list[(l_start, l_end)]
    layer_to_part      : list[int]                    -- length L
    prompt, pos_npz, neg_npz, drive_source_filter, N, svd_dir

‖μ_{l,t,j}‖ is *not* tracked here: doing it correctly would require keeping a
(J, L, D_flat) tensor in memory, and the per-(l, t) capture ratio averaged
across i and j is already available in <svd-dir>/svd_summary.pt.

V (and svd_summary.pt) on disk are NOT modified, so existing jacobians and
the existing per-(l,t) c_means stay valid.

This script is single-rank by design: 80 pairs × 2 forwards × ~3 s ≈ 8 min on
one GH200 (no Y / W / Halko work — only μ accumulation and a final projection).

Usage:
    python compute_contrastive_per_j.py \
        --svd-dir /path/to/.../libero10_task00_milk_first_chunks_dsall_N-1_k64_p10_ws8_tsall \
        --pos-npz /path/.../milk_first_chunks/positive.npz \
        --neg-npz /path/.../milk_first_chunks/negative.npz \
        --prompt "put both the milk and the tomato sauce in the basket"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _setup_env():
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd-dir", type=Path, required=True,
                    help="existing SVD output dir (config.json + V_part*_t*.pt). "
                         "c_means_per_j.pt is written into this dir.")
    ap.add_argument("--pos-npz", type=Path, default=None,
                    help="paired-pos npz. Defaults to the pos_npz recorded in "
                         "<svd-dir>/config.json.")
    ap.add_argument("--neg-npz", type=Path, default=None,
                    help="paired-neg npz. Defaults to the neg_npz recorded in "
                         "<svd-dir>/config.json.")
    ap.add_argument("--prompt", type=str, default=None,
                    help="single prompt used for both pos and neg passes. "
                         "Defaults to the prompt recorded in <svd-dir>/config.json.")
    ap.add_argument("--drive-source", type=str, default=None,
                    help="'all' | '0' | '1'. Defaults to the filter recorded in "
                         "<svd-dir>/config.json (drive_source_filter).")
    ap.add_argument("--N", type=int, default=None,
                    help="number of paired rows to use after drive-source filter. "
                         "Defaults to N recorded in <svd-dir>/config.json so the "
                         "row selection matches the original SVD.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt-path", type=str, default=None)
    ap.add_argument("--config-name", type=str, default=None)
    ap.add_argument("--config-file", type=str, default="cosmos_policy/config/config.py")
    ap.add_argument("--dataset-stats-path", type=str, default=None)
    ap.add_argument("--t5-cache-path", type=str, default=None)
    ap.add_argument("--output-name", type=str, default="c_means_per_j.pt")
    ap.add_argument("--v-device", choices=["cpu", "cuda", "auto"], default="auto",
                    help="where to keep V during projection. 'auto' = cuda if "
                         "available. CPU is fine if memory-pressured.")
    return ap.parse_args()


def parse_partitions(spec_list, L):
    parts = [(int(a), int(b)) for a, b in spec_list]
    covered = [l for s, e in parts for l in range(s, e + 1)]
    if sorted(covered) != list(range(L)):
        raise ValueError(f"partitions {parts} must tile [0, {L-1}] exactly")
    return parts


def _filter_indices(drive_source_arr, drive_filter):
    if drive_filter == "all":
        return np.arange(drive_source_arr.shape[0], dtype=np.int64)
    code = int(drive_filter)
    if code not in (0, 1):
        raise ValueError(f"drive-source must be 'all', '0', or '1'; got {drive_filter!r}")
    return np.nonzero(drive_source_arr == code)[0]


def load_paired_observations(pos_npz, neg_npz, drive_filter, N_request):
    pos = np.load(pos_npz)
    neg = np.load(neg_npz)
    for key in ("primary_images", "wrist_images", "proprios"):
        if pos[key].shape != neg[key].shape:
            raise ValueError(
                f"{key} shape mismatch: pos={pos[key].shape} neg={neg[key].shape}"
            )
    for key in ("episode_idx", "inference_idx", "drive_source"):
        if not np.array_equal(pos[key], neg[key]):
            raise ValueError(f"{key} differs between pos and neg npz — pairing broken")

    kept = _filter_indices(pos["drive_source"], drive_filter)
    n_avail = int(kept.shape[0])
    if N_request is None or N_request < 0:
        N_used = n_avail
    else:
        if N_request > n_avail:
            raise ValueError(
                f"requested N={N_request} but only {n_avail} rows available "
                f"after --drive-source={drive_filter}"
            )
        N_used = N_request
    selected = kept[:N_used]

    obs_pos_list, obs_neg_list = [], []
    for i in selected:
        obs_pos_list.append({
            "wrist_image":   pos["wrist_images"][i].copy(),
            "primary_image": pos["primary_images"][i].copy(),
            "proprio":       pos["proprios"][i].astype(np.float32).copy(),
        })
        obs_neg_list.append({
            "wrist_image":   neg["wrist_images"][i].copy(),
            "primary_image": neg["primary_images"][i].copy(),
            "proprio":       neg["proprios"][i].astype(np.float32).copy(),
        })
    inference_idx = pos["inference_idx"][selected].astype(np.int64)
    episode_idx   = pos["episode_idx"][selected].astype(np.int64)
    return obs_pos_list, obs_neg_list, N_used, n_avail, selected, inference_idx, episode_idx


def _v_file(svd_dir, p_idx, partitions, t_id, k_target):
    a, b = partitions[p_idx]
    pattern = f"V_part{p_idx}_layers{a}-{b}_t{t_id}_k*.pt"
    matches = sorted(svd_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no V file matching {pattern} in {svd_dir}")
    preferred = [m for m in matches if m.name.endswith(f"_k{k_target}.pt")]
    return (preferred or matches)[0]


def main():
    args = parse_args()

    cfg = json.loads((args.svd_dir / "config.json").read_text())
    L = cfg["L"]
    partitions = parse_partitions(cfg["partitions"], L)
    sel_t = list(cfg["selected_timesteps"])
    T_sel = len(sel_t)
    sel_t_set = set(sel_t)
    sel_t_pos = {t: i for i, t in enumerate(sel_t)}
    k_target = cfg["k_target"]
    sampling_steps = cfg["sampling_steps"]
    D_flat = cfg["D_flat"]
    denoise_t_start = cfg["denoise_t_start"]
    denoise_t_end = cfg["denoise_t_end"]

    layer_to_part = [0] * L
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for l in range(l_start, l_end + 1):
            layer_to_part[l] = p_idx
    part_l_start = {p: partitions[p][0] for p in range(len(partitions))}

    pos_npz = args.pos_npz or Path(cfg["pos_npz"])
    neg_npz = args.neg_npz or Path(cfg["neg_npz"])
    prompt = args.prompt or cfg["prompt"]
    drive_filter = args.drive_source or cfg["drive_source_filter"]
    N_request = args.N if args.N is not None else cfg["N"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    ckpt_path = args.ckpt_path or cfg["ckpt_path"]
    config_name = args.config_name or cfg["config_name"]
    dataset_stats_path = args.dataset_stats_path or f"{ckpt_path}/libero_dataset_statistics.json"
    t5_cache_path = args.t5_cache_path or f"{ckpt_path}/libero_t5_embeddings.pkl"

    (obs_pos_list, obs_neg_list, N_used, n_avail, selected_rows,
     inference_idx_per_pair, episode_idx_per_pair) = load_paired_observations(
        pos_npz, neg_npz, drive_filter, N_request,
    )

    inference_idx_unique = sorted(int(j) for j in np.unique(inference_idx_per_pair))
    J = int(inference_idx_per_pair.max()) + 1
    j_counts = np.bincount(inference_idx_per_pair, minlength=J).astype(np.int64)
    print(f"svd-dir : {args.svd_dir}")
    print(f"pos_npz : {pos_npz}")
    print(f"neg_npz : {neg_npz}")
    print(f"prompt  : {prompt!r}")
    print(f"drive   : {drive_filter}  N_used={N_used}/{n_avail}  L={L}  T_sel={T_sel}  "
          f"sampling_steps={sampling_steps}")
    print(f"J (max inference_idx + 1) = {J}; unique = {inference_idx_unique}")
    print(f"j_counts = {j_counts.tolist()}")

    # ---- Load V for every (partition, timestep) ----
    if args.v_device == "auto":
        v_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        v_device = torch.device(args.v_device)
    print(f"loading V matrices on {v_device} ...")
    V_table = {}  # (p_idx, t_id) -> (D_flat, k_target) bf16 tensor on v_device
    for p_idx in range(len(partitions)):
        for t_id in sel_t:
            fp = _v_file(args.svd_dir, p_idx, partitions, t_id, k_target)
            t0 = time.time()
            V_raw = torch.load(fp, map_location="cpu", weights_only=False)["V"]
            V = V_raw.to(dtype=torch.bfloat16, device=v_device).contiguous()
            V_table[(p_idx, t_id)] = V
            print(f"  loaded V part{p_idx} t{t_id}: {tuple(V.shape)} "
                  f"({fp.stat().st_size/1e9:.2f} GB) on {v_device}  "
                  f"({time.time()-t0:.1f}s)")

    # ---- Allocate per-j contrastive accumulators (small) ----
    # We project each pair's activation immediately and accumulate the
    # projected (k_target,) vector — no large per-j μ tensor.
    # c_per_j[(p_idx, t_id)] shape: (J, L_p, k_target) float32 on v_device.
    c_per_j_acc = {}
    for p_idx, (l_start, l_end) in enumerate(partitions):
        L_p = l_end - l_start + 1
        for t_id in sel_t:
            c_per_j_acc[(p_idx, t_id)] = torch.zeros(
                J, L_p, k_target, dtype=torch.float32, device=v_device
            )

    # ---- Build model ----
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        PolicyEvalConfig,
    )
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        get_t5_embedding_from_cache,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    eval_cfg = PolicyEvalConfig(
        config=config_name,
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
    print(f"loading dataset stats + T5 + model ...")
    dataset_stats = load_dataset_stats(eval_cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(eval_cfg.t5_text_embeddings_path, worker_id=0)
    model, _ = get_model(eval_cfg)
    _ = get_t5_embedding_from_cache(prompt)
    assert len(model.net.blocks) == L

    # ---- Forward hook: project activations through V on-the-fly. ----
    # State filled per run_one call.
    state = {"pass_idx": -1, "sign": 0.0, "j_inf": -1}
    PASSES_PER_STEP = 1

    def make_hook(layer_idx):
        p_idx = layer_to_part[layer_idx]
        l_local = layer_idx - part_l_start[p_idx]

        def hook(module, inputs, output):
            if layer_idx == 0:
                state["pass_idx"] += 1
            pass_idx = state["pass_idx"]
            if pass_idx % PASSES_PER_STEP != 0:
                return output
            step = pass_idx // PASSES_PER_STEP
            if step not in sel_t_set:
                return output

            act_bf = (
                output.detach()[0, denoise_t_start:denoise_t_end, :, :, :]
                .reshape(-1)
            )
            if act_bf.numel() != D_flat:
                raise RuntimeError(
                    f"activation size mismatch: got {act_bf.numel()}, "
                    f"expected D_flat={D_flat} at layer {layer_idx} step={step}"
                )
            t_id = step
            j_inf = state["j_inf"]
            sign = state["sign"]
            count_j = float(j_counts[j_inf])
            assert count_j > 0, f"unexpected j_inf={j_inf} with zero count"
            alpha = sign / count_j

            # Project on v_device. Use bf16 matmul, cast to fp32 for accumulator.
            # Projection is linear: Σ_n α_n · (act_n @ V) = (Σ_n α_n · act_n) @ V,
            # so accumulating projected vectors gives the same c as accumulating
            # μ first and projecting once.
            V = V_table[(p_idx, t_id)]            # (D_flat, k) bf16
            act_dev = act_bf.to(v_device, non_blocking=True)
            c_chunk = (act_dev.unsqueeze(0) @ V).squeeze(0).float()   # (k,)
            c_per_j_acc[(p_idx, t_id)][j_inf, l_local].add_(c_chunk, alpha=alpha)
            return output
        return hook

    handles = [
        b.register_forward_hook(make_hook(i))
        for i, b in enumerate(model.net.blocks)
    ]

    def run_one(prompt_str, obs, j_inf, sign):
        state["pass_idx"] = -1
        state["sign"] = float(sign)
        state["j_inf"] = int(j_inf)
        with torch.inference_mode():
            _ = get_action(
                eval_cfg, model, dataset_stats, obs, prompt_str,
                seed=seed,
                randomize_seed=False,
                num_denoising_steps_action=sampling_steps,
                generate_future_state_and_value_in_parallel=True,
            )
        expected = PASSES_PER_STEP * sampling_steps
        assert state["pass_idx"] + 1 == expected, (
            f"got {state['pass_idx']+1} layer-0 fires, expected {expected}"
        )

    # ---- Iterate pairs ----
    t0 = time.time()
    for i_local in range(N_used):
        tp = time.time()
        j_inf = int(inference_idx_per_pair[i_local])
        run_one(prompt, obs_pos_list[i_local], j_inf, sign=+1.0)
        run_one(prompt, obs_neg_list[i_local], j_inf, sign=-1.0)
        if (i_local + 1) % 8 == 0 or i_local == 0:
            print(f"  pair {i_local+1}/{N_used}  (ep={int(episode_idx_per_pair[i_local])}, "
                  f"j={j_inf}): {time.time()-tp:.1f}s", flush=True)
    print(f"forward passes done: {time.time()-t0:.1f}s")
    for h in handles:
        h.remove()
    del V_table
    torch.cuda.empty_cache() if v_device.type == "cuda" else None

    # ---- Stitch per-partition tiles into (L, J, T_sel, k_target) ----
    c_means_per_j = torch.zeros(L, J, T_sel, k_target, dtype=torch.float32)
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for t_pos, t_id in enumerate(sel_t):
            c_tile = c_per_j_acc[(p_idx, t_id)].to("cpu")   # (J, L_p, k)
            for l in range(l_start, l_end + 1):
                l_local = l - l_start
                c_means_per_j[l, :, t_pos, :] = c_tile[:, l_local, :]
    c_norms_per_j = c_means_per_j.norm(dim=-1)

    # ---- Save ----
    out_path = args.svd_dir / args.output_name
    torch.save({
        "c_means_per_j": c_means_per_j,
        "c_norms_per_j": c_norms_per_j,
        "j_counts": torch.from_numpy(j_counts),
        "inference_idx_set": inference_idx_unique,
        "selected_timesteps": sel_t,
        "layer_partitions": partitions,
        "layer_to_part": layer_to_part,
        "L": L, "J": J, "T_sel": T_sel, "k_target": k_target,
        "prompt": prompt,
        "pos_npz": str(pos_npz),
        "neg_npz": str(neg_npz),
        "drive_source_filter": drive_filter,
        "N_used": N_used,
        "selected_rows_first_8": [int(x) for x in selected_rows[:8]],
        "selected_rows_last_8": [int(x) for x in selected_rows[-8:]],
        "svd_dir": str(args.svd_dir),
        "source_script": "compute_contrastive_per_j.py",
    }, out_path)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  c_means_per_j {tuple(c_means_per_j.shape)}  "
          f"mean ‖c‖ = {c_norms_per_j.mean():.3e}")
    # Per-j summary printout: ‖c‖ averaged over layers & timesteps.
    per_j_mean = c_norms_per_j.mean(dim=(0, 2)).tolist()
    print("  ‖c‖ mean over (l, t) per j:")
    for jj, val in enumerate(per_j_mean):
        cnt = int(j_counts[jj])
        print(f"    j={jj}  n_pairs={cnt:3d}  ‖c‖_avg={val:.3e}")


if __name__ == "__main__":
    main()
