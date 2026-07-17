#!/usr/bin/env python
"""Per-task mean contrastive vectors using a shared multi-task SVD basis.

Reuses the V projection matrices written by
run_partition_svd_pairs_no_action_multitask.py (joint multi-task SVD) and
re-runs the paired forward passes from positive.npz / negative.npz, but this
time accumulates the (pos - neg) activation difference *separately for each
source task_id* (`pos_src_task_id` column in the paired npz).

For each pair n with task_id t_n:

    mu_{l, ts, t_n} += (act_pos[n, l, ts] - act_neg[n, l, ts]) / count(t_n)

then project once at the end (or equivalently project each row on-the-fly,
since projection is linear):

    c_{l, ts, t_n} = mu_{l, ts, t_n} @ V_{p(l), ts}

V (and svd_summary.pt) on disk are NOT modified, so the existing per-(l, ts)
c_means stay valid. This script writes a single new artifact:

    <svd-dir>/c_means_per_task.pt  with keys

        c_means_per_task   : float32 (L, T_tasks, T_sel, k_target)
        c_norms_per_task   : float32 (L, T_tasks, T_sel)
        task_counts        : int64   (T_tasks,)   -- # pairs per task
        unique_task_ids    : list[int]            -- aligned with axis 1
        unique_prompts     : list[str]            -- aligned with unique_task_ids
        selected_timesteps : list[int]
        layer_partitions   : list[(l_start, l_end)]
        layer_to_part      : list[int]
        prompt_per_task    : dict[str, str]       -- {task_id (str) -> prompt}
        pos_npz, neg_npz, drive_source_filter, N_used, svd_dir, source_script

Modes
-----
--mode sketch : one Python invocation per rank. Pin one GPU via
                CUDA_VISIBLE_DEVICES, pass --rank R --world-size W, and the
                process iterates pairs `range(R, N, W)`. Per-task projected
                c sums are accumulated and dumped to scratch.

--mode finalize : single-rank finalize. Sums per-rank partial c-sums and
                  task counts, divides by per-task counts, saves
                  c_means_per_task.pt into <svd-dir>.
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


# ====================================================================
# CLI
# ====================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sketch", "finalize"], required=True)
    ap.add_argument("--rank", type=int,
                    default=int(os.environ.get("RANK", 0)))
    ap.add_argument("--world-size", type=int,
                    default=int(os.environ.get("WORLD_SIZE", 1)))

    ap.add_argument("--svd-dir", type=Path, required=True,
                    help="existing multi-task SVD output dir (must contain "
                         "config.json + V_part*_t*.pt). c_means_per_task.pt "
                         "is written into this dir.")
    ap.add_argument("--pos-npz", type=Path, default=None,
                    help="paired-pos npz with `pos_src_task_id`. Defaults to "
                         "the pos_npz recorded in <svd-dir>/config.json.")
    ap.add_argument("--neg-npz", type=Path, default=None,
                    help="paired-neg npz with `neg_src_task_id`. Defaults to "
                         "the neg_npz recorded in <svd-dir>/config.json.")
    ap.add_argument("--task-prompts-json", type=Path, default=None,
                    help="optional JSON {task_id -> prompt}. If unset, the "
                         "task -> prompt mapping is read from <svd-dir>/"
                         "config.json (unique_task_ids + unique_prompts).")
    ap.add_argument("--drive-source", type=str, default=None,
                    help="'all' | '0' | '1'. Defaults to drive_source_filter "
                         "in <svd-dir>/config.json so the row selection "
                         "matches the original SVD.")
    ap.add_argument("--N", type=int, default=None,
                    help="number of paired rows to use after drive-source + "
                         "within-task filters. Defaults to N in config.json.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt-path", type=str, default=None)
    ap.add_argument("--config-name", type=str, default=None)
    ap.add_argument("--config-file", type=str,
                    default="cosmos_policy/config/config.py")
    ap.add_argument("--dataset-stats-path", type=str, default=None)
    ap.add_argument("--t5-cache-path", type=str, default=None)
    ap.add_argument("--output-name", type=str, default="c_means_per_task.pt")
    ap.add_argument("--v-device", choices=["cpu", "cuda", "auto"],
                    default="auto",
                    help="where to keep V during projection.")
    ap.add_argument("--scratch-dir", type=Path, default=None,
                    help="rank-partial files go here. Defaults to "
                         "<svd-dir>/scratch_per_task.")
    ap.add_argument("--keep-scratch", action="store_true")
    return ap.parse_args()


def parse_partitions(spec_list, L):
    parts = [(int(a), int(b)) for a, b in spec_list]
    covered = [l for s, e in parts for l in range(s, e + 1)]
    if sorted(covered) != list(range(L)):
        raise ValueError(f"partitions {parts} must tile [0, {L-1}] exactly")
    return parts


def scratch_root(args):
    return args.scratch_dir or (args.svd_dir / "scratch_per_task")


def my_indices(N, rank, world_size):
    return list(range(rank, N, world_size))


# ====================================================================
# task_id -> prompt resolution
# ====================================================================

def _load_task_prompts_from_json(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"--task-prompts-json must be a JSON object: {path}")
    return {int(k): str(v) for k, v in raw.items()}


def resolve_task_prompts(args, svd_cfg) -> tuple[dict, str]:
    """Return ({task_id: prompt}, source_description).

    Priority: explicit --task-prompts-json, else <svd-dir>/config.json's
    unique_task_ids + unique_prompts (aligned, written by the multitask SVD).
    """
    if args.task_prompts_json is not None:
        m = _load_task_prompts_from_json(args.task_prompts_json)
        return m, f"json:{args.task_prompts_json}"
    tids = svd_cfg.get("unique_task_ids")
    prompts = svd_cfg.get("unique_prompts")
    if tids is None or prompts is None or len(tids) != len(prompts):
        raise ValueError(
            "<svd-dir>/config.json is missing aligned unique_task_ids + "
            "unique_prompts (expected from the multitask SVD). Pass "
            "--task-prompts-json explicitly."
        )
    m = {int(t): str(p) for t, p in zip(tids, prompts)}
    return m, f"svd-config:{args.svd_dir / 'config.json'}"


# ====================================================================
# Observation loading — paired, per-row task_id (multi-task)
# ====================================================================

def _filter_indices(drive_source_arr, drive_filter):
    if drive_filter == "all":
        return np.arange(drive_source_arr.shape[0], dtype=np.int64)
    code = int(drive_filter)
    if code not in (0, 1):
        raise ValueError(
            f"drive-source must be 'all', '0', or '1'; got {drive_filter!r}"
        )
    return np.nonzero(drive_source_arr == code)[0]


def _require_keys(npz, npz_name, keys):
    missing = [k for k in keys if k not in npz.files]
    if missing:
        raise ValueError(
            f"{npz_name} is missing required keys for multi-task c-per-task: "
            f"{missing}. The paired NPZ must carry pos_src_task_id / "
            f"neg_src_task_id (produced by pair_inputs_by_similarity.py from "
            f"the multitask collector output)."
        )


def load_paired_observations(pos_npz: Path, neg_npz: Path, drive_filter: str,
                              N_request, task_to_prompt: dict):
    pos = np.load(pos_npz, allow_pickle=True)
    neg = np.load(neg_npz, allow_pickle=True)

    for key in ("primary_images", "wrist_images", "proprios"):
        if pos[key].shape != neg[key].shape:
            raise ValueError(
                f"{key} shape mismatch: pos={pos[key].shape} neg={neg[key].shape}"
            )
    for key in ("episode_idx", "inference_idx", "drive_source"):
        if not np.array_equal(pos[key], neg[key]):
            raise ValueError(f"{key} differs between pos and neg npz — pairing broken")

    _require_keys(pos, str(pos_npz), ("pos_src_task_id",))
    _require_keys(neg, str(neg_npz), ("neg_src_task_id",))
    pos_task = pos["pos_src_task_id"].astype(np.int64)
    neg_task = neg["neg_src_task_id"].astype(np.int64)
    if pos_task.shape != neg_task.shape:
        raise ValueError(
            f"task_id shape mismatch: pos={pos_task.shape} neg={neg_task.shape}"
        )

    within_mask = (pos_task == neg_task)
    n_total_pairs = int(within_mask.shape[0])
    n_cross = int((~within_mask).sum())
    if n_cross > 0:
        print(f"WARNING: dropping {n_cross}/{n_total_pairs} cross-task pairs "
              f"(pos and neg from different tasks).", flush=True)

    task_ids_used_arr = np.unique(pos_task[within_mask])
    unique_tasks = sorted(int(t) for t in task_ids_used_arr.tolist())
    missing = [t for t in unique_tasks if t not in task_to_prompt]
    if missing:
        raise ValueError(
            f"task_id -> prompt mapping is missing entries for tasks {missing}. "
            f"Resolved mapping has tasks {sorted(task_to_prompt)}."
        )
    prompts_lookup = [task_to_prompt[t] for t in unique_tasks]
    task_to_taskidx = {t: i for i, t in enumerate(unique_tasks)}

    kept_drive = _filter_indices(pos["drive_source"], drive_filter)
    kept_mask = np.zeros(n_total_pairs, dtype=bool)
    kept_mask[kept_drive] = True
    kept_mask &= within_mask
    kept = np.nonzero(kept_mask)[0]
    n_avail = int(kept.shape[0])
    if N_request is None or N_request < 0:
        N_used = n_avail
    else:
        if N_request > n_avail:
            raise ValueError(
                f"requested N={N_request} but only {n_avail} pairs available "
                f"after --drive-source={drive_filter} + within-task filters."
            )
        N_used = N_request
    selected = kept[:N_used]

    task_idx_selected = np.array(
        [task_to_taskidx[int(t)] for t in pos_task[selected]], dtype=np.int32
    )
    prompts_per_row = [prompts_lookup[int(c)] for c in task_idx_selected]

    pos_w  = pos["wrist_images"]
    pos_p  = pos["primary_images"]
    pos_pr = pos["proprios"]
    neg_w  = neg["wrist_images"]
    neg_p  = neg["primary_images"]
    neg_pr = neg["proprios"]
    obs_pos_list, obs_neg_list = [], []
    for i in selected:
        obs_pos_list.append({
            "wrist_image":   pos_w[i].copy(),
            "primary_image": pos_p[i].copy(),
            "proprio":       pos_pr[i].astype(np.float32).copy(),
        })
        obs_neg_list.append({
            "wrist_image":   neg_w[i].copy(),
            "primary_image": neg_p[i].copy(),
            "proprio":       neg_pr[i].astype(np.float32).copy(),
        })

    return {
        "obs_pos_list": obs_pos_list,
        "obs_neg_list": obs_neg_list,
        "prompts_per_row": prompts_per_row,
        "prompts_lookup": prompts_lookup,
        "unique_tasks": unique_tasks,
        "task_idx_selected": task_idx_selected,
        "task_id_selected": pos_task[selected].astype(np.int32),
        "N_used": N_used,
        "n_avail": n_avail,
        "selected_rows": selected,
        "drive_source_used": pos["drive_source"][selected].astype(np.int32),
    }


# ====================================================================
# V file discovery
# ====================================================================

def _v_file(svd_dir, p_idx, partitions, t_id, k_target):
    a, b = partitions[p_idx]
    pattern = f"V_part{p_idx}_layers{a}-{b}_t{t_id}_k*.pt"
    matches = sorted(svd_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no V file matching {pattern} in {svd_dir}")
    preferred = [m for m in matches if m.name.endswith(f"_k{k_target}.pt")]
    return (preferred or matches)[0]


# ====================================================================
# Sketch mode
# ====================================================================

def run_sketch(args):
    rank = args.rank
    world_size = args.world_size
    assert 0 <= rank < world_size, (
        f"--rank {rank} not in [0, --world-size {world_size})"
    )

    svd_cfg = json.loads((args.svd_dir / "config.json").read_text())
    L = svd_cfg["L"]
    partitions = parse_partitions(svd_cfg["partitions"], L)
    sel_t = list(svd_cfg["selected_timesteps"])
    T_sel = len(sel_t)
    sel_t_set = set(sel_t)
    k_target = svd_cfg["k_target"]
    sampling_steps = svd_cfg["sampling_steps"]
    D_flat = svd_cfg["D_flat"]
    denoise_t_start = svd_cfg["denoise_t_start"]
    denoise_t_end = svd_cfg["denoise_t_end"]

    layer_to_part = [0] * L
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for l in range(l_start, l_end + 1):
            layer_to_part[l] = p_idx
    part_l_start = {p: partitions[p][0] for p in range(len(partitions))}

    pos_npz = args.pos_npz or Path(svd_cfg["pos_npz"])
    neg_npz = args.neg_npz or Path(svd_cfg["neg_npz"])
    drive_filter = args.drive_source or svd_cfg["drive_source_filter"]
    N_request = args.N if args.N is not None else svd_cfg["N"]
    seed = args.seed if args.seed is not None else svd_cfg["seed"]
    ckpt_path = args.ckpt_path or svd_cfg["ckpt_path"]
    config_name = args.config_name or svd_cfg["config_name"]
    dataset_stats_path = (
        args.dataset_stats_path
        or f"{ckpt_path}/libero_dataset_statistics.json"
    )
    t5_cache_path = args.t5_cache_path or f"{ckpt_path}/libero_t5_embeddings.pkl"

    task_to_prompt, prompts_source = resolve_task_prompts(args, svd_cfg)
    print(f"[rank {rank}/{world_size}] task -> prompt source: {prompts_source}",
          flush=True)

    loaded = load_paired_observations(
        pos_npz, neg_npz, drive_filter, N_request, task_to_prompt,
    )
    obs_pos_list      = loaded["obs_pos_list"]
    obs_neg_list      = loaded["obs_neg_list"]
    prompts_per_row   = loaded["prompts_per_row"]
    prompts_lookup    = loaded["prompts_lookup"]
    unique_tasks      = loaded["unique_tasks"]
    task_idx_selected = loaded["task_idx_selected"]
    task_id_selected  = loaded["task_id_selected"]
    N_used            = loaded["N_used"]
    n_avail           = loaded["n_avail"]
    selected_rows     = loaded["selected_rows"]
    drive_source_used = loaded["drive_source_used"]
    T_tasks = len(unique_tasks)

    idx = my_indices(N_used, rank, world_size)
    print(f"[rank {rank}] svd_dir : {args.svd_dir}", flush=True)
    print(f"[rank {rank}] pos     : {pos_npz}", flush=True)
    print(f"[rank {rank}] neg     : {neg_npz}", flush=True)
    print(f"[rank {rank}] drive   : {drive_filter}  N_used={N_used}/{n_avail}  "
          f"L={L}  T_sel={T_sel}  sampling_steps={sampling_steps}", flush=True)
    print(f"[rank {rank}] assigned {len(idx)} pairs: "
          f"{idx[:5]}{'...' if len(idx) > 5 else ''}", flush=True)
    print(f"[rank {rank}] unique tasks ({T_tasks}):", flush=True)
    for i, tid in enumerate(unique_tasks):
        n_rows = int((task_id_selected == tid).sum())
        print(f"[rank {rank}]   taskidx{i}=task{tid} ({n_rows} rows) "
              f"{prompts_lookup[i]!r}", flush=True)

    # ---- Load V on this rank's GPU. ----
    if args.v_device == "auto":
        v_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        v_device = torch.device(args.v_device)
    print(f"[rank {rank}] loading V matrices on {v_device} ...", flush=True)
    V_table = {}
    for p_idx in range(len(partitions)):
        for t_id in sel_t:
            fp = _v_file(args.svd_dir, p_idx, partitions, t_id, k_target)
            t0 = time.time()
            V_raw = torch.load(fp, map_location="cpu", weights_only=False)["V"]
            V = V_raw.to(dtype=torch.bfloat16, device=v_device).contiguous()
            V_table[(p_idx, t_id)] = V
            print(f"[rank {rank}]   V part{p_idx} t{t_id}: {tuple(V.shape)} "
                  f"({fp.stat().st_size/1e9:.2f} GB) on {v_device}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    # ---- Per-rank per-task projected sums (signed). Normalize at finalize. ----
    # c_sum_partial[(p_idx, t_id)] : (T_tasks, L_p, k_target) f32 on v_device.
    c_sum_partial = {}
    for p_idx, (l_start, l_end) in enumerate(partitions):
        L_p = l_end - l_start + 1
        for t_id in sel_t:
            c_sum_partial[(p_idx, t_id)] = torch.zeros(
                T_tasks, L_p, k_target, dtype=torch.float32, device=v_device,
            )

    # ---- Build model. ----
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
    print(f"[rank {rank}] loading dataset stats + T5 + model ...", flush=True)
    dataset_stats = load_dataset_stats(eval_cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(eval_cfg.t5_text_embeddings_path,
                                  worker_id=rank)
    model, _ = get_model(eval_cfg)
    for p in prompts_lookup:
        _ = get_t5_embedding_from_cache(p)
    assert len(model.net.blocks) == L

    # ---- Forward hook: project on-the-fly through the loaded V table. ----
    # Linearity of projection: Σ_n s_n · (act_n @ V) = (Σ_n s_n · act_n) @ V,
    # so signed-sum-of-projections equals projection-of-signed-mu. We sum
    # WITHOUT normalizing here; the finalize step divides by per-task counts.
    state = {"pass_idx": -1, "sign": 0.0, "taskidx": -1}
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
            sign = state["sign"]
            taskidx = state["taskidx"]

            V = V_table[(p_idx, t_id)]                       # (D_flat, k) bf16
            act_dev = act_bf.to(v_device, non_blocking=True)
            c_chunk = (act_dev.unsqueeze(0) @ V).squeeze(0).float()   # (k,)
            c_sum_partial[(p_idx, t_id)][taskidx, l_local].add_(c_chunk,
                                                                  alpha=sign)
            return output
        return hook

    handles = [
        b.register_forward_hook(make_hook(i))
        for i, b in enumerate(model.net.blocks)
    ]

    def run_one(prompt_str, obs, taskidx, sign):
        state["pass_idx"] = -1
        state["sign"] = float(sign)
        state["taskidx"] = int(taskidx)
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

    # ---- Iterate this rank's pairs. ----
    t0 = time.time()
    task_count_partial = np.zeros(T_tasks, dtype=np.int64)
    for j_local, n in enumerate(idx):
        tp = time.time()
        prompt_n  = prompts_per_row[n]
        taskidx_n = int(task_idx_selected[n])
        task_count_partial[taskidx_n] += 1
        run_one(prompt_n, obs_pos_list[n], taskidx_n, sign=+1.0)
        run_one(prompt_n, obs_neg_list[n], taskidx_n, sign=-1.0)
        print(f"[rank {rank}] ({j_local+1}/{len(idx)}) pair {n} "
              f"(ds={int(drive_source_used[n])}, task={int(task_id_selected[n])}, "
              f"taskidx={taskidx_n}): {time.time()-tp:.1f}s", flush=True)
    print(f"[rank {rank}] sketch inference done: {time.time()-t0:.1f}s",
          flush=True)
    for h in handles:
        h.remove()

    # ---- Dump partial. ----
    rank_dir = scratch_root(args) / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    # Move c sums to CPU before saving (smaller .pt, no GPU dependency on load).
    c_sum_partial_cpu = {
        key: v.to("cpu").contiguous() for key, v in c_sum_partial.items()
    }
    partial_path = rank_dir / "c_partial.pt"
    torch.save({
        "c_sum_partial": c_sum_partial_cpu,
        "task_count_partial": torch.from_numpy(task_count_partial),
        "unique_tasks": [int(t) for t in unique_tasks],
        "prompts_lookup": prompts_lookup,
        "partitions": partitions,
        "selected_timesteps": sel_t,
        "L": L, "T_sel": T_sel, "T_tasks": T_tasks, "k_target": k_target,
        "rank": rank, "world_size": world_size,
        "N_used": N_used,
        "drive_source_filter": drive_filter,
        "pos_npz": str(pos_npz), "neg_npz": str(neg_npz),
        "svd_dir": str(args.svd_dir),
    }, partial_path)
    (rank_dir / "DONE").write_text(
        f"rank {rank} per-task sketch complete at {time.time()}\n"
        f"pairs={len(idx)}\n"
    )
    print(f"[rank {rank}] dumped {partial_path} "
          f"({partial_path.stat().st_size / 1e6:.2f} MB)", flush=True)
    print(f"[rank {rank}] task counts (this rank): "
          f"{task_count_partial.tolist()}", flush=True)
    print(f"[rank {rank}] sketch: done", flush=True)


# ====================================================================
# Finalize mode
# ====================================================================

def run_finalize(args):
    svd_cfg = json.loads((args.svd_dir / "config.json").read_text())
    L = svd_cfg["L"]
    partitions = parse_partitions(svd_cfg["partitions"], L)
    sel_t = list(svd_cfg["selected_timesteps"])
    T_sel = len(sel_t)
    k_target = svd_cfg["k_target"]
    scratch = scratch_root(args)

    layer_to_part = [0] * L
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for l in range(l_start, l_end + 1):
            layer_to_part[l] = p_idx

    # Discover world_size + verify all ranks done.
    rank_dirs = sorted(p for p in scratch.glob("rank*") if p.is_dir())
    if not rank_dirs:
        raise RuntimeError(f"no rank* subdirs in {scratch}")
    world_size = len(rank_dirs)
    for d in rank_dirs:
        if not (d / "DONE").exists():
            raise RuntimeError(f"rank dir missing DONE: {d}")
    print(f"finalize: scratch={scratch}  world_size={world_size}  "
          f"T_sel={T_sel}  partitions={partitions}", flush=True)

    # Sum across ranks.
    c_sum_total = None
    task_count_total = None
    unique_tasks = None
    prompts_lookup = None
    pos_npz = neg_npz = drive_filter = None
    N_used_first = None
    for i, d in enumerate(rank_dirs):
        partial_path = d / "c_partial.pt"
        part = torch.load(partial_path, map_location="cpu", weights_only=False)
        if unique_tasks is None:
            unique_tasks = part["unique_tasks"]
            prompts_lookup = part["prompts_lookup"]
            T_tasks = part["T_tasks"]
            pos_npz = part["pos_npz"]
            neg_npz = part["neg_npz"]
            drive_filter = part["drive_source_filter"]
            N_used_first = part["N_used"]
            c_sum_total = {}
            for key, v in part["c_sum_partial"].items():
                c_sum_total[key] = v.clone()
            task_count_total = part["task_count_partial"].clone()
        else:
            assert part["unique_tasks"] == unique_tasks, (
                f"unique_tasks mismatch at {d}"
            )
            assert part["prompts_lookup"] == prompts_lookup, (
                f"prompts_lookup mismatch at {d}"
            )
            assert part["N_used"] == N_used_first, (
                f"N_used mismatch at {d}: {part['N_used']} vs {N_used_first}"
            )
            for key, v in part["c_sum_partial"].items():
                c_sum_total[key].add_(v)
            task_count_total.add_(part["task_count_partial"])
        print(f"  + {partial_path}  pairs={int(part['task_count_partial'].sum())}  "
              f"({partial_path.stat().st_size / 1e6:.2f} MB)", flush=True)

    T_tasks = len(unique_tasks)
    print(f"unique tasks: {unique_tasks}", flush=True)
    print(f"task_counts:  {task_count_total.tolist()}", flush=True)
    if int(task_count_total.sum()) != N_used_first:
        raise RuntimeError(
            f"sum(task_counts)={int(task_count_total.sum())} != N_used="
            f"{N_used_first}; ranks may have overlapping/missing assignments."
        )
    if (task_count_total <= 0).any():
        bad = [unique_tasks[i] for i in range(T_tasks)
               if int(task_count_total[i]) <= 0]
        raise RuntimeError(f"tasks with zero pairs: {bad}")

    # Stitch per-partition tiles into (L, T_tasks, T_sel, k_target).
    c_means_per_task = torch.zeros(L, T_tasks, T_sel, k_target,
                                    dtype=torch.float32)
    inv_counts = (1.0 / task_count_total.to(torch.float64)).to(torch.float32)
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for t_pos, t_id in enumerate(sel_t):
            tile = c_sum_total[(p_idx, t_id)]                   # (T_tasks, L_p, k)
            tile = tile * inv_counts.view(-1, 1, 1)             # divide per task
            for l in range(l_start, l_end + 1):
                l_local = l - l_start
                c_means_per_task[l, :, t_pos, :] = tile[:, l_local, :]

    c_norms_per_task = c_means_per_task.norm(dim=-1)
    prompt_per_task = {str(int(t)): p
                       for t, p in zip(unique_tasks, prompts_lookup)}

    out_path = args.svd_dir / args.output_name
    torch.save({
        "c_means_per_task": c_means_per_task,
        "c_norms_per_task": c_norms_per_task,
        "task_counts": task_count_total,
        "unique_task_ids": [int(t) for t in unique_tasks],
        "unique_prompts": prompts_lookup,
        "prompt_per_task": prompt_per_task,
        "selected_timesteps": sel_t,
        "layer_partitions": partitions,
        "layer_to_part": layer_to_part,
        "L": L, "T_tasks": T_tasks, "T_sel": T_sel, "k_target": k_target,
        "pos_npz": pos_npz, "neg_npz": neg_npz,
        "drive_source_filter": drive_filter,
        "N_used": N_used_first,
        "svd_dir": str(args.svd_dir),
        "world_size": world_size,
        "source_script": "compute_contrastive_per_task.py",
    }, out_path)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)",
          flush=True)
    print(f"  c_means_per_task shape: {tuple(c_means_per_task.shape)}",
          flush=True)
    per_task_mean = c_norms_per_task.mean(dim=(0, 2)).tolist()
    print("  ‖c‖ mean over (l, ts) per task:", flush=True)
    for i, tid in enumerate(unique_tasks):
        cnt = int(task_count_total[i])
        print(f"    task{tid:>2}  n_pairs={cnt:4d}  ‖c‖_avg={per_task_mean[i]:.3e}  "
              f"{prompts_lookup[i]!r}", flush=True)

    if not args.keep_scratch:
        shutil.rmtree(scratch)
        print(f"cleaned scratch: {scratch}", flush=True)


def main():
    args = parse_args()
    if args.mode == "sketch":
        run_sketch(args)
    else:
        run_finalize(args)


if __name__ == "__main__":
    main()
