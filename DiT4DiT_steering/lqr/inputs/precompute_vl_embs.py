#!/usr/bin/env python
"""Precompute VLM backbone embeddings (vl_embs) for all observations in an NPZ file.

The VLM backbone (Cosmos-2.5-2B) is large (~20 GB) and slow to load from NFS.
This script loads it once, runs all observations through it, and caches the
resulting vl_embs tensor.  SVD sketch ranks can then load the small ~MB-scale
action DiT only, rather than the full 20 GB model.

Usage:
    python precompute_vl_embs.py --pos-npz /path/pos.npz --neg-npz /path/neg.npz \
        --out-path /path/vl_embs.pt --ckpt-path /path/pytorch_model.pt

Output:
    /path/vl_embs.pt  -- dict with keys:
        "pos": (N, seq_len, hidden_size) float16 tensor
        "neg": (N, seq_len, hidden_size) float16 tensor
        "prompt": str
        "ckpt_path": str
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

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
IMAGE_SIZE = 224


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def obs_to_example(primary_raw, wrist_raw, proprio_raw, prompt, max_state_dim=16):
    primary = cv2.resize(primary_raw, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    wrist   = cv2.resize(wrist_raw,   (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    concat_img = np.concatenate([primary, wrist], axis=1)

    sin_s = np.sin(proprio_raw[None])
    cos_s = np.cos(proprio_raw[None])
    state_enc = np.stack([sin_s, cos_s], axis=-1).reshape(1, -1).astype(np.float32)
    pad = max_state_dim - state_enc.shape[-1]
    if pad > 0:
        state_enc = np.pad(state_enc, ((0, 0), (0, pad)), "constant")
    return {"image": [concat_img], "lang": prompt, "state": state_enc}


CKPT_INTERVAL = 50  # save checkpoint every this many batches


def compute_vl_embs_for_npz(npz_path, model, prompt, device, max_state_dim, batch_size=8,
                             ckpt_path: Path | None = None,
                             obs_start: int = 0, obs_end: int | None = None):
    """Run VLM backbone on observations [obs_start:obs_end] in an NPZ.

    Returns (n, seq_len, H) float16. Checkpoints every CKPT_INTERVAL batches.
    """
    _log(f"  loading {npz_path.name} ...")
    data = np.load(npz_path)
    N_total = data["primary_images"].shape[0]
    if obs_end is None:
        obs_end = N_total
    N = obs_end - obs_start
    _log(f"  {N} observations (indices {obs_start}:{obs_end} of {N_total})")

    # Resume from checkpoint if one exists
    all_embs = []
    start_batch = 0
    if ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if ckpt.get("npz") == str(npz_path) and ckpt.get("obs_start") == obs_start:
            all_embs = ckpt["embs"]
            start_batch = ckpt["next_batch"]
            _log(f"  resuming from checkpoint: {start_batch * batch_size}/{N} observations done")

    for batch_idx, local_start in enumerate(range(start_batch * batch_size, N, batch_size),
                                            start=start_batch):
        abs_start = obs_start + local_start
        abs_end   = min(abs_start + batch_size, obs_end)
        local_end = abs_end - obs_start

        batch_images, batch_states = [], []
        for i in range(abs_start, abs_end):
            primary_raw = data["primary_images"][i]
            wrist_raw   = data["wrist_images"][i]
            proprio_raw = data["proprios"][i].astype(np.float32)
            ex = obs_to_example(primary_raw, wrist_raw, proprio_raw, prompt, max_state_dim)
            batch_images.append(ex["image"])
            batch_states.append(ex["state"])

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bi = model.backbone_interface.build_cosmos_inputs(
                    images=batch_images, instructions=[prompt] * len(batch_images),
                )
                bout = model.backbone_interface(
                    **bi, output_hidden_states=True, output_attentions=False, return_dict=True,
                )
                vl_embs = bout.hidden_states[-1]  # (B, seq_len, H)

        mem_peak_gb = torch.cuda.max_memory_allocated() / 1e9
        all_embs.append(vl_embs.half().cpu())
        if batch_idx % 10 == 0:
            _log(f"    [{local_end}/{N}]  embs shape: {vl_embs.shape}  peak GPU: {mem_peak_gb:.2f} GB")

        if ckpt_path is not None and (batch_idx + 1) % CKPT_INTERVAL == 0:
            torch.save({"npz": str(npz_path), "obs_start": obs_start, "embs": all_embs,
                        "next_batch": batch_idx + 1}, ckpt_path)
            _log(f"    [ckpt] saved {local_end}/{N} to {ckpt_path.name}")
        del bi, bout, vl_embs

    result = torch.cat(all_embs, dim=0)  # (N, seq_len, H)
    _log(f"  done: {result.shape}  {result.element_size() * result.numel() / 1e9:.2f} GB")
    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()
    return result


def _shard_range(N, worker_id, n_workers):
    """Return (start, end) for this worker's slice of N items."""
    chunk = math.ceil(N / n_workers)
    start = worker_id * chunk
    end = min(start + chunk, N)
    return start, end


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos-npz",  type=Path, required=True)
    ap.add_argument("--neg-npz",  type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True,
                    help="Output .pt file with precomputed vl_embs.")
    ap.add_argument("--prompt",   type=str,
                    default="put both the cream cheese box and the butter in the basket")
    ap.add_argument("--ckpt-path", type=str, default=CKPT_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="VLM batch size (lower if OOM).")
    ap.add_argument("--worker-id",  type=int, default=0,
                    help="Index of this worker (0-based).")
    ap.add_argument("--n-workers",  type=int, default=1,
                    help="Total number of parallel workers.")
    ap.add_argument("--merge", action="store_true",
                    help="Merge worker shard files into the final output and exit.")
    args = ap.parse_args()

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        _log(f"Merging {args.n_workers} worker shards → {args.out_path}")
        shards = []
        for w in range(args.n_workers):
            p = args.out_path.with_suffix(f".w{w}of{args.n_workers}.pt")
            if not p.exists():
                raise FileNotFoundError(f"Missing shard: {p}")
            shards.append(torch.load(p, map_location="cpu"))
            _log(f"  loaded shard {w}: pos {shards[-1]['pos_embs'].shape}  neg {shards[-1]['neg_embs'].shape}")
        pos_embs = torch.cat([s["pos_embs"] for s in shards], dim=0)
        neg_embs = torch.cat([s["neg_embs"] for s in shards], dim=0)
        torch.save({
            "pos": pos_embs, "neg": neg_embs,
            "prompt": shards[0]["prompt"], "ckpt_path": shards[0]["ckpt_path"],
            "pos_npz": str(args.pos_npz), "neg_npz": str(args.neg_npz),
        }, args.out_path)
        sz = args.out_path.stat().st_size / 1e9
        _log(f"Merged: pos={pos_embs.shape}  neg={neg_embs.shape}  {sz:.2f} GB → {args.out_path}")
        for w in range(args.n_workers):
            args.out_path.with_suffix(f".w{w}of{args.n_workers}.pt").unlink(missing_ok=True)
        _log("Done.")
        return

    # ── Worker mode ───────────────────────────────────────────────────────────
    shard_path = args.out_path.with_suffix(f".w{args.worker_id}of{args.n_workers}.pt")
    if shard_path.exists():
        _log(f"Shard already exists: {shard_path} — skipping.")
        return
    if args.out_path.exists() and args.n_workers == 1:
        _log(f"Output already exists: {args.out_path} — skipping.")
        return

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    # Figure out this worker's index ranges
    N_pos = np.load(args.pos_npz, mmap_mode="r")["primary_images"].shape[0]
    N_neg = np.load(args.neg_npz, mmap_mode="r")["primary_images"].shape[0]
    pos_start, pos_end = _shard_range(N_pos, args.worker_id, args.n_workers)
    neg_start, neg_end = _shard_range(N_neg, args.worker_id, args.n_workers)
    _log(f"worker {args.worker_id}/{args.n_workers}: "
         f"pos[{pos_start}:{pos_end}]  neg[{neg_start}:{neg_end}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"device: {device}")

    _log(f"loading DiT4DiT from {args.ckpt_path} ...")
    t0 = time.time()
    import DiT4DiT.model.framework.DiT4DiT  # noqa: F401  register framework
    from DiT4DiT.model.framework.base_framework import baseframework
    from DiT4DiT.model.framework.share_tools import read_mode_config
    model = baseframework.from_pretrained(args.ckpt_path).to(device).eval()
    cfg, _ = read_mode_config(args.ckpt_path)
    max_state_dim = cfg["framework"]["action_model"]["state_dim"]
    mem_model_gb = torch.cuda.memory_allocated() / 1e9
    _log(f"model loaded in {time.time()-t0:.1f}s  max_state_dim={max_state_dim}  "
         f"GPU after load: {mem_model_gb:.2f} GB allocated")

    pos_ckpt = args.out_path.with_suffix(f".pos.w{args.worker_id}.ckpt.pt")
    neg_ckpt = args.out_path.with_suffix(f".neg.w{args.worker_id}.ckpt.pt")

    _log(f"computing pos vl_embs [{pos_start}:{pos_end}] ...")
    pos_embs = compute_vl_embs_for_npz(
        args.pos_npz, model, args.prompt, device, max_state_dim, args.batch_size,
        ckpt_path=pos_ckpt, obs_start=pos_start, obs_end=pos_end,
    )

    _log(f"computing neg vl_embs [{neg_start}:{neg_end}] ...")
    neg_embs = compute_vl_embs_for_npz(
        args.neg_npz, model, args.prompt, device, max_state_dim, args.batch_size,
        ckpt_path=neg_ckpt, obs_start=neg_start, obs_end=neg_end,
    )

    torch.save({
        "worker_id": args.worker_id, "n_workers": args.n_workers,
        "pos_start": pos_start, "pos_end": pos_end, "pos_embs": pos_embs,
        "neg_start": neg_start, "neg_end": neg_end, "neg_embs": neg_embs,
        "prompt": args.prompt, "ckpt_path": args.ckpt_path,
    }, shard_path)
    _log(f"saved shard → {shard_path}")

    if args.n_workers == 1:
        # Single-worker: produce final output directly
        torch.save({
            "pos": pos_embs, "neg": neg_embs,
            "prompt": args.prompt, "ckpt_path": args.ckpt_path,
            "pos_npz": str(args.pos_npz), "neg_npz": str(args.neg_npz),
        }, args.out_path)
        shard_path.unlink(missing_ok=True)
        sz = args.out_path.stat().st_size / 1e9
        _log(f"saved {args.out_path} ({sz:.2f} GB)")
        _log("Done. Pass --vl-embs-path to run_partition_svd_pairs_no_action.sh.")


if __name__ == "__main__":
    main()
