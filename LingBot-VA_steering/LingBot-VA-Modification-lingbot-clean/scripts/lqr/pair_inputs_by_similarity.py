#!/usr/bin/env python
"""Pair unpaired positive/negative LingBot policy inputs by similarity.

This is the LingBot-local equivalent of the repo root's
notebooks/lqr/svd/pair_inputs_by_similarity.py. It is used for gripper XYZ
init-position perturbations, where positive rows come from successful
perturbed rollouts and negative rows come from failed perturbed rollouts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np


def _log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def _normalize_per_axis(x: np.ndarray, std_ref: np.ndarray) -> np.ndarray:
    return x / std_ref.clip(min=1e-6)


def _downsample_image_block(imgs: np.ndarray, block: int) -> np.ndarray:
    if block <= 1:
        return imgs.astype(np.float32)
    n, h, w, c = imgs.shape
    h2 = (h // block) * block
    w2 = (w // block) * block
    arr = imgs[:, :h2, :w2, :].astype(np.float32)
    return arr.reshape(n, h2 // block, block, w2 // block, block, c).mean(axis=(2, 4))


def build_features(
    pos: Dict[str, np.ndarray],
    neg: Dict[str, np.ndarray],
    feature: str,
    image_block: int,
) -> tuple[np.ndarray, np.ndarray]:
    pos_p = pos["proprios"].astype(np.float32)
    neg_p = neg["proprios"].astype(np.float32)
    pooled_p = np.concatenate([pos_p, neg_p], axis=0)
    p_std = pooled_p.std(axis=0)

    if feature == "proprio_raw":
        return pos_p, neg_p
    if feature == "proprio":
        return _normalize_per_axis(pos_p, p_std), _normalize_per_axis(neg_p, p_std)
    if feature == "proprio+wrist":
        pos_w = _downsample_image_block(pos["wrist_images"], image_block).reshape(pos_p.shape[0], -1)
        neg_w = _downsample_image_block(neg["wrist_images"], image_block).reshape(neg_p.shape[0], -1)
        pooled_w = np.concatenate([pos_w, neg_w], axis=0)
        w_std_scalar = float(pooled_w.std()) or 1.0
        pos_w = pos_w / w_std_scalar
        neg_w = neg_w / w_std_scalar
        pos_p_n = _normalize_per_axis(pos_p, p_std)
        neg_p_n = _normalize_per_axis(neg_p, p_std)
        scale = float(np.sqrt(pos_p_n.shape[1] / max(pos_w.shape[1], 1)))
        return np.concatenate([pos_p_n, pos_w * scale], axis=1), np.concatenate([neg_p_n, neg_w * scale], axis=1)
    raise ValueError(f"unknown feature: {feature}")


def pairwise_l2(a: np.ndarray, b: np.ndarray, chunk: int = 1024) -> np.ndarray:
    n_a = a.shape[0]
    out = np.empty((n_a, b.shape[0]), dtype=np.float32)
    b_sq = (b ** 2).sum(axis=1)
    for s in range(0, n_a, chunk):
        e = min(s + chunk, n_a)
        a_sub = a[s:e]
        d2 = (a_sub ** 2).sum(axis=1, keepdims=True) + b_sq[None, :] - 2.0 * (a_sub @ b.T)
        np.clip(d2, 0, None, out=d2)
        out[s:e] = np.sqrt(d2, dtype=np.float32)
    return out


def match_nn_replace(dists: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn_idx = dists.argmin(axis=1)
    return np.arange(dists.shape[0]), nn_idx, dists[np.arange(dists.shape[0]), nn_idx]


def match_nn_greedy(dists: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pos, n_neg = dists.shape
    n_pairs = min(n_pos, n_neg)
    flat_order = np.argsort(dists, axis=None)
    used_pos = np.zeros(n_pos, dtype=bool)
    used_neg = np.zeros(n_neg, dtype=bool)
    pos_pairs = np.empty(n_pairs, dtype=np.int64)
    neg_pairs = np.empty(n_pairs, dtype=np.int64)
    dist_pairs = np.empty(n_pairs, dtype=np.float32)
    k = 0
    for flat_idx in flat_order:
        i = flat_idx // n_neg
        j = flat_idx % n_neg
        if used_pos[i] or used_neg[j]:
            continue
        used_pos[i] = True
        used_neg[j] = True
        pos_pairs[k] = i
        neg_pairs[k] = j
        dist_pairs[k] = dists[i, j]
        k += 1
        if k == n_pairs:
            break
    return pos_pairs[:k], neg_pairs[:k], dist_pairs[:k]


def match_optimal(dists: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.optimize import linear_sum_assignment

    pos_idx, neg_idx = linear_sum_assignment(dists)
    return pos_idx, neg_idx, dists[pos_idx, neg_idx]


def _per_row_arrays(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    n = d["primary_images"].shape[0]
    return {k: v for k, v in d.items() if isinstance(v, np.ndarray) and v.shape[:1] == (n,)}


def _slice_per_row(d: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[idx] for k, v in _per_row_arrays(d).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature", choices=["proprio", "proprio_raw", "proprio+wrist"], default="proprio")
    parser.add_argument("--image-block", type=int, default=16)
    parser.add_argument("--match-mode", choices=["nn-replace", "nn-greedy", "optimal"], default="nn-greedy")
    parser.add_argument("--max-rows", type=int, default=-1)
    parser.add_argument("--max-distance", type=float, default=-1.0)
    args = parser.parse_args()

    in_dir = args.in_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = dict(np.load(in_dir / "positive.npz", allow_pickle=True))
    neg = dict(np.load(in_dir / "negative.npz", allow_pickle=True))
    n_pos_src = int(pos["primary_images"].shape[0])
    n_neg_src = int(neg["primary_images"].shape[0])
    if n_pos_src == 0 or n_neg_src == 0:
        raise RuntimeError(f"need both positive and negative rows; got pos={n_pos_src}, neg={n_neg_src}")

    _log(f"loaded {in_dir}: positive={n_pos_src}, negative={n_neg_src}")
    pos_feats, neg_feats = build_features(pos, neg, args.feature, args.image_block)
    dists = pairwise_l2(pos_feats, neg_feats)
    if args.match_mode == "nn-replace":
        pos_idx, neg_idx, dist_arr = match_nn_replace(dists)
    elif args.match_mode == "nn-greedy":
        pos_idx, neg_idx, dist_arr = match_nn_greedy(dists)
    else:
        pos_idx, neg_idx, dist_arr = match_optimal(dists)

    order = np.argsort(dist_arr)
    pos_idx = pos_idx[order]
    neg_idx = neg_idx[order]
    dist_arr = dist_arr[order]
    if args.max_distance > 0:
        keep = dist_arr <= args.max_distance
        pos_idx = pos_idx[keep]
        neg_idx = neg_idx[keep]
        dist_arr = dist_arr[keep]
    if args.max_rows > 0 and args.max_rows < len(pos_idx):
        pos_idx = pos_idx[: args.max_rows]
        neg_idx = neg_idx[: args.max_rows]
        dist_arr = dist_arr[: args.max_rows]
    if len(pos_idx) == 0:
        raise RuntimeError("no pairs survived filtering")

    pos_sub = _slice_per_row(pos, pos_idx)
    neg_sub = _slice_per_row(neg, neg_idx)
    n_pairs = len(pos_idx)
    common_episode = pos_sub["episode_idx"].astype(np.int32)
    common_inference = np.arange(n_pairs, dtype=np.int32)
    common_drive = np.zeros(n_pairs, dtype=np.int32)

    optional_keys = ("xyz_delta_m", "achieved_xyz_delta_m", "success", "task_id")

    def write_npz(npz_path: Path, sub: Dict[str, np.ndarray]) -> None:
        out = {
            "primary_images": sub["primary_images"],
            "wrist_images": sub["wrist_images"],
            "proprios": sub["proprios"],
            "episode_idx": common_episode,
            "inference_idx": common_inference,
            "drive_source": common_drive,
            "pos_src_episode_idx": pos_sub["episode_idx"].astype(np.int32),
            "pos_src_inference_idx": pos_sub["inference_idx"].astype(np.int32),
            "neg_src_episode_idx": neg_sub["episode_idx"].astype(np.int32),
            "neg_src_inference_idx": neg_sub["inference_idx"].astype(np.int32),
            "match_distance": dist_arr.astype(np.float32),
        }
        for key in optional_keys:
            if key in pos_sub:
                out[f"pos_src_{key}"] = pos_sub[key]
            if key in neg_sub:
                out[f"neg_src_{key}"] = neg_sub[key]
        np.savez_compressed(npz_path, **out)
        _log(f"wrote {npz_path} rows={n_pairs}")

    write_npz(out_dir / "positive.npz", pos_sub)
    write_npz(out_dir / "negative.npz", neg_sub)

    in_manifest_path = in_dir / "manifest.json"
    in_manifest = json.loads(in_manifest_path.read_text(encoding="utf-8")) if in_manifest_path.exists() else None
    task_prompt = None
    if isinstance(in_manifest, dict):
        task_prompt = in_manifest.get("task_language") or in_manifest.get("prompt")
    manifest = {
        "in_dir": str(in_dir),
        "feature": args.feature,
        "image_block": int(args.image_block),
        "match_mode": args.match_mode,
        "max_rows": int(args.max_rows),
        "max_distance": float(args.max_distance),
        "n_pos_source": n_pos_src,
        "n_neg_source": n_neg_src,
        "n_pairs": n_pairs,
        "match_distance_stats": {
            "min": float(dist_arr.min()),
            "median": float(np.median(dist_arr)),
            "mean": float(dist_arr.mean()),
            "max": float(dist_arr.max()),
        },
        "pairing": (
            "Rows are matched by feature-space L2 similarity. positive rows "
            "come from successful gripper-perturbed source rollouts; negative "
            "rows come from failed gripper-perturbed source rollouts. Source "
            "row identifiers are preserved in pos_src_* and neg_src_* fields."
        ),
        "input_manifest": in_manifest,
    }
    if task_prompt:
        manifest["task_language"] = str(task_prompt)
        manifest["prompt"] = str(task_prompt)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(
        "final pairs=%d distance min=%.4f median=%.4f mean=%.4f max=%.4f"
        % (n_pairs, dist_arr.min(), float(np.median(dist_arr)), dist_arr.mean(), dist_arr.max())
    )


if __name__ == "__main__":
    main()
