#!/usr/bin/env python
"""Pair unpaired positive/negative policy inputs by similarity so they can
be consumed by run_partition_svd_pairs*.py / compute_jacobians_*.py.

Input
-----
A directory containing positive.npz and negative.npz produced by
notebooks/lqr/inputs/collect_policy_inputs_gripper_xyz_perturbation.py
(or any other collector that writes the same schema but with different
positive/negative row counts):

    primary_images : (N, H, W, 3) uint8
    wrist_images   : (N, H, W, 3) uint8
    proprios       : (N, 9) float32
    episode_idx    : (N,) int32
    inference_idx  : (N,) int32
    drive_source   : (N,) int32
    success        : (N,) int32  (optional)
    ...                          (other per-row arrays are passed through)

Output
------
A new directory with positive.npz and negative.npz that have IDENTICAL
row counts and IDENTICAL `episode_idx` / `inference_idx` / `drive_source`
arrays (so they satisfy run_partition_svd_pairs*.py's pairing assertions).
The original source row tags are preserved as extra keys
(`pos_src_episode_idx`, `neg_src_episode_idx`, etc.).

Matching
--------
For each positive row we find a negative row "in a similar state" and
emit them as a pair. By default the feature space is proprio L2 distance
with per-axis std normalization (so the gripper-qpos[2] / eef_pos[3] /
eef_quat[4] components each contribute roughly equally). Alternative
feature spaces:
  --feature proprio              std-normalized 9-D proprio (default)
  --feature proprio_raw          raw 9-D proprio (no normalization)
  --feature proprio+wrist        9-D proprio concatenated with a
                                  downsampled wrist image flattened &
                                  rescaled to ~unit-std (catches scene
                                  state the proprio misses)

Match modes:
  --match-mode nn-replace        nearest neighbor with replacement (any
                                  negative may be used multiple times;
                                  produces N_pos pairs)
  --match-mode nn-greedy         nearest neighbor WITHOUT replacement
                                  via greedy lowest-distance-first
                                  (default; produces min(N_pos, N_neg))
  --match-mode optimal           optimal bipartite matching via
                                  scipy.optimize.linear_sum_assignment
                                  (globally minimum total distance;
                                  produces min(N_pos, N_neg))

After matching, the pairs are sorted by ascending match distance, so the
top --max-rows lowest-distance pairs are kept when truncating.

The output is plug-in compatible with
  notebooks/lqr/svd/run_partition_svd_pairs_no_action.sh
and downstream
  notebooks/lqr/jacobians/run_jacobians_full.sh
— point those at the OUT_DIR via POS_NPZ / NEG_NPZ.
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


# ====================================================================
# Feature extraction
# ====================================================================

def _normalize_per_axis(x: np.ndarray, std_ref: np.ndarray) -> np.ndarray:
    return x / std_ref.clip(min=1e-6)


def _downsample_image_block(imgs: np.ndarray, block: int) -> np.ndarray:
    """Box-average HxWxC uint8 images to (H/block)x(W/block)xC float32."""
    if block <= 1:
        return imgs.astype(np.float32)
    n, h, w, c = imgs.shape
    h2 = (h // block) * block
    w2 = (w // block) * block
    arr = imgs[:, :h2, :w2, :].astype(np.float32)
    arr = arr.reshape(n, h2 // block, block, w2 // block, block, c).mean(axis=(2, 4))
    return arr  # (N, h2/block, w2/block, C)


def build_features(pos: Dict[str, np.ndarray], neg: Dict[str, np.ndarray],
                    feature: str, image_block: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos_feats, neg_feats) of shape (N_pos, D), (N_neg, D).
    Features are scaled together (using pooled std) so matching distances
    are commensurable across positive and negative.
    """
    pos_p = pos["proprios"].astype(np.float32)
    neg_p = neg["proprios"].astype(np.float32)
    pooled_p = np.concatenate([pos_p, neg_p], axis=0)
    p_std = pooled_p.std(axis=0)

    if feature == "proprio_raw":
        return pos_p, neg_p

    if feature == "proprio":
        return (
            _normalize_per_axis(pos_p, p_std),
            _normalize_per_axis(neg_p, p_std),
        )

    if feature == "proprio+wrist":
        pos_w = _downsample_image_block(pos["wrist_images"], image_block)
        neg_w = _downsample_image_block(neg["wrist_images"], image_block)
        pos_w = pos_w.reshape(pos_w.shape[0], -1)
        neg_w = neg_w.reshape(neg_w.shape[0], -1)
        pooled_w = np.concatenate([pos_w, neg_w], axis=0)
        w_std_scalar = float(pooled_w.std()) or 1.0
        pos_w = pos_w / w_std_scalar
        neg_w = neg_w / w_std_scalar
        pos_p_n = _normalize_per_axis(pos_p, p_std)
        neg_p_n = _normalize_per_axis(neg_p, p_std)
        # Rescale image block so its total contribution roughly matches the
        # proprio's 9 dims (otherwise the high-dim image term dominates).
        scale = float(np.sqrt(pos_p_n.shape[1] / max(pos_w.shape[1], 1)))
        pos_w = pos_w * scale
        neg_w = neg_w * scale
        return (
            np.concatenate([pos_p_n, pos_w], axis=1),
            np.concatenate([neg_p_n, neg_w], axis=1),
        )

    raise ValueError(f"unknown --feature={feature!r}")


# ====================================================================
# Matching
# ====================================================================

def pairwise_l2(a: np.ndarray, b: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Pairwise L2 distance matrix (N_a, N_b), chunked along axis 0 to keep
    memory bounded when N_a * N_b * D is large."""
    n_a, d = a.shape
    n_b = b.shape[0]
    out = np.empty((n_a, n_b), dtype=np.float32)
    b_sq = (b ** 2).sum(axis=1)  # (n_b,)
    for s in range(0, n_a, chunk):
        e = min(s + chunk, n_a)
        a_sub = a[s:e]
        a_sq = (a_sub ** 2).sum(axis=1, keepdims=True)  # (chunk, 1)
        cross = a_sub @ b.T  # (chunk, n_b)
        d2 = a_sq + b_sq[None, :] - 2.0 * cross
        np.clip(d2, 0, None, out=d2)
        out[s:e] = np.sqrt(d2, dtype=np.float32)
    return out


def match_nn_replace(dists: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn_idx = dists.argmin(axis=1)
    nn_dist = dists[np.arange(dists.shape[0]), nn_idx]
    return np.arange(dists.shape[0]), nn_idx, nn_dist


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
        j = flat_idx %  n_neg
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


# ====================================================================
# I/O
# ====================================================================

def _per_row_arrays(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    n = d["primary_images"].shape[0]
    return {k: v for k, v in d.items()
            if isinstance(v, np.ndarray) and v.shape[:1] == (n,)}


def _slice_per_row(d: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[idx] for k, v in _per_row_arrays(d).items()}


# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="dir containing unpaired positive.npz and negative.npz")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="output dir for paired positive.npz and negative.npz")
    ap.add_argument("--feature", type=str, default="proprio",
                    choices=["proprio", "proprio_raw", "proprio+wrist"])
    ap.add_argument("--image-block", type=int, default=16,
                    help="box-average block size for proprio+wrist feature "
                         "(256 / 16 = 16x16 downsampled)")
    ap.add_argument("--match-mode", type=str, default="nn-greedy",
                    choices=["nn-replace", "nn-greedy", "optimal"])
    ap.add_argument("--max-rows", type=int, default=-1,
                    help="cap on number of paired rows (kept lowest-distance "
                         "first); -1 = keep all matched pairs")
    ap.add_argument("--max-distance", type=float, default=-1.0,
                    help="drop pairs whose match distance exceeds this "
                         "threshold (in feature units); -1 = no threshold")
    ap.add_argument("--within-task", action="store_true",
                    help="constrain matching so each pair shares the same "
                         "task_id (requires task_id column in both npz; "
                         "intended for multi-task collector output)")
    args = ap.parse_args()

    in_dir = args.in_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = dict(np.load(in_dir / "positive.npz", allow_pickle=True))
    neg = dict(np.load(in_dir / "negative.npz", allow_pickle=True))
    n_pos_src = int(pos["primary_images"].shape[0])
    n_neg_src = int(neg["primary_images"].shape[0])
    _log(f"loaded {in_dir}")
    _log(f"  positive: {n_pos_src} rows  (success bucket)")
    _log(f"  negative: {n_neg_src} rows  (failure bucket)")

    if n_pos_src == 0 or n_neg_src == 0:
        raise RuntimeError(
            f"need both pos ({n_pos_src}) and neg ({n_neg_src}) rows to pair"
        )

    # ---------------- features + pairwise distance ----------------
    _log(f"building features: {args.feature}")
    pos_feats, neg_feats = build_features(pos, neg, args.feature, args.image_block)
    _log(f"  pos_feats: {pos_feats.shape}  neg_feats: {neg_feats.shape}")
    _log(f"computing pairwise L2 distance "
         f"({n_pos_src} x {n_neg_src} = {n_pos_src * n_neg_src:,} entries)")
    dists = pairwise_l2(pos_feats, neg_feats)

    # ---------------- optional within-task constraint ----------------
    if args.within_task:
        if "task_id" not in pos or "task_id" not in neg:
            raise ValueError(
                "--within-task requires `task_id` column in BOTH positive.npz "
                "and negative.npz (produced by the multi-task collector). "
                "Got pos keys=%s neg keys=%s"
                % (sorted(pos.keys()), sorted(neg.keys()))
            )
        pos_task = pos["task_id"].astype(np.int64)
        neg_task = neg["task_id"].astype(np.int64)
        cross = pos_task[:, None] != neg_task[None, :]
        n_cross = int(cross.sum())
        _log(f"  --within-task: blocking {n_cross:,} cross-task entries "
             f"({n_cross / dists.size:.1%}) by setting distance to +inf")
        dists[cross] = np.inf
        # Sanity: ensure every positive row has at least one same-task neg
        no_partner = np.isinf(dists).all(axis=1)
        if no_partner.any():
            n_orphan = int(no_partner.sum())
            _log(f"  WARNING: {n_orphan} positive rows have no same-task "
                 f"negative partner and will be unmatchable")

    # ---------------- match ----------------
    _log(f"matching: mode={args.match_mode}")
    if args.match_mode == "nn-replace":
        pos_idx, neg_idx, dist_arr = match_nn_replace(dists)
    elif args.match_mode == "nn-greedy":
        pos_idx, neg_idx, dist_arr = match_nn_greedy(dists)
    else:
        pos_idx, neg_idx, dist_arr = match_optimal(dists)
    _log(f"  initial pairs: {len(pos_idx)}")

    # ---------------- filter / sort / cap ----------------
    order = np.argsort(dist_arr)
    pos_idx = pos_idx[order]
    neg_idx = neg_idx[order]
    dist_arr = dist_arr[order]

    if args.max_distance > 0:
        keep = dist_arr <= args.max_distance
        n_drop = int((~keep).sum())
        if n_drop > 0:
            _log(f"  dropping {n_drop} pairs with distance > {args.max_distance}")
        pos_idx = pos_idx[keep]
        neg_idx = neg_idx[keep]
        dist_arr = dist_arr[keep]

    if args.max_rows > 0 and args.max_rows < len(pos_idx):
        _log(f"  capping to top --max-rows={args.max_rows} lowest-distance pairs")
        pos_idx = pos_idx[:args.max_rows]
        neg_idx = neg_idx[:args.max_rows]
        dist_arr = dist_arr[:args.max_rows]

    n_pairs = len(pos_idx)
    if n_pairs == 0:
        raise RuntimeError("no pairs survived filtering")
    _log(f"  final pairs: {n_pairs}  distance "
         f"min={dist_arr.min():.4f} median={float(np.median(dist_arr)):.4f} "
         f"mean={dist_arr.mean():.4f} max={dist_arr.max():.4f}")

    # ---------------- pull rows + override tag arrays ----------------
    pos_sub = _slice_per_row(pos, pos_idx)
    neg_sub = _slice_per_row(neg, neg_idx)

    common_episode = pos_sub["episode_idx"].astype(np.int32)
    common_inference = np.arange(n_pairs, dtype=np.int32)
    common_drive = np.zeros(n_pairs, dtype=np.int32)

    # Optional pass-through keys that we expose with src_/dst_ prefixes so
    # downstream consumers can inspect what source row each paired row
    # actually came from. task_id is needed by the multitask SVD wrapper
    # to look up the per-row prompt.
    OPTIONAL_KEYS = ("xyz_delta_m", "achieved_xyz_delta_m", "success", "task_id")

    def _write(npz_path: Path, sub: Dict[str, np.ndarray]):
        out = {
            "primary_images": sub["primary_images"],
            "wrist_images":   sub["wrist_images"],
            "proprios":       sub["proprios"],
            "episode_idx":    common_episode,
            "inference_idx":  common_inference,
            "drive_source":   common_drive,
            "pos_src_episode_idx":   pos_sub["episode_idx"].astype(np.int32),
            "pos_src_inference_idx": pos_sub["inference_idx"].astype(np.int32),
            "neg_src_episode_idx":   neg_sub["episode_idx"].astype(np.int32),
            "neg_src_inference_idx": neg_sub["inference_idx"].astype(np.int32),
            "match_distance":        dist_arr.astype(np.float32),
        }
        for k in OPTIONAL_KEYS:
            if k in pos_sub:
                out[f"pos_src_{k}"] = pos_sub[k]
            if k in neg_sub:
                out[f"neg_src_{k}"] = neg_sub[k]
        np.savez_compressed(npz_path, **out)
        _log(f"wrote {npz_path.name} ({npz_path.stat().st_size/1e6:.1f} MB) "
             f"rows={n_pairs}")

    POSITIVE_NPZ = out_dir / "positive.npz"
    NEGATIVE_NPZ = out_dir / "negative.npz"
    MANIFEST_JSON = out_dir / "manifest.json"
    _write(POSITIVE_NPZ, pos_sub)
    _write(NEGATIVE_NPZ, neg_sub)

    # ---------------- manifest ----------------
    in_manifest_path = in_dir / "manifest.json"
    in_manifest = (
        json.loads(in_manifest_path.read_text())
        if in_manifest_path.exists() else None
    )
    manifest = {
        "in_dir": str(in_dir),
        "feature": args.feature,
        "image_block": args.image_block,
        "match_mode": args.match_mode,
        "max_rows": args.max_rows,
        "max_distance": args.max_distance,
        "within_task": bool(args.within_task),
        "n_pos_source": n_pos_src,
        "n_neg_source": n_neg_src,
        "n_pairs": n_pairs,
        "match_distance_stats": {
            "min":    float(dist_arr.min()),
            "median": float(np.median(dist_arr)),
            "mean":   float(dist_arr.mean()),
            "max":    float(dist_arr.max()),
            "p25":    float(np.percentile(dist_arr, 25)),
            "p75":    float(np.percentile(dist_arr, 75)),
        },
        "pairing": (
            "row i in positive.npz and row i in negative.npz are matched "
            "by feature-space L2 similarity. positive row is from a "
            "SUCCESSFUL source rollout; negative row is from a FAILED "
            "source rollout. Tag arrays (episode_idx / inference_idx / "
            "drive_source) are forced equal between pos and neg for "
            "run_partition_svd_pairs*.py compatibility; source row "
            "identifiers are preserved in pos_src_*/neg_src_*."
        ),
        "image_layout": "HWC uint8, inherited from input npz",
        "proprio_layout": "shape (9,) float32, inherited from input npz",
        "drive_sources": [
            {"code": 0, "name": "paired_by_similarity",
             "desc": "single virtual drive source covering all matched pairs"},
        ],
        "sets": {
            "positive": {"out_npz": str(POSITIVE_NPZ),
                          "role": "rows from successful rollouts (matched)"},
            "negative": {"out_npz": str(NEGATIVE_NPZ),
                          "role": "rows from failed rollouts (matched)"},
        },
        "input_manifest": in_manifest,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    _log(f"wrote {MANIFEST_JSON.name}")

    _log("=== summary ===")
    _log(f"  positive : {POSITIVE_NPZ}")
    _log(f"  negative : {NEGATIVE_NPZ}")
    _log(f"  manifest : {MANIFEST_JSON}")
    _log(f"Next: run notebooks/lqr/svd/run_partition_svd_pairs_no_action.sh with")
    _log(f"  POS_NPZ='{POSITIVE_NPZ}'")
    _log(f"  NEG_NPZ='{NEGATIVE_NPZ}'")


if __name__ == "__main__":
    main()
