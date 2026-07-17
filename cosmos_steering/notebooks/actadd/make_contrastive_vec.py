#!/usr/bin/env python
"""Build a per-DiT-block contrastive steering vector from collected
activations (positive - negative, averaged across rows).

Input  : .npz with `positive` and `negative` arrays
            - shape (N, n_blocks, D)        — per-layer mode
            - shape (N, n_blocks, T, D)     — per-timestep mode
         as written by collect_acts_passive.py. Mode is auto-detected from
         the array rank.
Output : .pt dict
            - per-layer:    {layer_idx: torch.Tensor(D,)}
            - per-timestep: {layer_idx: torch.Tensor(T, D)}
         Both formats are consumed by run_actadd_cosmos_policy.py's --v-path
         (the hook reshapes 2-D vectors to broadcast over [B, T, H, W, D]).

Usage:
    python make_contrastive_vec.py \
        --activations-path notebooks/lqr/artifacts/actadd/activations__libero_10__task06.npz \
        --out-path notebooks/lqr/artifacts/actadd/v__libero_10__task06.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--activations-path", type=Path, required=True,
                    help=".npz from collect_acts_passive.py (positive/negative "
                         "stacked arrays of shape (N, n_blocks, D))")
    ap.add_argument("--out-path", type=Path, required=True,
                    help=".pt output path; dict[int -> Tensor(D,)]")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.activations_path.exists():
        raise FileNotFoundError(args.activations_path)

    z = np.load(args.activations_path, allow_pickle=True)
    pos = z["positive"]  # (N_pos, n_blocks, D) or (N_pos, n_blocks, T, D)
    neg = z["negative"]  # (N_neg, n_blocks, D) or (N_neg, n_blocks, T, D)
    if pos.ndim not in (3, 4) or neg.ndim not in (3, 4) \
            or pos.shape[1:] != neg.shape[1:]:
        raise ValueError(
            f"unexpected activation shapes: positive={pos.shape}, "
            f"negative={neg.shape}; want (N, n_blocks, D) or "
            f"(N, n_blocks, T, D)"
        )

    per_timestep = (pos.ndim == 4)
    diff = pos.mean(axis=0) - neg.mean(axis=0)  # (n_blocks, [T,] D)
    n_blocks = diff.shape[0]
    contrastive_vecs = {
        int(layer): torch.from_numpy(diff[layer].astype(np.float32))
        for layer in range(n_blocks)
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(contrastive_vecs, args.out_path)
    shape_str = (f"(T={diff.shape[1]}, D={diff.shape[2]})" if per_timestep
                 else f"(D={diff.shape[1]})")
    print(
        f"[done] wrote {args.out_path}  "
        f"(n_blocks={n_blocks}, {shape_str}, per_timestep={per_timestep}, "
        f"N_pos={pos.shape[0]}, N_neg={neg.shape[0]})"
    )


if __name__ == "__main__":
    main()
