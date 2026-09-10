#!/usr/bin/env python
"""Inspect or extract one exact policy-inference input from a rollout bundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch


def digest(value) -> str | None:
    if not isinstance(value, np.ndarray):
        return None
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--inference", type=int, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="optional standalone .pt containing this exact inference")
    args = p.parse_args()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    rows = [r for r in bundle["inferences"] if int(r["inference_idx"]) == args.inference]
    if len(rows) != 1:
        available = [int(r["inference_idx"]) for r in bundle["inferences"]]
        raise ValueError(f"inference {args.inference} not found; available={available}")
    row = rows[0]
    exact = {
        "rollout_metadata": {k: v for k, v in bundle.items()
                             if k not in ("inferences", "executed_actions")},
        "inference": row,
        "actions_before_inference": bundle["executed_actions"][:row["executed_action_count"]],
    }
    print(f"bundle={args.bundle}")
    print(f"episode={bundle['episode']} condition={bundle['condition']} "
          f"inference={row['inference_idx']} env_step={row['env_step']}")
    for key, value in row["observation"].items():
        if isinstance(value, np.ndarray):
            print(f"observation.{key}: shape={value.shape} dtype={value.dtype} sha256={digest(value)}")
    for key, value in row["model_input"].items():
        if isinstance(value, np.ndarray):
            print(f"model_input.{key}: shape={value.shape} dtype={value.dtype} sha256={digest(value)}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(exact, args.out)
        print(f"saved={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
