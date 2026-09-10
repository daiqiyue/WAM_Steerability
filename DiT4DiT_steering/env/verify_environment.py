#!/usr/bin/env python
"""Fail-fast verification for the site-specific DiT4DiT environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--load-model", action="store_true")
    args = p.parse_args()

    import torch
    import transformers
    import diffusers
    import mujoco
    import robosuite
    from libero.libero import benchmark, get_libero_path

    import DiT4DiT.model.framework.DiT4DiT  # noqa: F401
    from DiT4DiT.model.framework.base_framework import baseframework

    checkpoint = Path(os.environ["CKPT_PATH"])
    assert checkpoint.is_file(), checkpoint
    assert Path(get_libero_path("bddl_files")).is_dir()
    assert Path(get_libero_path("init_states")).is_dir()
    assert "libero_10" in benchmark.get_benchmark_dict()
    print({
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "mujoco": mujoco.__version__,
        "robosuite": robosuite.__version__,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }, flush=True)

    if args.load_model:
        model = baseframework.from_pretrained(str(checkpoint)).cuda().eval()
        print({
            "action_blocks": len(model.action_model.model.transformer_blocks),
            "action_horizon": model.action_model.action_horizon,
            "action_dim": model.action_model.action_dim,
            "action_inner_dim": model.action_model.model.inner_dim,
        }, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
