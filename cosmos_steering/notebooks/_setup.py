"""Shared runtime setup for Cosmos notebook scripts.

This file keeps the input-collection scripts usable outside notebooks by
setting the same environment defaults used by the LQR/SVD/Jacobian entrypoints.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _set_hf_token_from_disk() -> None:
    if "HF_TOKEN" in os.environ:
        return

    for candidate in (
        Path.home() / ".huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "token",
    ):
        if candidate.exists():
            token = candidate.read_text().strip()
            if token:
                os.environ["HF_TOKEN"] = token
            return


def setup_env() -> Path:
    """Set conservative defaults before importing Cosmos/LIBERO modules."""

    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    hf_hub_cache = os.environ.get("HF_HUB_CACHE", str(Path(hf_home) / "hub"))
    Path(hf_hub_cache).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_hub_cache)
    _set_hf_token_from_disk()

    libero_cfg = os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))
    Path(libero_cfg).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", libero_cfg)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # LIBERO init-state files use an older pickle format. PyTorch 2.6+ defaults
    # torch.load(..., weights_only=True), which rejects those trusted files.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    notebooks_root = Path(__file__).resolve().parent
    repo_root = notebooks_root.parent
    workspace_root = repo_root.parent
    libero_home = Path(os.environ.get("LIBERO_HOME", workspace_root / "LIBERO"))
    import_roots = [repo_root, notebooks_root]
    if libero_home.exists():
        os.environ.setdefault("LIBERO_HOME", str(libero_home))
        import_roots.append(libero_home)

    for path in import_roots:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    os.chdir(repo_root)
    return repo_root
