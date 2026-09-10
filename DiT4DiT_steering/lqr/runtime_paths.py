"""Portable path setup shared by the DiT4DiT steering scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_runtime(local_root: Path) -> tuple[Path, str]:
    """Add the model checkout and optional LIBERO checkout to ``sys.path``.

    ``local_root`` is the in-repository ``DiT4DiT_steering`` directory.  Model
    source may live elsewhere and is selected with ``DIT4DIT_CODE_ROOT``.
    """

    code_root = Path(
        os.environ.get("DIT4DIT_CODE_ROOT", os.environ.get("DIT4DIT_ROOT", str(local_root)))
    ).expanduser().resolve()
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    libero_home = os.environ.get("LIBERO_HOME", "")
    if libero_home and libero_home not in sys.path:
        sys.path.insert(0, libero_home)

    extra_site = os.environ.get("DIT4DIT_EXTRA_SITE_PACKAGES", "")
    if extra_site and extra_site not in sys.path:
        sys.path.append(extra_site)

    os.environ.setdefault("DIT4DIT_CODE_ROOT", str(code_root))
    if libero_home:
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path(libero_home) / "libero"))
    return code_root, libero_home


def load_libero_init_states(task_suite, task_id: int):
    """Load trusted local LIBERO init states across PyTorch 2.6+.

    LIBERO's current benchmark helper relies on the pre-2.6 ``torch.load``
    default, but its files contain NumPy arrays and therefore are not accepted
    by the newer weights-only unpickler.  Resolve exactly the benchmark-owned
    local file and opt out explicitly without globally monkeypatching PyTorch.
    """

    import torch
    from libero.libero import get_libero_path

    task = task_suite.get_task(task_id)
    path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO init states not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)
