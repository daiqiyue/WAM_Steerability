import hashlib
import json
import os
import time
from typing import Any, Dict, Iterable, List


def maybe_load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"YAML requested but pyyaml is unavailable: {path}") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must decode to dict, got {type(data)}")
    return data


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_int_list(value: str) -> List[int]:
    if not value.strip():
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_partitions(spec: str, num_layers: int) -> List[tuple[int, int]]:
    """Parse repo-root-style partition spec, e.g. ``0-9,10-18,19-27``."""
    parts: List[tuple[int, int]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split("-")
        parts.append((int(a), int(b)))
    covered = [layer for l_start, l_end in parts for layer in range(l_start, l_end + 1)]
    if sorted(covered) != list(range(num_layers)):
        raise ValueError(
            f"partitions {parts} must tile [0, {num_layers - 1}] exactly; got layers {sorted(set(covered))}"
        )
    return parts


def default_partitions_three(num_layers: int) -> str:
    """Default 3-partition layout matching the repo root's original convention (contiguous layer groups)."""
    if num_layers == 28:
        return "0-9,10-18,19-27"
    chunk = num_layers // 3
    rem = num_layers % 3
    sizes = [chunk + (1 if i < rem else 0) for i in range(3)]
    tokens: List[str] = []
    start = 0
    for size in sizes:
        end = start + size - 1
        tokens.append(f"{start}-{end}")
        start = end + 1
    return ",".join(tokens)


def layer_to_part_from_partitions(partitions: List[tuple[int, int]], num_layers: int) -> List[int]:
    layer_to_part = [0] * num_layers
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for layer in range(l_start, l_end + 1):
            layer_to_part[layer] = p_idx
    return layer_to_part


def stable_run_id(parts: List[str]) -> str:
    digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"run_{digest}"


def now_ts() -> int:
    return int(time.time())


def default_slurm_port(base: int = 29056) -> int:
    """PORT_BASE + (SLURM_JOB_ID % 1000); honors PORT env if already set."""
    port_env = os.environ.get("PORT", "").strip()
    if port_env:
        return int(port_env)
    port_base = int(os.environ.get("PORT_BASE", str(base)))
    job_id = int(os.environ.get("SLURM_JOB_ID", "0") or "0")
    return port_base + (job_id % 1000)
