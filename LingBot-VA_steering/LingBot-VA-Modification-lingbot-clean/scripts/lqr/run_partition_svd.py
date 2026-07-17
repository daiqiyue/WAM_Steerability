import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from scripts.lqr.common import (
    default_partitions_three,
    layer_to_part_from_partitions,
    parse_partitions,
)


@dataclass
class _CallCtx:
    action_mode: bool
    update_cache: int
    step_idx: int


class LingbotActivationTracer:
    def __init__(self, layers: List[int], selected_timesteps: List[int], mode: str) -> None:
        self.layers = set(layers)
        self.selected_timesteps = set(selected_timesteps)
        self.mode = mode
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.current: Optional[_CallCtx] = None
        self._hook_handles = []
        self.captured: Dict[Tuple[int, int], torch.Tensor] = {}

    def reset_chunk(self) -> None:
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.current = None
        self.captured = {}

    def begin_call(self, action_mode: bool, update_cache: int) -> None:
        if action_mode:
            step_idx = self.action_step_idx
            self.action_step_idx += 1
        else:
            step_idx = self.video_step_idx
            self.video_step_idx += 1
        self.current = _CallCtx(action_mode=bool(action_mode), update_cache=int(update_cache), step_idx=step_idx)

    def end_call(self) -> None:
        self.current = None

    def _mode_allow(self, action_mode: bool) -> bool:
        if self.mode == "both":
            return True
        if self.mode == "action":
            return action_mode
        return not action_mode

    def register_hooks(self, transformer: torch.nn.Module) -> None:
        for idx, block in enumerate(transformer.blocks):
            if idx not in self.layers:
                continue
            self._hook_handles.append(block.register_forward_hook(self._hook_fn(idx)))

    def _hook_fn(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self.current is None:
                return output
            if not self._mode_allow(self.current.action_mode):
                return output
            if self.current.step_idx not in self.selected_timesteps:
                return output
            self.captured[(layer_idx, self.current.step_idx)] = output[0].detach().reshape(-1).float().cpu()
            return output

        return hook

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def _default_master_port() -> str:
    return str(12355 + (os.getpid() % 1000))


def _ensure_dist_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _default_master_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def _build_server(config_name: str):
    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.wan_va_server import VA_Server

    _ensure_dist_env()
    if not dist.is_initialized():
        init_distributed(world_size=1, local_rank=0, rank=0)
    cfg = copy.deepcopy(VA_CONFIGS[config_name])
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    return VA_Server(cfg)


def _obs_from_npz(npz_obj, idx: int, cam_keys: List[str]) -> Dict:
    cam0 = npz_obj["primary_images"][idx]
    cam1 = npz_obj["wrist_images"][idx]
    return {"obs": [{cam_keys[0]: np.ascontiguousarray(cam0), cam_keys[1]: np.ascontiguousarray(cam1)}]}


def _cache_prompt_for_reuse(server, prompt: str):
    """Encode a fixed prompt once, then reuse it across SVD pair rollouts."""
    with torch.no_grad():
        server._reset(prompt=prompt)
    prompt_embeds = server.prompt_embeds.detach() if server.prompt_embeds is not None else None
    negative_prompt_embeds = (
        server.negative_prompt_embeds.detach()
        if server.negative_prompt_embeds is not None
        else None
    )
    return prompt_embeds, negative_prompt_embeds


def _reset_with_cached_prompt(server, prompt_cache) -> None:
    with torch.no_grad():
        server._reset(prompt=None)
    server.prompt_embeds, server.negative_prompt_embeds = prompt_cache


def _parse_int_csv(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _resolve_num_samples(num_samples: int, n_pos: int, n_neg: int) -> int:
    """<=0 means use all available paired rows (the repo root's N=-1 convention)."""
    n_avail = min(int(n_pos), int(n_neg))
    if int(num_samples) <= 0:
        return n_avail
    return min(int(num_samples), n_avail)


def _prompt_from_manifest(manifest: Dict) -> Optional[str]:
    for key in ("task_language", "prompt"):
        value = manifest.get(key)
        if value:
            return str(value).strip()
    nested = manifest.get("input_manifest")
    if isinstance(nested, dict):
        for key in ("task_language", "prompt"):
            value = nested.get(key)
            if value:
                return str(value).strip()
    return None


def _load_prompt_from_pairs_dir(pairs_dir: Path, cli_prompt: Optional[str]) -> str:
    if cli_prompt and str(cli_prompt).strip():
        return str(cli_prompt).strip()

    manifest_path = pairs_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompt = _prompt_from_manifest(manifest)
        if prompt:
            return prompt
        in_dir = manifest.get("in_dir")
        if in_dir:
            parent_prompt = _load_prompt_from_pairs_dir(Path(str(in_dir)), None)
            if parent_prompt:
                return parent_prompt

    prompt_path = pairs_dir / "prompt.txt"
    if prompt_path.is_file():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text

    raise ValueError(
        f"Prompt is required for SVD. Pass --prompt or provide manifest/prompt.txt under {pairs_dir}."
    )


def _collect_diffs_from_activation_pairs(
    pos_payload: Dict,
    neg_payload: Dict,
    num_samples: int,
) -> Tuple[Dict[Tuple[int, int], List[torch.Tensor]], str, List[int], List[int], int]:
    pos_records = list(pos_payload.get("records", []))
    neg_records = list(neg_payload.get("records", []))
    n_total = _resolve_num_samples(num_samples, len(pos_records), len(neg_records))
    if n_total <= 0:
        raise ValueError("No activation pairs available for SVD.")

    prompt = pos_payload.get("prompt") or neg_payload.get("prompt")
    selected_timesteps = pos_payload.get("selected_timesteps") or neg_payload.get("selected_timesteps")
    layers = pos_payload.get("layers") or neg_payload.get("layers")
    if not selected_timesteps or not layers:
        sample_keys = sorted(list(pos_records[0]["activations"].keys()))
        layers = sorted({int(k[0]) for k in sample_keys})
        selected_timesteps = sorted({int(k[1]) for k in sample_keys})

    diffs: Dict[Tuple[int, int], List[torch.Tensor]] = {(l, t): [] for l in layers for t in selected_timesteps}
    for i in range(n_total):
        cap_pos = pos_records[i]["activations"]
        cap_neg = neg_records[i]["activations"]
        for key in diffs:
            if key not in cap_pos or key not in cap_neg:
                raise RuntimeError(f"Missing activation key {key} in activation-native pair index {i}")
            diffs[key].append((cap_pos[key] - cap_neg[key]).float().cpu())
    return diffs, str(prompt), list(layers), list(selected_timesteps), n_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lingbot partition SVD for LQR.")
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["video", "action", "both"], default="action")
    parser.add_argument("--layers", type=str, default="", help="Comma separated layer ids; empty means all layers.")
    parser.add_argument("--selected-timesteps", type=str, default="0,10,20,30,40")
    parser.add_argument("--num-samples", type=int, default=16, help="Rows for SVD; <=0 uses all paired rows.")
    parser.add_argument("--k-target", type=int, default=32)
    parser.add_argument("--p-over", type=int, default=8)
    parser.add_argument(
        "--partitions",
        type=str,
        default="",
        help="Repo-root-style layer groups, e.g. 0-9,10-19,20-29. Empty = auto 3 partitions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    pos_act = args.pairs_dir / "positive.pt"
    neg_act = args.pairs_dir / "negative.pt"
    activation_native = pos_act.exists() and neg_act.exists()

    if activation_native:
        pos_payload = torch.load(pos_act, map_location="cpu", weights_only=False)
        neg_payload = torch.load(neg_act, map_location="cpu", weights_only=False)
        diffs, prompt_from_pairs, layers, selected_timesteps, n_total = _collect_diffs_from_activation_pairs(
            pos_payload=pos_payload,
            neg_payload=neg_payload,
            num_samples=int(args.num_samples),
        )
        prompt = args.prompt or prompt_from_pairs
        n_layers = max(layers) + 1
        sampling_steps = max(selected_timesteps) + 1
    else:
        pos_path = args.pairs_dir / "positive.npz"
        neg_path = args.pairs_dir / "negative.npz"
        if not pos_path.exists() or not neg_path.exists():
            raise FileNotFoundError(f"Expected positive/negative pair files in {args.pairs_dir}")
        pos = np.load(pos_path)
        neg = np.load(neg_path)
        prompt = _load_prompt_from_pairs_dir(args.pairs_dir, args.prompt)
        print(f"[svd] prompt={prompt!r}")

        server = _build_server(args.config_name)
        transformer = server.transformer
        n_layers = len(transformer.blocks)
        layers = _parse_int_csv(args.layers) if args.layers.strip() else list(range(n_layers))
        selected_timesteps = _parse_int_csv(args.selected_timesteps)
        tracer = LingbotActivationTracer(layers=layers, selected_timesteps=selected_timesteps, mode=args.mode)
        tracer.register_hooks(transformer)
        original_forward = transformer.forward

        def patched_forward(*f_args, **f_kwargs):
            action_mode = bool(f_kwargs.get("action_mode", False))
            update_cache = int(f_kwargs.get("update_cache", 0))
            tracer.begin_call(action_mode=action_mode, update_cache=update_cache)
            try:
                return original_forward(*f_args, **f_kwargs)
            finally:
                tracer.end_call()

        transformer.forward = patched_forward
        cam_keys = list(server.job_config.obs_cam_keys)
        n_pos = int(pos["primary_images"].shape[0])
        n_neg = int(neg["primary_images"].shape[0])
        n_total = _resolve_num_samples(args.num_samples, n_pos, n_neg)
        if n_total <= 0:
            raise ValueError(f"No paired NPZ rows available for SVD under {args.pairs_dir}")
        print(f"[svd] using {n_total}/{n_pos} paired rows (num_samples={args.num_samples})")
        diffs = {(l, t): [] for l in layers for t in selected_timesteps}
        prompt_cache = _cache_prompt_for_reuse(server, prompt)
        try:
            for i in range(n_total):
                obs_pos = _obs_from_npz(pos, i, cam_keys)
                obs_neg = _obs_from_npz(neg, i, cam_keys)
                _reset_with_cached_prompt(server, prompt_cache)
                tracer.reset_chunk()
                with torch.no_grad():
                    server._infer(obs_pos, frame_st_id=0)
                cap_pos = dict(tracer.captured)
                _reset_with_cached_prompt(server, prompt_cache)
                tracer.reset_chunk()
                with torch.no_grad():
                    server._infer(obs_neg, frame_st_id=0)
                cap_neg = dict(tracer.captured)
                for key in diffs:
                    if key not in cap_pos or key not in cap_neg:
                        raise RuntimeError(f"Missing activation key {key} at sample {i}.")
                    diffs[key].append((cap_pos[key] - cap_neg[key]).float().cpu())
                del cap_pos, cap_neg, obs_pos, obs_neg
                if torch.cuda.is_available() and (i + 1) % 25 == 0:
                    torch.cuda.empty_cache()
        finally:
            tracer.close()
            transformer.forward = original_forward

        if args.mode == "action":
            sampling_steps = int(server.job_config.action_num_inference_steps)
        elif args.mode == "video":
            sampling_steps = int(server.job_config.num_inference_steps)
        else:
            sampling_steps = int(max(server.job_config.num_inference_steps, server.job_config.action_num_inference_steps))

    t_to_idx = {t: i for i, t in enumerate(selected_timesteps)}
    partition_spec = args.partitions.strip() or default_partitions_three(n_layers)
    partitions = parse_partitions(partition_spec, n_layers)
    layer_to_part = layer_to_part_from_partitions(partitions, n_layers)
    c_means = torch.zeros(n_layers, len(selected_timesteps), args.k_target, dtype=torch.float32)
    projected_diffs: Dict[Tuple[int, int, int], torch.Tensor] = {}

    vk_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    for p_idx, (l_start, l_end) in enumerate(partitions):
        for t in selected_timesteps:
            pooled_rows: List[torch.Tensor] = []
            for layer in range(l_start, l_end + 1):
                pooled_rows.extend(diffs[(layer, t)])
            if not pooled_rows:
                raise RuntimeError(f"No deltas for partition={p_idx}, t={t}.")
            x_pool = torch.stack(pooled_rows, dim=0)
            n_pool, d_pool = x_pool.shape
            if args.k_target > min(n_pool, d_pool):
                raise ValueError(
                    f"k_target={args.k_target} too large for partition={p_idx}, t={t} "
                    f"with N={n_pool}, D={d_pool}."
                )
            mean_pool = x_pool.mean(dim=0)
            xc_pool = x_pool - mean_pool
            q = min(args.k_target + args.p_over, min(n_pool, d_pool))
            _, _, v = torch.pca_lowrank(xc_pool, q=q, center=False)
            vk = v[:, : args.k_target].contiguous().float()
            vk_cache[(p_idx, t)] = vk
            v_path = args.out_dir / f"V_part{p_idx}_layers{l_start}-{l_end}_t{t}_k{args.k_target}.pt"
            torch.save({"V": vk}, v_path)
            print(
                f"[svd] partition {p_idx} layers {l_start}-{l_end} t={t}: "
                f"pooled_rows={n_pool} saved {v_path.name}"
            )

    for (layer, t), delta_list in diffs.items():
        if not delta_list:
            raise RuntimeError(f"No deltas for layer={layer}, t={t}.")
        p_idx = layer_to_part[layer]
        l_start, l_end = partitions[p_idx]
        vk = vk_cache[(p_idx, t)]
        x = torch.stack(delta_list, dim=0)
        n, d = x.shape
        if vk.shape[0] != d:
            raise RuntimeError(
                f"V/activation dim mismatch at layer={layer}, t={t}: V D={vk.shape[0]} vs delta D={d}"
            )
        mean = x.mean(dim=0)
        c_means[layer, t_to_idx[t]] = (mean @ vk).float()
        for s_idx in range(n):
            projected_diffs[(s_idx, layer, t)] = (x[s_idx] @ vk).float().cpu()

    cfg = {
        "config_name": args.config_name,
        "prompt": prompt,
        "mode": args.mode,
        "selected_timesteps": selected_timesteps,
        "sampling_steps": int(sampling_steps),
        "L": int(n_layers),
        "k_target": int(args.k_target),
        "partitions": [list(p) for p in partitions],
        "partition_spec": partition_spec,
        "num_samples": int(n_total),
        "input_type": "activation_pairs" if activation_native else "obs_pairs",
    }
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    torch.save(
        {
            "c_means": c_means,
            "layer_to_part": layer_to_part,
            "selected_timesteps": selected_timesteps,
            "mode": args.mode,
            "num_samples": n_total,
            "k_target": int(args.k_target),
        },
        args.out_dir / "svd_summary.pt",
    )
    torch.save(
        {
            "projected_diffs": projected_diffs,
            "mode": args.mode,
            "prompt": prompt,
            "selected_timesteps": selected_timesteps,
            "k_target": int(args.k_target),
        },
        args.out_dir / "projected_diffs.pt",
    )
    print(f"[svd] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
