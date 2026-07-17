import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

DEBUG_LOG_PATH = os.environ.get(
    "LINGBOT_DEBUG_LOG_PATH",
    "outputs/debug/build_all_pairs_debug.jsonl",
)
DEBUG_SESSION_ID = os.environ.get("LINGBOT_DEBUG_SESSION_ID", "local")


def _dbg_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    if not DEBUG_LOG_PATH:
        return
    Path(DEBUG_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"record not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _trajectory_id(rec: Dict[str, Any]) -> str:
    return f"{rec.get('variant_name', 'unknown')}|task{int(rec.get('task_id', -1))}|ep{int(rec.get('episode_idx', -1))}"


def _extract_sorted_captures(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    captures = list(rec.get("captures", []))
    captures.sort(key=lambda x: int(x.get("inference_idx_in_traj", 0)))
    return captures


def _normalize_to_k(captures: List[Dict[str, Any]], target_k: int) -> Tuple[List[Dict[str, Any]], int]:
    if not captures:
        return [], 0
    out = list(captures[:target_k])
    if len(out) >= target_k:
        return out, 0
    pad_count = target_k - len(out)
    last = out[-1]
    for _ in range(pad_count):
        out.append(
            {
                "inference_idx_in_traj": int(last.get("inference_idx_in_traj", 0)),
                "frame_st_id": int(last.get("frame_st_id", 0)),
                "activations": last.get("activations"),
                "_is_padded_repeat_last": True,
            }
        )
    return out, pad_count


def _make_item(
    traj: Dict[str, Any],
    cap: Dict[str, Any],
    pair_id: int,
    paired_timestep: int,
    pos_traj_id: str,
    neg_traj_id: str,
) -> Dict[str, Any]:
    return {
        "variant_name": str(traj["variant_name"]),
        "task_id": int(traj["task_id"]),
        "episode_idx": int(traj["episode_idx"]),
        "inference_idx_in_traj": int(paired_timestep),
        "source_inference_idx_in_traj": int(cap.get("inference_idx_in_traj", 0)),
        "frame_st_id": int(cap.get("frame_st_id", 0)),
        "activations": cap["activations"],
        "pair_id": int(pair_id),
        "paired_timestep": int(paired_timestep),
        "pos_traj_id": str(pos_traj_id),
        "neg_traj_id": str(neg_traj_id),
        "is_padded_repeat_last": bool(cap.get("_is_padded_repeat_last", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pool-based activation pairs from trajectory records.")
    parser.add_argument("--collect-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--pairing-mode", type=str, choices=["pool"], default="pool")
    args = parser.parse_args()
    run_id = f"pre-fix-{int(time.time())}"

    if args.pairing_mode != "pool":
        raise ValueError("Only pool pairing is supported.")

    manifest = _load_manifest(args.collect_dir / "manifest.json")
    records = manifest.get("records", [])
    if not records:
        raise ValueError("No records in collect manifest.")
    #region agent log
    _dbg_log(
        run_id=run_id,
        hypothesis_id="H4",
        location="scripts/lqr/build_all_pairs.py:main:init",
        message="Loaded collect manifest",
        data={
            "collect_dir": str(args.collect_dir),
            "records_count": len(records),
            "num_episodes": manifest.get("num_episodes"),
            "variants": [str(v.get("name", "")) for v in manifest.get("variants", [])],
        },
    )
    #endregion

    positive_trajs: List[Dict[str, Any]] = []
    negative_trajs: List[Dict[str, Any]] = []
    dropped_nominal_fail = 0
    dropped_perturb_success = 0
    total_activation_records = 0
    traj_counter = {
        "nominal_success_traj": 0,
        "nominal_fail_traj": 0,
        "perturb_success_traj": 0,
        "perturb_fail_traj": 0,
    }
    cap_counter = {
        "nominal_success_caps": 0,
        "nominal_fail_caps": 0,
        "perturb_success_caps": 0,
        "perturb_fail_caps": 0,
    }
    variant_traj: Dict[str, Dict[str, int]] = {}
    variant_caps: Dict[str, Dict[str, int]] = {}
    length_hist_pos: Dict[str, int] = {}
    length_hist_neg: Dict[str, int] = {}
    dropped_empty_capture_traj = 0

    for row in records:
        rec = _load_record(Path(row["path"]))
        is_nominal = bool(rec.get("is_nominal", False))
        traj_success = bool(rec.get("trajectory_success", False))
        variant_name = str(rec.get("variant_name", "unknown"))
        captures = _extract_sorted_captures(rec)
        total_activation_records += len(captures)
        if variant_name not in variant_traj:
            variant_traj[variant_name] = {"success": 0, "fail": 0}
            variant_caps[variant_name] = {"success_caps": 0, "fail_caps": 0}
        if traj_success:
            variant_traj[variant_name]["success"] += 1
            variant_caps[variant_name]["success_caps"] += len(captures)
        else:
            variant_traj[variant_name]["fail"] += 1
            variant_caps[variant_name]["fail_caps"] += len(captures)
        traj_obj = {
            "traj_id": _trajectory_id(rec),
            "variant_name": variant_name,
            "task_id": int(rec["task_id"]),
            "episode_idx": int(rec["episode_idx"]),
            "captures": captures,
        }
        if not captures:
            dropped_empty_capture_traj += 1
            continue
        if is_nominal:
            if traj_success:
                cap_counter["nominal_success_caps"] += len(captures)
                positive_trajs.append(traj_obj)
                length_hist_pos[str(len(captures))] = int(length_hist_pos.get(str(len(captures)), 0) + 1)
            else:
                cap_counter["nominal_fail_caps"] += len(captures)
                dropped_nominal_fail += len(captures)
        else:
            if traj_success:
                cap_counter["perturb_success_caps"] += len(captures)
                dropped_perturb_success += len(captures)
            else:
                cap_counter["perturb_fail_caps"] += len(captures)
                negative_trajs.append(traj_obj)
                length_hist_neg[str(len(captures))] = int(length_hist_neg.get(str(len(captures)), 0) + 1)
        if is_nominal and traj_success:
            traj_counter["nominal_success_traj"] += 1
        elif is_nominal and (not traj_success):
            traj_counter["nominal_fail_traj"] += 1
        elif (not is_nominal) and traj_success:
            traj_counter["perturb_success_traj"] += 1
        else:
            traj_counter["perturb_fail_traj"] += 1

    #region agent log
    _dbg_log(
        run_id=run_id,
        hypothesis_id="H1,H2,H3",
        location="scripts/lqr/build_all_pairs.py:main:post_scan",
        message="Pool precheck stats",
        data={
            "total_activation_records": total_activation_records,
            "traj_counter": traj_counter,
            "cap_counter": cap_counter,
            "variant_traj": variant_traj,
            "variant_caps": variant_caps,
            "positive_traj_count": len(positive_trajs),
            "negative_traj_count": len(negative_trajs),
            "length_hist_pos": length_hist_pos,
            "length_hist_neg": length_hist_neg,
            "dropped_empty_capture_traj": dropped_empty_capture_traj,
        },
    )
    #endregion

    if not positive_trajs:
        #region agent log
        _dbg_log(
            run_id=run_id,
            hypothesis_id="H1,H2,H3",
            location="scripts/lqr/build_all_pairs.py:main:raise_no_positive",
            message="No positive candidates",
            data={
                "positive_traj_count": len(positive_trajs),
                "traj_counter": traj_counter,
                "variant_traj": variant_traj,
            },
        )
        #endregion
        raise RuntimeError("No positive candidates after filtering: nominal success pool is empty.")
    if not negative_trajs:
        #region agent log
        _dbg_log(
            run_id=run_id,
            hypothesis_id="H1",
            location="scripts/lqr/build_all_pairs.py:main:raise_no_negative",
            message="No negative candidates",
            data={
                "negative_traj_count": len(negative_trajs),
                "traj_counter": traj_counter,
                "variant_traj": variant_traj,
            },
        )
        #endregion
        raise RuntimeError("No negative candidates after filtering: perturbation failure pool is empty.")

    target_k_raw = manifest.get("top_k_inference_per_traj")
    if target_k_raw is not None:
        target_k = int(target_k_raw)
        target_k_source = "manifest.top_k_inference_per_traj"
    else:
        max_len = max(len(t["captures"]) for t in positive_trajs + negative_trajs)
        target_k = int(max_len)
        target_k_source = "observed_max_capture_len"
    if target_k <= 0:
        raise RuntimeError(f"Invalid target_k for pairing: {target_k}")

    pos_pad_total = 0
    neg_pad_total = 0
    for traj in positive_trajs:
        norm_caps, pad_n = _normalize_to_k(traj["captures"], target_k=target_k)
        pos_pad_total += pad_n
        traj["captures_norm"] = norm_caps
    for traj in negative_trajs:
        norm_caps, pad_n = _normalize_to_k(traj["captures"], target_k=target_k)
        neg_pad_total += pad_n
        traj["captures_norm"] = norm_caps

    rng = np.random.default_rng(seed=int(args.pair_seed))
    neg_order = np.arange(len(negative_trajs), dtype=np.int64)
    rng.shuffle(neg_order)
    positive_records: List[Dict[str, Any]] = []
    negative_records: List[Dict[str, Any]] = []
    pos_traj_use_count: Dict[str, int] = {}
    timestep_counts: Dict[str, int] = {}
    for pair_id, neg_idx in enumerate(neg_order.tolist()):
        neg_traj = negative_trajs[int(neg_idx)]
        pos_choice = int(rng.integers(low=0, high=len(positive_trajs)))
        pos_traj = positive_trajs[pos_choice]
        pos_traj_id = str(pos_traj["traj_id"])
        neg_traj_id = str(neg_traj["traj_id"])
        pos_traj_use_count[pos_traj_id] = int(pos_traj_use_count.get(pos_traj_id, 0) + 1)
        for t in range(target_k):
            pos_cap = pos_traj["captures_norm"][t]
            neg_cap = neg_traj["captures_norm"][t]
            positive_records.append(
                _make_item(
                    traj=pos_traj,
                    cap=pos_cap,
                    pair_id=pair_id,
                    paired_timestep=t,
                    pos_traj_id=pos_traj_id,
                    neg_traj_id=neg_traj_id,
                )
            )
            negative_records.append(
                _make_item(
                    traj=neg_traj,
                    cap=neg_cap,
                    pair_id=pair_id,
                    paired_timestep=t,
                    pos_traj_id=pos_traj_id,
                    neg_traj_id=neg_traj_id,
                )
            )
            timestep_counts[str(t)] = int(timestep_counts.get(str(t), 0) + 1)

    assert len(positive_records) == len(negative_records), "Pair arrays must be equal length."

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pos_fp = args.out_dir / "positive.pt"
    neg_fp = args.out_dir / "negative.pt"
    torch.save(
        {
            "records": positive_records,
            "mode": manifest.get("mode"),
            "selected_timesteps": manifest.get("selected_timesteps"),
            "layers": manifest.get("layers"),
            "prompt": manifest.get("task_language"),
        },
        pos_fp,
    )
    torch.save(
        {
            "records": negative_records,
            "mode": manifest.get("mode"),
            "selected_timesteps": manifest.get("selected_timesteps"),
            "layers": manifest.get("layers"),
            "prompt": manifest.get("task_language"),
        },
        neg_fp,
    )

    out_manifest = {
        "mode": "temporal_aligned_pool",
        "collect_dir": str(args.collect_dir.resolve()),
        "pair_seed": int(args.pair_seed),
        "target_k": int(target_k),
        "target_k_source": target_k_source,
        "total_activation_records": int(total_activation_records),
        "dropped_nominal_fail": int(dropped_nominal_fail),
        "dropped_perturb_success": int(dropped_perturb_success),
        "dropped_empty_capture_traj": int(dropped_empty_capture_traj),
        "positive_traj_count": int(len(positive_trajs)),
        "negative_traj_count": int(len(negative_trajs)),
        "positive_unique_used": int(len(pos_traj_use_count)),
        "positive_reuse_rate": float(len(negative_trajs) / max(1, len(pos_traj_use_count))),
        "num_traj_pairs_total": int(len(negative_trajs)),
        "num_pairs_total": int(len(positive_records)),
        "pos_pad_total": int(pos_pad_total),
        "neg_pad_total": int(neg_pad_total),
        "length_hist_pos": length_hist_pos,
        "length_hist_neg": length_hist_neg,
        "timestep_counts": timestep_counts,
        "prompt": manifest.get("task_language"),
        "task_language": manifest.get("task_language"),
        "files": {
            "positive": str(pos_fp.resolve()),
            "negative": str(neg_fp.resolve()),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    #region agent log
    _dbg_log(
        run_id=run_id,
        hypothesis_id="H4",
        location="scripts/lqr/build_all_pairs.py:main:success",
        message="Pair construction success",
        data={
            "paired_count": int(len(negative_trajs)),
            "positive_traj_count": len(positive_trajs),
            "negative_traj_count": len(negative_trajs),
            "num_pairs_total": len(positive_records),
            "target_k": target_k,
            "target_k_source": target_k_source,
            "pos_pad_total": pos_pad_total,
            "neg_pad_total": neg_pad_total,
        },
    )
    #endregion
    print(f"[pair] wrote {pos_fp}")
    print(f"[pair] wrote {neg_fp}")
    print(f"[pair] wrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
