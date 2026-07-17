#!/usr/bin/env python
"""Collect Cosmos-Policy contrastive (positive, negative) input pairs across
the 10_object_pair_full_sweep stress-test rollouts, keeping only episodes
that SUCCEED in the uncluttered case but FAIL in the cluttered case.

Standalone .py variant of collect_policy_inputs_object_pairs.ipynb — same
2-pass per-config methodology (record env_pos states in pass 1, replay them
into env_neg in pass 2a, then run neg-drives in pass 2b), same NPZ schema,
same per-config subdir layout, but parameterized to sweep many more pairs
(parsing all_results.json from the full sweep instead of hard-coding 3
configs).

Filter: for each (target_a, target_b) pair in the sweep we look up the
matching `*_uncluttered` and `*_cluttered` configs and keep
`episodes = unc_succ_set - clu_succ_set` — episodes the uncluttered policy
solved that the cluttered policy didn't. These are the trajectories where
the cluttered environment actively degrades behavior, which is what the
contrastive direction should capture.

Output layout:

    <OUT_DIR>/
      positive.npz                # unified, all rows across all configs
      negative.npz                # unified, paired row-for-row with positive
      manifest.json
      derived__*.bddl             # SceneRetargetTask BDDLs (one per env per cfg)
      checkpoint_cfgN.{npz,json}  # incremental dumps after each config
      <name_hint>/                # per-config subdir (SVD-script compatible)
        positive.npz
        negative.npz
        prompt.txt

Pass 1 runs env_pos alone (no env_neg constructed yet) on the success-list
episodes — get_action is deterministic given (obs, desc) (every call uses
seed=1 by default, fresh np.RandomState for noise + fresh torch.Generator
for scheduler steps; no global RNG carries between calls), so this should
reproduce the stress test's pos trajectories bit-for-bit. Pass 2a builds
env_neg, replays each captured pos sim state via _expand_recorded_pos_to_neg,
and captures the matched neg render. Pass 2b runs neg-drives normally
(env_neg drives, env_pos state-injected per inference).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Match notebook _setup.py — must run BEFORE any cosmos_policy import. This
# script lives at notebooks/lqr/inputs/<this>.py; notebooks/_setup.py is two
# dirs up.
_HERE = Path(__file__).resolve().parent
_NOTEBOOKS_ROOT = _HERE.parent.parent
if str(_NOTEBOOKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_ROOT))
from _setup import setup_env  # noqa: E402

setup_env()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import torch as _torch  # noqa: E402

from libero.libero import benchmark, get_libero_path  # noqa: E402
from cosmos_policy.experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env, get_libero_dummy_action,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (  # noqa: E402
    PolicyEvalConfig, prepare_observation, TASK_MAX_STEPS,
)
from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    get_action, get_model, load_dataset_stats, init_t5_text_embeddings_cache,
)


# ====================================================================
# Logging — direct fd writes survive nbconvert / sbatch's buffering.
# ====================================================================

def _log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


# ====================================================================
# BDDL editing helpers (copied from 10_object_pair_selected.ipynb)
# ====================================================================

def _find_section_bounds(bddl, keyword):
    needle = f"(:{keyword}"
    start = bddl.find(needle)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(bddl)):
        c = bddl[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def _drop_section_lines(bddl, keyword, predicate):
    bounds = _find_section_bounds(bddl, keyword)
    if bounds is None:
        return bddl
    s, e = bounds
    section = bddl[s:e]
    keep = [line for line in section.split("\n") if not predicate(line.strip())]
    return bddl[:s] + "\n".join(keep) + bddl[e:]


def _parse_objects_order(bddl):
    bounds = _find_section_bounds(bddl, "objects")
    if bounds is None:
        raise ValueError("BDDL missing :objects section")
    s, e = bounds
    section = bddl[s:e]
    keys = []
    for line in section.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("(:") or stripped == ")":
            continue
        m = re.match(r"(\S+)\s*-\s*\S+", stripped)
        if m:
            keys.append(m.group(1))
    return keys


def _base_name(key):
    return re.sub(r"_\d+$", "", key)


def _strip_paren_block(text, opening_token):
    idx = text.find(opening_token)
    if idx == -1:
        return text
    depth = 0
    end = None
    for i in range(idx, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return text
    while end < len(text) and text[end] in " \t":
        end += 1
    if end < len(text) and text[end] == "\n":
        end += 1
    line_start = text.rfind("\n", 0, idx) + 1
    if text[line_start:idx].strip() == "":
        idx = line_start
    return text[:idx] + text[end:]


def _resolve_libero_problem_obj(env):
    cur, seen = env, set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "objects_dict") and hasattr(cur, "fixtures_dict"):
            return cur
        cur = getattr(cur, "env", None)
    return None


# ====================================================================
# SceneRetargetTask (copied from 10_object_pair_selected.ipynb)
# ====================================================================

class StressTest:
    slug: str = "stock"

    def transform_task(self, task, output_dir=None):
        return task

    def set_init_state(self, env, init_state, episode_idx=0):
        return env.set_init_state(init_state)

    def apply_to_env(self, env, episode_idx=0) -> None:
        return None

    def transform_task_desc(self, desc, env=None):
        return desc

    def manifest(self) -> dict:
        return {"kind": type(self).__name__, "slug": self.slug}


@dataclass
class SceneRetargetTask(StressTest):
    replacements: Tuple[Tuple[str, str, str], ...] = ()
    keep_only: Optional[Tuple[str, ...]] = None
    goal_pair: Tuple[str, str] = ("", "")
    container_key: str = "basket_1"
    prompt: str = ""
    name_hint: str = "retarget"

    _bddl_path: Optional[str] = field(default=None, init=False, repr=False)
    _orig_order: Tuple[str, ...] = field(default=(), init=False, repr=False)
    _final_order: Tuple[str, ...] = field(default=(), init=False, repr=False)
    _final_source: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _renamed_set: Tuple[str, ...] = field(default=(), init=False, repr=False)

    @property
    def slug(self) -> str:
        payload = (
            "r:" + "|".join(f"{a}=>{b}:{t}" for a, b, t in self.replacements) +
            "||k:" + (",".join(sorted(self.keep_only)) if self.keep_only is not None else "ALL") +
            "||g:" + ",".join(self.goal_pair) +
            "||c:" + self.container_key +
            "||p:" + self.prompt
        )
        h = hashlib.md5(payload.encode()).hexdigest()[:6]
        return f"{self.name_hint}_{h}"

    def transform_task(self, task, output_dir=None):
        src_path = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file,
        )
        with open(src_path) as f:
            bddl = f.read()

        self._orig_order = tuple(_parse_objects_order(bddl))
        rename_map = {old: (new, ntype) for old, new, ntype in self.replacements}
        self._renamed_set = tuple(new for _, new, _ in self.replacements)

        missing = set(rename_map) - set(self._orig_order)
        if missing:
            raise ValueError(f"replacement old_keys not in :objects: {sorted(missing)}")

        bounds = _find_section_bounds(bddl, "objects")
        if bounds is not None:
            s, e = bounds
            section = bddl[s:e]
            lines_out = []
            for line in section.split("\n"):
                stripped = line.strip()
                m = re.match(r"(\S+)\s*-\s*\S+", stripped)
                if m and m.group(1) in rename_map:
                    new, ntype = rename_map[m.group(1)]
                    indent = line[:len(line) - len(line.lstrip())]
                    lines_out.append(f"{indent}{new} - {ntype}")
                else:
                    lines_out.append(line)
            bddl = bddl[:s] + "\n".join(lines_out) + bddl[e:]

        bounds = _find_section_bounds(bddl, "init")
        if bounds is not None:
            s, e = bounds
            section = bddl[s:e]
            for old, new, _ in self.replacements:
                old_base = _base_name(old)
                section = re.sub(
                    rf"\(On\s+{re.escape(old)}\s+(\w+?)_{re.escape(old_base)}_init_region\)",
                    lambda m, nk=new: f"(On {nk} {m.group(1)}_{nk}_init_region)",
                    section,
                )
            bddl = bddl[:s] + section + bddl[e:]

        bounds = _find_section_bounds(bddl, "regions")
        if bounds is not None:
            s, e = bounds
            section = bddl[s:e]
            for old, new, _ in self.replacements:
                old_base = _base_name(old)
                section = section.replace(
                    f"({old_base}_init_region",
                    f"({new}_init_region",
                    1,
                )
            bddl = bddl[:s] + section + bddl[e:]

        post_rename_for_orig = {
            ok: (rename_map[ok][0] if ok in rename_map else ok)
            for ok in self._orig_order
        }
        post_rename_keys = [post_rename_for_orig[ok] for ok in self._orig_order]

        if self.keep_only is None:
            keep_set = set(post_rename_keys)
        else:
            keep_set = set(self.keep_only)
            missing_keep = {*self.goal_pair, self.container_key} - keep_set
            if missing_keep:
                raise ValueError(f"keep_only is missing required keys: {sorted(missing_keep)}")

        remove_origs = [ok for ok in self._orig_order if post_rename_for_orig[ok] not in keep_set]
        remove_posts = [post_rename_for_orig[ok] for ok in remove_origs]

        if remove_posts:
            bddl = _drop_section_lines(
                bddl, "objects",
                lambda L: any(re.match(rf"{re.escape(k)}\s*-", L) for k in remove_posts),
            )
            bddl = _drop_section_lines(
                bddl, "obj_of_interest",
                lambda L: L in remove_posts,
            )
            bddl = _drop_section_lines(
                bddl, "init",
                lambda L: any(L.startswith(f"(On {k} ") for k in remove_posts),
            )
            bounds = _find_section_bounds(bddl, "regions")
            if bounds is not None:
                s, e = bounds
                section = bddl[s:e]
                for ok, pk in zip(remove_origs, remove_posts):
                    region_id = pk if ok in rename_map else _base_name(ok)
                    section = _strip_paren_block(section, f"({region_id}_init_region")
                bddl = bddl[:s] + section + bddl[e:]

        bounds = _find_section_bounds(bddl, "goal")
        if bounds is not None:
            s, e = bounds
            a, b = self.goal_pair
            new_goal = (
                "(:goal\n"
                f"    (And (In {a} {self.container_key}_contain_region) "
                f"(In {b} {self.container_key}_contain_region))\n"
                "  )"
            )
            bddl = bddl[:s] + new_goal + bddl[e:]

        bounds = _find_section_bounds(bddl, "obj_of_interest")
        if bounds is not None:
            s, e = bounds
            new_section = (
                "(:obj_of_interest\n"
                f"    {self.goal_pair[0]}\n"
                f"    {self.goal_pair[1]}\n"
                f"    {self.container_key}\n"
                "  )"
            )
            bddl = bddl[:s] + new_section + bddl[e:]

        bddl = re.sub(
            r"\(:language[^)\n]*\)",
            f"(:language {self.prompt})",
            bddl, count=1,
        )

        self._final_order = tuple(_parse_objects_order(bddl))
        self._final_source = {
            pk: ok for ok, pk in zip(self._orig_order, post_rename_keys)
            if pk in keep_set
        }

        out_dir = Path(output_dir) if output_dir else Path("/tmp")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"derived__{self.slug}.bddl"
        out_path.write_text(bddl)
        self._bddl_path = str(out_path.resolve())

        if hasattr(task, "_replace"):
            return task._replace(bddl_file=self._bddl_path, problem_folder="")
        new_task = copy.copy(task)
        new_task.bddl_file = self._bddl_path
        new_task.problem_folder = ""
        return new_task

    def set_init_state(self, env, init_state, episode_idx=0):
        problem = _resolve_libero_problem_obj(env)
        if problem is None:
            return env.set_init_state(init_state)
        sim = problem.sim
        nq, nv = int(sim.model.nq), int(sim.model.nv)
        n_orig = len(self._orig_order)
        n_final = len(self._final_order)
        robot_nq = nq - 7 * n_final
        robot_nv = nv - 6 * n_final
        src_nq = robot_nq + 7 * n_orig
        src_nv = robot_nv + 6 * n_orig
        expected_src_len = 1 + src_nq + src_nv
        if len(init_state) != expected_src_len:
            raise ValueError(
                f"SceneRetargetTask.set_init_state: saved init_state size "
                f"{len(init_state)} != expected {expected_src_len}"
            )

        cur = sim.get_state()
        new_qpos = cur.qpos.copy()
        new_qvel = cur.qvel.copy()

        new_qpos[:robot_nq] = init_state[1 : 1 + robot_nq]
        new_qvel[:robot_nv] = init_state[1 + src_nq : 1 + src_nq + robot_nv]

        renamed_set = set(self._renamed_set)
        for final_key in self._final_order:
            source_key = self._final_source[final_key]
            j = self._orig_order.index(source_key)
            src_qpos_start = 1 + robot_nq + 7 * j
            src_qvel_start = 1 + src_nq + robot_nv + 6 * j

            mjobj = problem.objects_dict.get(final_key)
            if mjobj is None or not getattr(mjobj, "joints", None):
                continue
            addr = sim.model.get_joint_qpos_addr(mjobj.joints[0])
            qstart = addr[0] if isinstance(addr, tuple) else int(addr)
            vaddr = sim.model.get_joint_qvel_addr(mjobj.joints[0])
            vstart = vaddr[0] if isinstance(vaddr, tuple) else int(vaddr)

            new_qpos[qstart : qstart + 7] = init_state[src_qpos_start : src_qpos_start + 7]
            new_qvel[vstart : vstart + 6] = init_state[src_qvel_start : src_qvel_start + 6]

            if final_key in renamed_set:
                new_qpos[qstart + 2] = cur.qpos[qstart + 2]
                new_qpos[qstart + 3 : qstart + 7] = cur.qpos[qstart + 3 : qstart + 7]
                new_qvel[vstart : vstart + 6] = 0.0

        flat = np.concatenate([[init_state[0]], new_qpos, new_qvel])
        return env.set_init_state(flat)

    def transform_task_desc(self, desc, env=None):
        return self.prompt or desc

    def manifest(self) -> dict:
        return {
            "kind": "SceneRetargetTask",
            "slug": self.slug,
            "name_hint": self.name_hint,
            "replacements": [list(t) for t in self.replacements],
            "keep_only": list(self.keep_only) if self.keep_only is not None else None,
            "goal_pair": list(self.goal_pair),
            "container_key": self.container_key,
            "prompt": self.prompt,
            "orig_order": list(self._orig_order),
            "final_order": list(self._final_order),
            "derived_bddl_path": self._bddl_path,
        }


# ====================================================================
# pos -> neg expansion via captured arrays (no live env_pos needed in pass 2)
# ====================================================================

def get_kept_joint_addrs(env, kept_names):
    problem = _resolve_libero_problem_obj(env)
    sim = problem.sim
    out = {}
    for k in kept_names:
        obj = problem.objects_dict.get(k)
        if obj is None or not getattr(obj, "joints", None):
            continue
        ap = sim.model.get_joint_qpos_addr(obj.joints[0])
        qp = ap[0] if isinstance(ap, tuple) else int(ap)
        vap = sim.model.get_joint_qvel_addr(obj.joints[0])
        vp = vap[0] if isinstance(vap, tuple) else int(vap)
        out[k] = (qp, vp)
    return out


def expand_recorded_pos_to_neg(pos_qpos, pos_qvel, pos_time, init_state_flat,
                                pos_addrs, neg_addrs, neg_problem, n_kept):
    neg_sim = neg_problem.sim
    neg_nq, neg_nv = int(neg_sim.model.nq), int(neg_sim.model.nv)
    robot_nq = len(pos_qpos) - 7 * n_kept
    robot_nv = len(pos_qvel) - 6 * n_kept
    expected_neg_len = 1 + neg_nq + neg_nv
    if len(init_state_flat) != expected_neg_len:
        raise ValueError(
            f"init_state size {len(init_state_flat)} != expected {expected_neg_len}"
        )

    out = np.asarray(init_state_flat, dtype=np.float64).copy()
    out[0] = pos_time
    out[1 : 1 + robot_nq] = pos_qpos[:robot_nq]
    out[1 + neg_nq : 1 + neg_nq + robot_nv] = pos_qvel[:robot_nv]

    for k, (qp, vp) in pos_addrs.items():
        if k not in neg_addrs:
            continue
        qn, vn = neg_addrs[k]
        out[1 + qn : 1 + qn + 7] = pos_qpos[qp : qp + 7]
        out[1 + neg_nq + vn : 1 + neg_nq + vn + 6] = pos_qvel[vp : vp + 6]
    return out


# ====================================================================
# Rollout helpers — record env_pos alone, replay into env_neg, neg-drives
# ====================================================================

def _store_obs(obs_packed):
    return {
        "primary_image": np.ascontiguousarray(obs_packed["primary_image"]),
        "wrist_image":   np.ascontiguousarray(obs_packed["wrist_image"]),
        "proprio":       np.asarray(obs_packed["proprio"], dtype=np.float32),
    }


def record_pos_rollout(env_pos, init_state, pos_stress, prompt, eval_cfg,
                       model, dataset_stats, max_env_steps,
                       *, num_steps_wait=10):
    env_pos.reset()
    obs = pos_stress.set_init_state(env_pos, init_state)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env_pos.step(get_libero_dummy_action(eval_cfg.model_family))

    pos_problem = _resolve_libero_problem_obj(env_pos)
    pos_sim = pos_problem.sim

    queue = deque(maxlen=eval_cfg.num_open_loop_steps)
    records = []
    success = False
    t = 0
    while t < max_env_steps:
        if not queue:
            obs_packed = prepare_observation(obs, resize_size=224, flip_images=eval_cfg.flip_images)
            records.append({
                "pos_qpos":   np.asarray(pos_sim.data.qpos, dtype=np.float64).copy(),
                "pos_qvel":   np.asarray(pos_sim.data.qvel, dtype=np.float64).copy(),
                "pos_time":   float(pos_sim.data.time),
                "obs_stored": _store_obs(obs_packed),
            })
            out = get_action(
                eval_cfg, model, dataset_stats, obs_packed, prompt,
                num_denoising_steps_action=eval_cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            for a in out["actions"][:eval_cfg.num_open_loop_steps]:
                queue.append(np.asarray(a, dtype=np.float32))
        a = queue.popleft()
        obs, _, done, _ = env_pos.step(a.tolist())
        if done:
            success = True
            break
        t += 1
    return success, t + num_steps_wait, records


def render_pos_records_in_neg(env_neg, init_state, records,
                              pos_addrs, neg_addrs, n_kept, eval_cfg):
    neg_problem = _resolve_libero_problem_obj(env_neg)
    neg_stored = []
    for rec in records:
        flat = expand_recorded_pos_to_neg(
            rec["pos_qpos"], rec["pos_qvel"], rec["pos_time"],
            init_state, pos_addrs, neg_addrs, neg_problem, n_kept,
        )
        obs_neg = env_neg.set_init_state(flat)
        obs_neg_packed = prepare_observation(
            obs_neg, resize_size=224, flip_images=eval_cfg.flip_images,
        )
        neg_stored.append(_store_obs(obs_neg_packed))
    return neg_stored


def rollout_collect_neg_drives(init_state, env_neg, env_pos, pos_stress,
                                neg_stress, prompt, eval_cfg, model,
                                dataset_stats, max_env_steps,
                                *, num_steps_wait=10):
    env_neg.reset()
    obs_driver = neg_stress.set_init_state(env_neg, init_state)
    pos_stress.set_init_state(env_pos, init_state)

    for _ in range(num_steps_wait):
        obs_driver, _, _, _ = env_neg.step(get_libero_dummy_action(eval_cfg.model_family))

    queue = deque(maxlen=eval_cfg.num_open_loop_steps)
    neg_inputs, pos_inputs = [], []
    success = False
    t = 0
    while t < max_env_steps:
        if not queue:
            neg_packed = prepare_observation(obs_driver, resize_size=224, flip_images=eval_cfg.flip_images)
            driver_state_flat = env_neg.get_sim_state()
            obs_replay = pos_stress.set_init_state(env_pos, driver_state_flat)
            pos_packed = prepare_observation(obs_replay, resize_size=224, flip_images=eval_cfg.flip_images)
            neg_inputs.append(_store_obs(neg_packed))
            pos_inputs.append(_store_obs(pos_packed))
            out = get_action(
                eval_cfg, model, dataset_stats, neg_packed, prompt,
                num_denoising_steps_action=eval_cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            for a in out["actions"][:eval_cfg.num_open_loop_steps]:
                queue.append(np.asarray(a, dtype=np.float32))
        a = queue.popleft()
        obs_driver, _, done, _ = env_neg.step(a.tolist())
        if done:
            success = True
            break
        t += 1
    return success, t + num_steps_wait, neg_inputs, pos_inputs


# ====================================================================
# Discovery: unc-success ∧ clu-failure across the full sweep
# ====================================================================

def discover_scene_configs(rollouts_root: Path) -> List[dict]:
    """Parse all_results.json, pair uncluttered/cluttered configs by pair_names,
    and emit one entry per pair that has at least one (unc-success ∧ clu-failure)
    episode. Configs are 0-indexed in deterministic pair-name order.
    """
    all_results_path = rollouts_root / "all_results.json"
    if not all_results_path.exists():
        raise FileNotFoundError(f"missing {all_results_path}")
    res = json.loads(all_results_path.read_text())

    pairs = defaultdict(dict)
    for entry in res:
        key = tuple(entry["pair_names"])
        pairs[key][entry["setting"]] = entry

    configs = []
    for pair, settings in sorted(pairs.items()):
        if "uncluttered" not in settings or "cluttered" not in settings:
            continue
        unc = settings["uncluttered"]
        clu = settings["cluttered"]
        unc_succ = {r["episode"] for r in unc["results"] if r["success"]}
        clu_succ = {r["episode"] for r in clu["results"] if r["success"]}
        diff = sorted(unc_succ - clu_succ)
        if not diff:
            continue
        configs.append({
            "cfg_idx": len(configs),
            "name_hint": unc["name_hint"],
            "pair_names": list(pair),
            "goal_pair": list(unc["goal_pair"]),
            "prompt": unc["prompt"],
            "episodes": diff,
            "n_unc_success": len(unc_succ),
            "n_clu_success": len(clu_succ),
        })
    return configs


# ====================================================================
# Per-config NPZ subdirs (SVD-script compatible) — same as the notebook
# ====================================================================

_PER_CFG_KEYS = (
    "primary_images", "wrist_images", "proprios",
    "episode_idx",   "inference_idx", "drive_source",
    "config_idx",
)


def _slice_to_dict(arrs_dict, mask):
    return {k: arrs_dict[k][mask] for k in _PER_CFG_KEYS if k in arrs_dict}


# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts-root", type=Path,
                    default=_NOTEBOOKS_ROOT / "stress_test" / "rollouts" / "10_object_pair_full_sweep",
                    help="dir containing all_results.json + per-config subdirs")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="where to write positive.npz / negative.npz / per-config subdirs / manifest.json")
    ap.add_argument("--suite", type=str, default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--container-key", type=str, default="basket_1")
    ap.add_argument("--max-configs", type=int, default=-1,
                    help="cap on number of pairs to process (-1 = all)")
    ap.add_argument("--max-episodes-per-config", type=int, default=-1,
                    help="cap on episodes per pair (-1 = all qualifying)")
    ap.add_argument("--ckpt-path", type=str,
                    default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir: {out_dir}")
    _log(f"rollouts_root: {args.rollouts_root}")

    scene_configs = discover_scene_configs(args.rollouts_root)
    if args.max_configs > 0:
        scene_configs = scene_configs[:args.max_configs]
    if args.max_episodes_per_config > 0:
        for sc in scene_configs:
            sc["episodes"] = sc["episodes"][:args.max_episodes_per_config]

    n_total_episodes = sum(len(sc["episodes"]) for sc in scene_configs)
    _log(f"discovered {len(scene_configs)} pairs / {n_total_episodes} qualifying episodes "
         f"(unc-success ∧ clu-failure):")
    for sc in scene_configs:
        _log(f"  cfg{sc['cfg_idx']:>2d}  {sc['name_hint']:60s}  eps={sc['episodes']}")

    # ---------------- load libero suite + model ----------------
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    max_env_steps = TASK_MAX_STEPS[args.suite]
    _log(f"libero suite={args.suite} task={args.task_id}  "
         f"init_states={init_states.shape[0]}  max_env_steps={max_env_steps}")

    eval_cfg = PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=args.ckpt_path,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{args.ckpt_path}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{args.ckpt_path}/libero_t5_embeddings.pkl",
        use_wrist_image=True, use_proprio=True, normalize_proprio=True, unnormalize_actions=True,
        chunk_size=16, num_open_loop_steps=16, trained_with_image_aug=True,
        use_jpeg_compression=True, flip_images=True,
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1, num_denoising_steps_value=1,
        task_suite_name=args.suite,
    )
    dataset_stats = load_dataset_stats(eval_cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(eval_cfg.t5_text_embeddings_path)
    model, _ = get_model(eval_cfg)
    _log("model ready")

    # ---------------- accumulators ----------------
    all_pos_primary, all_pos_wrist, all_pos_proprio = [], [], []
    all_neg_primary, all_neg_wrist, all_neg_proprio = [], [], []
    all_episode_idx, all_inference_idx = [], []
    all_drive_source, all_config_idx = [], []
    rollout_summaries = []
    config_manifests = []

    DRIVE_SOURCES = (("neg_drives", 0), ("pos_drives", 1))

    def _checkpoint(suffix: str) -> None:
        if not all_episode_idx:
            return
        ck = out_dir / f"checkpoint_{suffix}.npz"
        np.savez_compressed(
            ck,
            pos_primary=np.stack(all_pos_primary, axis=0),
            pos_wrist=np.stack(all_pos_wrist, axis=0),
            pos_proprio=np.stack(all_pos_proprio, axis=0),
            neg_primary=np.stack(all_neg_primary, axis=0),
            neg_wrist=np.stack(all_neg_wrist, axis=0),
            neg_proprio=np.stack(all_neg_proprio, axis=0),
            episode_idx=np.asarray(all_episode_idx, dtype=np.int32),
            inference_idx=np.asarray(all_inference_idx, dtype=np.int32),
            drive_source=np.asarray(all_drive_source, dtype=np.int32),
            config_idx=np.asarray(all_config_idx, dtype=np.int32),
        )
        (out_dir / f"checkpoint_{suffix}.json").write_text(json.dumps({
            "rollout_summaries": rollout_summaries,
            "config_manifests":  config_manifests,
        }, indent=2))
        _log(f"checkpoint {ck.name} ({ck.stat().st_size/1e6:.1f} MB)")

    # ---------------- main loop ----------------
    _log("=== run loop starting ===")
    for scene in scene_configs:
        cfg_idx        = scene["cfg_idx"]
        name_hint      = scene["name_hint"]
        goal_pair      = tuple(scene["goal_pair"])
        prompt         = scene["prompt"]
        target_success = list(scene["episodes"])
        target_set     = set(target_success)

        _log(f"cfg{cfg_idx} {name_hint} START  prompt={prompt!r}  "
             f"unc-succ∧clu-fail={target_success}")

        pos_stress = SceneRetargetTask(
            replacements=(),
            keep_only=(goal_pair[0], goal_pair[1], args.container_key),
            goal_pair=goal_pair,
            container_key=args.container_key,
            prompt=prompt,
            name_hint=f"{name_hint}__pos",
        )
        neg_stress = SceneRetargetTask(
            replacements=(),
            keep_only=None,
            goal_pair=goal_pair,
            container_key=args.container_key,
            prompt=prompt,
            name_hint=f"{name_hint}__neg",
        )

        # --- Build env_pos alone (matches stress-test single-env construction) ---
        _log(f"cfg{cfg_idx} building env_pos")
        pos_task = pos_stress.transform_task(task, output_dir=out_dir)
        env_pos, base_task_desc_pos = get_libero_env(pos_task, "cosmos", resolution=args.resolution)
        pos_addrs = get_kept_joint_addrs(env_pos, pos_stress._final_order)
        n_kept = len(pos_stress._final_order)
        _log(f"cfg{cfg_idx} env_pos ready kept={list(pos_stress._final_order)}")

        # --- Pass 1: record_pos_rollout on the success-list episodes ---
        per_episode_records = {}
        per_episode_success = {}
        per_episode_env_steps = {}
        for ep in target_success:
            _log(f"cfg{cfg_idx} pass1 ep={ep} start")
            t0 = time.time()
            success, env_steps, records = record_pos_rollout(
                env_pos, init_states[ep], pos_stress, prompt,
                eval_cfg, model, dataset_stats, max_env_steps,
            )
            dt = time.time() - t0
            per_episode_success[ep] = bool(success)
            per_episode_env_steps[ep] = int(env_steps)
            per_episode_records[ep] = records
            tag = "SUCCESS" if success else "FAILURE"
            _log(f"cfg{cfg_idx} pass1 ep={ep} {tag} steps={env_steps} inf={len(records)} {dt:.1f}s")
            _torch.cuda.empty_cache()

        reproduced_succ = {ep for ep in target_set if per_episode_success[ep]}
        matched_set     = target_set & reproduced_succ
        missing_set     = target_set - reproduced_succ
        if missing_set:
            _log(f"cfg{cfg_idx} pass1: matched {len(matched_set)}/{len(target_set)}; "
                 f"missing={sorted(missing_set)}")

        # --- Build env_neg ---
        _log(f"cfg{cfg_idx} building env_neg")
        neg_task = neg_stress.transform_task(task, output_dir=out_dir)
        env_neg, base_task_desc_neg = get_libero_env(neg_task, "cosmos", resolution=args.resolution)
        neg_addrs = get_kept_joint_addrs(env_neg, pos_stress._final_order)
        _log(f"cfg{cfg_idx} env_neg ready")

        config_manifests.append({
            "cfg_idx": cfg_idx,
            "name_hint": name_hint,
            "pair_names": scene["pair_names"],
            "goal_pair": list(goal_pair),
            "prompt": prompt,
            "n_unc_success": scene["n_unc_success"],
            "n_clu_success": scene["n_clu_success"],
            "episodes_target_success": sorted(target_set),
            "episodes_reproduced_success": sorted(matched_set),
            "episodes_missing_success": sorted(missing_set),
            "positive_stress": pos_stress.manifest(),
            "negative_stress": neg_stress.manifest(),
            "env_base_task_desc_pos": base_task_desc_pos,
            "env_base_task_desc_neg": base_task_desc_neg,
        })

        # --- Pass 2a: pos-drives replay over matched episodes ---
        emit_eps = sorted(matched_set)
        _log(f"cfg{cfg_idx} pass2a START replay over {len(emit_eps)} matched episodes")
        for ep in emit_eps:
            _log(f"cfg{cfg_idx} pass2a ep={ep} replay start")
            records = per_episode_records[ep]
            t0 = time.time()
            neg_stored = render_pos_records_in_neg(
                env_neg, init_states[ep], records,
                pos_addrs, neg_addrs, n_kept, eval_cfg,
            )
            dt = time.time() - t0
            assert len(neg_stored) == len(records)

            for inf_idx, (rec, n_st) in enumerate(zip(records, neg_stored)):
                p_st = rec["obs_stored"]
                all_neg_primary.append(n_st["primary_image"])
                all_neg_wrist.append(n_st["wrist_image"])
                all_neg_proprio.append(n_st["proprio"])
                all_pos_primary.append(p_st["primary_image"])
                all_pos_wrist.append(p_st["wrist_image"])
                all_pos_proprio.append(p_st["proprio"])
                all_episode_idx.append(ep)
                all_inference_idx.append(inf_idx)
                all_drive_source.append(1)
                all_config_idx.append(cfg_idx)

            n_inf = len(records)
            _log(f"cfg{cfg_idx} pass2a ep={ep} replayed inf={n_inf} {dt:.1f}s")
            rollout_summaries.append({
                "cfg_idx": cfg_idx,
                "name_hint": name_hint,
                "drive_source": 1,
                "drive_name": "pos_drives",
                "episode": ep,
                "success": True,
                "env_steps": per_episode_env_steps[ep],
                "n_inferences": n_inf,
                "wall_time_s": dt,
                "reproduction_source": "pass1_record_replay",
            })

        per_episode_records.clear()
        _torch.cuda.empty_cache()

        # --- Pass 2b: neg-drives over target episodes ---
        _log(f"cfg{cfg_idx} pass2b START neg-drives over {len(target_success)} episodes")
        for ep in target_success:
            _log(f"cfg{cfg_idx} pass2b ep={ep} start")
            t0 = time.time()
            success, env_steps, neg_inputs, pos_inputs = rollout_collect_neg_drives(
                init_states[ep], env_neg, env_pos, pos_stress, neg_stress, prompt,
                eval_cfg, model, dataset_stats, max_env_steps,
            )
            dt = time.time() - t0
            assert len(neg_inputs) == len(pos_inputs)

            for inf_idx, (n_rec, p_rec) in enumerate(zip(neg_inputs, pos_inputs)):
                all_neg_primary.append(n_rec["primary_image"])
                all_neg_wrist.append(n_rec["wrist_image"])
                all_neg_proprio.append(n_rec["proprio"])
                all_pos_primary.append(p_rec["primary_image"])
                all_pos_wrist.append(p_rec["wrist_image"])
                all_pos_proprio.append(p_rec["proprio"])
                all_episode_idx.append(ep)
                all_inference_idx.append(inf_idx)
                all_drive_source.append(0)
                all_config_idx.append(cfg_idx)

            tag = "SUCCESS" if success else "FAILURE"
            n_inf = len(neg_inputs)
            _log(f"cfg{cfg_idx} pass2b ep={ep} {tag} steps={env_steps} inf={n_inf} {dt:.1f}s")
            rollout_summaries.append({
                "cfg_idx": cfg_idx,
                "name_hint": name_hint,
                "drive_source": 0,
                "drive_name": "neg_drives",
                "episode": ep,
                "success": bool(success),
                "env_steps": int(env_steps),
                "n_inferences": n_inf,
                "wall_time_s": dt,
            })
            _torch.cuda.empty_cache()

        env_neg.close()
        env_pos.close()
        _torch.cuda.empty_cache()
        _log(f"cfg{cfg_idx} DONE total_rows={len(all_episode_idx)}")
        _checkpoint(f"cfg{cfg_idx}")

    _log(f"all configs complete; total paired rows: {len(all_episode_idx)}")

    # ---------------- save unified NPZs ----------------
    POSITIVE_NPZ = out_dir / "positive.npz"
    NEGATIVE_NPZ = out_dir / "negative.npz"
    MANIFEST_JSON = out_dir / "manifest.json"

    prompts_lookup = np.array([s["prompt"] for s in scene_configs], dtype=object)
    name_hints_lookup = np.array([s["name_hint"] for s in scene_configs], dtype=object)

    def _stack_save(out_npz, primary_list, wrist_list, proprio_list):
        primary_arr = np.stack(primary_list, axis=0)
        wrist_arr = np.stack(wrist_list, axis=0)
        proprio_arr = np.stack(proprio_list, axis=0)
        np.savez_compressed(
            out_npz,
            primary_images=primary_arr,
            wrist_images=wrist_arr,
            proprios=proprio_arr,
            episode_idx=np.asarray(all_episode_idx, dtype=np.int32),
            inference_idx=np.asarray(all_inference_idx, dtype=np.int32),
            drive_source=np.asarray(all_drive_source, dtype=np.int32),
            config_idx=np.asarray(all_config_idx, dtype=np.int32),
            prompts=prompts_lookup,
            name_hints=name_hints_lookup,
        )
        _log(f"wrote {out_npz.name} ({out_npz.stat().st_size/1e6:.1f} MB) "
             f"primary_images={primary_arr.shape}")
        return primary_arr.shape[0]

    n_pos = _stack_save(POSITIVE_NPZ, all_pos_primary, all_pos_wrist, all_pos_proprio)
    n_neg = _stack_save(NEGATIVE_NPZ, all_neg_primary, all_neg_wrist, all_neg_proprio)
    assert n_pos == n_neg

    pos_proprio = np.stack(all_pos_proprio, axis=0)
    neg_proprio = np.stack(all_neg_proprio, axis=0)
    max_dproprio = float(np.max(np.abs(pos_proprio - neg_proprio)))
    _log(f"paired proprio max |Δ| = {max_dproprio:.3e}")

    # ---------------- per-config NPZ subdirs (SVD-compatible) ----------------
    unified_pos = dict(np.load(POSITIVE_NPZ, allow_pickle=True))
    unified_neg = dict(np.load(NEGATIVE_NPZ, allow_pickle=True))
    ci = unified_pos["config_idx"]
    _log("per-config NPZ subdirs:")
    for s in scene_configs:
        cfg_idx = s["cfg_idx"]
        name_hint = s["name_hint"]
        prompt = s["prompt"]
        mask = (ci == cfg_idx)
        n_rows = int(mask.sum())
        if n_rows == 0:
            _log(f"  cfg{cfg_idx} {name_hint}: 0 rows — SKIPPED")
            continue
        sub_dir = out_dir / name_hint
        sub_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sub_dir / "positive.npz", **_slice_to_dict(unified_pos, mask))
        np.savez_compressed(sub_dir / "negative.npz", **_slice_to_dict(unified_neg, mask))
        (sub_dir / "prompt.txt").write_text(prompt + "\n")
        _log(f"  cfg{cfg_idx} {name_hint}: {n_rows} rows")
    del unified_pos, unified_neg

    # ---------------- manifest ----------------
    manifest = {
        "suite": args.suite,
        "task_id": args.task_id,
        "resolution": args.resolution,
        "container_key": args.container_key,
        "rollouts_root": str(args.rollouts_root),
        "selection_rule": "uncluttered-success ∧ cluttered-failure",
        "pairing": (
            "row i in positive.npz and row i in negative.npz share the same MuJoCo "
            "state at capture time. drive_source distinguishes which env stepped: "
            "0 = env_neg drove, env_pos was state-injected via "
            "SceneRetargetTask.set_init_state; 1 = env_pos drove, env_neg was "
            "state-injected via expand_recorded_pos_to_neg (kept-body qpos via "
            "joint addrs, removed-body slots from the episode's saved libero "
            "init_state)."
        ),
        "image_layout": "HWC uint8, flip_images=True applied at capture time",
        "proprio_layout": ("concat(robot0_gripper_qpos[2], robot0_eef_pos[3], "
                            "robot0_eef_quat[4]) -> shape (9,) float32"),
        "drive_sources": [
            {"code": 0, "name": "neg_drives", "desc": "env_neg drives; env_pos state-injected"},
            {"code": 1, "name": "pos_drives", "desc": "env_pos drives; env_neg state-injected via captured arrays"},
        ],
        "sets": {
            "positive": {"out_npz": str(POSITIVE_NPZ),
                          "role": "uncluttered (keep-only) render at every captured pose"},
            "negative": {"out_npz": str(NEGATIVE_NPZ),
                          "role": "cluttered (full 8-object) render at every captured pose"},
        },
        "configs": config_manifests,
        "rollouts": rollout_summaries,
        "paired_proprio_max_abs_diff": max_dproprio,
        "total_paired_rows": int(n_pos),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    _log(f"wrote {MANIFEST_JSON.name}")

    # ---------------- summary ----------------
    _log("=== summary ===")
    ci_arr = np.asarray(all_config_idx, dtype=np.int32)
    ds_arr = np.asarray(all_drive_source, dtype=np.int32)
    for s in scene_configs:
        for drv_name, drv_code in DRIVE_SOURCES:
            n = int(((ci_arr == s["cfg_idx"]) & (ds_arr == drv_code)).sum())
            _log(f"  cfg{s['cfg_idx']:>2d} {s['name_hint']:60s} {drv_name:11s}: {n} rows")
    _log(f"total paired rows: {n_pos}")
    _log(f"positive (uncluttered) : {POSITIVE_NPZ}")
    _log(f"negative (cluttered)   : {NEGATIVE_NPZ}")
    _log(f"manifest               : {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
