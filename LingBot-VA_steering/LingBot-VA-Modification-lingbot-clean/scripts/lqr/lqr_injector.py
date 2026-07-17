import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from scripts.lqr.lqr_core import VCache, chained_riccati_per_chunk


class LQRInjector:
    def __init__(
        self,
        svd_dir: str,
        jac_dir_act: str,
        lambda_scale: float,
        q_scale: float,
        r_scale: float,
        r_scale_tau: float,
        r_scale_final: float,
        max_chunks: int,
        qf_scale: float,
        inject_mode: str = "auto",
        device: Optional[torch.device] = None,
    ) -> None:
        self.svd_dir = Path(svd_dir)
        self.jac_dir_act = str(jac_dir_act)
        self.lambda_scale = float(lambda_scale)
        self.q_scale = float(q_scale)
        self.r_scale = float(r_scale)
        self.r_scale_tau = float(r_scale_tau)
        self.r_scale_final = float(r_scale_final)
        self.max_chunks = int(max_chunks)
        self.qf_scale = float(qf_scale)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.inject_mode = str(inject_mode)
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.in_ad = False
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.current_action_mode: bool = False
        self.current_step_idx: int = -1
        self.u_step_pending: Optional[Dict[str, torch.Tensor]] = None
        self.u_norm_log: List[Tuple[int, int, float, float]] = []
        self._shape_warned = set()
        self.rollout_chunk_idx = 0
        self.active_chunk_idx = 0

        cfg = json.loads((self.svd_dir / "config.json").read_text(encoding="utf-8"))
        cfg_mode = str(cfg.get("mode", "action"))
        if self.inject_mode == "auto":
            self.inject_mode = cfg_mode
        if self.inject_mode not in {"action", "video", "both"}:
            raise ValueError(f"inject_mode must be one of action/video/both/auto, got {inject_mode}")
        self.sel_t = list(cfg["selected_timesteps"])
        self.sel_idx_of = {t: i for i, t in enumerate(self.sel_t)}
        self.T_diff = len(self.sel_t)
        self.L = int(cfg["L"])
        self.r = int(cfg["k_target"])

        summary = torch.load(self.svd_dir / "svd_summary.pt", map_location="cpu", weights_only=False)
        c_means = summary["c_means"].float()
        self.layer_to_part = list(summary["layer_to_part"])
        self.partitions = [tuple(p) for p in cfg["partitions"]]
        self.tilde_mu = c_means.norm(dim=-1)
        self.tilde_v = c_means / self.tilde_mu.unsqueeze(-1).clamp(min=1e-12)

        jac_fp = self.svd_dir / self.jac_dir_act / "A_tilde__full.pt"
        raw = torch.load(jac_fp, map_location="cpu", weights_only=False)
        A_dict = raw.get("A_tilde", {})
        B_dict = raw.get("B_tilde", {})

        A_tilde = torch.zeros(self.T_diff, self.L - 1, self.r, self.r, dtype=torch.float32)
        for (t, l_in), Atl in A_dict.items():
            if t in self.sel_idx_of:
                A_tilde[self.sel_idx_of[t], l_in] = Atl.float()

        B_tilde = torch.zeros(max(self.T_diff - 1, 0), self.r, self.r, dtype=torch.float32)
        for (t,), Bt in B_dict.items():
            if t in self.sel_idx_of and self.sel_idx_of[t] < self.T_diff - 1:
                B_tilde[self.sel_idx_of[t]] = Bt.float()

        if self.r_scale_tau <= 0:
            raise ValueError(f"r_scale_tau must be > 0, got {self.r_scale_tau}")
        if self.r_scale_final < self.r_scale:
            raise ValueError(
                f"r_scale_final ({self.r_scale_final:g}) must be >= r_scale ({self.r_scale:g})"
            )
        if self.max_chunks < 1:
            raise ValueError(f"max_chunks must be >= 1, got {self.max_chunks}")
        self.r_scale_schedule = [
            min(self.r_scale_final, self.r_scale * math.exp(c / self.r_scale_tau))
            for c in range(self.max_chunks)
        ]

        self.K_intra_per_chunk, self.K_step_per_chunk = chained_riccati_per_chunk(
            A_tilde=A_tilde,
            B_tilde=B_tilde,
            q_scale=self.q_scale,
            r_scale_schedule=self.r_scale_schedule,
            qf_scale=self.qf_scale,
            device=self.device,
        )
        K_intra = self.K_intra_per_chunk[0]
        K_step = self.K_step_per_chunk[0]
        self.lqr = {
            "K_intra": K_intra.to(device=self.device, dtype=torch.float32),
            "K_step": K_step.to(device=self.device, dtype=torch.float32),
            "v": self.tilde_v.to(device=self.device, dtype=torch.float32),
            "mu": self.tilde_mu.to(device=self.device, dtype=torch.float32),
        }
        self.vcache = VCache(
            svd_dir=str(self.svd_dir),
            partitions=self.partitions,
            layer_to_part=self.layer_to_part,
            sel_t=self.sel_t,
            k_target=self.r,
            device=self.device,
            dtype=torch.bfloat16,
        )

    def reset_rollout(self) -> None:
        self.rollout_chunk_idx = 0
        self.active_chunk_idx = 0
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.current_step_idx = -1
        self.u_step_pending = None

    def _swap_k_for_chunk(self, chunk_idx: int) -> None:
        c_eff = min(max(int(chunk_idx), 0), self.max_chunks - 1)
        self.active_chunk_idx = c_eff
        self.lqr["K_intra"].copy_(self.K_intra_per_chunk[c_eff].to(device=self.device, dtype=torch.float32))
        if self.K_step_per_chunk.numel() > 0:
            self.lqr["K_step"].copy_(self.K_step_per_chunk[c_eff].to(device=self.device, dtype=torch.float32))

    def on_chunk_start(self, chunk_id: int) -> None:
        del chunk_id
        self._swap_k_for_chunk(self.rollout_chunk_idx)
        self.rollout_chunk_idx += 1
        self.video_step_idx = 0
        self.action_step_idx = 0
        self.current_step_idx = -1
        self.u_step_pending = None

    def begin_call(self, action_mode: bool, update_cache: int) -> None:
        del update_cache
        self.current_action_mode = bool(action_mode)
        if self.current_action_mode:
            self.current_step_idx = self.action_step_idx
            self.action_step_idx += 1
        else:
            self.current_step_idx = self.video_step_idx
            self.video_step_idx += 1

    def end_call(self) -> None:
        pass

    def _mode_allow(self, action_mode: bool) -> bool:
        if self.inject_mode == "both":
            return True
        if self.inject_mode == "action":
            return bool(action_mode)
        return not bool(action_mode)

    def _selected(self) -> Tuple[Optional[int], Optional[int]]:
        if self.current_step_idx < 0:
            return None, None
        if not self._mode_allow(self.current_action_mode):
            return None, None
        step = self.current_step_idx
        sel = self.sel_idx_of.get(step)
        if sel is None:
            return None, None
        return step, sel

    def register_hooks(self, transformer_model: torch.nn.Module) -> None:
        blocks = transformer_model.blocks
        if len(blocks) != self.L:
            raise RuntimeError(f"Model block count mismatch: model={len(blocks)}, svd L={self.L}")
        self.vcache.preload([(p, t) for p in range(len(self.partitions)) for t in self.sel_t])
        self._hook_handles.append(blocks[0].register_forward_hook(self._cross_step_apply_hook()))
        for l_in in range(self.L - 1):
            self._hook_handles.append(blocks[l_in + 1].register_forward_hook(self._intra_hook(l_in)))
        self._hook_handles.append(blocks[self.L - 1].register_forward_hook(self._cross_step_compute_hook()))

    def _intra_hook(self, l_in: int):
        def hook(_block, _args, output):
            if self.in_ad:
                return output
            step, sel = self._selected()
            if step is None:
                return output
            z = output[0].detach().reshape(-1)
            z_dtype = output.dtype
            v_in = self.vcache.for_layer(l_in, step)
            if z.numel() != v_in.shape[0]:
                warn_key = (l_in, step, int(self.current_action_mode))
                if warn_key not in self._shape_warned:
                    self._shape_warned.add(warn_key)
                    print(
                        f"[lqr][skip] shape mismatch at layer={l_in} step={step} "
                        f"action_mode={self.current_action_mode}: z={z.numel()} v_in={v_in.shape[0]}"
                    )
                return output
            self.in_ad = True
            try:
                x_proj = (z.to(v_in.dtype) @ v_in).float()
                v_fp = self.lqr["v"][l_in, sel]
                mu_fp = self.lqr["mu"][l_in, sel]
                K_fp = self.lqr["K_intra"][sel, l_in]
                alpha = self.lambda_scale * mu_fp - v_fp @ x_proj
                u_tilde = K_fp @ (alpha * v_fp)
                self.u_norm_log.append((sel, l_in, float(u_tilde.norm()), float(alpha)))
            finally:
                self.in_ad = False
            v_out = self.vcache.for_layer(l_in + 1, step)
            self.in_ad = True
            try:
                u_full = v_out @ u_tilde.to(v_out.dtype)
            finally:
                self.in_ad = False
            delta = u_full.to(z_dtype).reshape_as(output[0])
            out = output.clone()
            out[0] = out[0] + delta
            return out

        return hook

    def _cross_step_compute_hook(self):
        def hook(_block, _args, output):
            if self.in_ad:
                return output
            step, sel = self._selected()
            if step is None or sel >= self.T_diff - 1:
                return output
            z = output[0].detach().reshape(-1)
            v_in = self.vcache.for_layer(self.L - 1, step)
            if z.numel() != v_in.shape[0]:
                warn_key = (self.L - 1, step, int(self.current_action_mode))
                if warn_key not in self._shape_warned:
                    self._shape_warned.add(warn_key)
                    print(
                        f"[lqr][skip] shape mismatch at layer={self.L - 1} step={step} "
                        f"action_mode={self.current_action_mode}: z={z.numel()} v_in={v_in.shape[0]}"
                    )
                return output
            self.in_ad = True
            try:
                x_proj = (z.to(v_in.dtype) @ v_in).float()
                v_fp = self.lqr["v"][self.L - 1, sel]
                mu_fp = self.lqr["mu"][self.L - 1, sel]
                K_fp = self.lqr["K_step"][sel]
                alpha = self.lambda_scale * mu_fp - v_fp @ x_proj
                u_tilde = K_fp @ (alpha * v_fp)
                self.u_norm_log.append((sel, -1, float(u_tilde.norm()), float(alpha)))
                self.u_step_pending = {"src_sel": torch.tensor(sel), "u_tilde": u_tilde.detach()}
            finally:
                self.in_ad = False
            return output

        return hook

    def _cross_step_apply_hook(self):
        def hook(_block, _args, output):
            if self.in_ad:
                return output
            step, sel = self._selected()
            pending = self.u_step_pending
            if step is None or pending is None or sel == 0:
                return output
            src_sel = int(pending["src_sel"].item())
            if src_sel != sel - 1:
                return output
            v_dest = self.vcache.for_layer(0, step)
            self.in_ad = True
            try:
                u_full = v_dest @ pending["u_tilde"].to(v_dest.dtype)
            finally:
                self.in_ad = False
            self.u_step_pending = None
            delta = u_full.to(output.dtype).reshape_as(output[0])
            out = output.clone()
            out[0] = out[0] + delta
            return out

        return hook

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
