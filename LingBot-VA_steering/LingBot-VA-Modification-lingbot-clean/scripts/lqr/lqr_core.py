import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch


def _solve_with_jitter(P: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    n = P.shape[-1]
    P_work = 0.5 * (P + P.transpose(-2, -1))
    P_work = torch.nan_to_num(P_work, nan=0.0, posinf=1e12, neginf=-1e12)
    F_work = torch.nan_to_num(F, nan=0.0, posinf=1e12, neginf=-1e12)
    eye = torch.eye(n, dtype=P_work.dtype, device=P_work.device)
    scale = torch.linalg.matrix_norm(P_work.detach(), ord=float("inf"))
    if not torch.isfinite(scale):
        scale = torch.ones((), dtype=P_work.dtype, device=P_work.device)
    scale = scale.clamp_min(1.0)

    for eps in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        P_reg = P_work + (eps * scale) * eye
        solution, info = torch.linalg.solve_ex(P_reg, F_work)
        if int(info.max().item()) == 0 and torch.isfinite(solution).all():
            return solution

    # If the local linearization is too ill-conditioned even after heavy
    # regularization, skip this local LQR correction instead of aborting eval.
    return torch.zeros_like(F_work)


def _finite_symmetric(mat: torch.Tensor) -> torch.Tensor:
    mat = torch.nan_to_num(mat, nan=0.0, posinf=1e12, neginf=-1e12)
    mat = mat.clamp(min=-1e12, max=1e12)
    return 0.5 * (mat + mat.transpose(-2, -1))


class VCache:
    def __init__(
        self,
        svd_dir: str,
        partitions: List[Tuple[int, int]],
        layer_to_part: List[int],
        sel_t: Iterable[int],
        k_target: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        max_gpu_tiles: Optional[int] = None,
    ) -> None:
        self.svd_dir = Path(svd_dir)
        self.partitions = partitions
        self.layer_to_part = layer_to_part
        self.sel_t = list(sel_t)
        self.k_target = int(k_target)
        self.device = device
        self.dtype = dtype
        self.max_gpu = max_gpu_tiles or (len(partitions) * max(1, len(self.sel_t)))
        self._cpu: Dict[Tuple[int, int], torch.Tensor] = {}
        self._gpu: "OrderedDict[Tuple[int, int], torch.Tensor]" = OrderedDict()
        self.stats = {"cpu_loads": 0, "gpu_swaps": 0, "gpu_hits": 0, "cpu_to_gpu_s": 0.0}

    def _vfile(self, p_idx: int, t_id: int) -> Path:
        a, b = self.partitions[p_idx]
        pattern = f"V_part{p_idx}_layers{a}-{b}_t{t_id}_k*.pt"
        matches = sorted(self.svd_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no V file matching {pattern} in {self.svd_dir}")
        preferred = [m for m in matches if m.name.endswith(f"_k{self.k_target}.pt")]
        return (preferred or matches)[0]

    def _load_cpu(self, p_idx: int, t_id: int) -> torch.Tensor:
        key = (p_idx, t_id)
        cur = self._cpu.get(key)
        if cur is not None:
            return cur
        fp = self._vfile(p_idx, t_id)
        raw = torch.load(fp, map_location="cpu", weights_only=False)["V"]
        v = raw.to(dtype=self.dtype).contiguous()
        self._cpu[key] = v
        self.stats["cpu_loads"] += 1
        return v

    def preload(self, partition_t_pairs: Iterable[Tuple[int, int]]) -> None:
        for p_idx, t_id in partition_t_pairs:
            self._load_cpu(p_idx, t_id)

    def for_layer(self, layer_idx: int, t_id: int) -> torch.Tensor:
        p_idx = self.layer_to_part[layer_idx]
        key = (p_idx, t_id)
        cur = self._gpu.get(key)
        if cur is not None:
            self._gpu.move_to_end(key)
            self.stats["gpu_hits"] += 1
            return cur
        cpu_v = self._load_cpu(p_idx, t_id)
        if self.device.type == "cpu":
            self._gpu[key] = cpu_v
            return cpu_v
        while len(self._gpu) >= self.max_gpu:
            _, old = self._gpu.popitem(last=False)
            del old
            torch.cuda.empty_cache()
            self.stats["gpu_swaps"] += 1
        t0 = time.time()
        v = cpu_v.to(device=self.device, non_blocking=False).contiguous()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stats["cpu_to_gpu_s"] += time.time() - t0
        self._gpu[key] = v
        return v


def chained_riccati(
    A_tilde: torch.Tensor,
    B_tilde: torch.Tensor,
    q_scale: float,
    r_scale: float,
    qf_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # A_tilde: [T_diff, L-1, r, r], B_tilde: [T_diff-1, r, r]
    T_diff, L_minus1, r, _ = A_tilde.shape
    L = L_minus1 + 1
    K_total = T_diff * L - 1 if T_diff > 0 else 0
    if K_total <= 0:
        return (
            torch.zeros(T_diff, L - 1, r, r, dtype=torch.float32),
            torch.zeros(max(T_diff - 1, 0), r, r, dtype=torch.float32),
        )

    _dtype = torch.float64
    I_r = torch.eye(r, dtype=_dtype, device=device)
    Q_chain = (float(q_scale) * I_r).expand(K_total, r, r).contiguous()
    R_chain = (float(r_scale) * I_r).expand(K_total, r, r).contiguous()
    S_T = (float(qf_scale) * I_r).contiguous()

    A_dev = A_tilde.to(device=device, dtype=_dtype)
    B_dev = B_tilde.to(device=device, dtype=_dtype)
    A_chain = torch.zeros(K_total, r, r, dtype=_dtype, device=device)
    for t in range(T_diff):
        for l in range(L - 1):
            A_chain[t * L + l] = A_dev[t, l]
        if t < T_diff - 1 and B_dev.numel() > 0:
            A_chain[t * L + (L - 1)] = B_dev[t]

    Tn = A_chain.shape[0]
    S = torch.zeros(Tn + 1, r, r, dtype=_dtype, device=device)
    K = torch.zeros(Tn, r, r, dtype=_dtype, device=device)
    S[Tn] = S_T
    for k in reversed(range(Tn)):
        Ak = A_chain[k]
        P = S[k + 1] + R_chain[k]
        F = S[k + 1] @ Ak
        G = Q_chain[k] + Ak.transpose(-2, -1) @ S[k + 1] @ Ak
        Kk = _solve_with_jitter(P, F)
        Kk = torch.nan_to_num(Kk, nan=0.0, posinf=1e6, neginf=-1e6).clamp(min=-1e6, max=1e6)
        K[k] = Kk
        Snew = G - F.transpose(-2, -1) @ Kk
        S[k] = _finite_symmetric(Snew)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    K_chain = K.float().cpu()
    K_intra = torch.zeros(T_diff, L - 1, r, r, dtype=torch.float32)
    K_step = torch.zeros(max(T_diff - 1, 0), r, r, dtype=torch.float32)
    for t in range(T_diff):
        for l in range(L - 1):
            K_intra[t, l] = K_chain[t * L + l]
        if t < T_diff - 1:
            K_step[t] = K_chain[t * L + (L - 1)]
    return K_intra, K_step


def chained_riccati_per_chunk(
    A_tilde: torch.Tensor,
    B_tilde: torch.Tensor,
    q_scale: float,
    r_scale_schedule: Iterable[float],
    qf_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Direct repo-root-style variant: one Riccati solve per policy chunk, with
    # the same A/B/Q/Qf and a chunk-dependent scalar R.
    T_diff, L_minus1, r, _ = A_tilde.shape
    L = L_minus1 + 1
    K_total = T_diff * L - 1 if T_diff > 0 else 0
    r_values = [float(x) for x in r_scale_schedule]
    n_chunks = len(r_values)
    if K_total <= 0 or n_chunks <= 0:
        return (
            torch.zeros(n_chunks, T_diff, L - 1, r, r, dtype=torch.float32),
            torch.zeros(n_chunks, max(T_diff - 1, 0), r, r, dtype=torch.float32),
        )

    _dtype = torch.float64
    I_r = torch.eye(r, dtype=_dtype, device=device)
    Q_chain = (float(q_scale) * I_r).expand(K_total, r, r).contiguous()
    S_T = (float(qf_scale) * I_r).contiguous()

    A_dev = A_tilde.to(device=device, dtype=_dtype)
    B_dev = B_tilde.to(device=device, dtype=_dtype)
    A_chain = torch.zeros(K_total, r, r, dtype=_dtype, device=device)
    for t in range(T_diff):
        for l in range(L - 1):
            A_chain[t * L + l] = A_dev[t, l]
        if t < T_diff - 1 and B_dev.numel() > 0:
            A_chain[t * L + (L - 1)] = B_dev[t]

    K_intra_per_chunk = torch.zeros(n_chunks, T_diff, L - 1, r, r, dtype=torch.float32)
    K_step_per_chunk = torch.zeros(n_chunks, max(T_diff - 1, 0), r, r, dtype=torch.float32)

    Tn = A_chain.shape[0]
    for c, r_scale_c in enumerate(r_values):
        R_chain = (r_scale_c * I_r).expand(K_total, r, r).contiguous()
        S = torch.zeros(Tn + 1, r, r, dtype=_dtype, device=device)
        K = torch.zeros(Tn, r, r, dtype=_dtype, device=device)
        S[Tn] = S_T
        for k in reversed(range(Tn)):
            Ak = A_chain[k]
            P = S[k + 1] + R_chain[k]
            F = S[k + 1] @ Ak
            G = Q_chain[k] + Ak.transpose(-2, -1) @ S[k + 1] @ Ak
            Kk = _solve_with_jitter(P, F)
            Kk = torch.nan_to_num(Kk, nan=0.0, posinf=1e6, neginf=-1e6).clamp(min=-1e6, max=1e6)
            K[k] = Kk
            Snew = G - F.transpose(-2, -1) @ Kk
            S[k] = _finite_symmetric(Snew)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        K_chain = K.float().cpu()
        for t in range(T_diff):
            for l in range(L - 1):
                K_intra_per_chunk[c, t, l] = K_chain[t * L + l]
            if t < T_diff - 1:
                K_step_per_chunk[c, t] = K_chain[t * L + (L - 1)]
        del R_chain, S, K, K_chain

    del A_chain, Q_chain, A_dev, B_dev, I_r
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return K_intra_per_chunk, K_step_per_chunk
