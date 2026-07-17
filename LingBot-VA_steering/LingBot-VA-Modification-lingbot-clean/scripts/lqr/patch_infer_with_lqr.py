import argparse
import os

import torch

from scripts.lqr.common import default_slurm_port
from scripts.lqr.lqr_injector import LQRInjector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lingbot server with LQR activation steering.")
    parser.add_argument("--config-name", type=str, default="libero")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--svd-dir", type=str, required=True)
    parser.add_argument("--jac-dir-act", type=str, default="A_tilde_lingbot")
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--q-scale", type=float, default=10000.0)
    parser.add_argument("--r-scale", type=float, default=75000.0)
    parser.add_argument("--r-scale-tau", type=float, default=3.0)
    parser.add_argument("--r-scale-final", type=float, default=1e9)
    parser.add_argument("--max-chunks", type=int, default=50)
    parser.add_argument("--qf-scale", type=float, default=1.0)
    parser.add_argument("--inject-mode", type=str, choices=["auto", "action", "video", "both"], default="auto")
    return parser.parse_args()


def _build_injector(args: argparse.Namespace) -> LQRInjector:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return LQRInjector(
        svd_dir=args.svd_dir,
        jac_dir_act=args.jac_dir_act,
        lambda_scale=args.lambda_scale,
        q_scale=args.q_scale,
        r_scale=args.r_scale,
        r_scale_tau=args.r_scale_tau,
        r_scale_final=args.r_scale_final,
        max_chunks=args.max_chunks,
        qf_scale=args.qf_scale,
        inject_mode=args.inject_mode,
        device=device,
    )


def main() -> None:
    args = _parse_args()
    if args.port is None:
        args.port = default_slurm_port()
    injector = _build_injector(args)
    os.environ.setdefault("PYTHONPATH", ".")

    import wan_va.wan_va_server as base_server

    original_init = base_server.VA_Server.__init__
    original_reset = base_server.VA_Server._reset
    original_infer_chunk = base_server.VA_Server._infer

    def patched_init(self, job_config):
        original_init(self, job_config)
        transformer = self.transformer
        injector.register_hooks(transformer)
        original_forward = transformer.forward

        def patched_forward(*f_args, **f_kwargs):
            action_mode = bool(f_kwargs.get("action_mode", False))
            update_cache = int(f_kwargs.get("update_cache", 0))
            injector.begin_call(action_mode=action_mode, update_cache=update_cache)
            try:
                return original_forward(*f_args, **f_kwargs)
            finally:
                injector.end_call()

        transformer.forward = patched_forward

    def patched_reset(self, prompt=None):
        injector.reset_rollout()
        return original_reset(self, prompt=prompt)

    def patched_infer(self, obs, frame_st_id=0):
        injector.on_chunk_start(chunk_id=int(frame_st_id))
        return original_infer_chunk(self, obs, frame_st_id=frame_st_id)

    base_server.VA_Server.__init__ = patched_init
    base_server.VA_Server._reset = patched_reset
    base_server.VA_Server._infer = patched_infer
    try:
        base_server.run(args)
    finally:
        injector.close()


if __name__ == "__main__":
    main()
