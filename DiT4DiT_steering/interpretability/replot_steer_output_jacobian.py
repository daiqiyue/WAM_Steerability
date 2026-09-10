#!/usr/bin/env python
"""Regenerate per-denoising-step Jacobian heatmaps from a saved result.

This is intentionally model-free: it reads the tensors already stored by
``compute_steer_output_jacobian.py`` and places the shared colorbar in a
dedicated right margin so it cannot overlap the fourth block column.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path, help="saved all-block/all-step .pt file")
    p.add_argument("--steps", default="all",
                   help="comma-separated steps to redraw, or all")
    p.add_argument("--no-pdf", action="store_true",
                   help="only regenerate PNG files")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = torch.load(args.result, map_location="cpu", weights_only=False)
    records = payload["records"]
    if not records:
        raise ValueError(f"no records in {args.result}")
    first = next(iter(records.values()))["d_action_tokens_d_steer_alpha"]
    horizon, output_dims = map(int, first.shape)
    labels = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"][:output_dims]
    available_steps = sorted({int(row["step"]) for row in records.values()})
    steps = available_steps if args.steps == "all" else [
        int(x) for x in args.steps.split(",")
    ]
    if any(step not in available_steps for step in steps):
        raise ValueError(f"requested steps={steps}, available={available_steps}")
    arrays = [row["d_action_tokens_d_steer_alpha"].numpy() for row in records.values()]
    vmax = max(float(np.abs(a).max()) for a in arrays) or 1.0
    pdf_path = args.result.with_name(args.result.stem + "_heatmaps_all.pdf")

    pdf_context = nullcontext(None) if args.no_pdf else PdfPages(pdf_path)
    with pdf_context as pdf:
        for step in steps:
            step_records = sorted(
                ((key, row) for key, row in records.items() if int(row["step"]) == step),
                key=lambda item: int(item[1]["block"]),
            )
            ncols = min(4, len(step_records))
            nrows = math.ceil(len(step_records) / ncols)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.1 * ncols, 3.7 * nrows), squeeze=False
            )
            image = None
            for ax, (_key, row) in zip(axes.ravel(), step_records):
                values = row["d_action_tokens_d_steer_alpha"].numpy()
                image = ax.imshow(
                    values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto"
                )
                ax.set_title(f"block {row['block']}   |Jv|={row['total_l2']:.3g}")
                ax.set_xlabel("action dimension")
                ax.set_ylabel("output token")
                ax.set_xticks(range(output_dims), labels, rotation=40, ha="right")
                ax.set_yticks(range(horizon))
            for ax in axes.ravel()[len(step_records):]:
                ax.set_visible(False)
            fig.suptitle(
                f"Directional output Jacobian — denoising step {step} (all blocks)"
            )
            # Reserve a real margin for the colorbar. Do not pass all axes to
            # fig.colorbar: that combination previously covered column four.
            fig.subplots_adjust(
                left=0.06, right=0.84, bottom=0.08, top=0.92, hspace=0.45, wspace=0.32
            )
            cbar_ax = fig.add_axes([0.875, 0.12, 0.018, 0.74])
            fig.colorbar(image, cax=cbar_ax,
                         label=r"$d\,action / d\,steer\;alpha$")
            step_path = args.result.with_name(args.result.stem + f"_heatmap_step{step}.png")
            fig.savefig(step_path, dpi=180, bbox_inches="tight")
            if pdf is not None:
                pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(step_path)
    if not args.no_pdf:
        print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
