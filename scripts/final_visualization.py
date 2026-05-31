#!/usr/bin/env python
"""Milestone figure: continuous CSI capture, model predictions, ground truth.

Loads the stitched capture from ``build_continuous_capture.py`` and the
trained BiLSTM, runs sliding-window inference, and renders three vertically
stacked panels sharing one time axis:

    Top    — CSI amplitude heatmap of the whole capture (subcarrier × time).
    Middle — stacked class-probability area, one band per class, over time.
    Bottom — ground-truth activity as a colored bar.

The middle and bottom panels share one color palette (same color = same
class) and one legend.

This renders **raw model output**: the probability bands are exactly the
softmax the network produced per window — no smoothing, no hysteresis, no
confidence gating.  Where the model is wrong or lags a transition, the
figure shows it.  A per-window accuracy (prediction vs. ground truth at each
window center) is printed so the alignment is auditable, not hidden.

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/final_visualization.py
    python scripts/final_visualization.py --window-size 250 --stride 25

Runnable directly (not via ``-m``): it prepends the repo root to sys.path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.streaming import sliding_window_predict
from src.models import build_model
from src.viz import plot_amplitude_heatmap

DEFAULT_CAPTURE = ROOT / "data" / "continuous" / "synthetic_capture.npz"
DEFAULT_CKPT = ROOT / "runs" / "best_bilstm.pt"
DEFAULT_OUT = ROOT / "figures" / "final_visualization.png"


def load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """Rebuild the trained model from a checkpoint (same recipe as evaluate.py)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, list(ckpt["class_names"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--window-size", type=int, default=250)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )

    # --- load capture + model ---------------------------------------------
    cap = np.load(args.capture, allow_pickle=True)
    stream = cap["stream"]                       # (T_total, S) raw CSI
    labels_per_step = cap["labels_per_step"]     # (T_total,) ground truth
    boundaries = cap["boundaries"]               # (N+1,) segment edges
    capture_classes = list(cap["class_names"])
    t_total, n_sub = stream.shape

    model, class_names = load_model(args.checkpoint, device)
    if class_names != capture_classes:
        raise ValueError(
            "Class-name mismatch between checkpoint and capture:\n"
            f"  checkpoint: {class_names}\n  capture:    {capture_classes}"
        )
    n_classes = len(class_names)

    # --- sliding-window inference (raw model output) -----------------------
    timestamps, probs = sliding_window_predict(
        model, stream, window_size=args.window_size, stride=args.stride, device=device
    )
    pred = probs.argmax(axis=1)

    # Honesty check: prediction vs. ground truth at each window center.
    truth_at_center = labels_per_step[np.clip(timestamps.astype(int), 0, t_total - 1)]
    win_acc = float(np.mean(pred == truth_at_center))
    print(f"Capture: {args.capture.name}  ({t_total} steps, {n_sub} subcarriers)")
    print(f"Windows: {len(timestamps)}  (size={args.window_size}, stride={args.stride})")
    print(f"Per-window accuracy (pred vs. truth at window center): {win_acc:.3f}\n")
    print("  center   truth        predicted      conf   match")
    for tc, tr, pr, pb in zip(timestamps, truth_at_center, pred, probs):
        mark = "✓" if tr == pr else "✗"
        print(
            f"  {tc:7.1f}  {class_names[tr]:<11}  {class_names[pr]:<11}  "
            f"{pb[pr]:.2f}   {mark}"
        )

    # --- consistent color palette (same color == same class) ---------------
    base = plt.get_cmap("tab10").colors
    palette = [base[i % len(base)] for i in range(n_classes)]
    listed = ListedColormap(palette)

    # --- figure layout -----------------------------------------------------
    # A narrow right column carries the heatmap colorbar without stealing
    # width from the lower panels, so all three axes stay x-aligned.
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[40, 1],
        height_ratios=[1.25, 1.25, 0.32],
        hspace=0.14, wspace=0.02,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[0, 1])

    # Top: CSI amplitude heatmap — reuse the viz primitive so this panel is
    # rendered identically to the rest of the project (see docs/visualization.md).
    plot_amplitude_heatmap(stream, ax=ax_top, cmap="viridis")
    ax_top.set_xlabel("")  # x label belongs on the shared bottom panel only
    ax_top.set_title("Continuous CSI capture — amplitude", fontsize=11, loc="left")
    fig.colorbar(ax_top.images[-1], cax=cax, label="amplitude")

    # Middle: stacked class probabilities over time.
    ax_mid.stackplot(
        timestamps, probs.T, colors=palette, labels=class_names, edgecolor="none"
    )
    ax_mid.set_ylim(0, 1)
    ax_mid.set_xlim(0, t_total)
    ax_mid.set_ylabel("P(class)")
    ax_mid.set_title(
        "Predicted class probability (raw sliding-window softmax, no smoothing)",
        fontsize=11, loc="left",
    )

    # Bottom: ground-truth activity bar (imshow over the same palette).
    ax_bot.imshow(
        labels_per_step[np.newaxis, :], aspect="auto", origin="lower",
        cmap=listed, vmin=0, vmax=n_classes - 1,
        extent=[0, t_total, 0, 1], interpolation="nearest",
    )
    ax_bot.set_yticks([])
    ax_bot.set_ylabel("Truth", rotation=0, ha="right", va="center")
    ax_bot.set_xlabel("Time (CSI time steps)")

    # Segment boundaries as alignment cues on every panel.
    for b in boundaries[1:-1]:
        for ax, color in ((ax_top, "white"), (ax_mid, "white"), (ax_bot, "black")):
            ax.axvline(b, color=color, lw=0.8, alpha=0.6, ls="--")

    # Hide x tick labels on the shared upper panels.
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_mid.get_xticklabels(), visible=False)

    # Single shared legend (colors identical to middle + bottom panels).
    handles = [Patch(facecolor=palette[i], label=class_names[i]) for i in range(n_classes)]
    fig.legend(
        handles=handles, loc="lower center", ncol=n_classes,
        bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9,
        title=f"Activity  (per-window accuracy: {win_acc:.0%})",
    )

    fig.suptitle(
        "UT-HAR BiLSTM — sliding-window inference over a stitched continuous capture",
        fontsize=13, y=0.97,
    )
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure → {args.out}  (dpi={args.dpi})")


if __name__ == "__main__":
    main()
