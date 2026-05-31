#!/usr/bin/env python
"""Compare temporal post-processing strategies on the continuous capture.

PURPOSE
    Chunk 6 visualized *raw* sliding-window softmax; chunk 7 showed the low
    continuous-capture accuracy is dominated by boundary windows that straddle
    two activities (``docs/diagnostics.md``). This script adds the
    post-processing layer deferred in chunk 6 and quantifies what each strategy
    buys. It runs three smoothers from ``src.inference.postprocess`` over the
    capture's per-window probabilities and scores them three ways:

        * Per-window accuracy   — before (raw argmax) and after each method,
          vs. ground truth at each window center (the headline chunk-6 metric).
        * In-segment accuracy   — using chunk 7's boundary-exclusion logic
          (windows whose full span lies inside one activity), so genuine
          mid-activity behaviour is separated from boundary churn.
        * Transition rate       — class flips per 100 windows; lower = smoother.

    It also re-renders the milestone figure with a second probability panel for
    the best method (deliverable 3), so raw vs. smoothed are directly
    comparable in the same colors as the chunk-6 figure.

USAGE (from the repo root, with the project env active)
    conda activate wifisense
    python scripts/compare_postprocessing.py
    python scripts/compare_postprocessing.py --k 5 --window-size 250 --stride 25

    Runnable directly (not via ``-m``): it prepends the repo root to sys.path.

INPUTS
    --capture     data/continuous/synthetic_capture.npz   (stream + ground truth)
    --checkpoint  runs/best_bilstm.pt                      (frozen BiLSTM)
    --data        data/processed/ut_har/ut_har.npz         (training labels for
                  the HMM transition matrix)

OUTPUTS
    notes/postprocessing.md                       — the comparison table.
    figures/final_visualization_smoothed.png      — 4-panel figure: CSI
                  heatmap / raw probabilities / best-method probabilities /
                  ground truth.

CAVEAT
    HMM/transition priors and smoothing help the *boundary* windows, but they
    can also paper over genuine model errors (a wrong-but-persistent prediction
    is smoothed into a longer wrong run). See docs/chunk8_postprocessing.md.
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

from scripts.diagnose_accuracy import window_segment_membership  # chunk-7 logic
from src.inference.postprocess import (
    hmm_decode,
    labels_to_onehot,
    learn_transition_matrix,
    majority_vote,
    moving_average,
    moving_average_probs,
    select_best_method,
    transition_rate,
)
from src.inference.streaming import sliding_window_predict
from src.models import build_model
from src.viz import plot_amplitude_heatmap

DEFAULT_CAPTURE = ROOT / "data" / "continuous" / "synthetic_capture.npz"
DEFAULT_CKPT = ROOT / "runs" / "best_bilstm.pt"
DEFAULT_DATA = ROOT / "data" / "processed" / "ut_har" / "ut_har.npz"
DEFAULT_NOTES = ROOT / "notes" / "postprocessing.md"
DEFAULT_FIG = ROOT / "figures" / "final_visualization_smoothed.png"


def load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """Rebuild the trained model from a checkpoint (same recipe as evaluate.py)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, list(ckpt["class_names"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def score(
    pred: np.ndarray,
    truth_center: np.ndarray,
    in_segment: np.ndarray,
    seg_truth: np.ndarray,
) -> dict[str, float]:
    """Per-window accuracy, in-segment accuracy, and transition rate for one pred."""
    window_acc = float(np.mean(pred == truth_center))
    in_acc = (
        float(np.mean(pred[in_segment] == seg_truth[in_segment]))
        if in_segment.any()
        else float("nan")
    )
    return {
        "window_acc": window_acc,
        "in_segment_acc": in_acc,
        "transition_rate": transition_rate(pred),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown(
    path: Path,
    results: dict[str, dict[str, float]],
    best: str,
    *,
    k: int,
    window_size: int,
    stride: int,
    laplace: float,
    n_windows: int,
    n_in_segment: int,
    transition_matrix: np.ndarray,
    class_names: list[str],
) -> None:
    """Write the comparison table (+ context) to a markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = results["raw"]

    def row(name: str, r: dict[str, float]) -> str:
        d_acc = r["window_acc"] - raw["window_acc"]
        delta = "—" if name == "raw" else f"{d_acc:+.3f}"
        in_acc = "n/a" if np.isnan(r["in_segment_acc"]) else f"{r['in_segment_acc']:.3f}"
        return (
            f"| {name} | {r['window_acc']:.3f} | {delta} | {in_acc} | "
            f"{r['transition_rate']:.1f} |"
        )

    lines = [
        "# Post-processing comparison (chunk 8)",
        "",
        "Temporal smoothing of the per-window classifier output over the stitched "
        "continuous capture. Generated by `scripts/compare_postprocessing.py` — "
        "do not edit by hand.",
        "",
        f"- Capture windows: **{n_windows}** "
        f"(window_size={window_size}, stride={stride}); "
        f"**{n_in_segment}** lie fully inside one activity (in-segment).",
        f"- Smoothing window: **k = {k}** windows "
        f"(~half a segment: {window_size}/{stride} = {window_size // stride} "
        "windows per segment).",
        f"- HMM transition matrix: learned from UT-HAR training labels "
        f"(Laplace alpha={laplace}).",
        "",
        "## Results",
        "",
        "Transition rate = class flips per 100 windows (lower = smoother). "
        "In-segment accuracy uses chunk 7's boundary-exclusion logic.",
        "",
        "| Method | Per-window acc | Δ vs raw | In-segment acc | Transition rate |",
        "|---|---|---|---|---|",
        row("raw", results["raw"]),
        row(f"moving_average (k={k})", results["moving_average"]),
        row(f"majority_vote (k={k})", results["majority_vote"]),
        row("hmm_decode (Viterbi)", results["hmm"]),
        "",
        f"**Best method (per-window acc, ties → smoother): `{best}`.**",
        "",
        "## Learned transition matrix",
        "",
        "Row `i`, column `j` = P(next window = `j` | current = `i`). The strong "
        "diagonal is the realistic prior (activities persist across windows); "
        "see the caveat in `docs/chunk8_postprocessing.md` about how it is "
        "estimated from isolated training clips.",
        "",
        "| from \\ to | " + " | ".join(class_names) + " |",
        "|" + "---|" * (len(class_names) + 1),
    ]
    for i, name in enumerate(class_names):
        cells = " | ".join(f"{transition_matrix[i, j]:.3f}" for j in range(len(class_names)))
        lines.append(f"| {name} | {cells} |")
    lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Figure (deliverable 3): raw vs. best-method probability panels
# ---------------------------------------------------------------------------


def render_figure(
    path: Path,
    *,
    stream: np.ndarray,
    labels_per_step: np.ndarray,
    boundaries: np.ndarray,
    timestamps: np.ndarray,
    raw_probs: np.ndarray,
    best_probs: np.ndarray,
    best_name: str,
    raw_acc: float,
    best_acc: float,
    class_names: list[str],
    dpi: int,
) -> None:
    """4-panel figure mirroring chunk 6, with raw and smoothed probability rows.

    Panels (top→bottom, shared x): CSI amplitude heatmap, raw sliding-window
    probabilities, best post-processed probabilities, ground-truth activity bar.
    Colors are the chunk-6 ``tab10`` palette so same color == same class across
    this and ``figures/final_visualization.png``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    t_total = stream.shape[0]
    n_classes = len(class_names)

    base = plt.get_cmap("tab10").colors
    palette = [base[i % len(base)] for i in range(n_classes)]
    listed = ListedColormap(palette)

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(
        4, 2,
        width_ratios=[40, 1],
        height_ratios=[1.25, 1.15, 1.15, 0.32],
        hspace=0.16, wspace=0.02,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_raw = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_smooth = fig.add_subplot(gs[2, 0], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[3, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[0, 1])

    # Top: CSI amplitude heatmap (same primitive as the rest of the project).
    plot_amplitude_heatmap(stream, ax=ax_top, cmap="viridis")
    ax_top.set_xlabel("")
    ax_top.set_title("Continuous CSI capture — amplitude", fontsize=11, loc="left")
    fig.colorbar(ax_top.images[-1], cax=cax, label="amplitude")

    # Middle-upper: raw probabilities.
    ax_raw.stackplot(timestamps, raw_probs.T, colors=palette, labels=class_names,
                     edgecolor="none")
    ax_raw.set_ylim(0, 1)
    ax_raw.set_xlim(0, t_total)
    ax_raw.set_ylabel("P(class)")
    ax_raw.set_title(
        f"Raw sliding-window softmax (no smoothing)  —  per-window acc {raw_acc:.0%}",
        fontsize=11, loc="left",
    )

    # Middle-lower: best post-processed result.
    ax_smooth.stackplot(timestamps, best_probs.T, colors=palette, labels=class_names,
                        edgecolor="none")
    ax_smooth.set_ylim(0, 1)
    ax_smooth.set_xlim(0, t_total)
    ax_smooth.set_ylabel("P(class)")
    ax_smooth.set_title(
        f"Post-processed: {best_name}  —  per-window acc {best_acc:.0%}",
        fontsize=11, loc="left",
    )

    # Bottom: ground-truth activity bar.
    ax_bot.imshow(
        labels_per_step[np.newaxis, :], aspect="auto", origin="lower",
        cmap=listed, vmin=0, vmax=n_classes - 1,
        extent=[0, t_total, 0, 1], interpolation="nearest",
    )
    ax_bot.set_yticks([])
    ax_bot.set_ylabel("Truth", rotation=0, ha="right", va="center")
    ax_bot.set_xlabel("Time (CSI time steps)")

    # Segment boundaries on every panel as alignment cues.
    for b in boundaries[1:-1]:
        for ax, color in (
            (ax_top, "white"), (ax_raw, "white"), (ax_smooth, "white"), (ax_bot, "black"),
        ):
            ax.axvline(b, color=color, lw=0.8, alpha=0.6, ls="--")

    for ax in (ax_top, ax_raw, ax_smooth):
        plt.setp(ax.get_xticklabels(), visible=False)

    handles = [Patch(facecolor=palette[i], label=class_names[i]) for i in range(n_classes)]
    fig.legend(
        handles=handles, loc="lower center", ncol=n_classes,
        bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9, title="Activity",
    )
    fig.suptitle(
        "UT-HAR BiLSTM — raw vs. post-processed sliding-window inference",
        fontsize=13, y=0.965,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="Preprocessed UT-HAR .npz (training labels for the HMM).")
    p.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    p.add_argument("--fig", type=Path, default=DEFAULT_FIG)
    p.add_argument("--window-size", type=int, default=250)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument(
        "--k", type=int, default=None,
        help="Smoothing window in windows. Default: ~half a segment "
             "(round(0.5 * window_size / stride)).",
    )
    p.add_argument("--laplace", type=float, default=1.0,
                   help="Laplace alpha for the HMM transition matrix.")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )

    model, class_names = load_model(args.checkpoint, device)
    n_classes = len(class_names)

    # --- capture + ground truth -------------------------------------------
    cap = np.load(args.capture, allow_pickle=True)
    stream = cap["stream"]
    labels_per_step = cap["labels_per_step"]
    boundaries = cap["boundaries"]
    sample_len = int(cap["sample_len"])
    t_total = stream.shape[0]

    # ~half a segment, in *window* units, is a defensible default (a segment is
    # window_size/stride windows wide). Chosen once; not tuned per the brief.
    seg_windows = max(1, args.window_size // args.stride)
    k = args.k if args.k is not None else max(1, round(0.5 * args.window_size / args.stride))

    # --- raw per-window inference -----------------------------------------
    timestamps, probs = sliding_window_predict(
        model, stream, window_size=args.window_size, stride=args.stride, device=device
    )
    raw_pred = probs.argmax(axis=1)
    starts = np.rint(timestamps - args.window_size / 2.0).astype(int)
    truth_center = labels_per_step[np.clip(timestamps.astype(int), 0, t_total - 1)]
    in_segment, _ = window_segment_membership(starts, args.window_size, boundaries)
    seg_truth = labels_per_step[np.clip(starts, 0, t_total - 1)]

    # --- HMM transition matrix from training labels ------------------------
    # Count adjacent transitions in the UT-HAR *training label sequence*,
    # Laplace-smoothed so no transition is impossible. IMPORTANT CAVEAT
    # (docs/chunk8_postprocessing.md): UT-HAR training samples are isolated,
    # single-activity clips stored grouped by class, so adjacent training
    # labels are almost always identical -> the learned matrix is near-identity
    # (self-transition ~0.998). It encodes a very strong "activities never
    # switch" prior, which over-smooths this deliberately short-segment capture.
    npz = np.load(args.data)
    y_train = npz["y_train"].astype(int)
    A = learn_transition_matrix(y_train, n_classes, alpha=args.laplace)

    # --- run the three strategies -----------------------------------------
    ma_pred = moving_average(probs, k)
    mv_pred = majority_vote(raw_pred, k)
    hmm_pred = hmm_decode(probs, A)

    preds = {"raw": raw_pred, "moving_average": ma_pred,
             "majority_vote": mv_pred, "hmm": hmm_pred}
    results = {name: score(p, truth_center, in_segment, seg_truth)
               for name, p in preds.items()}

    smoothing = {n: {"window_acc": results[n]["window_acc"],
                     "transition_rate": results[n]["transition_rate"]}
                 for n in ("moving_average", "majority_vote", "hmm")}
    best = select_best_method(smoothing)

    # --- console report ----------------------------------------------------
    print("=" * 72)
    print("POST-PROCESSING COMPARISON")
    print("=" * 72)
    print(f"  window_size={args.window_size}  stride={args.stride}  "
          f"segment_len={sample_len}  (~{seg_windows} windows/segment)")
    print(f"  smoothing k={k} windows   HMM Laplace alpha={args.laplace}")
    print(f"  windows: {len(timestamps)} total, {int(in_segment.sum())} in-segment\n")
    print(f"  {'method':<22} {'win_acc':>8} {'in_seg':>8} {'flips/100':>10}")
    for name in ("raw", "moving_average", "majority_vote", "hmm"):
        r = results[name]
        in_acc = "  n/a" if np.isnan(r["in_segment_acc"]) else f"{r['in_segment_acc']:.3f}"
        print(f"  {name:<22} {r['window_acc']:>8.3f} {in_acc:>8} "
              f"{r['transition_rate']:>10.1f}")
    print(f"\n  best smoothing method: {best}")

    # --- markdown table ----------------------------------------------------
    write_markdown(
        args.notes, results, best,
        k=k, window_size=args.window_size, stride=args.stride, laplace=args.laplace,
        n_windows=len(timestamps), n_in_segment=int(in_segment.sum()),
        transition_matrix=A, class_names=class_names,
    )

    # --- figure: raw vs. best method --------------------------------------
    # Soft methods keep their smoothed bands; hard methods (mode/HMM) are shown
    # as one-hot blocks so the panel format matches the raw stackplot.
    if best == "moving_average":
        best_probs = moving_average_probs(probs, k)
        best_label = f"moving average (k={k}), argmax"
    elif best == "majority_vote":
        best_probs = labels_to_onehot(mv_pred, n_classes)
        best_label = f"majority vote (k={k}), hard labels"
    else:
        best_probs = labels_to_onehot(hmm_pred, n_classes)
        best_label = "HMM Viterbi, hard labels"

    render_figure(
        args.fig,
        stream=stream, labels_per_step=labels_per_step, boundaries=boundaries,
        timestamps=timestamps, raw_probs=probs, best_probs=best_probs,
        best_name=best_label, raw_acc=results["raw"]["window_acc"],
        best_acc=results[best]["window_acc"], class_names=class_names, dpi=args.dpi,
    )

    print(f"\nWrote:\n  {args.notes}\n  {args.fig}")


if __name__ == "__main__":
    main()
