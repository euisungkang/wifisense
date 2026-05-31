#!/usr/bin/env python
"""Diagnose *why* continuous-capture accuracy is low — before any fix.

Chunk 6 reported ~56% per-window accuracy on the stitched continuous capture
(``figures/final_visualization.png``), far below the ~92% the same model
scores on isolated UT-HAR test clips.  This script decomposes that gap into
three candidate sources of error and decides which one dominates:

    (a) Genuine model error   — mistakes the model would make on clean,
        isolated test clips too (measured on the UT-HAR test split).
    (b) Sliding-window boundary effects — windows whose span straddles two
        activities, so the input is a *mixture* the model never saw in
        training.
    (c) Preprocessing edge effects — NaNs / blow-ups / degenerate
        normalization on the first/last windows of the capture.

Method:
    * Clean test:   run the model on the UT-HAR test set with no stitching
      and no sliding window.  Report per-class accuracy + macro F1.
    * In-segment:   re-score the continuous capture but keep only windows
      whose *full span* lies inside a single ground-truth segment.  This is
      the sliding-window pipeline with boundary windows removed.
    * Verdict:      if clean ≈ in-segment ≫ continuous, the gap is purely
      boundary effects (b).  If clean is also low, the issue is upstream (a).
    * Edge check:   inspect the first few windows for NaNs / Infs / extreme
      values after preprocessing, to rule in/out (c).

Outputs (under ``--out-dir``, default ``figures/``):
    * stdout + ``diagnostics_summary.json`` — the three accuracies and the
      verdict.
    * ``per_class_confusion_continuous.png`` — confusion on the continuous
      capture, boundary windows excluded (with an all-windows panel for
      contrast).
    * ``window_position_accuracy.png`` — accuracy vs. the window center's
      offset from the nearest segment boundary (the boundary-effect V-curve).
    * ``lie_down_failure_diagnosis.png`` — CSI amplitude of the failing
      lie_down-start windows beside correctly-classified lie_down test clips.

This is **diagnosis only**: it does not change the model, preprocessing,
training, ``window_size`` or ``stride``.  Findings go in
``docs/diagnostics.md``.

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/diagnose_accuracy.py
    python scripts/diagnose_accuracy.py --window-size 250 --stride 25

Runnable directly (not via ``-m``): it prepends the repo root to sys.path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocess import Pipeline
from src.inference.streaming import sliding_window_predict
from src.models import build_model
from src.viz import plot_amplitude_heatmap

DEFAULT_CAPTURE = ROOT / "data" / "continuous" / "synthetic_capture.npz"
DEFAULT_CKPT = ROOT / "runs" / "best_bilstm.pt"
DEFAULT_DATA = ROOT / "data" / "processed" / "ut_har" / "ut_har.npz"
DEFAULT_OUT = ROOT / "figures"


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------


def load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """Rebuild the trained model from a checkpoint (same recipe as evaluate.py)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, list(ckpt["class_names"])


@torch.no_grad()
def predict_clean(model: torch.nn.Module, X: torch.Tensor, device: torch.device,
                  batch_size: int = 256) -> np.ndarray:
    """Softmax probabilities ``(N, C)`` over already-preprocessed clips X."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(F.softmax(model(X[i : i + batch_size].to(device)), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Window bookkeeping for the continuous capture
# ---------------------------------------------------------------------------


def window_segment_membership(
    starts: np.ndarray, window_size: int, boundaries: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each window by the ground-truth segment(s) its span covers.

    A window covers ``[start, start + window_size)``.  ``boundaries`` are the
    segment edges (``(N+1,)``, monotonically increasing).

    Returns:
        in_segment: bool ``(n_windows,)`` — True iff the full span lies inside
                    a single segment (no boundary crossed).
        seg_label_idx: int ``(n_windows,)`` — the segment index containing the
                    window *start* (used to look up the ground-truth label for
                    in-segment windows).
    """
    seg_start = np.searchsorted(boundaries, starts, side="right") - 1
    seg_end = np.searchsorted(boundaries, starts + window_size - 1, side="right") - 1
    in_segment = seg_start == seg_end
    return in_segment, seg_start


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_confusion(
    y_true_all: np.ndarray, y_pred_all: np.ndarray,
    y_true_in: np.ndarray, y_pred_in: np.ndarray,
    class_names: list[str], path: Path,
) -> None:
    """Two confusion matrices: all windows vs. boundary-windows-excluded.

    Left panel uses *truth at window center* for every window — boundary
    windows smear probability mass off the diagonal.  Right panel keeps only
    windows fully inside one segment; a clean diagonal here means the residual
    error is entirely at the seams, not genuine misclassification.
    """
    labels = list(range(len(class_names)))
    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    cm_in = confusion_matrix(y_true_in, y_pred_in, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, cm, title in (
        (axes[0], cm_all, f"All windows (truth at center)  —  n={len(y_true_all)}"),
        (axes[1], cm_in, f"In-segment only, boundary windows excluded  —  n={len(y_true_in)}"),
    ):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={"label": "windows"}, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle(
        "Continuous-capture confusion — off-diagonal mass lives at the boundaries",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_window_position(
    offsets: np.ndarray, correct: np.ndarray,
    clean_acc: float, continuous_acc: float, half_window: int, path: Path,
) -> None:
    """Accuracy vs. the window center's signed offset from the nearest boundary.

    Offset 0 = window center sits on a segment boundary (maximal straddle).
    Offset ±``half_window`` = window fully inside one segment (no straddle).
    A V-shaped valley at 0 is the signature of boundary effects.
    """
    uniq = np.array(sorted(set(offsets)))
    acc = np.array([correct[offsets == o].mean() for o in uniq])
    n = np.array([(offsets == o).sum() for o in uniq])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axvspan(-half_window * 0.18, half_window * 0.18, color="crimson", alpha=0.07,
               label="window straddles two activities")
    ax.plot(uniq, acc, "-o", color="#1f77b4", lw=2, zorder=3)
    for o, a, c in zip(uniq, acc, n):
        ax.annotate(f"n={c}", (o, a), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="#444")
    ax.axhline(clean_acc, color="green", ls="--", lw=1.2,
               label=f"clean test acc ({clean_acc:.0%})")
    ax.axhline(continuous_acc, color="gray", ls=":", lw=1.2,
               label=f"continuous all-window acc ({continuous_acc:.0%})")
    ax.set_xlabel("Window-center offset from nearest segment boundary (time steps)")
    ax.set_ylabel("Per-window accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        "Boundary-effect V-curve: accuracy collapses as the window straddles a seam",
        fontsize=12,
    )
    ax.legend(loc="upper center", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_lie_down_failure(
    stream: np.ndarray, starts: np.ndarray, pred: np.ndarray, probs: np.ndarray,
    truth_center: np.ndarray, window_size: int, first_boundary: int,
    clean_lie_down: np.ndarray, class_names: list[str], path: Path,
) -> None:
    """Failing lie_down-start windows vs. correctly-classified lie_down clips.

    Both rows are shown in the **preprocessed** representation the model
    actually classifies (the capture windows are pushed through the same
    per-window ``Pipeline`` used in streaming inference; the clean clips are
    already preprocessed in the .npz).  This keeps the amplitude color scale
    comparable across panels — plotting raw stream amplitude beside z-scored
    clips would be apples-to-oranges.

    Top row: the early windows of the capture, with the lie_down→fall segment
    boundary marked where it falls inside the window.
    Bottom row: clean UT-HAR test clips the model labels lie_down correctly.
    """
    # Early windows spanning the first boundary (the reported failure region).
    early = np.where(starts < first_boundary)[0]
    early = early[: min(4, len(early))]
    n_cols = max(len(early), len(clean_lie_down))

    # Preprocess the capture windows exactly as streaming inference does, so
    # the color scale matches the already-preprocessed clean clips.
    pipe = Pipeline()
    early_windows = pipe.transform(
        np.stack([stream[starts[i] : starts[i] + window_size] for i in early])
    )

    stacked = np.concatenate([early_windows.reshape(-1), clean_lie_down.reshape(-1)])
    vmin = float(np.percentile(stacked, 1))
    vmax = float(np.percentile(stacked, 99))

    fig, axes = plt.subplots(2, n_cols, figsize=(3.4 * n_cols, 6.4), squeeze=False)

    for j in range(n_cols):
        # --- top: early capture windows (preprocessed) ---
        ax = axes[0][j]
        if j < len(early):
            i = early[j]
            s = starts[i]
            win = early_windows[j]
            ax.imshow(win.T, aspect="auto", origin="lower", cmap="viridis",
                      vmin=vmin, vmax=vmax, interpolation="nearest")
            # mark the segment boundary inside this window, if present
            bx = first_boundary - s
            if 0 < bx < window_size:
                ax.axvline(bx, color="red", lw=1.6, ls="--")
            mark = "OK" if truth_center[i] == pred[i] else "MISS"
            ax.set_title(
                f"win start={s} ({mark})\ntruth={class_names[truth_center[i]]} "
                f"pred={class_names[pred[i]]} ({probs[i, pred[i]]:.2f})",
                fontsize=8.5,
            )
            ax.set_xlabel("Time step")
            if j == 0:
                ax.set_ylabel("capture window\nSubcarrier")
        else:
            ax.axis("off")

        # --- bottom: clean correctly-classified lie_down clips ---
        ax = axes[1][j]
        if j < len(clean_lie_down):
            ax.imshow(clean_lie_down[j].T, aspect="auto", origin="lower", cmap="viridis",
                      vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title("clean lie_down clip (pred=lie_down)", fontsize=8.5)
            ax.set_xlabel("Time step")
            if j == 0:
                ax.set_ylabel("clean test\nSubcarrier")
        else:
            ax.axis("off")

    fig.suptitle(
        "lie_down → stand_up at capture start is a boundary artifact, not a bad clip\n"
        "(red dashed = lie_down/fall seam inside the window; first in-segment window is correct)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Edge-effect check
# ---------------------------------------------------------------------------


def edge_effect_report(stream: np.ndarray, starts: np.ndarray, window_size: int,
                       n_windows: int = 8) -> list[dict]:
    """Preprocess the first ``n_windows`` and report NaN/Inf/range per window."""
    pipe = Pipeline()
    sel = starts[:n_windows]
    wins = np.stack([stream[s : s + window_size] for s in sel])
    proc = pipe.transform(wins)
    rows = []
    for k, s in enumerate(sel):
        w = proc[k]
        rows.append({
            "start": int(s),
            "has_nan": bool(np.isnan(w).any()),
            "has_inf": bool(np.isinf(w).any()),
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "std": float(w.std()),
        })
    return rows


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
                   help="Preprocessed UT-HAR .npz (clean test set).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--window-size", type=int, default=250)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )

    model, class_names = load_model(args.checkpoint, device)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    # === (1) Clean test set — no stitching, no sliding window ===============
    npz = np.load(args.data)
    X_test = torch.from_numpy(npz["X_test"]).float()
    y_test = npz["y_test"].astype(int)
    clean_probs = predict_clean(model, X_test, device)
    clean_pred = clean_probs.argmax(axis=1)
    clean_acc = accuracy_score(y_test, clean_pred)
    clean_macro_f1 = f1_score(y_test, clean_pred, average="macro", labels=labels,
                              zero_division=0)
    per_class_acc = {}
    print("=" * 70)
    print("(1) CLEAN UT-HAR TEST SET  (isolated clips, no sliding window)")
    print("=" * 70)
    print(f"  overall accuracy : {clean_acc:.4f}")
    print(f"  macro F1         : {clean_macro_f1:.4f}")
    print("  per-class accuracy:")
    for c, name in enumerate(class_names):
        m = y_test == c
        a = float((clean_pred[m] == c).mean()) if m.any() else float("nan")
        per_class_acc[name] = a
        print(f"    {name:<11} {a:.4f}  (n={int(m.sum())})")

    # === (2) Continuous capture — full sliding window + in-segment subset ===
    cap = np.load(args.capture, allow_pickle=True)
    stream = cap["stream"]
    labels_per_step = cap["labels_per_step"]
    boundaries = cap["boundaries"]
    t_total = stream.shape[0]

    timestamps, probs = sliding_window_predict(
        model, stream, window_size=args.window_size, stride=args.stride, device=device
    )
    pred = probs.argmax(axis=1)
    starts = np.rint(timestamps - args.window_size / 2.0).astype(int)
    truth_center = labels_per_step[np.clip(timestamps.astype(int), 0, t_total - 1)]
    continuous_acc = float((pred == truth_center).mean())

    in_segment, seg_idx = window_segment_membership(starts, args.window_size, boundaries)
    # Ground-truth label for an in-segment window is its whole segment's label.
    seg_truth = labels_per_step[np.clip(starts, 0, t_total - 1)]
    in_acc = float((pred[in_segment] == seg_truth[in_segment]).mean())

    print()
    print("=" * 70)
    print("(2) CONTINUOUS CAPTURE  (sliding window)")
    print("=" * 70)
    print(f"  window_size={args.window_size}  stride={args.stride}  "
          f"segment_len={int(cap['sample_len'])}")
    print(f"  total windows         : {len(timestamps)}")
    print(f"  in-segment windows    : {int(in_segment.sum())}  "
          f"(full span inside one activity)")
    print(f"  boundary windows      : {int((~in_segment).sum())}  "
          f"(span straddles a seam)")
    print(f"  continuous accuracy (all windows) : {continuous_acc:.4f}")
    print(f"  in-segment accuracy               : {in_acc:.4f}")

    # === (3) Verdict ========================================================
    print()
    print("=" * 70)
    print("(3) VERDICT")
    print("=" * 70)
    gap_boundary = in_acc - continuous_acc
    gap_upstream = 1.0 - clean_acc
    if clean_acc >= 0.85 and in_acc >= 0.85 and continuous_acc < 0.7:
        verdict = (
            "Boundary effects DOMINATE. Clean test and in-segment accuracy are "
            "both high; the continuous drop is created almost entirely by windows "
            "straddling two activities (the model never trained on mixed windows)."
        )
    elif clean_acc < 0.85:
        verdict = (
            "Issue is UPSTREAM (genuine model error / preprocessing): clean test "
            "accuracy is itself low, so the continuous gap is not just boundaries."
        )
    else:
        verdict = "Mixed: see the three accuracies above; no single source dominates."
    print(f"  clean test acc      : {clean_acc:.3f}")
    print(f"  in-segment acc      : {in_acc:.3f}")
    print(f"  continuous acc      : {continuous_acc:.3f}")
    print(f"  boundary-attributable gap (in_seg - continuous): {gap_boundary:+.3f}")
    print(f"  upstream gap        (1 - clean_test)           : {gap_upstream:+.3f}")
    print(f"  → {verdict}")

    # === (4) Preprocessing edge-effect check (start of capture) =============
    edge_rows = edge_effect_report(stream, starts, args.window_size)
    any_nan = any(r["has_nan"] for r in edge_rows)
    any_inf = any(r["has_inf"] for r in edge_rows)
    print()
    print("=" * 70)
    print("(4) PREPROCESSING EDGE-EFFECT CHECK (first windows of capture)")
    print("=" * 70)
    print(f"  any NaN after preprocessing: {any_nan}    any Inf: {any_inf}")
    print("  start     min     max    mean    std")
    for r in edge_rows:
        print(f"  {r['start']:5d}  {r['min']:6.2f}  {r['max']:6.2f}  "
              f"{r['mean']:6.3f}  {r['std']:6.3f}")

    # === lie_down → stand_up failure focus ==================================
    first_boundary = int(boundaries[1])
    ld_miss = np.where((truth_center == 0) & (pred != 0) & (starts < first_boundary))[0]
    print()
    print("  lie_down-start windows (segment 0):")
    for i in np.where(starts < first_boundary)[0]:
        mk = "OK" if truth_center[i] == pred[i] else "MISS"
        print(f"    start={starts[i]:4d}  pred={class_names[pred[i]]:<11} "
              f"conf={probs[i, pred[i]]:.2f}  [{mk}]")

    # === Figures ============================================================
    fig_confusion(
        truth_center, pred,
        seg_truth[in_segment], pred[in_segment],
        class_names, args.out_dir / "per_class_confusion_continuous.png",
    )
    nearest_boundary = np.rint(timestamps / float(args.window_size)).astype(int) * args.window_size \
        if args.window_size == int(cap["sample_len"]) else \
        boundaries[np.argmin(np.abs(timestamps[:, None] - boundaries[None, :]), axis=1)]
    offsets = (timestamps - nearest_boundary).astype(int)
    fig_window_position(
        offsets, (pred == truth_center).astype(float),
        clean_acc, continuous_acc, args.window_size // 2,
        args.out_dir / "window_position_accuracy.png",
    )
    # Correctly-classified clean lie_down clips for the comparison panel.
    ld_correct = np.where((y_test == 0) & (clean_pred == 0))[0][:4]
    fig_lie_down_failure(
        stream, starts, pred, probs, truth_center, args.window_size, first_boundary,
        npz["X_test"][ld_correct], class_names,
        args.out_dir / "lie_down_failure_diagnosis.png",
    )

    # === Summary JSON =======================================================
    summary = {
        "checkpoint": str(args.checkpoint),
        "window_size": args.window_size,
        "stride": args.stride,
        "segment_len": int(cap["sample_len"]),
        "clean_test_accuracy": clean_acc,
        "clean_test_macro_f1": clean_macro_f1,
        "clean_test_per_class_accuracy": per_class_acc,
        "continuous_accuracy_all_windows": continuous_acc,
        "in_segment_accuracy": in_acc,
        "n_windows_total": int(len(timestamps)),
        "n_windows_in_segment": int(in_segment.sum()),
        "n_windows_boundary": int((~in_segment).sum()),
        "boundary_attributable_gap": gap_boundary,
        "upstream_gap": gap_upstream,
        "edge_check_any_nan": any_nan,
        "edge_check_any_inf": any_inf,
        "verdict": verdict,
    }
    summary_path = args.out_dir / "diagnostics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Wrote:")
    print(f"  {args.out_dir / 'per_class_confusion_continuous.png'}")
    print(f"  {args.out_dir / 'window_position_accuracy.png'}")
    print(f"  {args.out_dir / 'lie_down_failure_diagnosis.png'}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
