#!/usr/bin/env python
"""Assemble the UT-HAR <-> NTU-Fi domain-shift results figure.

Reads the four metric JSONs produced earlier in chunk 9 and renders a single
figure ``figures/domain_shift_matrix.png`` with three panels:

    1. the 2x2 train-set x test-set accuracy matrix (diagonal = full in-domain
       test accuracy; off-diagonal = zero-shot accuracy on the shared classes
       {fall, run, walk});
    2. the UT-HAR-trained model's confusion matrix on NTU-Fi;
    3. the NTU-Fi-trained model's confusion matrix on UT-HAR.

Inputs (must already exist):
    figures/eval_metrics_test.json            — UT-HAR in-domain (src.evaluate)
    figures/ntu/eval_metrics_test.json        — NTU-Fi in-domain (src.evaluate)
    figures/cross_uthar_on_ntu_metrics.json   — cross_dataset_eval.py
    figures/cross_ntu_on_uthar_metrics.json   — cross_dataset_eval.py

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/domain_shift_matrix.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def draw_matrix(ax, acc: np.ndarray, kind: np.ndarray) -> None:
    """2x2 accuracy heatmap; ``kind`` marks each cell in-domain vs zero-shot."""
    ax.imshow(acc, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    rows = ["UT-HAR-trained", "NTU-Fi-trained"]
    cols = ["tested on\nUT-HAR", "tested on\nNTU-Fi"]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{acc[i, j] * 100:.1f}%\n({kind[i, j]})",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )
    ax.set_xticks([0, 1], cols, fontsize=10)
    ax.set_yticks([0, 1], rows, fontsize=10)
    ax.set_title("Domain-shift matrix\n(diag = in-domain, off-diag = zero-shot)", fontsize=11)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)


def draw_confusion(ax, res: dict, title: str) -> None:
    """Row-normalized shared-true x all-source-pred heatmap."""
    cm = np.array(res["confusion_matrix"], dtype=float)
    rs = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
    im = ax.imshow(norm, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    cols, rows = res["confusion_cols"], res["confusion_rows"]
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}\n{norm[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if norm[i, j] < 0.6 else "white",
            )
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    ax.set_xlabel("Predicted (source classes)", fontsize=9)
    ax.set_ylabel("True (shared)", fontsize=9)
    acc = res["accuracy"] * 100
    ax.set_title(f"{title}\nzero-shot acc {acc:.1f}% (chance {res['chance']:.0%})", fontsize=10)
    return im


def main() -> None:
    uthar_in = load(FIG / "eval_metrics_test.json")["accuracy"]
    ntu_in = load(FIG / "ntu" / "eval_metrics_test.json")["accuracy"]
    u_on_n = load(FIG / "cross_uthar_on_ntu_metrics.json")
    n_on_u = load(FIG / "cross_ntu_on_uthar_metrics.json")

    acc = np.array(
        [
            [uthar_in, u_on_n["accuracy"]],
            [n_on_u["accuracy"], ntu_in],
        ]
    )
    kind = np.array(
        [["in-domain", "zero-shot"], ["zero-shot", "in-domain"]]
    )

    fig = plt.figure(figsize=(17, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.4], wspace=0.45)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    draw_matrix(ax0, acc, kind)
    draw_confusion(ax1, u_on_n, "UT-HAR-trained -> NTU-Fi")
    im = draw_confusion(ax2, n_on_u, "NTU-Fi-trained -> UT-HAR")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="row-normalized")

    fig.suptitle(
        "UT-HAR <-> NTU-Fi domain shift: strong in-domain, collapse across domains",
        fontsize=14,
        y=1.02,
    )
    out = FIG / "domain_shift_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")

    # Echo the headline table to stdout.
    print("\n                 tested on UT-HAR     tested on NTU-Fi")
    print(f"UT-HAR-trained        {acc[0,0]*100:5.1f}% (in)         {acc[0,1]*100:5.1f}% (0-shot)")
    print(f"NTU-Fi-trained        {acc[1,0]*100:5.1f}% (0-shot)     {acc[1,1]*100:5.1f}% (in)")


if __name__ == "__main__":
    main()
