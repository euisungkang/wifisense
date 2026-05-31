#!/usr/bin/env python
"""Zero-shot cross-dataset evaluation for CSI activity classifiers.

Takes a model trained on one dataset (the *source*) and runs it, with no
fine-tuning, on another dataset's test split (the *target*). Because the two
label sets only partially overlap (see ``notes/class_mapping.md``), the metric
is computed **only over target test samples whose true class also exists in the
source** — the shared classes ``{fall, run, walk}``. The model may still
predict a non-shared source class for those inputs; such predictions count as
wrong and appear in the confusion matrix's off-diagonal, which is the
domain-shift signal we want to surface.

Both datasets must already be in the common ``(250, 90)`` representation
(UT-HAR native; NTU-Fi via ``scripts/preprocess_ntu_fi.py``) so a single
``input_size=90`` model runs on either.

Outputs (``--tag`` distinguishes the two directions):
    figures/cross_{tag}_confusion.png   — shared-true x all-source-pred heatmap
    figures/cross_{tag}_metrics.json    — accuracy, per-class, confusion matrix

Examples (from the repo root, with the project env active)::

    conda activate wifisense

    # UT-HAR-trained model, tested on NTU-Fi
    python scripts/cross_dataset_eval.py \
        --checkpoint runs/best_bilstm.pt \
        --data data/processed/ntu_fi/ntu_fi.npz \
        --tag uthar_on_ntu

    # NTU-Fi-trained model, tested on UT-HAR
    python scripts/cross_dataset_eval.py \
        --checkpoint runs/best_bilstm_ntu.pt \
        --data data/processed/ut_har/ut_har.npz \
        --tag ntu_on_uthar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.data.loader import UT_HAR_CLASSES
from src.models import build_model

FIGURES_DIR = ROOT / "figures"


def load_target_classes(npz) -> list[str]:
    """Class names for a processed .npz, falling back to UT-HAR's order."""
    if "class_names" in npz.files:
        return [str(c) for c in npz["class_names"]]
    return list(UT_HAR_CLASSES)


@torch.no_grad()
def _predict(model, X: torch.Tensor, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size].to(device)
        out.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def cross_eval(checkpoint: Path, data: Path, split: str, device: torch.device) -> dict:
    """Run zero-shot cross-domain inference and assemble the metrics dict.

    Returns a dict with source/target/shared class lists, the per-shared-class
    confusion matrix (rows = shared true classes, cols = all source classes),
    overall and per-class accuracy, and the share of predictions that leaked to
    non-shared source classes.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    source_classes = list(ckpt["class_names"])

    npz = np.load(data, allow_pickle=True)
    target_classes = load_target_classes(npz)
    X = torch.from_numpy(npz[f"X_{split}"]).float()
    y_target = npz[f"y_{split}"].astype(int)

    shared = [c for c in source_classes if c in target_classes]

    # Keep only target test samples whose true class is shared.
    target_idx_to_name = {i: c for i, c in enumerate(target_classes)}
    keep = np.array([target_idx_to_name[t] in shared for t in y_target])
    X_keep = X[keep]
    true_names = np.array([target_idx_to_name[t] for t in y_target[keep]])

    probs = _predict(model, X_keep, device)
    pred_src_idx = probs.argmax(axis=1)
    pred_names = np.array([source_classes[i] for i in pred_src_idx])

    correct = pred_names == true_names
    accuracy = float(correct.mean()) if len(correct) else float("nan")

    per_class_acc = {}
    for c in shared:
        m = true_names == c
        per_class_acc[c] = float(correct[m].mean()) if m.any() else float("nan")

    # Confusion: rows = shared true classes, cols = ALL source classes.
    cm = np.zeros((len(shared), len(source_classes)), dtype=int)
    row_of = {c: r for r, c in enumerate(shared)}
    col_of = {c: i for i, c in enumerate(source_classes)}
    for tn, pn in zip(true_names, pred_names):
        cm[row_of[tn], col_of[pn]] += 1

    # Fraction of shared-input predictions that landed on a non-shared class.
    leaked = float(np.mean([pn not in shared for pn in pred_names])) if len(pred_names) else 0.0

    return {
        "checkpoint": str(checkpoint),
        "data": str(data),
        "split": split,
        "source_classes": source_classes,
        "target_classes": target_classes,
        "shared_classes": shared,
        "n_eval": int(keep.sum()),
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "leak_to_nonshared_frac": leaked,
        "confusion_rows": shared,
        "confusion_cols": source_classes,
        "confusion_matrix": cm.tolist(),
        "chance": 1.0 / len(shared) if shared else float("nan"),
    }


def plot_confusion(result: dict, title: str, path: Path) -> None:
    """Heatmap of shared-true (rows) x all-source-pred (cols), row-normalized."""
    cm = np.array(result["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    annot = np.array(
        [[f"{int(c)}\n{p:.0%}" for c, p in zip(crow, prow)] for crow, prow in zip(cm, norm)]
    )
    fig, ax = plt.subplots(figsize=(1.4 * len(result["confusion_cols"]) + 1.5, 3.6))
    sns.heatmap(
        norm,
        annot=annot,
        fmt="",
        cmap="Reds",
        vmin=0,
        vmax=1,
        xticklabels=result["confusion_cols"],
        yticklabels=result["confusion_rows"],
        cbar_kws={"label": "row-normalized"},
        ax=ax,
    )
    ax.set_xlabel("Predicted (source classes)")
    ax.set_ylabel("True (shared)")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="Source-trained best.pt.")
    p.add_argument("--data", type=Path, required=True, help="Target processed .npz.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--tag", required=True, help="Label for output filenames (e.g. uthar_on_ntu).")
    p.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
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

    res = cross_eval(args.checkpoint, args.data, args.split, device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target:     {args.data}  (split={args.split})")
    print(f"Source classes: {res['source_classes']}")
    print(f"Target classes: {res['target_classes']}")
    print(f"Shared classes: {res['shared_classes']}  (chance {res['chance']:.3f})")
    print(f"Eval samples (shared-true only): {res['n_eval']}")
    print(f"\nOverall zero-shot accuracy: {res['accuracy']:.4f}")
    for c, a in res["per_class_accuracy"].items():
        print(f"  {c:>10}: {a:.4f}")
    print(f"Predictions leaking to non-shared classes: {res['leak_to_nonshared_frac']:.1%}")

    fig_path = args.out_dir / f"cross_{args.tag}_confusion.png"
    plot_confusion(res, f"Zero-shot: {args.tag}", fig_path)
    print(f"\nConfusion matrix -> {fig_path}")

    json_path = args.out_dir / f"cross_{args.tag}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Metrics JSON     -> {json_path}")


if __name__ == "__main__":
    main()
