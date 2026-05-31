#!/usr/bin/env python
"""Evaluate a trained CSI classifier checkpoint.

Loads a ``best.pt`` written by ``src.train`` (which carries the model name,
config and class names so the model is reconstructed without guessing
hyperparameters), runs it over a split, and produces:

    * stdout + JSON: overall accuracy, macro F1, per-class precision /
      recall / F1 (sklearn ``classification_report``);
    * ``figures/confusion_matrix.png`` — seaborn heatmap;
    * ``figures/predictions_{split}.csv`` — one row per sample with true /
      predicted label, correctness, and per-class softmax probabilities,
      for downstream qualitative analysis.

Example (from the repo root, with the project env active)::

    // promote desired run to best
    cp runs/***/best.pt runs/best_bilstm.pt

    conda activate wifisense
    python -m src.evaluate --checkpoint runs/best_bilstm.pt --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.models import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "ut_har" / "ut_har.npz"
DEFAULT_CKPT = PROJECT_ROOT / "runs" / "best_bilstm.pt"
FIGURES_DIR = PROJECT_ROOT / "figures"


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Return softmax probabilities ``(N, num_classes)`` for inputs X."""
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size].to(device)
        probs.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    """Save a seaborn heatmap of the confusion matrix (row-normalized overlay)."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "count"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CKPT, help="best.pt checkpoint."
    )
    p.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="Preprocessed .npz path."
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--out-dir", type=Path, default=FIGURES_DIR, help="Where to write figures/CSV."
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        or args.device == "cuda"
        else "cpu"
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    class_names = ckpt["class_names"]

    npz = np.load(args.data)
    X = torch.from_numpy(npz[f"X_{args.split}"]).float()
    y_true = npz[f"y_{args.split}"].astype(int)

    probs = predict(model, X, device)
    y_pred = probs.argmax(axis=1)

    labels = list(range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --- stdout summary -----------------------------------------------------
    print(
        f"Checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch')}, "
        f"val_acc {ckpt.get('val_acc'):.4f})"
    )
    print(f"Split: {args.split}  (N={len(y_true)})")
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}\n")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    # --- confusion matrix figure -------------------------------------------
    cm_path = args.out_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"Confusion matrix → {cm_path}")

    # --- per-sample predictions CSV ----------------------------------------
    df = pd.DataFrame(
        {
            "index": np.arange(len(y_true)),
            "true_label": y_true,
            "true_class": [class_names[i] for i in y_true],
            "pred_label": y_pred,
            "pred_class": [class_names[i] for i in y_pred],
            "correct": (y_true == y_pred),
            "confidence": probs.max(axis=1),
        }
    )
    for j, name in enumerate(class_names):
        df[f"prob_{name}"] = probs[:, j]
    csv_path = args.out_dir / f"predictions_{args.split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Predictions CSV → {csv_path}")

    # --- metrics JSON -------------------------------------------------------
    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_samples": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {name: report[name] for name in class_names if name in report},
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    metrics_path = args.out_dir / f"eval_metrics_{args.split}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON    → {metrics_path}")


if __name__ == "__main__":
    main()
