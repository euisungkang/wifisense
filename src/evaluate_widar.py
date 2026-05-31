#!/usr/bin/env python
"""Evaluate the BVP CNN-RNN across all four Widar3.0 cross-domain splits.

This is the real test of BVP's environment-invariance claim. ``src/train_widar.py``
fits one model per split (``runs/best_bvp_<split>.pt``); this script loads each
checkpoint, rebuilds *its own* held-out test partition from the ``split_config``
the checkpoint carries, and scores it. The accuracy gap between ``in_domain`` and
the three cross-domain splits *is* the domain gap — and the whole point of BVP is
that this gap should be small, in stark contrast to chunk 9's raw-CSI collapse
(``figures/domain_shift_matrix.png``: UT-HAR<->NTU-Fi fell from ~92-98% in-domain
to ~1.6-42% across domains).

Method (per split):
    * Reconstruct the exact held-out test set by re-invoking the split builder
      with the stored filters / held-out values / seed (and the same
      max-per-gesture cap, if any). Augmentation is forced off on test.
    * Run the model; compute overall accuracy, macro F1, a full per-class
      precision/recall/F1 report, and the confusion matrix.

Outputs (under ``--out-dir``, default ``figures/``):
    * stdout + ``widar/<split>_metrics.json`` — metrics per split.
    * ``widar/<split>_predictions.csv`` — one row per test sample (true/pred/
      correct/confidence) for qualitative drill-down (consumed by spatial_viz).
    * ``widar_domain_results.png`` — the 2x2 grid of confusion matrices, one per
      evaluation mode, each titled with its accuracy. Compare against chunk 9's
      domain-shift matrix: BVP should hold up dramatically better.

Examples (from the repo root, with the project env active)::

    conda activate wifisense
    python src/evaluate_widar.py                       # all four splits
    python src/evaluate_widar.py --splits in_domain cross_user

Runnable via ``-m`` or directly (it prepends the repo root to sys.path itself).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import make_dataloader  # noqa: E402
from src.data.widar_dataset import (  # noqa: E402
    cross_orientation,
    cross_position,
    cross_user,
    in_domain,
)
from src.models import build_model  # noqa: E402
from src.train_widar import SPLITS, cap_index  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
FIGURES_DIR = PROJECT_ROOT / "figures"

PRETTY = {
    "in_domain": "In-domain (i.i.d.)",
    "cross_user": "Cross-user",
    "cross_position": "Cross-position",
    "cross_orientation": "Cross-orientation",
}


def build_test_ds(cfg: dict):
    """Rebuild the held-out *test* dataset for a split from its stored config."""
    cap_index(cfg.get("max_per_gesture"), cfg["seed"])
    filters = cfg.get("filters", {})
    common = dict(
        target_T=cfg["target_T"], normalize=cfg["normalize"], seed=cfg["seed"]
    )
    split = cfg["split"]
    if split == "in_domain":
        _, test = in_domain(test_frac=cfg["test_frac"], **filters, **common)
    elif split == "cross_user":
        _, test = cross_user(test_users=cfg["test_users"], **filters, **common)
    elif split == "cross_position":
        _, test = cross_position(test_positions=cfg["test_positions"], **filters, **common)
    elif split == "cross_orientation":
        _, test = cross_orientation(
            test_orientations=cfg["test_orientations"], **filters, **common
        )
    else:
        raise ValueError(f"unknown split {split!r}")
    return test


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (softmax probs (N, C), true labels (N,)) over a loader."""
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        probs.append(F.softmax(model(xb), dim=1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def eval_split(split: str, args: argparse.Namespace, device: torch.device) -> dict | None:
    """Load a split's checkpoint, score its held-out test set, dump artifacts."""
    ckpt_path = RUNS_ROOT / f"best_bvp_{split}.pt"
    if not ckpt_path.exists():
        print(f"[skip] {split}: no checkpoint at {ckpt_path.name} "
              f"(train it with: python src/train_widar.py --split {split})")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    class_names = ckpt["class_names"]

    test_ds = build_test_ds(ckpt["split_config"])
    # The test set must encode gestures the same way the model was trained on.
    if test_ds.classes != class_names:
        print(f"[warn] {split}: test label order differs from checkpoint; "
              "using checkpoint class order for the report.")
    loader = make_dataloader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    probs, y_true = predict(model, loader, device)
    y_pred = probs.argmax(axis=1)

    labels = list(range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        digits=4, zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    held = (ckpt["split_config"].get("test_users")
            or ckpt["split_config"].get("test_positions")
            or ckpt["split_config"].get("test_orientations")
            or f"{ckpt['split_config'].get('test_frac', 0.2):.0%} random")
    print(f"\n=== {PRETTY[split]} (held out: {held}) ===")
    print(f"  checkpoint val_acc {ckpt.get('val_acc', float('nan')):.4f} | "
          f"test N={len(y_true)}")
    print(f"  test accuracy {acc:.4f} | macro F1 {macro_f1:.4f}")

    out_dir = args.out_dir / "widar"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "split": split,
        "checkpoint": str(ckpt_path),
        "split_config": ckpt["split_config"],
        "n_samples": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": {n: report[n] for n in class_names if n in report},
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "held_out": held if isinstance(held, str) else list(held),
    }
    with open(out_dir / f"{split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    df = pd.DataFrame({
        "index": np.arange(len(y_true)),
        "true_label": y_true,
        "true_class": [class_names[i] for i in y_true],
        "pred_label": y_pred,
        "pred_class": [class_names[i] for i in y_pred],
        "correct": (y_true == y_pred),
        "confidence": probs.max(axis=1),
    })
    df.to_csv(out_dir / f"{split}_predictions.csv", index=False)

    return {"split": split, "accuracy": acc, "macro_f1": macro_f1,
            "cm": cm, "class_names": class_names, "n": int(len(y_true))}


def draw_confusion(ax, res: dict) -> None:
    """Row-normalized confusion heatmap for one split."""
    cm = res["cm"].astype(float)
    rs = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
    names = res["class_names"]
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(
        f"{PRETTY[res['split']]} — acc {res['accuracy'] * 100:.1f}% "
        f"(macro F1 {res['macro_f1'] * 100:.1f}, N={res['n']})",
        fontsize=10,
    )
    return im


def assemble_figure(results: list[dict], path: Path) -> None:
    """2x2 grid of confusion matrices, ordered in_domain -> cross_*."""
    order = {s: i for i, s in enumerate(SPLITS)}
    results = sorted(results, key=lambda r: order[r["split"]])
    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    axes = axes.ravel()
    im = None
    for ax, res in zip(axes, results):
        im = draw_confusion(ax, res)
    for ax in axes[len(results):]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02,
                     label="row-normalized (recall)")
    fig.suptitle(
        "Widar3.0 BVP — gesture recognition across domain splits\n"
        "BVP is environment-invariant: cross-domain accuracy should stay high, "
        "unlike raw-CSI (chunk 9)",
        fontsize=14, y=0.98,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
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

    results = [r for s in args.splits if (r := eval_split(s, args, device)) is not None]
    if not results:
        print("\nNo trained split checkpoints found — run src/train_widar.py first.")
        return

    assemble_figure(results, args.out_dir / "widar_domain_results.png")

    # Headline comparison table (vs. chunk 9's raw-CSI cross-domain collapse).
    print("\n  split                accuracy   macro F1")
    for r in sorted(results, key=lambda r: list(SPLITS).index(r["split"])):
        print(f"  {PRETTY[r['split']]:20} {r['accuracy'] * 100:6.1f}%   {r['macro_f1'] * 100:5.1f}")
    if len(results) > 1:
        ind = next((r["accuracy"] for r in results if r["split"] == "in_domain"), None)
        cross = [r["accuracy"] for r in results if r["split"] != "in_domain"]
        if ind is not None and cross:
            gap = ind - min(cross)
            print(f"\n  in-domain -> worst cross-domain gap: {gap * 100:.1f} pts "
                  f"(chunk 9 raw-CSI gap was ~50-90 pts).")


if __name__ == "__main__":
    main()
