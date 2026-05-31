#!/usr/bin/env python
"""Train a CSI activity classifier on preprocessed UT-HAR.

Standard PyTorch loop: Adam + cross-entropy, early stopping on validation
accuracy.  Reads the preprocessed ``data/processed/ut_har/ut_har.npz``
(train / val / test tensors produced by ``scripts/preprocess_data.py``) and
uses the dataset's *provided* val split for early stopping — UT-HAR ships a
standard train(3977)/val(496)/test(500) partition, so we honour it rather
than re-carving 10% of train, keeping runs reproducible and comparable to
SenseFi.

Each run writes to ``runs/{timestamp}/``:
    best.pt              — best-val-accuracy checkpoint (model-agnostic)
    metrics.json         — per-epoch history + run summary
    training_curves.png  — loss & accuracy curves

Example (from the repo root, with the project env active)::

    conda activate wifisense
    python -m src.train --model bilstm --epochs 80 --batch-size 64 --lr 1e-3 --patience 15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import UT_HAR_CLASSES
from src.models import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "ut_har" / "ut_har.npz"
RUNS_ROOT = PROJECT_ROOT / "runs"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_split(npz: np.lib.npyio.NpzFile, split: str) -> TensorDataset:
    """Build a TensorDataset from ``X_{split}`` / ``y_{split}`` arrays."""
    X = torch.from_numpy(npz[f"X_{split}"]).float()
    y = torch.from_numpy(npz[f"y_{split}"]).long()
    return TensorDataset(X, y)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean loss, accuracy) over a loader."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """One training pass; returns (mean loss, accuracy)."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total


def plot_curves(history: list[dict], path: Path) -> None:
    """Save train/val loss and accuracy curves side by side."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    best = max(history, key=lambda h: h["val_acc"])
    ax_acc.axvline(best["epoch"], color="grey", ls="--", lw=1)
    ax_acc.annotate(
        f"best val={best['val_acc']:.3f}\n@ epoch {best['epoch']}",
        xy=(best["epoch"], best["val_acc"]),
        xytext=(0.45, 0.1),
        textcoords="axes fraction",
        fontsize=9,
    )

    fig.suptitle("Training curves", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="bilstm", help="Model name (registry key).")
    p.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="Preprocessed .npz path."
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Max epochs (early stopping may stop sooner).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Early-stop after N epochs with no val-acc gain.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Run directory (default: runs/{timestamp}).",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-workers", type=int, default=0)
    # Model hyperparameters (BiLSTM defaults match the project config).
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        or args.device == "cuda"
        else "cpu"
    )

    run_dir = args.save_dir or (RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.data, allow_pickle=True)
    train_ds = load_split(npz, "train")
    val_ds = load_split(npz, "val")
    n_classes = int(max(npz["y_train"].max(), npz["y_val"].max())) + 1
    input_size = train_ds.tensors[0].shape[-1]

    # Class names default to UT-HAR's; a non-UT-HAR .npz (e.g. NTU-Fi) may
    # carry its own ``class_names`` array so checkpoints self-describe.
    if "class_names" in npz.files:
        class_names = [str(c) for c in npz["class_names"]][:n_classes]
    else:
        class_names = UT_HAR_CLASSES[:n_classes]

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model_cfg = {
        "input_size": input_size,
        "hidden_size": args.hidden,
        "num_layers": args.layers,
        "num_classes": n_classes,
        "dropout": args.dropout,
    }
    model = build_model(args.model, **model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device} | run dir: {run_dir}")
    print(f"Model: {args.model} {model.config}")
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | classes {n_classes}")

    history: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    ckpt_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            epochs_no_improve = 0
            flag = " *"
            torch.save(
                {
                    "model_name": args.model,
                    "model_config": model.config,
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": vars(args)
                    | {"data": str(args.data), "save_dir": str(run_dir)},
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}{flag}"
        )

        if epochs_no_improve >= args.patience:
            print(f"Early stopping: no val-acc improvement in {args.patience} epochs.")
            break

    summary = {
        "model": args.model,
        "model_config": model.config,
        "args": vars(args) | {"data": str(args.data), "save_dir": str(run_dir)},
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "epochs_run": history[-1]["epoch"],
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_curves(history, run_dir / "training_curves.png")

    print(f"\nBest val acc {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Metrics:    {run_dir / 'metrics.json'}")
    print(f"Curves:     {run_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
