#!/usr/bin/env python
"""Run the default preprocessing pipeline on UT-HAR and save results.

Outputs:
    data/processed/ut_har/ut_har.npz   — X_train, y_train, X_val, y_val, X_test, y_test
    figures/preprocessing/ut_har_before_after.png — CSI heatmap per class

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/preprocess_data.py

Runnable directly (not via ``-m``): it prepends the repo root to sys.path
itself so the ``src`` package imports resolve.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_ut_har, UT_HAR_CLASSES
from src.data.preprocess import Pipeline


def plot_before_after(
    X_raw: np.ndarray,
    X_proc: np.ndarray,
    y: np.ndarray,
    fig_dir: Path,
) -> None:
    """CSI heatmap (subcarrier × time) before/after for one sample per class."""
    n_classes = len(UT_HAR_CLASSES)
    fig, axes = plt.subplots(n_classes, 2, figsize=(14, 3 * n_classes))

    for cls_idx, cls_name in enumerate(UT_HAR_CLASSES):
        sample_idx = int(np.where(y == cls_idx)[0][0])
        raw = X_raw[sample_idx].T   # (S, T) for imshow
        proc = X_proc[sample_idx].T

        ax_raw, ax_proc = axes[cls_idx, 0], axes[cls_idx, 1]

        im0 = ax_raw.imshow(raw, aspect="auto", origin="lower", interpolation="nearest")
        ax_raw.set_ylabel(cls_name, fontsize=11)
        fig.colorbar(im0, ax=ax_raw, fraction=0.046, pad=0.04)

        im1 = ax_proc.imshow(proc, aspect="auto", origin="lower", interpolation="nearest")
        fig.colorbar(im1, ax=ax_proc, fraction=0.046, pad=0.04)

        if cls_idx == 0:
            ax_raw.set_title("Raw")
            ax_proc.set_title("Preprocessed")
        if cls_idx == n_classes - 1:
            ax_raw.set_xlabel("Time step")
            ax_proc.set_xlabel("Time step")

    fig.suptitle(
        "UT-HAR CSI: Raw vs Preprocessed (one sample per class)", y=1.01, fontsize=13
    )
    fig.tight_layout()
    path = fig_dir / "ut_har_before_after.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {path}")


def main() -> None:
    out_dir = ROOT / "data" / "processed" / "ut_har"
    fig_dir = ROOT / "figures" / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline()

    raw = {}
    for split in ("train", "val", "test"):
        print(f"Loading {split} …")
        X, y = load_ut_har(split)
        raw[split] = (X.numpy(), y.numpy())

    X_train_raw = raw["train"][0].copy()

    print("Preprocessing train …")
    X_train = pipe.fit_transform(raw["train"][0])
    print("Preprocessing val …")
    X_val = pipe.transform(raw["val"][0])
    print("Preprocessing test …")
    X_test = pipe.transform(raw["test"][0])

    npz_path = out_dir / "ut_har.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        y_train=raw["train"][1],
        X_val=X_val,
        y_val=raw["val"][1],
        X_test=X_test,
        y_test=raw["test"][1],
    )
    print(f"Saved processed data → {npz_path}")
    print(f"  X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")

    print("Generating before/after plots …")
    plot_before_after(X_train_raw, X_train, raw["train"][1], fig_dir)


if __name__ == "__main__":
    main()
