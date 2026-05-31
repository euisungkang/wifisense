#!/usr/bin/env python3
"""Characterize the UT-HAR and NTU-Fi HAR datasets.

Prints sample counts, per-class distributions, tensor stats, and saves
5 random samples per class as .npy files under data/samples/.

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/explore_data.py

Runnable directly (not via ``-m``): it prepends the repo root to sys.path
itself so the ``src`` package imports resolve.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import (
    load_ut_har,
    load_ntu_fi_har,
    UT_HAR_CLASSES,
    NTU_FI_HAR_CLASSES,
    PROJECT_ROOT,
)


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def describe_tensor(name: str, X) -> None:
    arr = X.numpy()
    print(f"\n  {name}:")
    print(f"    shape:   {tuple(X.shape)}")
    print(f"    dtype:   {X.dtype}")
    print(f"    min:     {arr.min():.4f}")
    print(f"    max:     {arr.max():.4f}")
    print(f"    mean:    {arr.mean():.4f}")
    print(f"    std:     {arr.std():.4f}")
    print(f"    NaN:     {np.isnan(arr).any()}")
    print(f"    Inf:     {np.isinf(arr).any()}")
    print(f"    complex: {np.iscomplexobj(arr)}")

    all_positive = (arr >= 0).all()
    has_negative = (arr < 0).any()
    print(f"    all positive (amplitude-only): {all_positive}")
    if has_negative:
        print(f"    negative values present — NOT raw amplitude")

    mean_close_zero = abs(arr.mean()) < 0.5
    std_close_one = abs(arr.std() - 1.0) < 0.2
    in_unit_range = arr.min() >= -0.01 and arr.max() <= 1.01
    if mean_close_zero and std_close_one:
        print(f"    appears z-score normalized (mean≈0, std≈1)")
    elif in_unit_range:
        print(f"    appears min-max normalized to [0, 1]")
    else:
        print(f"    NOT normalized (neither z-score nor [0,1])")


def print_class_counts(y, class_names: list[str]) -> None:
    arr = y.numpy()
    print(f"\n  Per-class counts:")
    total = len(arr)
    for idx, name in enumerate(class_names):
        count = int((arr == idx).sum())
        pct = 100.0 * count / total
        print(f"    {idx} ({name:>10s}): {count:5d}  ({pct:5.1f}%)")
    print(f"    {'total':>14s}: {total:5d}")

    counts = [(arr == i).sum() for i in range(len(class_names))]
    ratio = max(counts) / max(min(counts), 1)
    if ratio > 2.0:
        print(f"    ⚠ imbalance ratio (max/min): {ratio:.1f}×")
    else:
        print(f"    balance ratio (max/min): {ratio:.1f}×")


def save_samples(X, y, class_names: list[str], dataset_name: str, n: int = 5) -> None:
    rng = np.random.default_rng(42)
    samples_dir = PROJECT_ROOT / "data" / "samples"

    for idx, name in enumerate(class_names):
        mask = y.numpy() == idx
        indices = np.where(mask)[0]
        chosen = rng.choice(indices, size=min(n, len(indices)), replace=False)

        out_dir = samples_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, sample_idx in enumerate(chosen):
            filename = f"{dataset_name}_{name}_{i}.npy"
            np.save(out_dir / filename, X[sample_idx].numpy())

    print(f"\n  Saved {n} samples/class to {samples_dir.relative_to(PROJECT_ROOT)}/")


def main() -> None:
    # ---------------------------------------------------------------
    # UT-HAR
    # ---------------------------------------------------------------
    print_header("UT-HAR Dataset")

    for split in ("train", "val", "test"):
        X, y = load_ut_har(split)
        print(f"\n  [{split}] samples: {len(y)}")
        describe_tensor(f"X_{split}", X)
        print_class_counts(y, UT_HAR_CLASSES)

    X_train, y_train = load_ut_har("train")
    save_samples(X_train, y_train, UT_HAR_CLASSES, "ut_har")

    # ---------------------------------------------------------------
    # NTU-Fi HAR
    # ---------------------------------------------------------------
    print_header("NTU-Fi HAR Dataset")

    for split in ("train", "test"):
        print(f"\n  [{split}] loading (this may use ~2.5 GB for train)...")
        X, y = load_ntu_fi_har(split)
        print(f"  [{split}] samples: {len(y)}")
        describe_tensor(f"X_{split}", X)
        print_class_counts(y, NTU_FI_HAR_CLASSES)

    X_train_ntu, y_train_ntu = load_ntu_fi_har("train")
    save_samples(X_train_ntu, y_train_ntu, NTU_FI_HAR_CLASSES, "ntu_fi")

    print_header("Done")


if __name__ == "__main__":
    main()
