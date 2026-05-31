#!/usr/bin/env python
"""Stitch UT-HAR test samples into one continuous synthetic capture.

UT-HAR ships as isolated 250-step samples, but the milestone deliverable
needs a *continuous* recording to slide a window over.  This script picks N
test samples, concatenates them along the time axis, and records which
sample (hence which activity) is "active" at every time step — the
ground-truth timeline the final visualization plots against.

The capture is stored **raw** (no preprocessing): the streaming inference
step applies the training preprocessing pipeline per window, so the saved
stream must be the un-normalized CSI those windows are cut from.

Sample selection is round-robin over classes (deterministic given --seed)
so the stitched capture shows a variety of activities rather than, say,
eight "walk" segments in a row.

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/build_continuous_capture.py            # N=8
    python scripts/build_continuous_capture.py --n 12 --seed 1

Runnable directly (not via ``-m``): it prepends the repo root to sys.path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_ut_har, UT_HAR_CLASSES

DEFAULT_OUT = ROOT / "data" / "continuous" / "synthetic_capture.npz"


def select_indices(y: np.ndarray, n: int, seed: int) -> list[int]:
    """Pick ``n`` sample indices, round-robin across classes for variety.

    Within each class the available indices are shuffled (seeded) so repeated
    runs with the same seed are reproducible but different seeds give
    different captures.  Classes are visited in a fixed cyclic order, so the
    resulting timeline cycles through activities.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for cls in np.unique(y):
        idxs = np.where(y == cls)[0]
        rng.shuffle(idxs)
        by_class[int(cls)] = list(idxs)

    classes = sorted(by_class)
    chosen: list[int] = []
    ci = 0
    while len(chosen) < n:
        cls = classes[ci % len(classes)]
        if by_class[cls]:
            chosen.append(by_class[cls].pop())
        ci += 1
        if ci > n * len(classes) * 2:  # safety: ran out of samples
            raise RuntimeError("Not enough samples to fulfil request")
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=8, help="Number of samples to stitch.")
    p.add_argument("--seed", type=int, default=0, help="Selection RNG seed.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output .npz path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_ut_har(args.split)
    X = X.numpy()
    y = y.numpy().astype(int)
    sample_len = X.shape[1]  # 250 for UT-HAR
    n_subcarriers = X.shape[2]

    idxs = select_indices(y, args.n, args.seed)
    sample_labels = y[idxs]

    # Concatenate raw samples along time → (N * sample_len, S).
    stream = np.concatenate([X[i] for i in idxs], axis=0).astype(np.float32)

    # Ground-truth timeline: one label per time step, plus segment boundaries.
    labels_per_step = np.repeat(sample_labels, sample_len).astype(np.int64)
    boundaries = np.arange(args.n + 1) * sample_len  # (N+1,) start..end indices

    np.savez_compressed(
        args.out,
        stream=stream,
        labels_per_step=labels_per_step,
        sample_labels=sample_labels.astype(np.int64),
        sample_indices=np.asarray(idxs, dtype=np.int64),
        boundaries=boundaries.astype(np.int64),
        sample_len=np.int64(sample_len),
        class_names=np.asarray(UT_HAR_CLASSES),
    )

    print(f"Wrote continuous capture → {args.out}")
    print(f"  stream shape: {stream.shape}  ({args.n} × {sample_len} steps, "
          f"{n_subcarriers} subcarriers)")
    print("  segment timeline:")
    for k, (i, lab) in enumerate(zip(idxs, sample_labels)):
        t0, t1 = boundaries[k], boundaries[k + 1]
        print(f"    [{t0:5d}, {t1:5d})  sample #{i:3d}  → {UT_HAR_CLASSES[lab]}")


if __name__ == "__main__":
    main()
