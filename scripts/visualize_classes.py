#!/usr/bin/env python
"""Render per-class CSI visualization grids for sanity-checking the data.

For each UT-HAR class, picks 3 random samples and lays them out in a grid
(rows = classes, cols = samples):

    figures/class_grid.png    — CSI amplitude heatmaps
    figures/doppler_grid.png  — Doppler spectrograms (STFT)

Samples are drawn from the *preprocessed* training set
(data/processed/ut_har/ut_har.npz), i.e. exactly what the model will see,
so these grids double as a preprocessing sanity check.

Usage:
    python scripts/visualize_classes.py [--seed N] [--fs HZ]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import UT_HAR_CLASSES
from src.viz import plot_amplitude_heatmap, plot_doppler_spectrogram

# UT-HAR's nominal Intel-5300 packet rate.  The benchmark resamples each
# clip to 250 steps, so the *effective* rate is uncertain; this only
# rescales the Doppler axis and does not affect relative class structure.
DEFAULT_FS = 1000.0

N_PER_CLASS = 3


def pick_samples(
    y: np.ndarray, rng: np.random.Generator
) -> dict[int, np.ndarray]:
    """Choose N_PER_CLASS random sample indices for each class label."""
    chosen = {}
    for cls_idx in range(len(UT_HAR_CLASSES)):
        idxs = np.where(y == cls_idx)[0]
        n = min(N_PER_CLASS, len(idxs))
        chosen[cls_idx] = rng.choice(idxs, size=n, replace=False)
    return chosen


def render_grid(
    X: np.ndarray,
    chosen: dict[int, np.ndarray],
    plot_fn,
    out_path: Path,
    suptitle: str,
    add_colorbar: bool,
) -> None:
    """Render a (classes x samples) grid using ``plot_fn`` for each cell."""
    n_rows = len(UT_HAR_CLASSES)
    fig, axes = plt.subplots(
        n_rows, N_PER_CLASS, figsize=(4 * N_PER_CLASS, 2.6 * n_rows)
    )
    axes = np.atleast_2d(axes)

    for row, cls_name in enumerate(UT_HAR_CLASSES):
        for col in range(N_PER_CLASS):
            ax = axes[row, col]
            sample_idx = chosen[cls_name_to_idx(cls_name)][col]
            ax = plot_fn(X[sample_idx], ax=ax)
            if add_colorbar and ax.images:
                fig.colorbar(ax.images[-1], ax=ax, fraction=0.046, pad=0.04)
            # Label the leftmost column with the class name.
            if col == 0:
                ax.set_ylabel(f"{cls_name}\n{ax.get_ylabel()}", fontsize=10)
            if row == 0:
                ax.set_title(f"sample {col + 1}", fontsize=10)

    fig.suptitle(suptitle, y=1.005, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def cls_name_to_idx(name: str) -> int:
    return UT_HAR_CLASSES.index(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sample picks")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS, help="CSI packet rate (Hz)")
    args = parser.parse_args()

    npz_path = ROOT / "data" / "processed" / "ut_har" / "ut_har.npz"
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    X, y = data["X_train"], data["y_train"]
    print(f"Loaded {X.shape[0]} samples, shape {X.shape[1:]}")

    rng = np.random.default_rng(args.seed)
    chosen = pick_samples(y, rng)

    render_grid(
        X,
        chosen,
        plot_fn=plot_amplitude_heatmap,
        out_path=fig_dir / "class_grid.png",
        suptitle="UT-HAR CSI amplitude heatmaps (3 random samples / class)",
        add_colorbar=True,
    )

    render_grid(
        X,
        chosen,
        plot_fn=lambda x, ax: plot_doppler_spectrogram(x, fs=args.fs, ax=ax),
        out_path=fig_dir / "doppler_grid.png",
        suptitle=f"UT-HAR Doppler spectrograms (fs={args.fs:g} Hz, 3 random samples / class)",
        add_colorbar=True,
    )


if __name__ == "__main__":
    main()
