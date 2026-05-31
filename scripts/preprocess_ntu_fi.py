#!/usr/bin/env python
"""Preprocess NTU-Fi HAR into UT-HAR's (250, 90) common representation.

For chunk 9 we need a *single* BiLSTM input format so one architecture can be
trained / cross-evaluated on both datasets.  UT-HAR is the reference shape:
``(250 time steps, 90 features)`` where 90 = 30 subcarriers x 3 RX antennas.

NTU-Fi ships ``(342, 2000)`` per sample — 342 = 114 subcarriers x 3 antennas
(feature axis), 2000 = time packets.  We treat each sample as a CSI
spectrogram and bilinearly resize it ``(342, 2000) -> (90, 250)`` (feat, time),
then transpose to ``(250, 90)`` to match UT-HAR's (time, feat) layout.  After
resizing we run the *identical* UT-HAR preprocessing pipeline
(amplitude -> hampel -> median -> per-sample z-score).

NTU-Fi has no validation split (only train/test), so — to mirror chunk 5's
early-stopping-on-val recipe — we carve a stratified 10% validation set out of
the 936 train samples (seeded).  The 264-sample test set is left untouched.

Outputs:
    data/processed/ntu_fi/ntu_fi.npz   — X_train,y_train,X_val,y_val,
                                          X_test,y_test, class_names
    figures/preprocessing/ntu_fi_resize.png — native vs resized, one per class

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/preprocess_ntu_fi.py

Loads NTU-Fi lazily (one .mat at a time) so peak memory stays small even
though the raw train split is ~2.5 GB.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import NTUFiHARDataset, NTU_FI_HAR_CLASSES
from src.data.preprocess import Pipeline

# UT-HAR reference shape (time, feat).
TARGET_T = 250
TARGET_S = 90
VAL_FRACTION = 0.10
SEED = 42


def resize_to_uthar(csi: np.ndarray) -> np.ndarray:
    """Resize one NTU-Fi sample (342 feat, 2000 time) -> (250 time, 90 feat).

    Bilinear interpolation over the 2-D CSI spectrogram, then transpose so the
    time axis comes first (matching UT-HAR's (T, S) layout).
    """
    t = torch.from_numpy(csi).float()[None, None]  # (1, 1, 342, 2000)
    resized = F.interpolate(
        t, size=(TARGET_S, TARGET_T), mode="bilinear", align_corners=False
    )
    feat_time = resized[0, 0].numpy()  # (90, 250) = (feat, time)
    return feat_time.T.copy()  # (250, 90) = (time, feat)


def load_resized(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an NTU-Fi split lazily and resize every sample to (250, 90)."""
    ds = NTUFiHARDataset(split)
    n = len(ds)
    X = np.empty((n, TARGET_T, TARGET_S), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    for i in range(n):
        csi, label = ds[i]
        X[i] = resize_to_uthar(csi.numpy())
        y[i] = label
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  resized {i + 1}/{n} {split} samples")
    return X, y


def plot_resize(
    raw_examples: dict[int, np.ndarray],
    proc_examples: dict[int, np.ndarray],
    fig_dir: Path,
) -> None:
    """Native (342x2000) vs resized+preprocessed (250x90), one row per class."""
    classes = NTU_FI_HAR_CLASSES
    fig, axes = plt.subplots(len(classes), 2, figsize=(13, 3 * len(classes)))
    for row, cls_idx in enumerate(range(len(classes))):
        raw = raw_examples[cls_idx]  # (342, 2000) feat x time
        proc = proc_examples[cls_idx].T  # (90, 250) feat x time for imshow
        ax_raw, ax_proc = axes[row, 0], axes[row, 1]
        im0 = ax_raw.imshow(raw, aspect="auto", origin="lower", interpolation="nearest")
        ax_raw.set_ylabel(classes[cls_idx], fontsize=11)
        fig.colorbar(im0, ax=ax_raw, fraction=0.046, pad=0.04)
        im1 = ax_proc.imshow(
            proc, aspect="auto", origin="lower", interpolation="nearest"
        )
        fig.colorbar(im1, ax=ax_proc, fraction=0.046, pad=0.04)
        if row == 0:
            ax_raw.set_title("Native (342 x 2000)")
            ax_proc.set_title("Resized + preprocessed (90 x 250)")
        if row == len(classes) - 1:
            ax_raw.set_xlabel("Time packet")
            ax_proc.set_xlabel("Time step")
    fig.suptitle(
        "NTU-Fi HAR: native vs UT-HAR-format (one sample per class)",
        y=1.005,
        fontsize=13,
    )
    fig.tight_layout()
    path = fig_dir / "ntu_fi_resize.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {path}")


def main() -> None:
    out_dir = ROOT / "data" / "processed" / "ntu_fi"
    fig_dir = ROOT / "figures" / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading + resizing NTU-Fi train (936) ...")
    X_train_full, y_train_full = load_resized("train")
    print("Loading + resizing NTU-Fi test (264) ...")
    X_test, y_test = load_resized("test")

    # Stratified 10% val carved from train (mirrors UT-HAR's ~11% val split).
    idx = np.arange(len(y_train_full))
    tr_idx, val_idx = train_test_split(
        idx, test_size=VAL_FRACTION, stratify=y_train_full, random_state=SEED
    )
    X_tr_raw, y_tr = X_train_full[tr_idx], y_train_full[tr_idx]
    X_val_raw, y_val = X_train_full[val_idx], y_train_full[val_idx]
    print(
        f"Split: train {len(y_tr)} | val {len(y_val)} | test {len(y_test)} "
        f"(stratified, seed={SEED})"
    )

    # Identical UT-HAR pipeline (all transforms are per-sample, so fit==transform).
    pipe = Pipeline()
    print("Preprocessing train ...")
    X_train = pipe.fit_transform(X_tr_raw)
    print("Preprocessing val ...")
    X_val = pipe.transform(X_val_raw)
    print("Preprocessing test ...")
    X_test_p = pipe.transform(X_test)

    npz_path = out_dir / "ntu_fi.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test_p,
        y_test=y_test,
        class_names=np.array(NTU_FI_HAR_CLASSES),
    )
    print(f"Saved processed data -> {npz_path}")
    print(
        f"  X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test_p.shape}"
    )

    # Before/after figure: grab one native + one processed sample per class.
    print("Generating resize figure ...")
    test_ds = NTUFiHARDataset("test")
    raw_examples, proc_examples = {}, {}
    seen_raw, seen_proc = set(), set()
    for i in range(len(test_ds)):
        csi, label = test_ds[i]
        if label not in seen_raw:
            raw_examples[label] = csi.numpy()
            seen_raw.add(label)
        if len(seen_raw) == len(NTU_FI_HAR_CLASSES):
            break
    for cls_idx in range(len(NTU_FI_HAR_CLASSES)):
        first = int(np.where(y_test == cls_idx)[0][0])
        proc_examples[cls_idx] = X_test_p[first]
    plot_resize(raw_examples, proc_examples, fig_dir)


if __name__ == "__main__":
    main()
