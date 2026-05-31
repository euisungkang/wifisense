#!/usr/bin/env python3
"""Verify that the wifisense environment and datasets are set up correctly.

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python verify_setup.py
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


PROJECT_ROOT = Path(__file__).resolve().parent

UT_HAR_DIR = PROJECT_ROOT / "data" / "raw" / "ut_har" / "UT_HAR"
NTU_FI_HAR_DIR = PROJECT_ROOT / "data" / "raw" / "ntu_fi_har" / "NTU-Fi_HAR"


def check_ut_har():
    print("=" * 60)
    print("UT-HAR Dataset")
    print("=" * 60)

    data_dir = UT_HAR_DIR / "data"
    label_dir = UT_HAR_DIR / "label"

    # X = CSI samples, y = activity labels. Despite the .csv name these are
    # numpy binary dumps, hence np.load rather than a text parser.
    for name in ["X_train", "X_val", "X_test"]:
        path = data_dir / f"{name}.csv"
        with open(path, "rb") as f:
            arr = np.load(f)
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")

    for name in ["y_train", "y_val", "y_test"]:
        path = label_dir / f"{name}.csv"
        with open(path, "rb") as f:
            arr = np.load(f)
        unique = np.unique(arr).astype(int)
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"classes={unique.tolist()}")

    # One sample shaped the way the model expects it: (channel, time, subcarrier).
    with open(data_dir / "X_train.csv", "rb") as f:
        sample = np.load(f)
    sample = sample[0].reshape(1, 250, 90)
    tensor = torch.tensor(sample, dtype=torch.float32)
    print(f"\n  Single sample as tensor: shape={tuple(tensor.shape)}, "
          f"dtype={tensor.dtype}")
    print(f"  CSI matrix: {sample.shape[-2]} time steps x {sample.shape[-1]} subcarriers")


def check_ntu_fi_har():
    print("\n" + "=" * 60)
    print("NTU-Fi HAR Dataset")
    print("=" * 60)

    # Unlike UT-HAR, each sample is its own MATLAB .mat file, grouped into one
    # subdirectory per activity class (the folder name IS the label).
    for split in ["train_amp", "test_amp"]:
        split_dir = NTU_FI_HAR_DIR / split
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        total = sum(1 for _ in split_dir.rglob("*.mat"))
        print(f"  {split}: {total} samples, classes={classes}")

    first_mat = next((NTU_FI_HAR_DIR / "train_amp").rglob("*.mat"))
    mat = sio.loadmat(str(first_mat))
    csi = mat["CSIamp"]  # raw amplitude, (342 subcarriers, 2000 packets)
    print(f"\n  Sample file: {first_mat.name}")
    print(f"  Raw CSI amplitude: shape={csi.shape}, dtype={csi.dtype}, "
          f"range=[{csi.min():.4f}, {csi.max():.4f}]")

    # Keep every 4th packet (2000 -> 500), then split 342 rows into 3 antennas
    # x 114 subcarriers to match the model's expected (antenna, subcarrier, time).
    csi_reshaped = csi[:, ::4].reshape(3, 114, 500)
    tensor = torch.tensor(csi_reshaped, dtype=torch.float32)
    print(f"  After downsample+reshape: shape={tuple(tensor.shape)}, "
          f"dtype={tensor.dtype}")
    print(f"  3 antennas x 114 subcarriers x 500 time steps")


def check_pytorch():
    print("\n" + "=" * 60)
    print("PyTorch Environment")
    print("=" * 60)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  (CPU-only — expected for this setup)")
    print(f"  Python version: {sys.version}")


def main():
    ok = True

    if not UT_HAR_DIR.exists():
        print(f"MISSING: {UT_HAR_DIR}")
        ok = False
    else:
        check_ut_har()

    if not NTU_FI_HAR_DIR.exists():
        print(f"MISSING: {NTU_FI_HAR_DIR}")
        ok = False
    else:
        check_ntu_fi_har()

    check_pytorch()

    print("\n" + "=" * 60)
    if ok:
        print("All checks passed.")
    else:
        print("Some checks FAILED — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
