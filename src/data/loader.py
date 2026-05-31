"""Data loading for UT-HAR and NTU-Fi HAR WiFi CSI datasets.

Exposes raw CSI tensors without preprocessing — no normalization, no
downsampling, no reshaping beyond what the file format requires.
Preprocessing is a separate step.

UT-HAR (Intel 5300, 7 activities):
    Per-sample shape: (250, 90)
        250 time steps, 90 = 30 subcarriers × 3 RX antennas (flattened).
    Values: pre-processed CSI amplitudes, real-valued, roughly [-11, +31].
    Labels 0-6: lie_down, fall, walk, pickup, run, sit_down, stand_up.
    Splits: train (3977), val (496), test (500).

NTU-Fi HAR (6 activities):
    Per-sample shape: (342, 2000)
        342 = 3 antennas × 114 subcarriers (flattened), 2000 time packets.
    Values: raw CSI amplitudes, real-valued, all positive, roughly [10, 53].
    Labels 0-5 (alphabetical): box, circle, clean, fall, run, walk.
    Splits: train (936), test (264). Perfectly balanced across classes.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

UT_HAR_DIR = PROJECT_ROOT / "data" / "raw" / "ut_har" / "UT_HAR"
NTU_FI_HAR_DIR = PROJECT_ROOT / "data" / "raw" / "ntu_fi_har" / "NTU-Fi_HAR"

UT_HAR_CLASSES = ["lie_down", "fall", "walk", "pickup", "run", "sit_down", "stand_up"]
NTU_FI_HAR_CLASSES = ["box", "circle", "clean", "fall", "run", "walk"]


def load_ut_har(
    split: Literal["train", "val", "test"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load UT-HAR CSI data for the given split.

    Files are numpy binary dumps despite .csv extension.

    Returns:
        X: float32 tensor, shape (N, 250, 90).
        y: int64 tensor, shape (N,). Labels 0-6, see UT_HAR_CLASSES.
    """
    with open(UT_HAR_DIR / "data" / f"X_{split}.csv", "rb") as f:
        X = np.load(f)
    with open(UT_HAR_DIR / "label" / f"y_{split}.csv", "rb") as f:
        y = np.load(f)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.int64),
    )


def load_ntu_fi_har(
    split: Literal["train", "test"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load NTU-Fi HAR CSI amplitude data for the given split.

    Each sample is a separate .mat file; this stacks them all into one tensor.
    ~2.5 GB for the train split. Use NTUFiHARDataset for lazy loading.

    Returns:
        X: float32 tensor, shape (N, 342, 2000).
        y: int64 tensor, shape (N,). Labels 0-5, see NTU_FI_HAR_CLASSES.
    """
    split_name = "train_amp" if split == "train" else "test_amp"
    split_dir = NTU_FI_HAR_DIR / split_name
    classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())

    X_list, y_list = [], []
    for label_idx, class_name in enumerate(classes):
        for mat_path in sorted((split_dir / class_name).glob("*.mat")):
            csi = sio.loadmat(str(mat_path))["CSIamp"]
            X_list.append(csi.astype(np.float32))
            y_list.append(label_idx)

    return (
        torch.from_numpy(np.stack(X_list)),
        torch.tensor(y_list, dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset wrappers
# ---------------------------------------------------------------------------


class UTHARDataset(Dataset):
    """PyTorch Dataset for UT-HAR. Loads entire split into memory on init."""

    def __init__(self, split: Literal["train", "val", "test"]) -> None:
        self.X, self.y = load_ut_har(split)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.X[idx], int(self.y[idx].item())


class NTUFiHARDataset(Dataset):
    """PyTorch Dataset for NTU-Fi HAR. Loads .mat files lazily per __getitem__."""

    def __init__(self, split: Literal["train", "test"]) -> None:
        split_name = "train_amp" if split == "train" else "test_amp"
        split_dir = NTU_FI_HAR_DIR / split_name
        classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples: list[tuple[Path, int]] = []
        for class_name, label_idx in self.class_to_idx.items():
            for mat_path in sorted((split_dir / class_name).glob("*.mat")):
                self.samples.append((mat_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        csi = sio.loadmat(str(path))["CSIamp"]
        return torch.tensor(csi, dtype=torch.float32), label


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Convenience wrapper around DataLoader with sensible defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
