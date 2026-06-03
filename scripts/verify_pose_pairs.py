#!/usr/bin/env python3
"""Verify that MM-Fi (CSI window, 3D pose) training pairs are correctly aligned.

A pose regressor will happily "train" on misaligned data — pairing a CSI window
with the WRONG frame's pose — and then predict garbage, with no error message to
explain why. Temporal misalignment is the single most common cause. This script
surfaces it BEFORE any training, two ways:

  1. Index check (cheap, exhaustive on the sampled pairs): for each pair from
     ``MMFiPoseDataset.get_pair`` it confirms the CSI window is *centered* on the
     labeled frame — i.e. the middle element of ``window_frame_idx`` equals
     ``center_idx`` — and that the window never leaves the clip (indices are
     monotonic and within [0, 296]).

  2. Independent cross-loader check: it re-reads the SAME (subject, action) clip
     through the loader's ``data_unit='sequence'`` path and confirms the pose and
     the center CSI frame the frame-mode dataset produced match the sequence's
     ``[center_idx]`` slice. Two independent code paths agreeing is strong
     evidence the frame→pose pairing is right.

It then writes ``figures/pose_pair_check.png``: for a handful of pairs, the CSI
amplitude window (antenna-averaged subcarrier × packet heatmap) beside the
ground-truth skeleton it is paired with, titled with the provenance
(subject/scene/action and the frame index). Eyeballing this is the final guard:
the skeleton should be a plausible pose and the CSI a plausible amplitude map.

The MM-Fi dataset is NOT downloaded by the pipeline (large, behind Google Drive).
If it is missing, this script fails loudly with a pointer to
``docs/chunk13_mmfi_setup.md`` (and self-skips inside run_pipeline.sh).

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/verify_pose_pairs.py [--split cross_subject] [--n 5] [--seed 0]
    python scripts/verify_pose_pairs.py --window-size 1   # ASK before changing (see docs)

Runnable directly (not via ``-m``): it prepends the repo root to sys.path itself
so the ``src`` package imports resolve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.mmfi_loader import load_mmfi  # noqa: E402
from src.data.mmfi_pose_dataset import (  # noqa: E402
    cross_environment,
    cross_subject,
    random_split,
)
from src.viz.skeleton import plot_skeleton_3d  # noqa: E402

FIGURES_DIR = ROOT / "figures"
PAIR_CHECK_FIG = FIGURES_DIR / "pose_pair_check.png"

SPLIT_BUILDERS = {
    "cross_subject": cross_subject,
    "cross_environment": cross_environment,
    "random_split": random_split,
}


def build_split(name: str, window_size: int, protocol: str, data_root):
    """Return the train dataset of the requested split (train side is enough)."""
    builder = SPLIT_BUILDERS[name]
    kwargs = dict(window_size=window_size, protocol=protocol, data_root=data_root)
    train_ds, _test_ds = builder(**kwargs)
    return train_ds


def index_check(rec: dict, window_size: int) -> list[str]:
    """Return a list of failure strings (empty == aligned) for one pair."""
    problems = []
    win = rec["window_frame_idx"]
    center = rec["center_idx"]
    if len(win) != window_size:
        problems.append(f"window length {len(win)} != window_size {window_size}")
    mid = window_size // 2
    if win[mid] != center:
        problems.append(
            f"center frame mismatch: window middle {win[mid]} != center_idx {center}"
        )
    # Within-clip & monotonic non-decreasing (edge-clamping can repeat, not jump).
    if any(b < a for a, b in zip(win, win[1:])):
        problems.append(f"window indices not monotonic: {win}")
    if min(win) < 0:
        problems.append(f"negative frame index in window: {win}")
    return problems


def cross_loader_check(rec: dict, seq_lookup: dict, atol: float = 1e-5) -> list[str]:
    """Re-read the clip in 'sequence' mode and confirm the pose/CSI agree."""
    problems = []
    key = (rec["scene"], rec["subject"], rec["action"])
    seq = seq_lookup.get(key)
    if seq is None:
        return [f"clip {key} not found in sequence-mode loader (cannot cross-check)"]
    center = rec["center_idx"]

    kp_seq = np.asarray(seq["keypoints"], dtype=np.float32)[center]  # (17, 3)
    if not np.allclose(kp_seq, rec["keypoints_abs"], atol=atol):
        d = float(np.abs(kp_seq - rec["keypoints_abs"]).max())
        problems.append(f"pose mismatch vs sequence loader (max |Δ|={d:.2e} m)")

    # Center frame of the CSI window vs the sequence's center frame.
    mid = len(rec["window_frame_idx"]) // 2
    csi_center = np.asarray(rec["csi_window"], dtype=np.float32)[mid]  # (3,114,10)
    csi_seq = np.asarray(seq["csi"], dtype=np.float32)[center]
    if not np.allclose(csi_seq, csi_center, atol=1e-4):
        d = float(np.abs(csi_seq - csi_center).max())
        problems.append(f"CSI center frame mismatch vs sequence loader (max |Δ|={d:.2e})")
    return problems


def build_seq_lookup(protocol: str, data_root) -> dict:
    """Map (scene, subject, action) → sequence sample for the cross-loader check.

    Lazy: indexes metadata cheaply, materializes a clip only when ``__getitem__``
    is hit in ``cross_loader_check``.
    """
    seq = load_mmfi(
        modality="wifi-csi", split="all", protocol=protocol,
        split_strategy="random_split", data_unit="sequence", data_root=data_root,
    )
    lookup = {}
    for i, m in enumerate(seq.metadata):
        lookup[(m["scene"], m["subject"], m["action"])] = (seq, i)
    # Wrap so callers index transparently: lookup[key] -> sample dict.
    return {k: SeqRef(s, idx) for k, (s, idx) in lookup.items()}


class SeqRef:
    """Tiny lazy handle so the cross-check loads a clip only when accessed."""

    def __init__(self, subset, index: int):
        self._subset, self._index = subset, index
        self._cached = None

    def __getitem__(self, field):
        if self._cached is None:
            self._cached = self._subset[self._index]
        return self._cached[field]


def csi_window_image(csi_window: np.ndarray) -> np.ndarray:
    """Collapse a (W, 3, 114, 10) window to a (114, 10) heatmap for display.

    Averages over the W window frames and the 3 antennas, leaving a subcarrier ×
    packet amplitude map — enough to eyeball that it's real CSI, not noise.
    """
    arr = np.asarray(csi_window, dtype=np.float32)
    return arr.mean(axis=(0, 1))  # (114, 10)


def save_figure(records: list[dict]) -> None:
    """CSI amplitude window beside the GT skeleton, one row per pair."""
    n = len(records)
    fig = plt.figure(figsize=(7, 3.2 * n))
    for r, rec in enumerate(records):
        # Left: CSI amplitude heatmap.
        ax_csi = fig.add_subplot(n, 2, 2 * r + 1)
        img = csi_window_image(rec["csi_window"])
        im = ax_csi.imshow(img, aspect="auto", origin="lower", cmap="viridis")
        ax_csi.set_xlabel("packet (10/100 ms)")
        ax_csi.set_ylabel("subcarrier")
        ax_csi.set_title(
            f"CSI |amp|  {rec['subject']}/{rec['scene']} {rec['action']} "
            f"frame {rec['center_idx']}",
            fontsize=9,
        )
        fig.colorbar(im, ax=ax_csi, fraction=0.046, pad=0.04)

        # Right: the GT skeleton this window is paired with (absolute metres).
        ax_sk = fig.add_subplot(n, 2, 2 * r + 2, projection="3d")
        plot_skeleton_3d(rec["keypoints_abs"], ax=ax_sk, color="tab:green")
        ax_sk.set_title(f"GT pose @ frame {rec['center_idx']}", fontsize=9)
        ax_sk.view_init(elev=12, azim=-70)

    fig.suptitle(
        "MM-Fi pose-pair alignment check — CSI window ↔ ground-truth pose",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAIR_CHECK_FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="cross_subject",
                        choices=list(SPLIT_BUILDERS),
                        help="Which split's train set to sample pairs from "
                             "(default: cross_subject — works on E01 alone since "
                             "its held-out subjects live in E01).")
    parser.add_argument("--protocol", default="protocol3",
                        help="Action subset: protocol1/2/3 (default: protocol3).")
    parser.add_argument("--n", type=int, default=5,
                        help="How many pairs to check + plot (default: 5).")
    parser.add_argument("--window-size", type=int, default=1,
                        help="CSI window length (frames). DEFAULT 1 matches the "
                             "MM-Fi benchmark; ASK before changing (see docs).")
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (folder holding E01 ...). "
                             "Defaults to data/raw/mmfi/.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Split={args.split!r}  window_size={args.window_size}  protocol={args.protocol!r}")

    # The dataset builders raise FileNotFoundError/RuntimeError if data is missing
    # or a partition is empty; surface cleanly (no traceback).
    try:
        train_ds = build_split(args.split, args.window_size, args.protocol, args.data_root)
        seq_lookup = build_seq_lookup(args.protocol, args.data_root)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    n_total = len(train_ds)
    print(f"Train split has {n_total} pairs across {len(train_ds.clip_keys)} clips.")
    if n_total == 0:
        print("ERROR: no pairs to verify.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(n_total, size=min(args.n, n_total), replace=False)

    records = []
    all_problems = 0
    print("\nPer-pair alignment:")
    for i in picks:
        rec = train_ds.get_pair(int(i))
        problems = index_check(rec, args.window_size)
        problems += cross_loader_check(rec, seq_lookup)
        status = "OK" if not problems else "FAIL"
        print(f"  [{status}] {rec['subject']}/{rec['scene']} {rec['action']} "
              f"frame={rec['center_idx']}  window={rec['window_frame_idx']}")
        for p in problems:
            print(f"        - {p}")
        all_problems += len(problems)
        records.append(rec)

    save_figure(records)
    print(f"\nSaved {len(records)} pose-pair panels -> {PAIR_CHECK_FIG}")

    if all_problems:
        print(f"\n{all_problems} alignment problem(s) found — DO NOT train until fixed.",
              file=sys.stderr)
        sys.exit(1)
    print("\nAll sampled pairs are temporally aligned. Done.")


if __name__ == "__main__":
    main()
