#!/usr/bin/env python3
"""Characterize the MM-Fi WiFi-CSI → 3D-pose subset and render real poses.

Phase 3 milestone. Unlike the classification chunks (1–12), MM-Fi is a
REGRESSION dataset: WiFi CSI in, continuous 3D joint coordinates out. This
script is the first time the project draws actual human poses.

It prints, for whatever MM-Fi data is present under ``data/raw/mmfi/``:
  * sample counts by environment / subject / action (from the cheap metadata
    census — no arrays loaded),
  * CSI tensor shape + value range and keypoint array shape + per-axis range,
    computed over a small random sample of frames,
  * a sanity check that joint count (17) and CSI shape (3×114×10) match the
    documented MM-Fi format.

It then saves ``figures/mmfi_gt_skeletons.png``: a grid of 8 GROUND-TRUTH
skeletons drawn across 8 different actions with ``plot_skeleton_3d`` — the
poses a model will later try to predict from CSI alone.

The MM-Fi dataset is NOT downloaded by the pipeline (it is large and behind a
Google Drive link). If the data is missing, this script fails loudly with a
pointer to ``docs/chunk13_mmfi_setup.md``.

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/explore_mmfi.py [--split all] [--sample N] [--seed S]

Runnable directly (not via ``-m``): it prepends the repo root to sys.path
itself so the ``src`` package imports resolve.
"""

import argparse
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mmfi_loader import (  # noqa: E402
    CSI_FRAME_SHAPE,
    NUM_KEYPOINTS,
    available_scenes,
    load_mmfi,
)
from src.viz.skeleton import JOINT_NAMES, SKELETON_EDGES, plot_skeleton_3d  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
GT_SKELETON_FIG = FIGURES_DIR / "mmfi_gt_skeletons.png"


def print_header(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def print_counts(meta: list[dict], key: str) -> None:
    counts = Counter(m[key] for m in meta)
    print(f"\n  Counts by {key}:")
    total = sum(counts.values())
    for name, count in sorted(counts.items()):
        print(f"    {str(name):>8s}: {count:7d}  ({100.0 * count / total:5.1f}%)")
    print(f"    {'(distinct)':>8s}: {len(counts):7d}")


def describe_arrays(subset, n_sample: int, seed: int) -> None:
    """Open a random subset of frames and report CSI + keypoint stats."""
    rng = np.random.default_rng(seed)
    n = len(subset)
    if n == 0:
        print("\n  (no samples to describe)")
        return
    pick = rng.choice(n, size=min(n_sample, n), replace=False)

    csi_min, csi_max = np.inf, -np.inf
    kp_mins = np.full(3, np.inf)
    kp_maxs = np.full(3, -np.inf)
    csi_shape = kp_shape = None
    for i in pick:
        s = subset[int(i)]
        csi, kp = s["csi"], s["keypoints"]
        csi_shape, kp_shape = csi.shape, kp.shape
        csi_min, csi_max = min(csi_min, float(csi.min())), max(csi_max, float(csi.max()))
        flat = kp.reshape(-1, 3)
        kp_mins = np.minimum(kp_mins, flat.min(0))
        kp_maxs = np.maximum(kp_maxs, flat.max(0))

    print(f"\n  Sampled {len(pick)} of {n} frames:")
    print(f"    CSI       shape={csi_shape}  range=[{csi_min:.3f}, {csi_max:.3f}]")
    print(f"    keypoints shape={kp_shape}  (joints={NUM_KEYPOINTS}, dim=3)")
    for axis, name in enumerate("xyz"):
        print(f"      {name}: [{kp_mins[axis]:+.3f}, {kp_maxs[axis]:+.3f}] m")

    ok_csi = tuple(csi_shape) == CSI_FRAME_SHAPE
    ok_kp = tuple(kp_shape) == (NUM_KEYPOINTS, 3)
    print(f"\n    sanity: CSI shape == {CSI_FRAME_SHAPE}? {ok_csi}  | "
          f"keypoints == ({NUM_KEYPOINTS}, 3)? {ok_kp}")


def pick_one_per_action(subset, n: int, seed: int) -> list[int]:
    """Choose up to ``n`` sample indices, each from a distinct action."""
    rng = np.random.default_rng(seed)
    by_action: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, m in enumerate(subset.metadata):
        by_action.setdefault(m["action"], []).append(i)
    actions = sorted(by_action)
    chosen = []
    for a in actions:
        idxs = by_action[a]
        chosen.append(int(rng.choice(idxs)))
        if len(chosen) >= n:
            break
    # If fewer than n distinct actions exist, top up with random other frames.
    if len(chosen) < n:
        extra_pool = [i for i in range(len(subset)) if i not in set(chosen)]
        rng.shuffle(extra_pool)
        chosen.extend(extra_pool[: n - len(chosen)])
    return chosen[:n]


def save_skeleton_grid(subset, seed: int) -> None:
    """Render a 2×4 grid of ground-truth skeletons across different actions."""
    idxs = pick_one_per_action(subset, 8, seed)
    if not idxs:
        print("  (no samples available to plot)")
        return

    fig = plt.figure(figsize=(16, 8))
    for k, i in enumerate(idxs):
        s = subset[i]
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        plot_skeleton_3d(s["keypoints"], ax=ax, color="tab:blue")
        ax.set_title(f"{s['action']}  {s['subject']}/{s['scene']}", fontsize=10)
        ax.view_init(elev=12, azim=-70)
    fig.suptitle(
        "MM-Fi ground-truth 3D poses (regression targets) — "
        "17 joints, Human3.6M skeleton",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GT_SKELETON_FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {len(idxs)} ground-truth skeletons -> {GT_SKELETON_FIG}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="all",
                        choices=["train", "val", "all"],
                        help="Partition to explore (default: all).")
    parser.add_argument("--split-strategy", default="random_split",
                        help="MM-Fi split protocol (default: random_split).")
    parser.add_argument("--protocol", default="protocol3",
                        help="Action subset: protocol1/2/3 (default: protocol3).")
    parser.add_argument("--sample", type=int, default=200,
                        help="How many frames to open for shape/range stats.")
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (folder holding E01 ...). "
                             "Defaults to data/raw/mmfi/.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print_header("MM-Fi setup")
    print(f"  Environments present on disk: "
          f"{available_scenes(args.data_root) or '(none)'}")
    print(f"  Joint table ({NUM_KEYPOINTS}): " + ", ".join(JOINT_NAMES))
    print(f"  Kinematic tree: {len(SKELETON_EDGES)} bones")

    # Loader raises FileNotFoundError with a helpful message if data is missing;
    # surface it cleanly (no traceback) so the pipeline fails loudly but readably.
    try:
        subset = load_mmfi(
            modality="wifi-csi",
            split=args.split,
            protocol=args.protocol,
            split_strategy=args.split_strategy,
            data_unit="frame",
            data_root=args.data_root,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    meta = subset.metadata

    print_header(f"Census — split={args.split!r} ({len(subset)} frames)")
    print_counts(meta, "scene")
    print_counts(meta, "subject")
    print_counts(meta, "action")

    print_header("CSI + keypoint shapes / ranges")
    describe_arrays(subset, args.sample, args.seed)

    print_header("Ground-truth skeletons")
    save_skeleton_grid(subset, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
