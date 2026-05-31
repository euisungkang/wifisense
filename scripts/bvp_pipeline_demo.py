#!/usr/bin/env python
"""Show the Widar3.0 BVP preprocessing pipeline end-to-end on one gesture.

Chunk 11 turns variable-length BVP volumes into model-ready tensors via three
composable transforms (``src/data/bvp_preprocess.py``: normalize → pad/truncate,
plus train-only augment).  This script renders those stages on a single
Draw-O(H) instance so the effect of each is legible, and overlays the **motion
trajectory** that makes BVP feel spatial: a BVP frame is *where the body is
moving in velocity space right now*, so tracing each frame's energy centroid
across time draws the hand's path through the 2-D velocity plane.

Method:
    * Load one Draw-O(H) sample (user1) as a raw (T, 20, 20) energy volume.
    * Stage 1 raw:        per-frame L1-normalized energy, as loaded from disk.
    * Stage 2 normalized: ``normalize_bvp`` per-sample z-score (model input).
    * Stage 3 padded:     ``pad_or_truncate`` to a fixed T (zero frames appended).
    * For each stage, draw the time-summed 20x20 energy map over the body-frame
      velocity plane (x-velocity →, y-velocity ↑, ±2 m/s).
    * Trajectory overlay: per-frame energy-weighted centroid of the top-quartile
      cells, in m/s, traced over time (green=start → red=end).  It is a physical
      property of the motion, so it is computed once from the raw non-negative
      energy and drawn identically on all three panels — normalization rescales
      the colorbar without moving the path; padding only appends empty frames.

Outputs:
    * ``figures/bvp_pipeline_demo.png`` — the three-panel raw → normalized →
      padded figure with the motion-trajectory overlay.

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/bvp_pipeline_demo.py

Runnable directly (not via ``-m``): it prepends the repo root to sys.path itself.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.bvp_preprocess import normalize_bvp, pad_or_truncate  # noqa: E402
from src.data.widar_loader import (  # noqa: E402
    GRID_SIZE,
    PROJECT_ROOT,
    VELOCITY_RANGE_MPS,
    load_widar_bvp,
)

GESTURE = "Draw-O(H)"  # a loop, so the trajectory overlay closes on itself
TARGET_T = 32          # > the sample's T, so the padded panel shows zero frames
CENTROID_Q = 0.75      # keep cells in the top quartile when locating the centroid

# Velocity (m/s) at each grid-cell center, for placing the trajectory in physical
# coordinates. Edges span VELOCITY_RANGE_MPS across GRID_SIZE bins.
_EDGES = np.linspace(*VELOCITY_RANGE_MPS, GRID_SIZE + 1)
_CENTERS = (_EDGES[:-1] + _EDGES[1:]) / 2


def centroid_trajectory(x: np.ndarray, q: float = CENTROID_Q) -> np.ndarray:
    """Per-frame centroid of high-magnitude cells, in velocity (m/s) coords.

    For each frame, keeps cells at or above the q-quantile of that frame and
    takes their energy-weighted centroid. Frames with no energy (e.g. zero
    padding) yield NaN so they break the traced line rather than collapsing it
    to the origin.

    Args:
        x: BVP volume, shape (T, 20, 20) = (time, v_x, v_y).
        q: quantile threshold per frame.

    Returns:
        (T, 2) array of (v_x, v_y) centroids; rows are NaN for empty frames.
    """
    gx, gy = np.meshgrid(_CENTERS, _CENTERS, indexing="ij")  # match (v_x, v_y)
    out = np.full((x.shape[0], 2), np.nan, dtype=np.float64)
    for t, frame in enumerate(x):
        frame = np.clip(frame, 0.0, None)  # centroid is defined on energy only
        thr = np.quantile(frame, q)
        w = np.where(frame >= thr, frame, 0.0)
        total = w.sum()
        if total > 0:
            out[t] = (np.sum(w * gx) / total, np.sum(w * gy) / total)
    return out


def draw_panel(ax, agg: np.ndarray, traj: np.ndarray, title: str, cbar_label: str):
    """Render one stage: time-aggregated energy map + colored trajectory."""
    vmin, vmax = VELOCITY_RANGE_MPS
    # agg is (v_x, v_y); .T puts v_x horizontal, v_y vertical with origin lower.
    im = ax.imshow(
        agg.T, origin="lower", cmap="inferno",
        extent=[vmin, vmax, vmin, vmax], aspect="equal",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    valid = ~np.isnan(traj[:, 0])
    pts = traj[valid]
    if len(pts) >= 2:
        # Line segments colored by (valid-frame) time order.
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="winter", linewidths=2.2, zorder=3)
        lc.set_array(np.arange(len(segs)))
        ax.add_collection(lc)
    if len(pts) >= 1:
        order = np.arange(len(pts))
        ax.scatter(pts[:, 0], pts[:, 1], c=order, cmap="winter", s=28,
                   edgecolors="white", linewidths=0.6, zorder=4)
        ax.scatter(*pts[0], marker="o", s=70, facecolors="none",
                   edgecolors="lime", linewidths=1.8, zorder=5, label="start")
        ax.scatter(*pts[-1], marker="s", s=70, facecolors="none",
                   edgecolors="red", linewidths=1.8, zorder=5, label="end")

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$v_x$ (m/s)", fontsize=9)
    ax.set_ylabel(r"$v_y$ (m/s)", fontsize=9)
    ax.tick_params(labelsize=8)


def main() -> None:
    X, md = load_widar_bvp(user=1, gesture=GESTURE, limit=1)
    if not X:  # fall back to any user lacking it for user1
        X, md = load_widar_bvp(gesture=GESTURE, limit=1)
    raw = X[0]                                   # (T, 20, 20), raw energy
    meta = md[0]
    T = raw.shape[0]

    normed = normalize_bvp(raw, mode="per_sample")
    padded = pad_or_truncate(normed, TARGET_T)   # (TARGET_T, 20, 20)

    stages = [
        (raw, raw.sum(0), "raw (per-frame L1-normalized)", "summed energy",
         f"raw  ·  T = {T}"),
        (normed, normed.sum(0), "normalized (per-sample z-score)",
         "summed z-score", f"normalized  ·  T = {T}"),
        (padded, padded.sum(0), f"padded to T = {TARGET_T}", "summed z-score",
         f"padded  ·  {TARGET_T - T} zero frames appended"),
    ]

    # The trajectory is a physical property of the motion — identical across
    # stages — so compute it once from the raw non-negative energy and overlay
    # the same path on every panel.
    traj = centroid_trajectory(raw)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (vol, agg, title, cbar_label, sub) in zip(axes, stages):
        draw_panel(ax, agg, traj, f"{title}\n{sub}", cbar_label)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.85)

    fig.suptitle(
        f"BVP preprocessing pipeline — '{meta['gesture']}' "
        f"(user{meta['user']}, pos{meta['position']}, ori{meta['orientation']})."
        "  Overlay: per-frame energy centroid traced over time "
        "(green=start → red=end).",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = PROJECT_ROOT / "figures" / "bvp_pipeline_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
