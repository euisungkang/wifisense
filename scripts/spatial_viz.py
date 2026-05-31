#!/usr/bin/env python
"""Spatial-motion visualization for Widar3.0 BVP — the chunk-12 milestone figure.

This is the tier-2 analog of chunk 6's 3-panel continuous-capture figure. Where
chunk 6 showed *time vs. predicted activity* over a raw-CSI stream, BVP lets us
show something raw CSI never could: the **spatial shape of the motion itself**.
A BVP frame is a 20x20 map of where the body is moving in velocity space *right
now*; integrating that velocity over time reconstructs the path the hand traced.

For six representative gestures the figure draws, per gesture, a two-row block:

    Row 1 — six BVP frames evenly sampled across the gesture (20x20 each), the
            raw per-frame energy over the body-frame velocity plane (+/-2 m/s).
    Row 2 — the integrated motion trajectory: the cumulative path obtained by
            integrating each frame's energy-centroid velocity over time
            (position = sum of v*dt, dt = 1/10 s), with arrows marking direction
            of travel (green = start, red = end).

Each block is titled with the gesture name, the model's predicted class, and the
ground truth. Misclassifications are **left visible and flagged** (red title) —
the same raw-output-first honesty as chunk 6: no cherry-picking, no smoothing.

The model is the trained BVP CNN-RNN (``runs/best_bvp_<split>.pt``); predictions
use the chunk-11 preprocessing (per-sample z-score + pad/truncate) the model was
trained on. The trajectory is a *physical* property of the motion, computed from
the raw non-negative energy independent of the model.

Outputs:
    * ``figures/spatial_motion.png`` — the six-gesture composite.

Examples (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/spatial_viz.py
    python scripts/spatial_viz.py --checkpoint runs/best_bvp_cross_user.pt --room 1

Runnable directly (not via ``-m``): it prepends the repo root to sys.path itself.
"""

from __future__ import annotations

import argparse
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
    BVP_SAMPLE_RATE_HZ,
    GRID_SIZE,
    PROJECT_ROOT,
    VELOCITY_RANGE_MPS,
    index_widar_bvp,
    load_bvp_file,
)
from src.models import build_model  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Six representative, well-populated, visually distinct gestures (some linear,
# some looping/zig-zag) so the trajectories read differently from one another.
DEFAULT_GESTURES = [
    "Push&Pull", "Sweep", "Slide", "Clap", "Draw-O(H)", "Draw-Zigzag(H)",
]
N_FRAMES = 6          # BVP timesteps shown in row 1
CENTROID_Q = 0.75     # keep top-quartile cells when locating each frame's centroid
RUNS_ROOT = PROJECT_ROOT / "runs"

_EDGES = np.linspace(*VELOCITY_RANGE_MPS, GRID_SIZE + 1)
_CENTERS = (_EDGES[:-1] + _EDGES[1:]) / 2


def centroid_velocity(x: np.ndarray, q: float = CENTROID_Q) -> np.ndarray:
    """Per-frame energy-centroid velocity (m/s); NaN rows for empty frames.

    Same construction as the chunk-11 pipeline demo: per frame, threshold at the
    q-quantile and take the energy-weighted centroid over the velocity grid.
    """
    gx, gy = np.meshgrid(_CENTERS, _CENTERS, indexing="ij")  # (v_x, v_y)
    out = np.full((x.shape[0], 2), np.nan, dtype=np.float64)
    for t, frame in enumerate(x):
        frame = np.clip(frame, 0.0, None)
        thr = np.quantile(frame, q)
        w = np.where(frame >= thr, frame, 0.0)
        total = w.sum()
        if total > 0:
            out[t] = (np.sum(w * gx) / total, np.sum(w * gy) / total)
    return out


def integrate_path(vel: np.ndarray, dt: float) -> np.ndarray:
    """Integrate per-frame velocity (m/s) into a cumulative position path (m).

    Empty frames (NaN velocity) contribute zero displacement so the path stays
    continuous. Starts at the origin.
    """
    v = np.nan_to_num(vel, nan=0.0)
    disp = v * dt
    pos = np.cumsum(disp, axis=0)
    return np.vstack([[0.0, 0.0], pos])  # prepend the origin (t=0)


@torch.no_grad()
def predict(model, raw: np.ndarray, target_T: int, device) -> tuple[int, float]:
    """Run the model on one raw BVP volume; return (pred_idx, confidence)."""
    x = pad_or_truncate(normalize_bvp(raw, mode="per_sample"), target_T)
    xb = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, T, 20, 20)
    p = F.softmax(model(xb), dim=1).cpu().numpy()[0]
    return int(p.argmax()), float(p.max())


def pick_sample(gesture: str, filters: dict, seed: int) -> tuple[dict, np.ndarray] | None:
    """Deterministically pick one (metadata, raw volume) for *gesture*.

    Tries candidates in a fixed shuffled order and returns the first that loads
    cleanly with at least one timestep, so a truncated/empty file (a known
    corpus defect) never breaks the figure.
    """
    idx = index_widar_bvp(gesture=gesture, **filters)
    if not idx:
        return None
    order = np.random.default_rng(seed).permutation(len(idx))
    for k in order:
        md = idx[int(k)]
        try:
            raw = load_bvp_file(md["path"])
        except Exception:
            continue
        if raw.shape[0] > 0:
            return md, raw
    return None


def draw_frames(axes, raw: np.ndarray) -> None:
    """Row 1: N_FRAMES evenly-sampled raw BVP frames over the velocity plane."""
    vmin, vmax = VELOCITY_RANGE_MPS
    T = raw.shape[0]
    ts = np.linspace(0, T - 1, N_FRAMES).round().astype(int)
    vmax_e = float(raw.max()) or 1.0
    for ax, t in zip(axes, ts):
        ax.imshow(raw[t].T, origin="lower", cmap="inferno",
                  extent=[vmin, vmax, vmin, vmax], aspect="equal", vmin=0, vmax=vmax_e,
                  interpolation="nearest")  # raw cells, no smoothing (chunk-6 principle)
        ax.set_title(f"t={t} ({t / BVP_SAMPLE_RATE_HZ:.1f}s)", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])


def draw_trajectory(ax, raw: np.ndarray) -> None:
    """Row 2: the integrated (cumulative) motion path with direction arrows."""
    vel = centroid_velocity(raw)
    path = integrate_path(vel, dt=1.0 / BVP_SAMPLE_RATE_HZ)  # (T+1, 2), metres

    segs = np.stack([path[:-1], path[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidths=2.4, zorder=2)
    lc.set_array(np.arange(len(segs)))
    ax.add_collection(lc)

    # Sparse direction arrows along the path.
    n_arrows = min(6, max(1, len(path) - 1))
    for k in np.linspace(0, len(path) - 2, n_arrows).round().astype(int):
        p0, p1 = path[k], path[k + 1]
        if np.allclose(p0, p1):
            continue
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.1))
    ax.scatter(*path[0], marker="o", s=60, facecolors="none", edgecolors="lime",
               linewidths=1.8, zorder=4, label="start")
    ax.scatter(*path[-1], marker="s", s=60, facecolors="none", edgecolors="red",
               linewidths=1.8, zorder=4, label="end")

    # Symmetric, equal-aspect extent around the path so shape isn't distorted.
    span = float(np.abs(path).max()) * 1.25 or 0.1
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.axhline(0, color="0.85", lw=0.8, zorder=1)
    ax.axvline(0, color="0.85", lw=0.8, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlabel("integrated $v_x$ (m)", fontsize=7)
    ax.set_ylabel("integrated $v_y$ (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="BVP CNN-RNN checkpoint (default: first runs/best_bvp_*.pt).")
    ap.add_argument("--gestures", nargs="+", default=DEFAULT_GESTURES)
    ap.add_argument("--room", type=int, default=None, help="Scope samples to one room.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        cands = sorted(RUNS_ROOT.glob("best_bvp_*.pt"))
        if not cands:
            raise SystemExit(
                "No BVP checkpoint found (runs/best_bvp_*.pt). "
                "Train one first: python src/train_widar.py")
        ckpt_path = cands[0]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    class_names = ckpt["class_names"]
    target_T = ckpt.get("split_config", {}).get("target_T", 32)
    filters = {"room": args.room} if args.room is not None else {}

    gestures = list(args.gestures)
    n = len(gestures)
    fig = plt.figure(figsize=(2.1 * N_FRAMES, 4.6 * n))
    # Per gesture: a frames sub-row (N_FRAMES cols) + a trajectory/label sub-row.
    outer = fig.add_gridspec(n, 1, hspace=0.55)

    for gi, gesture in enumerate(gestures):
        block = outer[gi].subgridspec(2, N_FRAMES, height_ratios=[1, 1.35], hspace=0.45)
        frame_axes = [fig.add_subplot(block[0, c]) for c in range(N_FRAMES)]
        text_ax = fig.add_subplot(block[1, 0:2]); text_ax.axis("off")
        traj_ax = fig.add_subplot(block[1, 2:5])

        picked = pick_sample(gesture, filters, args.seed)
        if picked is None:
            for ax in frame_axes:
                ax.axis("off")
            traj_ax.axis("off")
            text_ax.text(0.0, 0.5, f"'{gesture}': no usable samples"
                         + (f" in room {args.room}" if args.room else ""),
                         fontsize=10, color="grey")
            continue

        md, raw = picked
        pred_idx, conf = predict(model, raw, target_T, device)
        pred_name = class_names[pred_idx]
        gt = md["gesture"]
        correct = (pred_name == gt)
        known = gt in class_names

        draw_frames(frame_axes, raw)
        draw_trajectory(traj_ax, raw)

        mark = "OK" if correct else "MISS"
        color = "#1a7d1a" if correct else "#c0271a"
        text_ax.text(
            0.0, 0.92,
            f"{gt}",
            fontsize=13, fontweight="bold", va="top",
        )
        note = "" if known else "  (gesture not in model's classes)"
        text_ax.text(
            0.0, 0.62,
            f"ground truth : {gt}\n"
            f"predicted    : {pred_name}  ({conf * 100:.0f}%)\n"
            f"result       : {mark}{note}",
            fontsize=9.5, va="top", family="monospace", color=color,
        )
        text_ax.text(
            0.0, 0.12,
            f"user{md['user']} · pos{md['position']} · ori{md['orientation']} · T={raw.shape[0]}",
            fontsize=8, va="top", color="0.4",
        )
        # A red frame border on a miss so failures pop out at a glance.
        if not correct:
            for ax in (*frame_axes, traj_ax):
                for s in ax.spines.values():
                    s.set_color("#c0271a"); s.set_linewidth(1.8)

    fig.suptitle(
        f"Widar3.0 BVP — spatial motion of six gestures  (model: {ckpt_path.name})\n"
        "Row 1: BVP energy in velocity space over time   |   "
        "Row 2: integrated motion path (green=start, red=end).  Misses flagged in red.",
        fontsize=12, y=0.997,
    )
    out = PROJECT_ROOT / "figures" / "spatial_motion.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
