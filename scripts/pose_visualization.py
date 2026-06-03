#!/usr/bin/env python
"""The Phase-3 capstone: a 3D human skeleton inferred from WiFi, animated.

This is the deliverable the whole pose phase has been building toward — a moving
human skeleton *predicted from WiFi CSI alone*, rendered beside the camera-derived
ground truth so the gap is honest and visible, animated across a real motion
sequence.

What it does
------------
1. Loads the trained regressor (``runs/best_pose.pt``) and rebuilds it from the
   checkpoint's own config.
2. Builds the **cross_subject** split and picks ONE held-out motion clip — a
   person the model never saw in training (the only honest setting for "what
   would this do in the wild"). By default it auto-selects the held-out clip with
   the most body movement (most worth animating); ``--subject`` / ``--action``
   override.
3. Runs the model frame-by-frame over that clip to get a predicted pose per
   frame, and reads the ground-truth pose per frame.
4. Renders an animation with two 3D panels:
     * LEFT  — predicted skeleton (red) overlaid on ground truth (green) via
       ``src/viz/skeleton.plot_skeleton_pair``, titled with the per-frame MPJPE;
     * RIGHT — prediction ONLY (no ground truth), i.e. exactly what the system
       outputs "in the wild" with no camera present — the actual use case for
       WiFi pose.
   Each frame is annotated with the running (cumulative) MPJPE over the whole
   sequence, and the figure is titled with the action label, subject id, and
   "WiFi-predicted vs ground-truth 3D pose".
5. Saves ``figures/pose_prediction.gif`` (Pillow writer). If GIF writing is
   unavailable it falls back to a multi-frame PNG strip. A static strip
   (``figures/pose_prediction_strip.png``) is always written too, as a
   doc-friendly companion to the animation.

Honesty (same principle as every prior visualization chunk)
-----------------------------------------------------------
The predicted skeleton is **raw model output** — no temporal smoothing, no
filtering. WiFi pose is coarse: the prediction jitters and the extremities
(wrists especially) wander. That is the real result, not a bug to hide. Per the
project brief, smoothing would be asked-about first; this script never adds it.
Axis limits ARE fixed across the sequence, but that is only camera framing (so
the view doesn't jump scale) — it does not touch the poses.

Example (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/pose_visualization.py
    python scripts/pose_visualization.py --subject S05 --action A17 --fps 12

Data: MM-Fi is large and not auto-downloaded (see docs/chunk13_mmfi_setup.md).
The cross_subject default held-out subjects (S05, S10) live in E01, so this works
on the E01-only partial download. Runs directly (prepends repo root to sys.path).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mmfi_pose_dataset import MMFiPoseDataset, cross_subject  # noqa: E402
from src.models import build_model  # noqa: E402
from src.viz.skeleton import plot_skeleton_3d, plot_skeleton_pair  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = PROJECT_ROOT / "runs" / "best_pose.pt"
FIGURES_DIR = PROJECT_ROOT / "figures"
GIF_PATH = FIGURES_DIR / "pose_prediction.gif"
STRIP_PATH = FIGURES_DIR / "pose_prediction_strip.png"


def _to_plot_frame(kp: np.ndarray) -> np.ndarray:
    """Same camera→upright remap src/viz/skeleton uses: (x, y, z) → (x, z, -y).

    Replicated here only to precompute fixed axis limits over the whole clip; the
    actual skeleton drawing is delegated to ``src/viz/skeleton`` (which applies
    the identical transform internally).
    """
    return np.stack([kp[..., 0], kp[..., 2], -kp[..., 1]], axis=-1)


# ---------------------------------------------------------------------------
# Clip selection + inference
# ---------------------------------------------------------------------------


def clip_motion(subset, global_indices: list[int], n_probe: int = 4) -> float:
    """Cheap movement score for a clip: total spread of joints over time.

    Reads ``n_probe`` evenly-spaced frames' ground-truth keypoints and returns
    the summed per-joint coordinate standard deviation across them. Higher =
    more body movement = a more interesting clip to animate.
    """
    if not global_indices:
        return 0.0
    picks = np.linspace(0, len(global_indices) - 1, min(n_probe, len(global_indices)))
    poses = [np.asarray(subset[global_indices[int(p)]]["keypoints"], dtype=np.float32)
             for p in picks]
    return float(np.stack(poses, axis=0).std(axis=0).sum())


def choose_clip(test_ds, subject: str | None, action: str | None):
    """Return the ``(scene, subject, action)`` clip key to animate.

    If ``subject`` and/or ``action`` are given, find the matching held-out clip
    (error if absent). Otherwise auto-pick the held-out clip with the most motion.
    """
    keys = test_ds.clip_keys
    if subject or action:
        cand = [k for k in keys
                if (subject is None or k[1] == subject)
                and (action is None or k[2] == action)]
        if not cand:
            raise SystemExit(
                f"No held-out clip matches subject={subject} action={action}.\n"
                f"Available held-out clips: "
                f"{sorted({(k[1], k[2]) for k in keys})}"
            )
        return cand[0]
    print(f"Auto-selecting the most-motion clip among {len(keys)} held-out clips ...")
    scored = [(clip_motion(test_ds.subset, test_ds.clips[k]), k) for k in keys]
    scored.sort(reverse=True)
    return scored[0][1]


@torch.no_grad()
def run_clip(model, clip_ds, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over every frame of a single clip, in temporal order.

    Returns ``(gt, pred, per_frame_mpjpe_mm)`` where gt/pred are
    ``(T, 17, 3)`` root-relative metres and the MPJPE is per frame (mm). One
    batched forward pass over the whole clip (~297 frames) — no smoothing.
    """
    model.eval()
    csis, gts = [], []
    for i in range(len(clip_ds)):
        csi, kp = clip_ds[i]
        csis.append(csi)
        gts.append(kp.numpy())
    csi_batch = torch.stack(csis, dim=0).to(device)
    pred = model(csi_batch).cpu().numpy()  # (T, 17, 3)
    gt = np.stack(gts, axis=0)             # (T, 17, 3)
    per_frame = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1) * 1000.0  # (T,)
    return gt, pred, per_frame


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fixed_limits(gt: np.ndarray, pred: np.ndarray):
    """Cubic, sequence-wide plot-frame limits so the view never jumps scale."""
    pts = _to_plot_frame(np.concatenate([gt.reshape(-1, 3), pred.reshape(-1, 3)], 0))
    mins, maxs = pts.min(0), pts.max(0)
    centers = (mins + maxs) / 2.0
    radius = float((maxs - mins).max()) / 2.0 or 1.0
    return centers, radius


def _apply_limits(ax, centers, radius):
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=8, azim=-72)


def render(gt, pred, per_frame, key, out_gif, out_strip, *, stride, fps, dpi):
    """Animate predicted-vs-GT (left) + prediction-only (right); save GIF + strip.

    The running MPJPE annotated on each rendered frame is the cumulative mean over
    the FULL-resolution sequence up to that frame (faithful even when ``stride``
    subsamples which frames are drawn).
    """
    scene, subject, action = key
    centers, radius = _fixed_limits(gt, pred)
    running = np.cumsum(per_frame) / np.arange(1, len(per_frame) + 1)  # full-res
    seq_mpjpe = float(per_frame.mean())
    frame_ids = list(range(0, len(gt), max(stride, 1)))

    title = (f"WiFi-predicted vs ground-truth 3D pose\n"
             f"action {action} | subject {subject} ({scene}, held out) | "
             f"sequence MPJPE {seq_mpjpe:.0f} mm")

    fig = plt.figure(figsize=(11, 6))
    ax_pair = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")

    def draw(t: int):
        ax_pair.cla()
        ax_pred.cla()
        # LEFT: prediction overlaid on ground truth (shared axes, shared frame).
        plot_skeleton_pair(gt[t], pred[t], ax=ax_pair, show_error=True)
        ax_pair.set_title(f"frame {t:3d}/{len(gt) - 1}   "
                          f"MPJPE {per_frame[t]:.0f} mm   "
                          f"(running {running[t]:.0f} mm)", fontsize=9)
        # RIGHT: prediction only — what WiFi outputs with no camera present.
        plot_skeleton_3d(pred[t], ax=ax_pred, color="tab:red", label="WiFi prediction")
        ax_pred.set_title("prediction only — 'in the wild' (no camera)", fontsize=9)
        ax_pred.legend(loc="upper right", fontsize=7)
        for ax in (ax_pair, ax_pred):
            _apply_limits(ax, centers, radius)
        fig.suptitle(title, fontsize=11)
        return []

    # --- GIF (the deliverable) ---
    saved_gif = False
    try:
        anim = FuncAnimation(fig, draw, frames=frame_ids, interval=1000 / fps, blit=False)
        anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=dpi)
        saved_gif = True
        print(f"Saved animation: {out_gif}  ({len(frame_ids)} frames @ {fps} fps)")
    except Exception as e:  # pragma: no cover — environment-dependent
        print(f"GIF writing failed ({e}); falling back to the PNG strip only.")

    # --- Static multi-frame strip (always; fallback if GIF failed) ---
    n_panels = min(6, len(gt))
    strip_ids = np.linspace(0, len(gt) - 1, n_panels).astype(int)
    fig_s = plt.figure(figsize=(3.2 * n_panels, 4.2))
    for col, t in enumerate(strip_ids):
        ax = fig_s.add_subplot(1, n_panels, col + 1, projection="3d")
        plot_skeleton_pair(gt[t], pred[t], ax=ax, show_error=False)
        ax.set_title(f"frame {t}\nMPJPE {per_frame[t]:.0f} mm", fontsize=8)
        if col != n_panels - 1:
            leg = ax.get_legend()
            if leg:
                leg.remove()
        _apply_limits(ax, centers, radius)
    fig_s.suptitle(title, fontsize=11)
    fig_s.tight_layout(rect=(0, 0, 1, 0.92))
    fig_s.savefig(out_strip, dpi=dpi, bbox_inches="tight")
    plt.close(fig_s)
    plt.close(fig)
    print(f"Saved strip:     {out_strip}")
    return saved_gif


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT,
                   help="Pose checkpoint (default: runs/best_pose.pt).")
    p.add_argument("--subject", default=None,
                   help="Held-out subject to animate (default: auto-pick by motion).")
    p.add_argument("--action", default=None,
                   help="Action code e.g. A17 (default: auto-pick by motion).")
    p.add_argument("--stride", type=int, default=3,
                   help="Render every Nth frame in the GIF (running MPJPE stays full-res).")
    p.add_argument("--fps", type=int, default=10, help="GIF frames per second.")
    p.add_argument("--dpi", type=int, default=110, help="Output DPI.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train one first:  python src/train_pose.py   (see docs/chunk15_pose_model.md)"
        )
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    ckpt_args = ckpt.get("args", {})

    ds_kwargs = dict(
        window_size=ckpt["model_config"].get("window_size", 1),
        csi_normalize=ckpt_args.get("csi_normalize", "none"),
        pose_scale=None,
    )
    # Build the cross_subject split (held-out = the bodies the model never saw).
    _, test_ds = cross_subject(
        test_subjects=ckpt_args.get("test_subjects"),
        protocol=ckpt_args.get("protocol", "protocol3"),
        data_root=ckpt_args.get("data_root"),
        limit=ckpt_args.get("limit"),
        **ds_kwargs,
    )

    key = choose_clip(test_ds, args.subject, args.action)
    print(f"Animating held-out clip: scene={key[0]} subject={key[1]} action={key[2]}")

    # A single-clip dataset whose items are exactly this clip in temporal order.
    clip_ds = MMFiPoseDataset(test_ds.subset, {key: test_ds.clips[key]}, **ds_kwargs)
    gt, pred, per_frame = run_clip(model, clip_ds, device)
    print(f"Frames: {len(gt)} | sequence MPJPE {per_frame.mean():.1f} mm "
          f"(min {per_frame.min():.0f} / max {per_frame.max():.0f})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    render(gt, pred, per_frame, key, GIF_PATH, STRIP_PATH,
           stride=args.stride, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
