"""3D human-skeleton rendering — the key new primitive for Phase 3 (pose).

Chunks 1–12 visualized *signals* (CSI heatmaps, Doppler, BVP velocity planes).
Phase 3 is pose estimation, so for the first time the project draws *people*:
the 3D joint coordinates that a model will later regress from WiFi CSI.

MM-Fi 3D pose = 17 joints in the **Human3.6M** ordering (MM-Fi paper §3:
``P_3D = {p_i ∈ R^{17×3}}``). The joint-index table and the kinematic tree
(which joints connect by a bone) are defined once here and reused everywhere.

Joint index table (Human3.6M 17-joint convention)
-------------------------------------------------
     0  Pelvis (root)        9  Neck / nose
     1  Right hip           10  Head (top)
     2  Right knee          11  Left shoulder
     3  Right ankle         12  Left elbow
     4  Left hip            13  Left wrist
     5  Left knee           14  Right shoulder
     6  Left ankle          15  Right elbow
     7  Spine (mid)         16  Right wrist
     8  Thorax (neck base)

Kinematic tree (parent → child bones), grouped by limb::

      right leg : 0–1, 1–2, 2–3
      left  leg : 0–4, 4–5, 5–6
      spine     : 0–7, 7–8, 8–9, 9–10
      left  arm : 8–11, 11–12, 12–13
      right arm : 8–14, 14–15, 15–16

                       10  head
                        |
                        9  neck/nose
                        |
        13--12--11----- 8 -----14--15--16     (L wrist..L sh | thorax | R sh..R wrist)
                        |
                        7  spine
                        |
                        0  pelvis
                       / \
                      4   1
                      |   |
                      5   2     (L / R knees)
                      |   |
                      6   3     (L / R ankles)

NOTE on axes: MM-Fi keypoints are in metres in the camera coordinate frame
(x right, y down, z forward-ish). For an upright, intuitive view we plot
(x, z, -y) so "up" on the page is up on the body; pass ``raw_axes=True`` to plot
the unmodified (x, y, z) instead. These functions never call ``plt.show()``.

Public API
----------
    plot_skeleton_3d(keypoints, ax=None, color=..., label=...) -> Axes3D
    plot_skeleton_pair(gt, pred, ax=None, ...) -> Axes3D
"""

from __future__ import annotations

import matplotlib.pyplot as plt  # noqa: F401  (registers 3D projection)
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

__all__ = [
    "JOINT_NAMES",
    "SKELETON_EDGES",
    "LEFT_JOINTS",
    "RIGHT_JOINTS",
    "plot_skeleton_3d",
    "plot_skeleton_pair",
]

# Human3.6M 17-joint names, index == position in the keypoints array.
JOINT_NAMES = [
    "pelvis",       # 0
    "r_hip",        # 1
    "r_knee",       # 2
    "r_ankle",      # 3
    "l_hip",        # 4
    "l_knee",       # 5
    "l_ankle",      # 6
    "spine",        # 7
    "thorax",       # 8
    "neck_nose",    # 9
    "head",         # 10
    "l_shoulder",   # 11
    "l_elbow",      # 12
    "l_wrist",      # 13
    "r_shoulder",   # 14
    "r_elbow",      # 15
    "r_wrist",      # 16
]

# Bones as (parent, child) index pairs — the MM-Fi / Human3.6M kinematic tree.
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),        # right leg
    (0, 4), (4, 5), (5, 6),        # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine -> head
    (8, 11), (11, 12), (12, 13),   # left arm
    (8, 14), (14, 15), (15, 16),   # right arm
]

# Left / right joint sets — used to tint limbs so orientation is readable.
LEFT_JOINTS = {4, 5, 6, 11, 12, 13}
RIGHT_JOINTS = {1, 2, 3, 14, 15, 16}


def _as_joints(keypoints) -> np.ndarray:
    """Coerce input to a clean (17, 3) float array, with clear errors."""
    kp = np.asarray(keypoints, dtype=np.float64)
    kp = np.squeeze(kp)
    if kp.shape == (3, len(JOINT_NAMES)):  # tolerate (3, 17)
        kp = kp.T
    if kp.shape != (len(JOINT_NAMES), 3):
        raise ValueError(
            f"Expected a single ({len(JOINT_NAMES)}, 3) pose, got shape "
            f"{np.asarray(keypoints).shape}."
        )
    return kp


def _to_plot_frame(kp: np.ndarray, raw_axes: bool) -> np.ndarray:
    """Map MM-Fi camera coords to a body-upright plot frame, unless raw_axes."""
    if raw_axes:
        return kp
    # (x, y, z) camera -> (x, z, -y): keep left/right (x), depth becomes the
    # second horizontal axis, and "up" (-y, since camera y points down) is the
    # vertical axis.
    return np.stack([kp[:, 0], kp[:, 2], -kp[:, 1]], axis=1)


def _new_ax(ax):
    if ax is None:
        fig = plt.figure(figsize=(4, 5))
        ax = fig.add_subplot(111, projection="3d")
    return ax


def _set_equal_aspect(ax, pts: np.ndarray) -> None:
    """Give the 3D axes a cubic aspect so the body isn't distorted."""
    mins, maxs = pts.min(0), pts.max(0)
    centers = (mins + maxs) / 2.0
    radius = float((maxs - mins).max()) / 2.0 or 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # older matplotlib
        pass


def plot_skeleton_3d(
    keypoints,
    ax=None,
    color="tab:blue",
    label=None,
    *,
    raw_axes: bool = False,
    joints: bool = True,
    tint_sides: bool = True,
    linewidth: float = 2.0,
    alpha: float = 1.0,
):
    """Draw one 3D skeleton (joints + bones) using the MM-Fi kinematic tree.

    Parameters
    ----------
    keypoints : array-like, shape (17, 3)
        One pose, MM-Fi/H36M joint order. ``(3, 17)`` is also accepted.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Target axes. A new 3D figure is created if omitted.
    color : color
        Base bone/joint color. When ``tint_sides`` is True, left-side limbs are
        drawn slightly lighter and right-side darker so orientation is legible;
        the spine uses ``color`` as-is.
    label : str, optional
        Legend label (attached to the spine line).
    raw_axes : bool
        If True, plot raw (x, y, z); otherwise remap to a body-upright frame.
    joints : bool
        Scatter the joint positions on top of the bones.
    tint_sides, linewidth, alpha
        Cosmetic controls.

    Returns
    -------
    Axes3D
        The axes drawn on (for further composition / saving by the caller).
    """
    ax = _new_ax(ax)
    kp = _to_plot_frame(_as_joints(keypoints), raw_axes)

    base = np.array(plt.matplotlib.colors.to_rgb(color))
    light = tuple(np.clip(base + 0.35 * (1 - base), 0, 1))
    dark = tuple(np.clip(base * 0.6, 0, 1))

    labelled = False
    for a, b in SKELETON_EDGES:
        if tint_sides and a in LEFT_JOINTS and b in LEFT_JOINTS:
            c = light
        elif tint_sides and a in RIGHT_JOINTS and b in RIGHT_JOINTS:
            c = dark
        else:
            c = tuple(base)
        line_label = None
        # Attach the legend label once, to a central (spine) bone.
        if label is not None and not labelled and (a, b) == (7, 8):
            line_label = label
            labelled = True
        ax.plot(
            [kp[a, 0], kp[b, 0]],
            [kp[a, 1], kp[b, 1]],
            [kp[a, 2], kp[b, 2]],
            color=c, linewidth=linewidth, alpha=alpha, label=line_label,
        )
    # Fallback: if the spine bone wasn't drawn for some reason, label the scatter.
    if label is not None and not labelled and joints:
        labelled = True

    if joints:
        ax.scatter(
            kp[:, 0], kp[:, 1], kp[:, 2],
            color=tuple(base), s=18, alpha=alpha, depthshade=True,
            label=(label if (label is not None and not labelled) else None),
        )

    _set_equal_aspect(ax, kp)
    ax.set_xlabel("x")
    ax.set_ylabel("z" if not raw_axes else "y")
    ax.set_zlabel("up" if not raw_axes else "z")
    return ax


def plot_skeleton_pair(
    gt,
    pred,
    ax=None,
    *,
    gt_color="tab:green",
    pred_color="tab:red",
    gt_label="ground truth",
    pred_label="prediction",
    raw_axes: bool = False,
    show_error: bool = True,
):
    """Overlay two skeletons (e.g. ground truth vs prediction) in shared 3D axes.

    Both poses are drawn in the SAME axes and the SAME plot frame so they are
    directly comparable. Optionally annotates the per-pose MPJPE (mean per-joint
    position error), the standard pose-estimation metric, computed in the raw
    metric coordinates regardless of the display frame.

    Parameters
    ----------
    gt, pred : array-like, shape (17, 3)
        Ground-truth and predicted poses (MM-Fi/H36M order).
    ax : Axes3D, optional
        Target axes; a new 3D figure is created if omitted.
    gt_color, pred_color, gt_label, pred_label : styling / legend.
    raw_axes : bool
        Passed through to ``plot_skeleton_3d``.
    show_error : bool
        If True, set the axes title to the MPJPE in millimetres.

    Returns
    -------
    Axes3D
    """
    ax = _new_ax(ax)
    gt_kp = _as_joints(gt)
    pred_kp = _as_joints(pred)

    plot_skeleton_3d(gt_kp, ax=ax, color=gt_color, label=gt_label,
                     raw_axes=raw_axes, tint_sides=False, alpha=0.9)
    plot_skeleton_3d(pred_kp, ax=ax, color=pred_color, label=pred_label,
                     raw_axes=raw_axes, tint_sides=False, alpha=0.9,
                     linewidth=1.6)

    # Re-fit the view to both skeletons together.
    both = np.concatenate(
        [_to_plot_frame(gt_kp, raw_axes), _to_plot_frame(pred_kp, raw_axes)], axis=0
    )
    _set_equal_aspect(ax, both)

    if show_error:
        mpjpe_m = float(np.mean(np.linalg.norm(gt_kp - pred_kp, axis=1)))
        ax.set_title(f"MPJPE = {mpjpe_m * 1000:.0f} mm")
    ax.legend(loc="upper right", fontsize=7)
    return ax
