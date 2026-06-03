"""CSI and 3D-keypoint preprocessing for the MM-Fi pose-regression task.

This is the Phase-3 (REGRESSION) analogue of ``src/data/preprocess.py`` (which
serves the classification chunks). It has two independent halves that meet in
``src/data/mmfi_pose_dataset.py``:

  * **CSI side** — turn the raw MM-Fi WiFi frames into a fixed-shape model input:
    amplitude handling, optional normalization, and the *centered temporal
    window* that pairs a short run of CSI frames with a single pose.
  * **Keypoint side** — turn an absolute (17, 3) camera-frame pose into a
    root-relative (and optionally scale-normalized) regression target, plus the
    exact inverse so predictions can be put back into metric camera space for
    MPJPE and visualization.

Coordinate conventions (READ THIS — getting it wrong scrambles every skeleton)
------------------------------------------------------------------------------
MM-Fi keypoints are **(17, 3) float32, metres, in the camera coordinate frame**:
    x → right,  y → DOWN,  z → forward (away from the camera).
The joint order is the Human3.6M 17-joint convention; index **0 is the pelvis**
(the root). The kinematic tree and the index table live in ``src/viz/skeleton.py``
and ``docs/chunk13_mmfi_setup.md`` — this module never reorders joints, it only
translates/scales coordinates, so the axis semantics above are preserved exactly.

  * We keep raw camera axes here (we do NOT apply the (x, z, -y) "body-upright"
    remap — that is a *display-only* transform done in ``src/viz/skeleton.py``).
  * Units stay in **metres** unless you opt into scale normalization, because the
    headline metric MPJPE is reported in metres/millimetres. If you scale-
    normalize the target for training you MUST un-normalize the prediction before
    computing MPJPE — see ``denormalize_pose`` and the "un-normalization recipe"
    in ``docs/chunk14_pose_pipeline.md``.

CSI conventions
---------------
Per MM-Fi, one WiFi frame is ``(3, 114, 10)`` = 3 antennas × 114 subcarriers ×
10 packets/100 ms, and is **amplitude only** — there is no phase in this dataset
(``CSIamp``). The official reader (``vendor/MMFi/mmfi_lib/mmfi.py``) already
imputes NaN/inf and min-max normalizes each frame to ``[0, 1]``. So
``csi_amplitude`` is effectively a guarded pass-through, and ``normalize_csi``
defaults to ``"none"`` to preserve that benchmark-faithful [0, 1] scaling; opt
into z-scoring if a model trains better with it.

A *window* of ``W`` frames is stacked on a new leading axis → ``(W, 3, 114, 10)``;
with the benchmark default ``W = 1`` that is ``(1, 3, 114, 10)``.
"""

from __future__ import annotations

import numpy as np

from .preprocess import normalize as _normalize_2d  # reuse the z-score/minmax math

__all__ = [
    "ROOT_JOINT",
    "csi_amplitude",
    "normalize_csi",
    "window_frame_indices",
    "center_pose",
    "pose_scale",
    "normalize_pose",
    "denormalize_pose",
]

# Pelvis is joint 0 in the Human3.6M ordering MM-Fi uses (see src/viz/skeleton.py).
ROOT_JOINT = 0


# ---------------------------------------------------------------------------
# CSI side
# ---------------------------------------------------------------------------


def csi_amplitude(x: np.ndarray) -> np.ndarray:
    """Return CSI amplitude as float32.

    MM-Fi WiFi is amplitude only (``CSIamp``), so for this dataset this is a
    pass-through cast. The ``np.abs`` guard is kept so the function is also
    correct if ever handed complex CSI from another source (cf.
    ``preprocess.amplitude``). Shape is preserved (any shape).
    """
    if np.iscomplexobj(x):
        return np.abs(x).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def normalize_csi(
    x: np.ndarray,
    mode: str = "none",
    per_sample: bool = True,
    stats: dict | None = None,
) -> np.ndarray:
    """Normalize a CSI frame or window of arbitrary shape.

    ``mode="none"`` (default) returns the input unchanged as float32 — the MM-Fi
    reader already min-max normalizes each frame to ``[0, 1]``, and keeping that
    scaling is what matches the published benchmark. ``"zscore"`` / ``"minmax"``
    reuse the exact statistics math from ``preprocess.normalize`` but over *all*
    elements of the (possibly N-D) array rather than the 2-D ``(T, S)`` layout
    that function documents.

    Input/output: same shape (e.g. ``(3, 114, 10)`` or ``(W, 3, 114, 10)``).
    """
    x = np.asarray(x, dtype=np.float32)
    if mode == "none":
        return x
    # _normalize_2d treats the array as one statistics pool when per_sample=True
    # (it uses x.mean()/x.std() / x.min()/x.max()), so it is shape-agnostic; for
    # per_sample=False it broadcasts the provided scalar stats. Flatten-safe.
    return _normalize_2d(x, mode=mode, per_sample=per_sample, stats=stats)


def window_frame_indices(center: int, n_frames: int, window_size: int) -> list[int]:
    """Frame indices for a window of ``window_size`` centered on ``center``.

    This is the single most alignment-critical function in the pipeline: it
    decides which CSI frames are paired with the pose at ``center``.

    Contract
    --------
    * The returned list has length ``window_size`` and ``center`` sits at its
      MIDDLE (index ``window_size // 2``) for odd sizes. The pose target is
      always the pose at ``center`` — never an end of the window — so the network
      sees symmetric past/future context around the labeled frame.
    * Indices are **clamped to ``[0, n_frames - 1]``**, i.e. frames that would
      fall before the clip start or after its end are replaced by the nearest
      valid frame (edge replication). The caller MUST pass ``n_frames`` for the
      *single clip* the center belongs to, so the window NEVER crosses a
      (subject, action) boundary — mixing frames from two clips would silently
      pair CSI with the wrong pose.
    * ``window_size == 1`` returns ``[center]`` (the MM-Fi benchmark default:
      one frame → one pose).

    Examples
    --------
    >>> window_frame_indices(0, 297, 5)      # clip start, clamped on the left
    [0, 0, 0, 1, 2]
    >>> window_frame_indices(150, 297, 5)
    [148, 149, 150, 151, 152]
    >>> window_frame_indices(296, 297, 5)    # clip end, clamped on the right
    [294, 295, 296, 296, 296]
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1; got {window_size}")
    if not 0 <= center < n_frames:
        raise IndexError(f"center {center} out of range [0, {n_frames})")
    half = window_size // 2
    # Odd sizes are symmetric. For even sizes (only reachable via direct calls;
    # the Dataset forbids them) the window leans left: start = center - half puts
    # ``half`` frames before the center and ``half - 1`` after.
    start = center - half
    idxs = [min(max(start + k, 0), n_frames - 1) for k in range(window_size)]
    return idxs


# ---------------------------------------------------------------------------
# Keypoint side
# ---------------------------------------------------------------------------


def center_pose(
    kp: np.ndarray, root: int = ROOT_JOINT
) -> tuple[np.ndarray, np.ndarray]:
    """Translate a pose so the root joint sits at the origin.

    Args:
        kp: ``(17, 3)`` (or ``(..., 17, 3)``) absolute camera-frame pose(s), metres.
        root: index of the root joint (pelvis = 0).

    Returns:
        ``(centered, offset)`` where ``offset`` is the original root position
        (``kp[..., root, :]`` with the joint axis kept as size-1 for broadcasting)
        and ``centered = kp - offset``. Store ``offset`` to restore absolute
        position later (``denormalize_pose``).
    """
    kp = np.asarray(kp, dtype=np.float32)
    offset = kp[..., root : root + 1, :]  # (..., 1, 3) — keepdims for broadcast
    centered = kp - offset
    return centered.astype(np.float32), offset.astype(np.float32)


def pose_scale(kp: np.ndarray, root: int = ROOT_JOINT) -> float:
    """A per-pose size scalar: RMS distance of all joints from the root.

    Useful as the divisor for scale normalization so people of different sizes /
    distances-to-camera map to a comparable target magnitude. Returns a Python
    float; guards against a degenerate zero (returns 1.0) so division is safe.
    """
    kp = np.asarray(kp, dtype=np.float64)
    rel = kp - kp[..., root : root + 1, :]
    scale = float(np.sqrt(np.mean(np.sum(rel * rel, axis=-1))))
    return scale if scale > 1e-6 else 1.0


def normalize_pose(
    kp: np.ndarray,
    root: int = ROOT_JOINT,
    scale: float | str | None = None,
) -> tuple[np.ndarray, dict]:
    """Center (and optionally scale) a pose into a regression target.

    Args:
        kp: ``(17, 3)`` absolute camera-frame pose, metres.
        root: root joint index (pelvis = 0).
        scale: how to scale after centering —
            * ``None`` (default): no scaling. Target stays in **metres**, so
              MPJPE is directly meaningful. RECOMMENDED unless a model needs it.
            * ``"rms"``: divide by this pose's own ``pose_scale`` (size/distance
              invariant; target becomes unitless).
            * a ``float``: divide by a fixed constant (e.g. a dataset-wide metre
              scale) — keeps a consistent unit across all samples.

    Returns:
        ``(norm_kp, info)`` where ``info`` has everything ``denormalize_pose``
        needs: ``{"offset": (1, 3) array, "scale": float, "root": int}``. The
        root joint of ``norm_kp`` is exactly zero.
    """
    centered, offset = center_pose(kp, root=root)
    if scale is None:
        s = 1.0
    elif scale == "rms":
        s = pose_scale(kp, root=root)
    elif isinstance(scale, (int, float)):
        s = float(scale)
        if s <= 0:
            raise ValueError(f"fixed scale must be > 0; got {scale}")
    else:
        raise ValueError(f"scale must be None, 'rms', or a positive float; got {scale!r}")
    norm = (centered / s).astype(np.float32)
    info = {"offset": offset.astype(np.float32), "scale": float(s), "root": int(root)}
    return norm, info


def denormalize_pose(
    norm_kp: np.ndarray,
    offset: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Invert ``normalize_pose`` — back to absolute camera-frame metres.

    The un-normalization recipe (see ``docs/chunk14_pose_pipeline.md``):

        absolute = norm_kp * scale + offset

    Apply this to BOTH the prediction and the ground truth (or just keep GT in
    absolute form) before computing MPJPE or plotting with
    ``src/viz/skeleton.py`` — otherwise the skeleton sits at the origin at the
    wrong size. ``offset`` and ``scale`` come straight from the ``info`` dict
    returned by ``normalize_pose``.
    """
    norm_kp = np.asarray(norm_kp, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    return (norm_kp * float(scale) + offset).astype(np.float32)
