"""Composable Widar3.0 BVP preprocessing transforms.

These mirror the purity conventions of ``src/data/preprocess.py``: every
function is pure, takes and returns a single ``float32`` numpy array, and never
mutates its input. They operate on **one BVP sample** of shape ``(T, 20, 20)``
as produced by ``widar_loader.load_bvp_file`` — leading axis time, then the
20x20 body-frame velocity grid (x-velocity on axis 1, y-velocity on axis 2).

Three jobs, kept separate so they compose:

    normalize_bvp   — rescale values (per-sample or with global stats)
    pad_or_truncate — fix the variable T to a common length for batching
    augment_bvp     — *training-only* stochastic perturbations

Order matters in a training pipeline. ``WidarBVPDataset`` applies them as
``augment_bvp`` (in raw, non-negative energy space) → ``normalize_bvp`` →
``pad_or_truncate``. augment_bvp must run on raw energy so its non-negativity
clip is meaningful and its noise scale matches the L1-normalized frames; never
call it at eval time.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_bvp",
    "pad_or_truncate",
    "augment_bvp",
]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_bvp(
    x: np.ndarray,
    mode: str = "per_sample",
    stats: dict | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Z-score normalize a BVP volume.

    On disk each 20x20 frame is roughly L1-normalized (cells sum to ~1), so raw
    cell magnitudes are tiny (~1e-3) and vary little across samples. Z-scoring
    to zero-mean/unit-variance gives the model a better-conditioned input; it
    does sacrifice non-negativity, which is why any energy-space augmentation
    must happen *before* this step.

    Args:
        x: BVP sample, shape (T, 20, 20).
        mode: ``"per_sample"`` computes mean/std from this volume alone;
            ``"global"`` uses a pre-computed *stats* dict (so train and test
            share one scale — see ``widar_dataset.compute_global_stats``).
        stats: required when ``mode="global"``; dict with ``"mean"``/``"std"``.
        eps: numerical floor added to the standard deviation.

    Returns:
        float32 array, same shape as input.
    """
    x = x.astype(np.float64, copy=True)
    if mode == "per_sample":
        return ((x - x.mean()) / (x.std() + eps)).astype(np.float32)
    if mode == "global":
        if stats is None:
            raise ValueError("stats dict required when mode='global'")
        return ((x - stats["mean"]) / (stats["std"] + eps)).astype(np.float32)
    raise ValueError(f"Unknown mode: {mode!r} (expected 'per_sample'/'global')")


# ---------------------------------------------------------------------------
# Fixed-length padding / truncation
# ---------------------------------------------------------------------------


def pad_or_truncate(x: np.ndarray, target_T: int) -> np.ndarray:
    """Force the time axis of a BVP volume to exactly ``target_T`` frames.

    Variable-length gestures cannot be stacked into a batch, so we anchor every
    sample at its start (t=0) and fix the length at the tail:

    - ``T < target_T``: append ``target_T - T`` zero frames. After per-sample
      z-scoring zero ≈ the volume mean, so padded frames read as "no motion".
    - ``T > target_T``: keep the first ``target_T`` frames. With ``target_T``
      set to a high percentile of the corpus (max observed T is 28), truncation
      only ever touches the longest gestures and merely drops their tail.

    Args:
        x: BVP sample, shape (T, 20, 20).
        target_T: desired number of frames.

    Returns:
        float32 array, shape (target_T, 20, 20).
    """
    if target_T <= 0:
        raise ValueError(f"target_T must be positive, got {target_T}")
    x = x.astype(np.float32, copy=False)
    T = x.shape[0]
    if T == target_T:
        return np.array(x, dtype=np.float32, copy=True)
    if T > target_T:
        return np.array(x[:target_T], dtype=np.float32, copy=True)
    pad = np.zeros((target_T - T, *x.shape[1:]), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)


# ---------------------------------------------------------------------------
# Training-only augmentation
# ---------------------------------------------------------------------------


def augment_bvp(
    x: np.ndarray,
    rng: np.random.Generator | None = None,
    temporal_crop: float = 0.15,
    flip_prob: float = 0.5,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Stochastically perturb a BVP sample for training data augmentation.

    Operates in raw (non-negative) energy space, so call this *before*
    ``normalize_bvp``. **Never apply at eval time** — it is non-deterministic
    and would corrupt held-out measurements.

    Three independent perturbations, each disabled by setting its parameter to
    0 (or ``flip_prob=0``):

    - **random temporal crop**: keep a contiguous sub-window covering a random
      fraction in ``[1 - temporal_crop, 1]`` of the frames, at a random start.
      Simulates slightly different gesture segmentation. Changes T, so a
      downstream ``pad_or_truncate`` is expected.
    - **horizontal flip**: with probability ``flip_prob``, mirror the x-velocity
      axis (``v_x → -v_x``). Valid only for left/right-symmetric gestures, where
      a leftward and rightward execution are the same class.
    - **Gaussian noise**: add ``N(0, noise_std)`` per cell and clip back to
      non-negative. ``noise_std`` is in raw BVP units; frames sum to ~1 over 400
      cells (peak cells ~0.15), so the 0.01 default is a mild jitter.

    Args:
        x: BVP sample, shape (T, 20, 20).
        rng: numpy Generator for reproducibility. If None, a fresh
            ``default_rng()`` is created (non-reproducible).
        temporal_crop: max fraction of frames that may be dropped (0 disables).
        flip_prob: probability of a horizontal flip.
        noise_std: standard deviation of additive Gaussian noise (0 disables).

    Returns:
        float32 array, shape (T', 20, 20) with ``T' <= T``.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = x.astype(np.float32, copy=True)
    T = x.shape[0]

    # Random temporal crop: contiguous window of length in [(1-c)*T, T].
    if temporal_crop > 0 and T > 1:
        lo = max(1, int(round((1.0 - temporal_crop) * T)))
        win = int(rng.integers(lo, T + 1)) if lo < T else T
        start = int(rng.integers(0, T - win + 1)) if win < T else 0
        x = x[start : start + win]

    # Horizontal flip across the x-velocity axis (axis 1).
    if rng.random() < flip_prob:
        x = np.ascontiguousarray(x[:, ::-1, :])

    # Additive Gaussian noise, clipped to keep the energy map non-negative.
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        np.clip(x, 0.0, None, out=x)

    return x.astype(np.float32, copy=False)
