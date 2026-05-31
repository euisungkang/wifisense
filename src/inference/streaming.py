"""Sliding-window inference over a continuous CSI stream.

A trained UT-HAR classifier expects one fixed-length sample of shape
``(T, S)`` (``T``=250 time steps, ``S``=90 subcarriers).  To label a
*continuous* capture we slide a fixed window across the time axis, run the
classifier on each window, and collect the per-window class probabilities.

Crucially, the model was trained on data put through
``src.data.preprocess.Pipeline`` (amplitude → hampel → median → per-sample
z-score).  To keep inference honest we apply *the same* pipeline to *each
window independently* — per-window normalization mirrors the per-sample
normalization used in training, rather than normalizing the whole stream
once.  Pass a custom ``pipeline`` to override.

This module emits raw model output only.  No temporal smoothing, no
hysteresis, no confidence gating — what the model says is what you get.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.data.preprocess import Pipeline

__all__ = ["sliding_window_predict"]


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    csi_stream: np.ndarray,
    window_size: int = 250,
    stride: int = 25,
    pipeline: Pipeline | None = None,
    device: torch.device | str | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window across a CSI stream and classify each window.

    Args:
        model:       Trained classifier accepting ``(B, T, S)`` and returning
                     ``(B, num_classes)`` logits.
        csi_stream:  Continuous capture, shape ``(T_total, S)``.  May be raw
                     (complex or real) CSI; the pipeline handles amplitude.
        window_size: Window length in time steps (250 matches training).
        stride:      Hop between consecutive window starts, in time steps.
        pipeline:    Preprocessing applied per window.  Defaults to a fresh
                     ``Pipeline()`` — the same default chain used in training.
                     A fresh pipeline carries no fitted PCA, matching the
                     per-sample (PCA-free) default training preprocessing.
        device:      Torch device; inferred from the model if None.
        batch_size:  Windows per forward pass.

    Returns:
        timestamps: ``(n_windows,)`` float array of window-*center* positions
                    in stream time-step units.  Centering aligns each
                    probability estimate with the middle of the evidence it
                    was computed from, which is what you want when overlaying
                    on a ground-truth timeline.
        probs:      ``(n_windows, num_classes)`` softmax probabilities.

    Raises:
        ValueError: if the stream is not 2-D or is shorter than one window.
    """
    csi_stream = np.asarray(csi_stream)
    if csi_stream.ndim != 2:
        raise ValueError(
            f"Expected a (T_total, S) stream, got shape {csi_stream.shape}"
        )
    t_total = csi_stream.shape[0]
    if t_total < window_size:
        raise ValueError(
            f"Stream length {t_total} is shorter than window_size {window_size}"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    if pipeline is None:
        pipeline = Pipeline()
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    # Window start indices: include the final fully-contained window even when
    # (t_total - window_size) is not an exact multiple of stride.
    starts = list(range(0, t_total - window_size + 1, stride))
    if starts[-1] != t_total - window_size:
        starts.append(t_total - window_size)
    starts_arr = np.asarray(starts)
    timestamps = starts_arr + window_size / 2.0

    # Stack windows → (n_windows, window_size, S), then apply the SAME
    # per-sample preprocessing the model was trained on, to each window.
    windows = np.stack([csi_stream[s : s + window_size] for s in starts])
    windows = pipeline.transform(windows)

    model.eval()
    X = torch.from_numpy(np.ascontiguousarray(windows)).float()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size].to(device)
        logits = model(xb)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)

    return timestamps, probs
