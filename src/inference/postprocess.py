"""Temporal post-processing of per-window classifier output.

Chunk 6 rendered the continuous-capture predictions as *raw* sliding-window
softmax — no smoothing — and chunk 7 showed that the resulting ~56% per-window
accuracy is almost entirely a **boundary artifact**: windows whose span
straddles two activities are fed a mixture the model never trained on, so they
flip to confident-but-wrong labels (see ``docs/diagnostics.md``).

A trained classifier looks at each window in isolation, but activities have
*temporal structure*: they persist for many windows and only occasionally
switch. This module adds the post-processing layer deliberately deferred in
chunk 6 — three ways to inject that prior over the per-window output:

    * ``moving_average`` — average the probability vectors over a sliding
      window of ``k`` windows, then argmax. Smooths confident single-window
      spikes; treats every neighbour equally.
    * ``majority_vote`` — mode filter over the argmax labels in a window of
      ``k``. Pure label-space denoiser; ignores confidence entirely.
    * ``hmm_decode`` — Viterbi decoding with an explicit transition matrix.
      The most principled option: it encodes *which* transitions are realistic
      (e.g. activities persist) rather than just "neighbours agree", and trades
      emission confidence against transition cost globally over the whole
      sequence instead of in a fixed local window.

All three operate on the ``(n_windows, n_classes)`` probability array (or its
argmax) produced by :func:`src.inference.streaming.sliding_window_predict` and
return a post-processed label sequence ``(n_windows,)``.

numpy-only, no torch — these are pure array transforms over already-computed
probabilities.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "moving_average",
    "moving_average_probs",
    "majority_vote",
    "hmm_decode",
    "viterbi_decode",
    "learn_transition_matrix",
    "transition_rate",
    "labels_to_onehot",
    "select_best_method",
]


# ---------------------------------------------------------------------------
# Strategy 1: probability moving average
# ---------------------------------------------------------------------------


def moving_average_probs(probs: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average of probability vectors along the time axis.

    Args:
        probs: ``(n_windows, n_classes)`` per-window probabilities (rows sum
            to 1).
        k:     Window length in *windows*. ``k <= 1`` is a no-op.

    Returns:
        ``(n_windows, n_classes)`` smoothed probabilities. Because each output
        row is a plain mean of input rows that each sum to 1, the output rows
        also sum to 1 — no renormalization needed. At the sequence ends the
        window simply shrinks (equivalent to ``mode="nearest"`` clamping),
        keeping the estimate centered without wrapping or zero-padding.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"Expected (n_windows, n_classes), got {probs.shape}")
    k = int(k)
    if k <= 1:
        return probs.copy()
    half = k // 2
    n = probs.shape[0]
    out = np.empty_like(probs)
    for t in range(n):
        lo = max(0, t - half)
        hi = min(n, t + half + 1)
        out[t] = probs[lo:hi].mean(axis=0)
    return out


def moving_average(probs: np.ndarray, k: int) -> np.ndarray:
    """Smooth ``probs`` over ``k`` windows, then argmax → label sequence.

    See :func:`moving_average_probs`. Returns ``(n_windows,)`` int labels.
    """
    return moving_average_probs(probs, k).argmax(axis=1).astype(int)


# ---------------------------------------------------------------------------
# Strategy 2: label-space majority (mode) filter
# ---------------------------------------------------------------------------


def majority_vote(preds: np.ndarray, k: int) -> np.ndarray:
    """Centered mode filter over a label sequence.

    Args:
        preds: ``(n_windows,)`` integer labels (e.g. ``probs.argmax(1)``).
        k:     Window length in windows. ``k <= 1`` is a no-op.

    Returns:
        ``(n_windows,)`` int labels: each position replaced by the most common
        label in its centered window of width ``k``. Ties are broken toward the
        smallest label index (``np.bincount(...).argmax()`` semantics). Edges
        shrink the window rather than pad it.
    """
    preds = np.asarray(preds).ravel().astype(int)
    if preds.size and preds.min() < 0:
        raise ValueError("majority_vote expects non-negative integer labels")
    k = int(k)
    if k <= 1:
        return preds.copy()
    half = k // 2
    n = preds.shape[0]
    out = np.empty(n, dtype=int)
    for t in range(n):
        lo = max(0, t - half)
        hi = min(n, t + half + 1)
        out[t] = np.bincount(preds[lo:hi]).argmax()
    return out


# ---------------------------------------------------------------------------
# Strategy 3: HMM / Viterbi decoding
# ---------------------------------------------------------------------------


def learn_transition_matrix(
    label_sequence: np.ndarray, n_classes: int, alpha: float = 1.0
) -> np.ndarray:
    """Estimate a row-stochastic transition matrix from a label sequence.

    Counts adjacent ``(i -> j)`` transitions in ``label_sequence`` and
    row-normalizes with **Laplace (add-alpha) smoothing**, so every transition
    keeps a non-zero probability — Viterbi can never be hard-blocked from a
    path the training data merely never happened to show.

    Args:
        label_sequence: 1-D array of integer labels sampled at the cadence the
            HMM will step over (one label per Viterbi step). See
            ``scripts/compare_postprocessing.py`` for how the UT-HAR training
            labels are turned into such a sequence, and ``docs/chunk8_*`` for
            the (important) caveat that UT-HAR training clips are isolated
            single-activity samples.
        n_classes: Number of classes ``C`` (fixes the matrix shape even if some
            class is absent from the sequence).
        alpha: Laplace pseudo-count added to every cell before normalizing.

    Returns:
        ``(C, C)`` array ``A`` with ``A[i, j] = P(next=j | cur=i)`` and each
        row summing to 1.
    """
    seq = np.asarray(label_sequence).ravel().astype(int)
    counts = np.zeros((n_classes, n_classes), dtype=float)
    if seq.size >= 2:
        np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
    counts += float(alpha)
    return counts / counts.sum(axis=1, keepdims=True)


def viterbi_decode(
    log_emit: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray
) -> np.ndarray:
    """Standard Viterbi most-likely-path decoder, all inputs in log space.

    Args:
        log_emit:  ``(T, C)`` log emission scores per time step / state.
        log_trans: ``(C, C)`` log transition probabilities ``log P(j | i)``.
        log_init:  ``(C,)`` log initial-state probabilities.

    Returns:
        ``(T,)`` int array — the highest-scoring state path.
    """
    T, C = log_emit.shape
    dp = np.empty((T, C), dtype=float)
    back = np.zeros((T, C), dtype=int)
    dp[0] = log_init + log_emit[0]
    for t in range(1, T):
        # scores[i, j] = best score ending in i at t-1, then i -> j
        scores = dp[t - 1][:, None] + log_trans
        back[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + log_emit[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def hmm_decode(
    probs: np.ndarray,
    transition_matrix: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Viterbi-decode a per-window probability sequence into a label path.

    The classifier's softmax outputs are used directly as emission scores. This
    treats the posterior ``P(class | window)`` as a stand-in for the emission
    likelihood ``P(window | class)`` — exact only under a uniform class prior,
    a standard and pragmatic approximation when smoothing a classifier with an
    HMM (documented as a caveat in ``docs/chunk8_postprocessing.md``).

    Args:
        probs: ``(n_windows, n_classes)`` per-window probabilities.
        transition_matrix: ``(C, C)`` row-stochastic transitions, normally from
            :func:`learn_transition_matrix`. If ``None``, a uniform matrix is
            used, which makes Viterbi degenerate to per-window argmax (no
            temporal prior) — so a meaningful call passes a learned matrix.
        eps: Floor added before taking logs, to keep zeros finite.

    Returns:
        ``(n_windows,)`` int label path.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"Expected (n_windows, n_classes), got {probs.shape}")
    T, C = probs.shape
    if transition_matrix is None:
        transition_matrix = np.full((C, C), 1.0 / C)
    A = np.asarray(transition_matrix, dtype=float)
    if A.shape != (C, C):
        raise ValueError(
            f"transition_matrix shape {A.shape} != (n_classes, n_classes) {(C, C)}"
        )
    log_emit = np.log(probs + eps)
    log_trans = np.log(A + eps)
    log_init = np.full(C, -np.log(C))  # uniform start state
    return viterbi_decode(log_emit, log_trans, log_init)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def transition_rate(preds: np.ndarray) -> float:
    """Class flips per 100 windows (lower = smoother).

    Counts adjacent positions where the label changes, normalized to a
    per-100-window rate so it is comparable across captures of different
    length.
    """
    preds = np.asarray(preds).ravel()
    if preds.size < 2:
        return 0.0
    flips = int(np.count_nonzero(preds[1:] != preds[:-1]))
    return flips / (preds.size - 1) * 100.0


def labels_to_onehot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """``(n_windows,)`` labels → ``(n_windows, n_classes)`` one-hot "probabilities".

    Used to render a hard-label strategy (majority vote / HMM) in the same
    stacked-probability panel format as the soft raw output, so the two are
    directly comparable.
    """
    labels = np.asarray(labels).ravel().astype(int)
    onehot = np.zeros((labels.size, n_classes), dtype=float)
    onehot[np.arange(labels.size), labels] = 1.0
    return onehot


def select_best_method(metrics: dict[str, dict[str, float]]) -> str:
    """Pick the best smoothing method: highest window accuracy, then smoothest.

    Args:
        metrics: ``{name: {"window_acc": float, "transition_rate": float}}``.

    Returns:
        The key maximizing per-window accuracy, ties broken by the *lower*
        transition rate (smoother output wins a tie).
    """
    return max(
        metrics,
        key=lambda n: (metrics[n]["window_acc"], -metrics[n]["transition_rate"]),
    )
