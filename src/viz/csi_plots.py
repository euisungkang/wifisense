"""Single-sample CSI rendering primitives.

Each function takes one CSI sample of shape (T, S) — T time steps, S
subcarriers (90 for UT-HAR: 30 subcarriers x 3 RX antennas, flattened) —
and either draws onto a provided matplotlib Axes or creates its own.  All
functions return the Axes so callers can compose them into multi-panel
figures and attach colorbars to the returned mappable where noted.

Conventions:
    * Heatmaps put subcarrier on Y, time on X, with origin at lower-left.
    * Doppler/STFT views put frequency on Y, time on X.
    * Functions never call plt.show() or set a backend; that is the
      caller's responsibility.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

__all__ = [
    "plot_amplitude_heatmap",
    "plot_doppler_spectrogram",
    "plot_subcarrier_traces",
]


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Coerce input to a real-valued (T, S) float array.

    Complex CSI is reduced to amplitude; this makes the primitives robust
    whether they are handed raw complex CSI or already-amplitude data.
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        x = np.abs(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a single (T, S) CSI sample, got shape {x.shape}")
    return x.astype(np.float64)


def plot_amplitude_heatmap(
    x: np.ndarray,
    ax: plt.Axes | None = None,
    title: str | None = None,
    cmap: str = "viridis",
) -> plt.Axes:
    """Heatmap of CSI amplitude: subcarrier on Y, time on X, colour = amplitude.

    Args:
        x:     CSI sample, shape (T, S).
        ax:    Axes to draw on; a new one is created if None.
        title: Optional Axes title.
        cmap:  Matplotlib colormap name.

    Returns:
        The Axes.  The drawn image is stored on ``ax.images[-1]`` for
        attaching a colorbar (``fig.colorbar(ax.images[-1], ax=ax)``).
    """
    x = _as_2d(x)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    # Transpose to (S, T) so subcarrier is the vertical axis.
    ax.imshow(
        x.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Subcarrier")
    if title:
        ax.set_title(title)
    return ax


def plot_doppler_spectrogram(
    x: np.ndarray,
    fs: float,
    ax: plt.Axes | None = None,
    title: str | None = None,
    nperseg: int = 64,
    noverlap: int | None = None,
    cmap: str = "magma",
) -> plt.Axes:
    """STFT-derived Doppler spectrogram averaged across subcarriers.

    Each subcarrier's amplitude time series is mean-removed (so the static
    component does not dominate the DC bin), transformed with a Short-Time
    Fourier Transform, and the resulting power spectra are averaged over
    subcarriers.  The vertical axis is the Doppler-shift frequency in Hz;
    its absolute scale depends on ``fs`` (the CSI packet rate).

    Args:
        x:        CSI sample, shape (T, S).
        fs:       Sampling rate of the time axis, in Hz.
        ax:       Axes to draw on; a new one is created if None.
        title:    Optional Axes title.
        nperseg:  STFT window length (samples).
        noverlap: STFT overlap; defaults to 75% of ``nperseg``.
        cmap:     Matplotlib colormap name.

    Returns:
        The Axes.  The drawn image is stored on ``ax.images[-1]``.
    """
    x = _as_2d(x)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    nperseg = min(nperseg, x.shape[0])
    if noverlap is None:
        noverlap = nperseg * 3 // 4

    # STFT along time (axis 0): Zxx has shape (n_freq, S, n_seg).
    x_detrended = x - x.mean(axis=0, keepdims=True)
    f, t, Zxx = stft(
        x_detrended,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=0,
    )

    # Average power across subcarriers, then convert to dB.
    power = np.mean(np.abs(Zxx) ** 2, axis=1)  # (n_freq, n_seg)
    power_db = 10.0 * np.log10(power + 1e-12)

    ax.pcolormesh(t, f, power_db, shading="auto", cmap=cmap)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Doppler freq (Hz)")
    if title:
        ax.set_title(title)
    return ax


def plot_subcarrier_traces(
    x: np.ndarray,
    subcarrier_idxs,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Amplitude-over-time line traces for selected subcarriers.

    Args:
        x:               CSI sample, shape (T, S).
        subcarrier_idxs: Iterable of subcarrier indices to plot.
        ax:              Axes to draw on; a new one is created if None.
        title:           Optional Axes title.

    Returns:
        The Axes (with a legend identifying each trace).
    """
    x = _as_2d(x)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    T, S = x.shape
    time = np.arange(T)
    for idx in subcarrier_idxs:
        if not 0 <= idx < S:
            raise IndexError(f"subcarrier index {idx} out of range [0, {S})")
        ax.plot(time, x[:, idx], linewidth=0.9, label=f"sc {idx}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    if title:
        ax.set_title(title)
    return ax
