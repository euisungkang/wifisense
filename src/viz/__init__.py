"""Visualization layer for CSI samples.

See :mod:`src.viz.csi_plots` for the single-sample rendering primitives
that get composed into multi-panel figures.
"""

from src.viz.csi_plots import (
    plot_amplitude_heatmap,
    plot_doppler_spectrogram,
    plot_subcarrier_traces,
)

__all__ = [
    "plot_amplitude_heatmap",
    "plot_doppler_spectrogram",
    "plot_subcarrier_traces",
]
