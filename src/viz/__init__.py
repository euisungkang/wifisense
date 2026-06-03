"""Visualization layer.

See :mod:`src.viz.csi_plots` for single-sample CSI rendering primitives, and
:mod:`src.viz.skeleton` for the 3D human-pose skeleton primitives used in
Phase 3 (pose estimation).
"""

from src.viz.csi_plots import (
    plot_amplitude_heatmap,
    plot_doppler_spectrogram,
    plot_subcarrier_traces,
)
from src.viz.skeleton import plot_skeleton_3d, plot_skeleton_pair

__all__ = [
    "plot_amplitude_heatmap",
    "plot_doppler_spectrogram",
    "plot_subcarrier_traces",
    "plot_skeleton_3d",
    "plot_skeleton_pair",
]
