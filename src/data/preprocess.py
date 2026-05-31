"""Composable CSI preprocessing transforms and pipeline.

Per-sample transforms operate on 2-D numpy arrays of shape (T, S):
    T = time steps  (250 for UT-HAR)
    S = subcarriers  (90 for UT-HAR)

pca_subcarriers operates on batched arrays (N, T, S) → (N, T, n).
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy.ndimage import median_filter as _scipy_median_filter
from sklearn.decomposition import PCA

__all__ = [
    "amplitude",
    "hampel_filter",
    "median_filter",
    "dwt_denoise",
    "pca_subcarriers",
    "normalize",
    "Pipeline",
]


# ---------------------------------------------------------------------------
# Per-sample transforms: (T, S) → (T, S)
# ---------------------------------------------------------------------------


def amplitude(x: np.ndarray) -> np.ndarray:
    """Compute |z| for complex CSI; pass-through for real-valued amplitudes.

    Input/output: (T, S)
    """
    if np.iscomplexobj(x):
        return np.abs(x).astype(np.float32)
    return x


def hampel_filter(
    x: np.ndarray, k: int = 5, n_sigma: float = 3.0
) -> np.ndarray:
    """Outlier rejection via Hampel identifier along the time axis.

    For each time step, computes the median and MAD in a symmetric window
    of 2k+1 steps.  Points deviating by more than n_sigma × MAD are
    replaced with the window median.  All subcarriers are processed in
    one vectorised pass.

    Input/output: (T, S)
    """
    x = x.astype(np.float64, copy=True)
    padded = np.pad(x, ((k, k), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, 2 * k + 1, axis=0
    )  # (T, S, 2k+1)
    med = np.median(windows, axis=2)
    mad = 1.4826 * np.median(np.abs(windows - med[:, :, np.newaxis]), axis=2)
    outlier = (mad > 0) & (np.abs(x - med) > n_sigma * mad)
    return np.where(outlier, med, x).astype(np.float32)


def median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Smooth along the time axis with a 1-D median filter of width k.

    The kernel shape is (k, 1) so subcarriers are filtered independently.

    Input/output: (T, S)
    """
    return _scipy_median_filter(x, size=(k, 1)).astype(np.float32)


def dwt_denoise(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
    mode: str = "soft",
) -> np.ndarray:
    """Discrete wavelet denoising (VisuShrink) applied per subcarrier.

    Decomposes each subcarrier's time series, estimates noise σ from the
    finest detail coefficients, applies universal thresholding, and
    reconstructs.

    Input/output: (T, S)
    """
    x = x.astype(np.float64, copy=True)
    T, S = x.shape
    for s in range(S):
        coeffs = pywt.wavedec(x[:, s], wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(T))
        coeffs[1:] = [
            pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]
        ]
        rec = pywt.waverec(coeffs, wavelet)
        x[:, s] = rec[:T]
    return x.astype(np.float32)


def normalize(
    x: np.ndarray,
    mode: str = "zscore",
    per_sample: bool = True,
    stats: dict | None = None,
) -> np.ndarray:
    """Z-score or min-max normalization.

    per_sample=True  — statistics computed from this (T, S) sample alone.
    per_sample=False — uses a pre-computed *stats* dict:
                       {'mean', 'std'} for zscore, {'min', 'max'} for minmax.

    Input/output: (T, S)
    """
    x = x.astype(np.float64, copy=True)
    eps = 1e-8
    if per_sample:
        if mode == "zscore":
            return ((x - x.mean()) / (x.std() + eps)).astype(np.float32)
        if mode == "minmax":
            lo, hi = x.min(), x.max()
            return ((x - lo) / (hi - lo + eps)).astype(np.float32)
        raise ValueError(f"Unknown mode: {mode}")
    if stats is None:
        raise ValueError("stats dict required when per_sample=False")
    if mode == "zscore":
        return ((x - stats["mean"]) / (stats["std"] + eps)).astype(np.float32)
    if mode == "minmax":
        lo, hi = stats["min"], stats["max"]
        return ((x - lo) / (hi - lo + eps)).astype(np.float32)
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Dataset-level transform
# ---------------------------------------------------------------------------


def pca_subcarriers(
    X: np.ndarray,
    n: int = 30,
    fitted_pca: PCA | None = None,
) -> tuple[np.ndarray, PCA]:
    """Reduce the subcarrier dimension via PCA.

    Reshapes (N, T, S) → (N×T, S), fits or applies PCA, reshapes back.
    For a single sample pass (T, S); a batch dim is added internally.

    Returns:
        X_out:  (N, T, n) or (T, n)
        pca:    the fitted sklearn PCA object (reuse for test data)
    """
    single = X.ndim == 2
    if single:
        X = X[np.newaxis]
    N, T, S = X.shape
    flat = X.reshape(N * T, S).astype(np.float64)
    if fitted_pca is None:
        pca = PCA(n_components=n)
        flat_t = pca.fit_transform(flat).astype(np.float32)
    else:
        pca = fitted_pca
        flat_t = pca.transform(flat).astype(np.float32)
    out = flat_t.reshape(N, T, -1)
    if single:
        out = out[0]
    return out, pca


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Chains per-sample CSI transforms with optional dataset-level PCA.

    Default chain (matches UT-HAR literature):
        amplitude → hampel_filter → median_filter → zscore normalize

    Usage::

        pipe = Pipeline()
        X_train = pipe.fit_transform(X_train_raw)   # (N, T, S) → (N, T, S')
        X_test  = pipe.transform(X_test_raw)
    """

    def __init__(
        self,
        steps: list[tuple[callable, dict]] | None = None,
        pca_components: int | None = None,
    ):
        if steps is None:
            steps = [
                (amplitude, {}),
                (hampel_filter, {"k": 5}),
                (median_filter, {"k": 5}),
                (normalize, {"mode": "zscore", "per_sample": True}),
            ]
        self.steps = steps
        self.pca_components = pca_components
        self._pca: PCA | None = None

    def _apply_per_sample(self, X: np.ndarray) -> np.ndarray:
        """Apply the per-sample transform chain. (N, T, S) → (N, T, S)."""
        results = []
        n = len(X)
        for i in range(n):
            x = X[i]
            for fn, kwargs in self.steps:
                x = fn(x, **kwargs)
            results.append(x)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{n} samples")
        if n >= 1000:
            print(f"  {n}/{n} samples")
        return np.stack(results)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply transforms to training data, fitting PCA if configured.

        Input:  (N, T, S)
        Output: (N, T, S) or (N, T, pca_components) if PCA enabled
        """
        X = self._apply_per_sample(X)
        if self.pca_components is not None:
            X, self._pca = pca_subcarriers(X, self.pca_components)
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply transforms to new data, reusing any fitted PCA.

        Input:  (N, T, S)
        Output: (N, T, S) or (N, T, pca_components) if PCA enabled
        """
        X = self._apply_per_sample(X)
        if self._pca is not None:
            X, _ = pca_subcarriers(X, fitted_pca=self._pca)
        return X
