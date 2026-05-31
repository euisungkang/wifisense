# Preprocessing: CSI Signal Cleaning

Raw CSI is noisy. WiFi signals bounce off walls, furniture, and people, and
the NIC captures all of that plus thermal noise, hardware artifacts, and
occasional outlier spikes. Preprocessing cleans the signal so the model can
focus on the activity patterns instead of learning to ignore noise.

The goal: remove noise and outliers **without collapsing the time dimension**.
The model (BiLSTM) needs the full 250-step temporal sequence — we're cleaning
each time series, not summarizing it.

## The default pipeline

```
amplitude → hampel_filter → median_filter → zscore normalize
```

This matches what UT-HAR papers commonly use. Each step feeds into the next;
order matters.

### 1. Amplitude extraction

**What:** If the CSI data is complex-valued (I/Q components), compute the
magnitude `|z| = sqrt(real² + imag²)`. If the data is already real-valued
amplitudes, pass through unchanged.

**Why:** Models work on signal strength over time, not raw complex phasors.
Phase information is noisy in commodity WiFi hardware and usually hurts more
than it helps.

**UT-HAR:** Already real-valued (range ≈ −11 to +31), so this is a no-op.
It's in the pipeline so the same code works on complex CSI from other sources.

```python
amplitude(x)  # (250, 90) → (250, 90), no change for UT-HAR
```

### 2. Hampel filter (outlier rejection)

**What:** Sliding-window outlier detector along the time axis. For each time
step, computes the median and MAD (median absolute deviation) within a window
of `2k+1` steps. Any point that deviates by more than `n_sigma × MAD` from
the window median is replaced with that median.

**Why:** CSI occasionally has single-sample spikes from hardware glitches,
interference bursts, or packet retransmissions. These spikes are rare but
large, and they confuse gradient-based training. The Hampel filter is more
robust than mean-based outlier detection because it uses medians, so a single
extreme value can't distort the reference statistics.

**Parameters:**
- `k=5` → window of 11 time steps (~44ms at typical CSI rates). Wide enough
  to establish a local baseline, narrow enough not to blur fast transitions
  like a fall.
- `n_sigma=3.0` → only replace points >3 MADs from the median. Conservative
  enough to keep real signal variation.

**Implementation note:** All 90 subcarriers are processed in one vectorized
pass using `numpy.lib.stride_tricks.sliding_window_view`, not a Python loop
over subcarriers.

```python
hampel_filter(x, k=5, n_sigma=3.0)  # (250, 90) → (250, 90)
```

### 3. Median filter (smoothing)

**What:** 1-D median filter of width `k` along the time axis. Each value is
replaced with the median of the `k` surrounding time steps. Subcarriers are
filtered independently (kernel shape is `(k, 1)`).

**Why:** After outlier rejection, the signal still has high-frequency noise
that doesn't carry activity information. Median smoothing reduces this noise
while preserving edges (sudden transitions like the start of a fall), unlike
Gaussian smoothing which blurs edges.

**Parameters:**
- `k=5` → same 11-step window as the Hampel filter. Larger values smooth
  more aggressively but risk blurring short activities.

```python
median_filter(x, k=5)  # (250, 90) → (250, 90)
```

### 4. Z-score normalization (per-sample)

**What:** Subtract the sample mean, divide by the sample standard deviation.
Each `(250, 90)` sample is normalized independently using its own statistics.

**Why:** CSI amplitude depends on distance, orientation, and environment —
two identical activities recorded in different rooms will have very different
absolute values. Per-sample z-scoring removes this offset so the model learns
*patterns of variation*, not absolute signal levels.

**Per-sample vs. dataset-wide:** We normalize each sample independently
(`per_sample=True`) rather than using global train-set statistics. This is
standard in CSI literature because the absolute scale carries no
activity-discriminative information, and per-sample normalization makes the
model robust to deployment in new environments.

```python
normalize(x, mode='zscore', per_sample=True)  # (250, 90) → (250, 90)
```

After normalization, each sample has mean ≈ 0 and std ≈ 1.

## Additional transforms (not in default pipeline)

These are available in `src/data/preprocess.py` but not enabled by default.
They're useful for experimentation — use them by constructing a custom
`Pipeline` with different steps.

### DWT denoising

**What:** Discrete wavelet transform (VisuShrink) per subcarrier. Decomposes
each subcarrier's time series into wavelet coefficients, estimates noise σ
from the finest detail coefficients, applies soft thresholding at the
universal threshold `σ√(2 log T)`, and reconstructs.

**Why:** Wavelets separate signal from noise better than simple filters for
signals with both smooth regions and sharp transitions — exactly what CSI
looks like during activity changes. More aggressive than median filtering but
also more computationally expensive.

**Parameters:**
- `wavelet='db4'` — Daubechies-4, a standard choice for smooth signals.
- `level=None` — auto-selects maximum decomposition level.
- `mode='soft'` — soft thresholding (shrink coefficients toward zero) vs.
  hard (zero them out). Soft produces smoother reconstructions.

**Not in the default pipeline** because the Hampel + median combination
already handles UT-HAR noise well, and DWT adds ~10× processing time per
sample.

```python
dwt_denoise(x, wavelet='db4', level=None, mode='soft')  # (250, 90) → (250, 90)
```

### PCA subcarrier reduction

**What:** Reduces the 90 subcarrier features to `n` principal components via
PCA. Reshapes `(N, 250, 90)` → `(N×250, 90)`, fits PCA, transforms, reshapes
back to `(N, 250, n)`.

**Why:** Many of the 90 subcarriers (30 subcarriers × 3 antennas) are highly
correlated. PCA can reduce dimensionality without losing much variance,
potentially speeding up training and reducing overfitting.

**Trade-off:** Reduces feature dimension but mixes physical subcarrier
channels, which may hurt interpretability. Also introduces a fit/transform
dependency — PCA must be fit on training data and the same projection applied
to test data.

**Not in the default pipeline** because the BiLSTM can handle 90 features,
and keeping the full subcarrier set preserves spatial diversity across
antennas.

```python
# Dataset-level operation — returns (X_out, fitted_pca)
X_reduced, pca = pca_subcarriers(X, n=30)       # (N, 250, 90) → (N, 250, 30)
X_test_reduced, _ = pca_subcarriers(X_test, fitted_pca=pca)
```

## Running the pipeline

```bash
conda activate wifisense
python scripts/preprocess_data.py
```

**Inputs:** Raw UT-HAR data from `data/raw/ut_har/UT_HAR/` (loaded via
`src/data/loader.py`).

**Outputs:**

| Path | Contents |
|------|----------|
| `data/processed/ut_har/ut_har.npz` | `X_train (3977, 250, 90)`, `y_train (3977,)`, `X_val (496, 250, 90)`, `y_val (496,)`, `X_test (500, 250, 90)`, `y_test (500,)` — all float32/int64 |
| `figures/preprocessing/ut_har_before_after.png` | CSI heatmap (subcarrier × time) for one sample of each class, raw vs preprocessed |

**Runtime:** ~60 seconds on a modern CPU. The Hampel filter is the bottleneck.

## What to check in the before/after figure

- **Temporal patterns should be sharper**, not blurred. Look for horizontal
  streaks and transitions (activity boundaries) being preserved or enhanced.
- **No dead subcarriers.** If an entire row goes flat, the pipeline killed
  a channel — that's a bug.
- **Classes should still look distinct.** If all 7 heatmaps look identical
  after preprocessing, the pipeline over-smoothed.
- **Value range is normalized.** Raw shows values in [−11, 31]; preprocessed
  should be centered around 0 with unit variance.

## Custom pipelines

```python
from src.data.preprocess import (
    Pipeline, amplitude, hampel_filter, median_filter,
    dwt_denoise, normalize, pca_subcarriers,
)

# Heavier denoising with DWT instead of median filter
pipe = Pipeline(steps=[
    (amplitude, {}),
    (hampel_filter, {"k": 7, "n_sigma": 2.5}),
    (dwt_denoise, {"wavelet": "db4"}),
    (normalize, {"mode": "minmax", "per_sample": True}),
])

X_train = pipe.fit_transform(X_train_raw)
X_test = pipe.transform(X_test_raw)

# Add PCA on top
pipe_pca = Pipeline(pca_components=30)
X_train = pipe_pca.fit_transform(X_train_raw)  # (N, 250, 30)
X_test = pipe_pca.transform(X_test_raw)
```

## What feeds into the next stage

The `.npz` file is the input to model training. Load it as:

```python
import numpy as np
d = np.load("data/processed/ut_har/ut_har.npz")
X_train, y_train = d["X_train"], d["y_train"]
```

Each sample is `(250, 90)` float32 — a sequence of 250 time steps with 90
features per step. For a BiLSTM, this is a sequence of length 250 with
90-dimensional input vectors. Labels are `int64` in `[0, 6]` mapping to the
7 UT-HAR activity classes.

## Code reference

| File | Role |
|------|------|
| `src/data/preprocess.py` | All transforms + `Pipeline` class |
| `scripts/preprocess_data.py` | Runs default pipeline on UT-HAR, saves outputs + figure |
| `src/data/loader.py` | Raw data loading (upstream of preprocessing) |
