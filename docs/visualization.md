# Visualization: Looking at CSI Before Trusting It

Preprocessing produces a clean `.npz`, but "clean" is a claim you should
*see* before training on it. This phase builds the visualization layer that
turns a single `(250, 90)` CSI sample into a picture, so we can (1)
sanity-check that the data and preprocessing make sense, and (2) reuse the
exact same renderer as the **top panel of the final deliverable** — the CSI
spectrogram that sits above the activity-prediction and ground-truth tracks.

The guiding idea: a human activity is a *pattern of signal variation over
time*. If two activities are genuinely different, you should be able to see
the difference. If you can't, the model probably can't either — better to
find that out now than after a training run.

## Three ways to look at one sample

Every primitive takes a single CSI sample of shape `(250, 90)` — 250 time
steps × 90 features (30 subcarriers × 3 RX antennas, flattened) — and either
draws onto a matplotlib `Axes` you pass in or creates its own. Each returns
the `Axes`, so they compose into multi-panel figures. They never call
`plt.show()` or set a backend; that's the caller's job.

They live in `src/viz/csi_plots.py` and are re-exported from `src.viz`.

### 1. Amplitude heatmap

**What:** The raw view — subcarrier on the Y axis, time on the X axis, color
= CSI amplitude. The sample is transposed to `(90, 250)` so each horizontal
row is one subcarrier's time series and each vertical column is one moment.

**Why:** This is the most direct picture of the data. Motion shows up as
vertical texture (the whole channel fluctuating over time); a static pose
shows up as smooth horizontal bands; a discrete event (a fall, a pickup)
shows up as a localized disturbance. It's also the natural top panel for the
deliverable because it reads left-to-right as time, lining up with the
prediction track below it.

```python
plot_amplitude_heatmap(x, ax=None, title=None, cmap="viridis")
# returns the Axes; image is at ax.images[-1] for fig.colorbar(...)
```

### 2. Doppler spectrogram

**What:** A short-time Fourier view of how fast the signal is changing. Each
subcarrier's amplitude series is mean-removed (so the static component
doesn't swamp the DC bin), run through `scipy.signal.stft`, and the resulting
power spectra are **averaged across all 90 subcarriers**. The result is
plotted as frequency (Y) against time (X), in dB.

**Why:** Body motion induces Doppler shifts in the reflected WiFi signal —
fast, whole-body motion (run, walk) produces sustained broadband energy,
while a static class (lie down) collapses toward DC. This view separates
classes by *how* they move rather than by absolute amplitude, which is often
more discriminative.

**Parameters:**
- `fs` (**required**) — the time-axis sampling rate in Hz. This sets the
  absolute frequency scale. See the calibration caveat below.
- `nperseg=64` — STFT window length, clamped to the sequence length. With 250
  steps this gives a usable time–frequency trade-off (~17 time bins).
- `noverlap=None` → defaults to 75% of `nperseg` for smooth time resolution.

```python
plot_doppler_spectrogram(x, fs, ax=None, title=None,
                         nperseg=64, noverlap=None, cmap="magma")
```

**Calibration caveat:** UT-HAR clips are resampled to 250 steps, so the
*effective* packet rate is unknown. We pass a nominal `fs=1000` Hz (the Intel
5300's nominal rate). `fs` only rescales the frequency axis linearly — it does
not change the visual pattern or relative comparison between classes — but it
means the absolute Hz values are **not physically meaningful**. Read the
Doppler grid for relative structure, not for true Doppler frequencies.

### 3. Subcarrier traces

**What:** Amplitude-over-time line plots for a chosen handful of subcarriers,
overlaid on one Axes with a legend.

**Why:** The heatmap shows all 90 subcarriers at once but hides exact values;
the traces zoom in on a few so you can read the actual signal — spot a dead
(flat) subcarrier, confirm an outlier spike was removed, or compare how
different subcarriers respond to the same motion. Mostly a debugging tool, not
a deliverable panel.

```python
plot_subcarrier_traces(x, subcarrier_idxs, ax=None, title=None)
# raises IndexError if any index is outside [0, 90)
```

All three coerce complex CSI to amplitude internally, so they work whether
handed raw complex CSI or the already-real UT-HAR data.

## Running the class grids

```bash
conda activate wifisense
python scripts/visualize_classes.py            # seed=0, fs=1000
python scripts/visualize_classes.py --seed 7 --fs 500
```

For each of the 7 classes, the script picks 3 random samples (seeded for
reproducibility) from the **preprocessed** train set and lays them out in a
grid — rows = classes, cols = samples. Drawing from `ut_har.npz` rather than
the raw data means the grids show exactly what the model will see, so they
double as a preprocessing check.

**Inputs:** `data/processed/ut_har/ut_har.npz` (from the preprocessing phase).

**Outputs:**

| Path | Contents |
|------|----------|
| `figures/class_grid.png` | 7×3 grid of amplitude heatmaps, one row per class |
| `figures/doppler_grid.png` | 7×3 grid of Doppler spectrograms (fs nominal) |

**Runtime:** a few seconds on CPU.

## What to check in the grids

- **High-motion vs. static separation.** walk/run should fill their panels
  with sustained texture/energy; lie_down should be the calmest. If a motion
  class looks as flat as lie_down, suspect over-smoothing upstream.
- **Transient classes show a burst.** fall and pickup should read as a quiet
  baseline punctuated by one localized disturbance.
- **No dead subcarriers.** A fully flat horizontal row in the heatmap means a
  channel got killed — that's a bug, not a feature.
- **Within-class consistency.** The 3 samples in a row should rhyme. If they
  look like 3 different activities, the labels or the sampling are suspect.
- **Watch for collinear pairs.** If two *different* classes look identical in
  both views, flag it before training — see below.

## Separability findings

The full write-up is in
[`docs/class_visual_separability.md`](class_visual_separability.md).
Headline:

- **Distinct:** walk and run (sustained broadband motion) vs. lie_down (near
  static) vs. fall/pickup (single transient burst) — these separate well
  along an activity-intensity axis.
- **⚠ Hard pair — sit_down vs stand_up:** nearly identical in both views.
  They're essentially time-reverses of each other, and both the amplitude
  heatmap and the (direction-blind) power spectrogram give the model little to
  separate them on. Expect these to dominate the confusion matrix. If they
  must be split, an amplitude-only representation likely won't do it — a
  direction-aware feature (phase or signed Doppler) would be the lever.
- **Softer overlaps:** walk vs run differ mainly in intensity; fall vs pickup
  share the "one burst" signature.

The preprocessing is broadly trustworthy: clean heatmaps, no clipping/NaN
artifacts, residual horizontal banding is physical (subcarrier-dependent
gain), and the motion texture survived the Hampel + median smoothing.

## Composing into multi-panel figures

The primitives are designed to drop into a larger figure — this is how the
final deliverable's top panel gets built:

```python
import matplotlib.pyplot as plt
from src.viz import (
    plot_amplitude_heatmap, plot_doppler_spectrogram, plot_subcarrier_traces,
)

fig, axes = plt.subplots(3, 1, figsize=(10, 9))
plot_amplitude_heatmap(x, ax=axes[0], title="CSI amplitude")
plot_doppler_spectrogram(x, fs=1000, ax=axes[1], title="Doppler")
plot_subcarrier_traces(x, [0, 30, 60, 89], ax=axes[2], title="Traces")

# attach a colorbar to the heatmap's image
fig.colorbar(axes[0].images[-1], ax=axes[0], fraction=0.046, pad=0.04)
fig.tight_layout()
```

## What feeds into the next stage

Nothing downstream *consumes* the figures — they're for human eyes. But the
`plot_amplitude_heatmap` primitive is the reusable building block for the
deliverable's top panel, and the separability findings are an input to
*training decisions*: class-weighted loss for the imbalance (see
`data_summary.md`) and realistic expectations for the sit_down/stand_up
confusion.

## Code reference

| File | Role |
|------|------|
| `src/viz/csi_plots.py` | The three single-sample rendering primitives |
| `src/viz/__init__.py` | Re-exports the primitives as `src.viz` |
| `scripts/visualize_classes.py` | Builds the per-class heatmap + Doppler grids |
| `docs/class_visual_separability.md` | Honest separability assessment |
