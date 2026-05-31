# Class visual separability (UT-HAR)

Source: 3 random samples/class from the **preprocessed** train set
(`data/processed/ut_har/ut_har.npz`), rendered by
`scripts/visualize_classes.py` into `figures/class_grid.png` (amplitude
heatmaps) and `figures/doppler_grid.png` (STFT Doppler, fs=1000 Hz nominal).

**Assessment.** The amplitude heatmaps and Doppler views separate the data
mostly along an *activity-intensity* axis rather than by specific action.
The cleanly distinct classes are the high-motion ones: **walk** and **run**
show sustained, broadband temporal fluctuation across the whole window
(bright, time-filling Doppler energy), and **lie_down** is distinct in the
other direction — low variation, energy collapsed toward DC, the calmest
panels. **fall** and **pickup** are recognizable as *transient* events: a
calm baseline punctuated by one localized burst (often a strong horizontal
band in the heatmap), which sets them apart from the sustained-motion and
static classes. The worrying pairs are (1) **sit_down vs stand_up**, which
look nearly identical in both views — each is a single short transition, and
since they are essentially time-reverses of one another, the power
spectrogram (direction-blind) and the amplitude heatmap give the model
almost nothing to latch onto; and (2) **walk vs run**, which differ mainly
in degree (run is a bit more intense/dense) and overlap heavily sample-to-
sample. To a lesser extent fall vs pickup share the "one burst on a quiet
baseline" signature and could be confused. I broadly **trust the
preprocessing**: the heatmaps are clean with no obvious clipping or NaN
artifacts, per-sample z-scoring makes panels comparable, and the remaining
horizontal banding is physical (subcarrier-dependent gain), not a bug — the
median/Hampel smoothing did not wash out the motion texture. Two caveats
worth flagging before training: the Doppler frequency axis is not physically
calibrated (clips are resampled to 250 steps, so fs=1000 Hz is nominal and
energy spreads unrealistically high — fine for relative comparison, not for
absolute Doppler), and given the sit_down/stand_up near-collinearity I'd
expect those two to dominate the confusion matrix. That's a data/label
ambiguity the model can't fully fix; if those two must be separated, we may
need a direction-aware feature (e.g. phase or signed Doppler) rather than
the amplitude-only representation we have now.
