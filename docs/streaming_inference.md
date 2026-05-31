# Streaming Inference: From Isolated Clips to a Continuous Timeline

The model was trained and evaluated on isolated `(250, 90)` clips, one
activity each. But a real deployment sees a *continuous* CSI stream with no
clip boundaries — activities start, run, and stop whenever the person moves.
This phase bridges that gap: it slides the trained classifier across a
continuous capture and produces a probability-over-time track, then stacks
that against the CSI heatmap and the ground-truth labels in the milestone
figure.

The guiding idea: a classifier that scores 90%+ on neatly-trimmed clips is
not the same thing as a classifier that can label a stream. The moment you
slide a fixed window across continuous data, most windows straddle two
activities, and you find out how the model behaves at the seams. This phase
is built to **show that honestly** — it renders raw model output, with no
temporal smoothing layered on top (see [What's deliberately not
here](#whats-deliberately-not-here)).

## The continuous capture

UT-HAR has no continuous recordings, so we synthesize one by stitching test
clips end-to-end. `scripts/build_continuous_capture.py` picks N clips
(default 8), concatenates them along the time axis, and records which clip —
hence which activity — is active at every time step.

**What it stores:** the capture is saved **raw** (un-normalized CSI). The
sliding-window step applies the training preprocessing per window, so the
stream must be the raw signal those windows are cut from — not already
preprocessed. Storing preprocessed data here would double-normalize.

**Sample selection:** indices are chosen round-robin across classes (seeded,
so reproducible) rather than at random, so the stitched capture cycles
through a variety of activities instead of, say, eight `walk` clips in a row.
With the default N=8 and 7 classes, the timeline shows all seven activities
plus one repeat.

**Why synthetic stitching is honest enough — and where it isn't.** Each
250-step segment is a real recording, so within a segment the signal is
genuine. The artifice is at the *joins*: real activity transitions are
gradual, whereas a stitch is a hard cut. That makes the seams *harder* than
reality, not easier — so boundary errors in the figure are a conservative
view, not a flattering one.

## Sliding-window inference

`sliding_window_predict` in `src/inference/streaming.py` is the core
primitive:

```python
timestamps, probs = sliding_window_predict(
    model, csi_stream, window_size=250, stride=25,
)
# timestamps: (n_windows,) window-center positions, in stream time steps
# probs:      (n_windows, n_classes) softmax per window
```

**What:** slide a window of `window_size` steps across the `(T_total, S)`
stream in hops of `stride`, classify each window, and collect the per-window
softmax. The final fully-contained window is always included even when
`(T_total - window_size)` isn't an exact multiple of `stride`.

**Why per-window preprocessing.** The model was trained on clips put through
`Pipeline` (amplitude → hampel → median → **per-sample** z-score). To keep
inference faithful, each window gets that *same* pipeline applied
independently — per-window normalization mirrors the per-sample
normalization training used. Normalizing the whole stream once instead would
feed the model statistics it never saw in training. Pass a custom `pipeline=`
to override the default.

**Why window centers, not starts.** A window's prediction is evidence about
the *middle* of the span it covers, so `timestamps` returns window centers
(`start + window_size/2`). Centering is what makes the probability track line
up with the ground-truth bar underneath it — a prediction from steps
`[0, 250)` is plotted at step 125, against the truth there.

**Window size vs. segment length — read this before trusting the accuracy.**
With `window_size=250` and 250-step segments, a window only fully covers a
single activity when it is centered exactly on a segment's midpoint.
Everywhere else it overlaps two activities. So the per-window accuracy is
dominated by transition windows by construction — it is a stress test, not an
estimate of clip accuracy. A smaller window would have less straddle but less
context per decision; that trade-off is left for a follow-up.

## The milestone figure

`scripts/final_visualization.py` ties it together into one PNG with three
vertically stacked panels sharing the time axis:

| Panel | Content |
|-------|---------|
| Top | CSI amplitude heatmap of the whole capture (subcarrier × time) |
| Middle | stacked class-probability area, one band per class, over time |
| Bottom | ground-truth activity as a colored bar |

**Shared palette + one legend.** The middle and bottom panels draw from one
`tab10`-derived palette via a shared `ListedColormap`, so *the same color
means the same class* in both, and a single figure-level legend covers both.
This is the one rule that makes the figure readable: your eye matches the
probability band's color to the truth bar's color without a second lookup.

**Layout note.** The heatmap's colorbar lives in a dedicated narrow gridspec
column on the right, not attached to the top axis — attaching it there would
shrink only the top panel and break x-alignment with the two below. Segment
boundaries are drawn as dashed lines on all three panels as alignment cues.
The middle panel intentionally spans only `[125, 1875]` (the valid
window-center range); the white margins are "no window can be centered here,"
not a rendering gap.

## Running

```bash
conda activate wifisense
python scripts/build_continuous_capture.py            # N=8, seed=0
python scripts/build_continuous_capture.py --n 12 --seed 1
python scripts/final_visualization.py                 # window=250, stride=25
python scripts/final_visualization.py --stride 10 --dpi 300
```

Both scripts are runnable directly (not via `-m`); they prepend the repo root
to `sys.path`. Run the capture builder first — the visualization consumes its
`.npz`.

**Inputs:**

| Path | Role |
|------|------|
| `data/raw/ut_har/UT_HAR/` | Raw test clips, stitched into the capture (via `src/data/loader.py`) |
| `runs/best_bilstm.pt` | Trained BiLSTM checkpoint (carries config + class names) |
| `data/continuous/synthetic_capture.npz` | Capture consumed by the visualization |

**Outputs:**

| Path | Contents |
|------|----------|
| `data/continuous/synthetic_capture.npz` | `stream (2000, 90)`, `labels_per_step (2000,)`, `sample_labels (8,)`, `sample_indices (8,)`, `boundaries (9,)`, `sample_len`, `class_names` |
| `figures/final_visualization.png` | 3-panel figure (heatmap / probabilities / ground truth), 200 DPI |

The visualization also prints a per-window table (truth vs. prediction vs.
confidence at each window center) and the overall per-window accuracy to
stdout, so the alignment is auditable, not just visual.

**Runtime:** a few seconds on CPU for the default settings.

## What to check in the figure

- **Confidence inside segments, confusion at seams.** Windows centered well
  inside a segment should be confident and correct; windows near a boundary
  should be wrong or transitional. That gradient *is* the expected behavior
  given a 250-step window on 250-step segments — see the window-size note
  above. If predictions were uniformly wrong even mid-segment, suspect a
  preprocessing mismatch.
- **Probability bands should lag each transition by ~half a window.** Because
  the window is centered, a band can't switch to the new activity until the
  window center crosses the boundary. Visible lag is correct; its absence
  would be suspicious.
- **Colors must agree across the middle and bottom panels.** If a probability
  band's color doesn't match the truth bar beneath it for the same activity,
  the palette wiring is broken.

### Findings from the default run (N=8, seed=0, window=250, stride=25)

- **Per-window accuracy ≈ 0.56.** Low *by design* — over half the 71 windows
  straddle a boundary. Mid-segment windows are largely correct and confident
  (`run` at 0.99, `pickup` at 1.00).
- **A genuine model error, not a boundary artifact:** the first half of the
  `sit_down` segment is confidently misread as `run` (0.94–0.96) before
  snapping to `sit_down`. This is consistent with the
  [separability findings](class_visual_separability.md) — `sit_down`/`stand_up`
  are the hard pair, and amplitude-only features give the model little to
  separate the seated-motion classes on.

## What's deliberately not here

The spec for this phase says to ask before adding *"smoothing,
post-processing, or any heuristic that isn't pure model output"* — to see
what the model actually says first. So this figure is **raw sliding-window
softmax**: no temporal smoothing, no hysteresis, no confidence gating, no
boundary-aware re-weighting. The half-window transition lag and the
boundary-window errors above are consequences of that choice and are left
visible on purpose.

Post-processing is therefore a deliberate **follow-up**, not an omission. Any
such layer (the spec calls these out as a category; specific techniques are
to be decided together, not assumed here) belongs in a separate pass on top
of these raw `(timestamps, probs)`, so the before/after is measurable against
this baseline.

## What feeds into the next stage

`sliding_window_predict` returns `(timestamps, probs)` as plain arrays —
that's the clean seam a post-processing phase would build on, and the same
arrays the figure is rendered from. The synthetic capture format
(`stream` + `labels_per_step` + `boundaries`) is also the natural input for
quantifying any future smoothing against the raw baseline.

## Code reference

| File | Role |
|------|------|
| `src/inference/streaming.py` | `sliding_window_predict` — windowing + per-window preprocessing + softmax |
| `src/inference/__init__.py` | Re-exports `sliding_window_predict` as `src.inference` |
| `scripts/build_continuous_capture.py` | Stitches raw test clips into a continuous capture |
| `scripts/final_visualization.py` | Runs inference and renders the 3-panel milestone figure |
| `src/viz/csi_plots.py` | Heatmap primitive reused for the top panel (see `visualization.md`) |
