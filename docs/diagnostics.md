# Accuracy Diagnosis: Decomposing the Continuous-Capture Gap

The milestone figure (`figures/final_visualization.png`) reports ~56%
per-window accuracy on the stitched continuous capture, against the ~92% the
same model scores on isolated UT-HAR test clips. Before reaching for a fix,
this phase (chunk 7) asks a narrower question: **where does that gap come
from?** `scripts/diagnose_accuracy.py` answers it by separating three error
sources. It is **diagnosis only** — it does not change the model,
preprocessing, training, `window_size`, or `stride`.

Reproduce with:

```bash
conda activate wifisense
python scripts/diagnose_accuracy.py
```

## TL;DR

The 56% continuous accuracy is **almost entirely a sliding-window boundary
artifact**, not a model or preprocessing problem.

| Measurement | Accuracy |
|---|---|
| Clean UT-HAR test set (isolated clips, no sliding window) | **0.920** (macro F1 0.887) |
| Continuous capture — in-segment windows only (boundary windows excluded) | **1.000** (8/8) |
| Continuous capture — all windows (the headline number) | **0.563** |

Clean ≈ in-segment ≫ continuous ⇒ the gap is boundary effects. The upstream
gap is only `1 − 0.92 = 0.08`, while the boundary-attributable gap is
`1.00 − 0.563 = 0.44`.

## The three error sources

A continuous-capture window can be wrong for three distinct reasons, and they
call for completely different fixes:

| Source | What it is | How we measure it |
|---|---|---|
| (a) Genuine model error | Mistakes the model makes on clean, isolated clips too | Accuracy on the UT-HAR test split (no stitching, no sliding window) |
| (b) Boundary effects | Windows whose span straddles two activities — a mixed input never seen in training | In-segment accuracy (boundary windows excluded) vs. all-window accuracy |
| (c) Preprocessing edge effects | NaNs / blow-ups / degenerate normalization at the start/end of the capture | NaN/Inf/range check on the first windows after the pipeline |

### The decision rule

The three accuracies localize the problem:

- **clean ≈ in-segment ≫ continuous** → the gap is **purely boundary effects**.
  The model is fine on whole clips; it only fails where windows mix two
  activities.
- **clean is also low** → the issue is **upstream** (genuine model error or a
  preprocessing problem), and no amount of boundary handling will fix it.

The script prints both gaps explicitly: the *upstream gap* (`1 − clean`) and
the *boundary-attributable gap* (`in_segment − continuous`). Whichever is
larger names the dominant source.

## Findings, source by source

**(a) Genuine model error — small (~8%).** The model scores 92% on the clean
test set. Weakest classes are `sit_down` (0.73) and `stand_up` (0.74); the
rest are ≥0.86. None of these clean errors land in the 8 stitched segments,
so on this particular capture genuine model error contributes **0** to the
continuous number. It would matter more on a capture that happened to stitch
a hard `sit_down`/`stand_up` clip.

**(b) Sliding-window boundary effects — DOMINANT.** This is the whole story.
With `window_size=250` and `stride=25` over eight 250-step segments, only
**8 of 71 windows (11%)** lie fully inside one activity. The other 63 straddle
a seam, so the model is fed a *mixture* of two activities — an input it never
saw in training — and emits a confident-but-wrong label. Exclude those 63 and
accuracy is a perfect 1.000 (right panel of
`per_class_confusion_continuous.png`); include them and the off-diagonal mass
appears (left panel).

`window_position_accuracy.png` makes the mechanism literal: accuracy is a
clean **V-curve** in the window center's offset from the nearest segment
boundary.
- Offset ±125 (window fully inside a segment): **acc = 1.00**.
- Offset 0 (window center sits exactly on a seam, 50/50 mixture): **acc = 0.00**.
The valley bottom is dead zero — *every* window centered on a boundary is
wrong. There is a mild left/right asymmetry (offset −100 dips to 0.29) because
the incoming activity and per-window re-normalization make some approaches to a
seam harder than others, but the V dominates. Note also that some seam errors
are *confident* (e.g. `start=100 → walk @ 0.88`), not low-probability hedges —
so the wrongness is not flagged by low softmax confidence alone.

**(c) Preprocessing edge effects — none found.** The first 8 windows of the
capture were pushed through the training `Pipeline` and checked: **no NaNs, no
Infs**, and per-window z-score behaves exactly as designed (mean ≈ 0, std ≈ 1
for every window). The Hampel/median filters' `reflect` padding at window edges
introduces no blow-ups. Edge-of-capture preprocessing is **not** a contributor.

## The `lie_down → stand_up` failure at capture start

See `figures/lie_down_failure_diagnosis.png` (both rows shown in the
preprocessed representation the model classifies, so the color scale is
comparable). This is **a boundary effect, not a bad clip and not an edge
artifact**:

- The first in-segment window (`start=0`, the genuine clean `lie_down` clip)
  is classified **correctly** as `lie_down` (conf 0.78). The in-segment
  evaluation confirms it — both `lie_down` segments (start 0 and 1750) score
  correct.
- The misclassifications are the *sliding* windows that follow:
  `start=25 → walk`, `start=50 → stand_up (0.60)`, `start=75 → stand_up (0.41)`,
  `start=100 → walk`. Each has the lie_down→fall seam *inside* the window (red
  dashed line in the figure): they contain 25–100 steps of the next `fall`
  segment mixed into `lie_down`.
- So "lie_down predicted as stand_up at the start" is a window straddling the
  first seam, producing a mixed input the model resolves to the wrong class —
  the same mechanism as every other boundary window. It is *not* evidence of a
  corrupted first clip or a preprocessing problem at the capture edge.

## In-segment vs. boundary windows (and an important caveat)

A window covers `[start, start + window_size)`. It is **in-segment** if that
whole span falls inside one ground-truth segment, and a **boundary window**
otherwise. Only in-segment windows have a single correct answer; boundary
windows are fed a mixture, so scoring them against the truth-at-center is
inherently unfair to the model. Re-scoring with boundary windows excluded
isolates source (b) from source (a).

> **Caveat — window/segment alignment.** Here `window_size (250) ==
> segment_len (250)`. Because the window is exactly as long as a segment, the
> **only** in-segment windows are those starting exactly on a boundary
> (start = 0, 250, 500, …) — i.e. the original clean clips. That makes the
> in-segment set tiny (n=8) and trivially equal to a clean-clip evaluation, and
> it forces 89% of windows to straddle a seam. The diagnostic *conclusion*
> (boundary effects dominate) is robust, but the *severity* (56%) is partly an
> artifact of this size/segment alignment plus the synthetic hard-cut stitching
> (real transitions are gradual, not instantaneous).

## Outputs

Run from the repo root with the project env active:

```bash
conda activate wifisense
python scripts/diagnose_accuracy.py
python scripts/diagnose_accuracy.py --window-size 250 --stride 25   # explicit
```

- **stdout + `figures/diagnostics_summary.json`** — clean / in-segment /
  continuous accuracy, per-class clean accuracy, the two gaps, the verdict
  string, and the edge-check flags.
- **`figures/per_class_confusion_continuous.png`** — two confusion matrices:
  all windows (truth at center) beside in-segment-only. Off-diagonal mass in
  the left panel collapsing to a clean diagonal on the right is the visual
  signature of boundary effects.
- **`figures/window_position_accuracy.png`** — accuracy vs. the window
  center's signed offset from the nearest segment boundary; the boundary-effect
  **V-curve**.
- **`figures/lie_down_failure_diagnosis.png`** — the early capture windows
  (preprocessed, as the model sees them) beside correctly-classified clean
  `lie_down` clips, with the seam marked inside each window.

## Code reference

| File | Role |
|------|------|
| `scripts/diagnose_accuracy.py` | Runs the full decomposition and renders the three figures |
| `src/evaluate.py` | Clean-split evaluation this builds on (same checkpoint-reconstruction recipe) |
| `src/inference/streaming.py` | `sliding_window_predict` — the windowing being diagnosed |
| `scripts/build_continuous_capture.py` | Builds the stitched capture (`stream` + `labels_per_step` + `boundaries`) |
| `run_pipeline.sh` | Orchestrator; `./run_pipeline.sh diagnose` runs this chunk (see `pipeline.md`) |
