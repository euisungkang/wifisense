# Post-Processing: Temporal Smoothing of Per-Window Predictions

Chunk 6 rendered the continuous-capture predictions as **raw** sliding-window
softmax — one independent forward pass per window, no temporal coupling. Chunk 7
([`docs/diagnostics.md`](diagnostics.md)) showed the resulting ~56% per-window
accuracy is dominated by **boundary windows**: a window whose span straddles two
activities is a mixture the model never trained on, so it flips to a
confident-but-wrong label.

This chunk adds the post-processing layer deliberately deferred in chunk 6. A
classifier sees each window in isolation, but activities have temporal
structure: they persist for many windows and only occasionally switch. All three
strategies below inject that prior over the per-window output. They are pure
array transforms over the already-computed probabilities — no retraining, no
change to `window_size`/`stride`.

Reproduce with:

```bash
conda activate wifisense
python scripts/compare_postprocessing.py
```

Outputs: [`notes/postprocessing.md`](../notes/postprocessing.md) (the table) and
`figures/final_visualization_smoothed.png` (raw vs. best-method panels). Code in
[`src/inference/postprocess.py`](../src/inference/postprocess.py).

## The three strategies

### 1. Moving average — `moving_average(probs, k)`
Average the probability vectors over a centered window of `k` windows, then
argmax. Because each smoothed row is a mean of rows that each sum to 1, the
result is still a valid distribution (no renormalization). Edges shrink the
window rather than pad it.

- **What it encodes:** "nearby windows should agree", in *probability* space, so
  a single confident spike is outvoted by its neighbours but a sustained
  high-confidence run survives.
- **When appropriate:** a good default when you trust the softmax magnitudes and
  want a cheap, order-1 smoother. Degrades gracefully — it never makes a hard
  commitment, just shifts mass.

### 2. Majority vote — `majority_vote(preds, k)`
Mode filter over the argmax labels in a centered window of `k`. Ties break toward
the smaller label index.

- **What it encodes:** "nearby windows should agree", in *label* space.
  Confidence is discarded entirely — a 0.99 and a 0.34 prediction count the same.
- **When appropriate:** when the softmax is poorly calibrated (so probability
  magnitudes are untrustworthy) but the argmax is usually right. It is the
  **conservative** choice here: it left in-segment accuracy untouched (1.000),
  because it only changes a label when a *majority* of its neighbours disagree.

### 3. HMM / Viterbi decoding — `hmm_decode(probs, transition_matrix)`
Treat the sequence as a hidden Markov model: the per-window softmax is the
emission score, an explicit transition matrix is the prior over label changes,
and Viterbi finds the single highest-scoring label path over the whole sequence.

- **What it encodes:** not just "neighbours agree" but *which* transitions are
  plausible and how costly a switch is, optimized **globally** rather than in a
  fixed local window. This is the most principled formulation — the others are
  special cases of "prefer temporal consistency" without an explicit transition
  model.
- **Transition matrix:** learned from the UT-HAR **training** label sequence by
  counting adjacent transitions, Laplace-smoothed (`alpha=1`) so no transition
  is impossible. See the caveat below.
- **Emission approximation:** we feed the classifier posterior `P(class|window)`
  directly as the emission score. That equals the true emission likelihood
  `P(window|class)` only under a uniform class prior — a standard, pragmatic
  approximation when smoothing a classifier with an HMM.
- **When appropriate:** when you have a trustworthy transition model (realistic
  dwell times, plausible activity orderings). It is the strongest smoother when
  the prior is right and the most dangerous when it is wrong (see caveats).

## Comparison results

`window_size=250`, `stride=25`, `k=5` windows, HMM Laplace `alpha=1.0`. 71
windows total, 8 in-segment. Transition rate = class flips per 100 windows
(lower = smoother); in-segment accuracy uses chunk 7's boundary-exclusion logic.

| Method | Per-window acc | Δ vs raw | In-segment acc | Transition rate |
|---|---|---|---|---|
| raw | 0.563 | — | 1.000 | 32.9 |
| moving_average (k=5) | 0.676 | +0.113 | 0.875 | 17.1 |
| majority_vote (k=5) | 0.620 | +0.056 | 1.000 | 20.0 |
| **hmm_decode (Viterbi)** | **0.718** | **+0.155** | 0.875 | **10.0** |

(Numbers are regenerated into `notes/postprocessing.md` on every run.)

Reading the table:
- Every method **raises per-window accuracy** over raw (all the gain is on
  boundary windows — raw is already perfect in-segment).
- Every method **cuts the transition rate** sharply: the raw 32.9 flips/100 is
  mostly boundary jitter, not real activity changes (the capture has ~7 true
  transitions over 70 steps ≈ 10/100).
- **HMM wins per-window accuracy and is the smoothest** (10.0 ≈ the true
  transition rate — it recovers almost exactly the right number of switches).
- **Majority vote is the conservative option**: smaller accuracy gain but it is
  the only smoother that leaves in-segment accuracy at 1.000.
- Moving average and HMM each cost one in-segment window (1.000 → 0.875): the
  smoothing pulls a neighbour's label across a seam into a window that was
  individually correct. This is the price of temporal coupling.

## Chosen default: HMM (`hmm_decode`)

It has the best per-window accuracy (0.718) and the smoothest output (10.0
flips/100, matching the true transition rate), and it is the most principled — it
encodes an explicit, inspectable prior over transitions rather than an implicit
"neighbours agree" rule. `figures/final_visualization_smoothed.png` shows the raw
panel (jagged, boundary-driven churn) above the HMM panel (clean blocks that
track the ground-truth timeline).

If you do **not** trust the transition prior on a given deployment, prefer
**majority vote**: it gave a real accuracy bump without ever degrading a window
the model already had right.

## Honest caveats

- **The transition matrix is barely "learned".** UT-HAR training samples are
  isolated, single-activity clips stored grouped by class, so adjacent training
  labels are almost always identical. The counted matrix is therefore
  **near-identity** (self-transition ≈ 0.98–0.995); its off-diagonal is
  essentially the Laplace prior plus a handful of class-group boundaries, not
  observed activity dynamics. We have **no continuous, multi-activity training
  recording** to estimate realistic transition probabilities or dwell times
  from. The matrix works here mainly as a generic "activities persist" prior,
  not as a calibrated model of *which* activity follows which.

- **HMM can over-smooth — we saw it.** During development, building the matrix
  by first expanding each clip to its window-cadence run made the self-transition
  prior even stickier (≈0.999). Viterbi then **merged adjacent activities**,
  collapsing the deliberately short (10-window) segments and dropping accuracy to
  **0.479 — below raw** — with in-segment accuracy falling to 0.500. The same
  failure would appear on any capture whose segments are short relative to the
  prior's implied dwell time. A strong transition prior papers over real
  transitions exactly as readily as it cleans up boundary noise.

- **Smoothing papers over genuine model errors too.** These methods help the
  *boundary* windows, which are the bulk of the error here. But where the model
  is genuinely, persistently wrong over several windows (a real
  misclassification, not a seam artifact), every smoother will keep — and
  lengthen — that wrong run rather than fix it. Higher per-window accuracy after
  smoothing is not evidence the underlying model improved.

- **`k` is untuned by design.** `k=5` is ~half a segment (a segment is
  `window_size/stride = 10` windows). It was chosen once on this principle, not
  swept to maximize the figure — overfitting a single visualization to a hand-
  picked `k` would be exactly the temptation to avoid. `--k` is exposed for
  inspection, not for cherry-picking.

- **The synthetic-capture caveats from chunk 7 still apply.** `window_size ==
  segment_len` and hard-cut stitching make boundary windows a large, artificially
  abrupt fraction of the total. On real captures with gradual transitions and
  longer dwell times the *absolute* numbers would differ; the *ordering* of the
  methods (smoothing helps boundaries; an over-strong prior over-smooths) is the
  portable lesson.

## Code reference

| File | Role |
|------|------|
| `src/inference/postprocess.py` | The three strategies + `learn_transition_matrix`, `viterbi_decode`, `transition_rate` |
| `scripts/compare_postprocessing.py` | Runs the comparison, writes the table + smoothed figure |
| `scripts/diagnose_accuracy.py` | Chunk 7; supplies the boundary-exclusion logic reused here |
| `src/inference/streaming.py` | `sliding_window_predict` — the per-window output being smoothed |
| `notes/postprocessing.md` | Generated comparison table (do not hand-edit) |
| `figures/final_visualization_smoothed.png` | Raw vs. best-method 4-panel figure |
| `run_pipeline.sh` | `./run_pipeline.sh postprocess` runs this chunk |
