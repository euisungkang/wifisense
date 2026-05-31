# Chunk 12 — Widar3.0 BVP: gesture model, cross-domain evaluation & spatial-motion figure

Chunk 11 turned the raw BVP corpus into model-ready tensors and the four
canonical cross-domain splits. This chunk closes the Widar3.0 phase: it adds a
gesture classifier built for BVP's structure, trains and evaluates it across all
four splits — the real test of BVP's environment-invariance claim — and produces
the **spatial-motion** milestone figure, the tier-2 analog of chunk 6's 3-panel
continuous-capture visualization.

New pieces:

- [`src/models/bvp_cnn_rnn.py`](../src/models/bvp_cnn_rnn.py) — the `BVPCNNRNN`
  architecture (registered as `bvp_cnn_rnn` in `build_model`).
- [`src/train_widar.py`](../src/train_widar.py) — trains one model per split.
- [`src/evaluate_widar.py`](../src/evaluate_widar.py) — scores all four splits and
  assembles `figures/widar_domain_results.png`.
- [`scripts/spatial_viz.py`](../scripts/spatial_viz.py) — the spatial-motion figure
  `figures/spatial_motion.png`.

## 1. Why CNN-RNN over BVP

A BVP sample is a `(T, 20, 20)` volume: `T` timesteps, each a 20×20 grid of
motion energy over a 2-D **body-frame velocity** space (x-velocity × y-velocity,
±2 m/s, sampled at 10 Hz). That shape is explicitly *spatial-then-temporal*, and
the architecture mirrors it:

```
per-timestep 2-D CNN     ->  "which velocities are active right now" (a spatial feature)
GRU over the T features  ->  how that velocity pattern evolves through the gesture
linear head              ->  one of 22 gestures
```

The CNN is applied to **every** one of the `T` frames (time folded into the batch
axis), so it learns a shared per-frame velocity-pattern encoder; the GRU then
consumes the resulting length-`T` sequence and the final (bi)directional hidden
state is classified. Concretely (defaults, `target_T = 32`):

| stage | shape | notes |
|---|---|---|
| input | `(B, 32, 20, 20)` | chunk-11 normalized + padded volume |
| fold time | `(B·32, 1, 20, 20)` | one frame per CNN forward |
| conv block 1 | `(B·32, 16, 10, 10)` | 3×3, BN, ReLU, MaxPool2 |
| conv block 2 | `(B·32, 32, 5, 5)` | 3×3, BN, ReLU, MaxPool2 |
| conv block 3 | `(B·32, 64, 2, 2)` | 3×3, BN, ReLU, AdaptiveAvgPool(2) |
| flatten / unfold | `(B, 32, 256)` | per-timestep feature sequence |
| BiGRU | `(B, 2·128)` | concat final fwd+bwd hidden states |
| dropout + FC | `(B, 22)` | logits |

**~326k parameters** — inside the 100k–500k target. This is the SenseFi
`Widar_CNN_GRU` idea (`vendor/SenseFi/widar_model.py`) brought into this project's
conventions: a `config` property so checkpoints self-describe and `build_model`
can reconstruct the model, an explicit `num_classes`, and exposed hyperparameters.
Unlike the SenseFi reference — which fixes `T = 22` by folding time into an MLP/CNN
channel axis — this model is **time-length agnostic**: the same CNN runs on
however many frames a sample has, so chunk-11's `target_T` is a free choice.

Design notes:

- **AdaptiveAvgPool(2)** on the third block fixes the flattened width at
  `64·2·2 = 256` regardless of the exact conv arithmetic, decoupling the GRU input
  size from kernel/stride bookkeeping.
- **BatchNorm** over the `B·T` folded batch is well-conditioned (2048 samples at
  the default batch).
- **Augmentation: horizontal flip is off by default** (`flip_prob=0`). The chunk-11
  flip (`v_x → −v_x`) is only label-preserving for left/right-symmetric gestures;
  across the 22-gesture set many are direction-defined (Slide, Draw-N, the
  digits), so flipping would corrupt labels. Temporal crop and mild Gaussian noise
  stay on.

## 2. Training: one model per split

The whole point of Widar3.0 is *cross-domain* generalization, so each split
defines a **different** train/test partition and therefore needs its **own**
trained model. `src/train_widar.py` with no `--split` trains all four in turn:

| split | held out (default) | question it answers |
|---|---|---|
| `in_domain` | random 20% | baseline accuracy with **no** domain shift |
| `cross_user` | user 3 | recognize gestures from **people** never trained on? |
| `cross_position` | torso location 5 | generalize to **room locations** never seen? |
| `cross_orientation` | face orientation 5 | is the representation truly **facing-invariant**? |

Each split:

1. builds its in-domain training partition via the chunk-11 split builder;
2. carves a **gesture-stratified** validation set out of it (same domain as
   train) to drive early stopping — the true domain-shift test is held back and
   never touched at train time;
3. fits the CNN-RNN with Adam + cross-entropy, checkpointing best-val-accuracy
   weights to `runs/best_bvp_<split>.pt` (and a timestamped run dir with
   `metrics.json` + `training_curves.png`), exactly mirroring `src/train.py`.

The checkpoint stores the full `split_config` (split name, held-out values,
scoping filters, `target_T`, `normalize`, `seed`, any `max_per_gesture` cap) so
the evaluator reconstructs the *identical* held-out partition without guessing.

```bash
conda activate wifisense
python src/train_widar.py                         # all four splits, full corpus
python src/train_widar.py --split cross_user --test-users 3 7
```

**Compute note (CPU).** The corpus is ~43.7k samples and this machine is
CPU-only, so a full four-split run is a multi-hour job. Two levers make it
tractable without changing any code:

- `--room 1` (or `2`/`3`) scopes to one capture room;
- `--max-per-gesture N` caps samples per gesture (random subset) before splitting;
- the chunk-11 `Dataset` gained an in-memory **raw-volume cache** (on by default
  here via `cache=True`), which removes the `.mat` IO cost on every epoch after
  the first — the dominant cost of a CPU epoch. Use `num_workers=0` so the cache
  is shared (it is the default).

## 3. Evaluation across splits — the environment-invariance test

`src/evaluate_widar.py` loads each `runs/best_bvp_<split>.pt`, rebuilds its
held-out test set from the stored `split_config`, scores it (overall accuracy,
macro F1, full per-class report, confusion matrix), and assembles
`figures/widar_domain_results.png`: a 2×2 grid of row-normalized confusion
matrices, one per split, each titled with its accuracy.

```bash
python src/evaluate_widar.py            # writes figures/widar_domain_results.png + figures/widar/*.json,csv
```

### What to expect, and the comparison that matters

This is where BVP earns its keep. In **chunk 9** we measured raw-CSI models across
a domain shift and watched them collapse
([`figures/domain_shift_matrix.png`](../figures/domain_shift_matrix.png)):

| | tested in-domain | tested cross-domain (zero-shot) |
|---|---|---|
| UT-HAR BiLSTM | **92.0%** | **42.4%** on NTU-Fi |
| NTU-Fi BiLSTM | **98.5%** | **1.6%** on UT-HAR |

A 50–90 point cliff: raw CSI encodes the *room's* multipath as much as the
motion, so the model memorizes its capture environment. BVP is the principled fix
— a body-frame velocity representation that factors out the TX/RX geometry and
static multipath — so the cross-domain drop should be **far** smaller.

The Widar3.0 paper's targets (the bar this chunk aims at) are roughly:

| split | target accuracy |
|---|---|
| in-domain | > 85% |
| cross-user | > 75% |
| cross-position | similar range |
| cross-orientation | usually the **hardest** (don't panic if it lags) |

> **Status:** the concrete numbers and the figure are produced by running
> `src/train_widar.py` + `src/evaluate_widar.py` on your machine — they are *not*
> baked into this doc, because the four full-corpus trainings are a multi-hour CPU
> job left to run when convenient (see the compute note above). When you run them,
> read the result the same way as chunk 9's matrix: there the off-diagonal
> (cross-domain) cells were nearly empty; here the **cross-\* panels should stay
> strongly diagonal**. The accuracy gap between `in_domain` and the cross-\*
> splits *is* the residual domain gap BVP leaves behind — and it should be a small
> fraction of chunk 9's 50–90 point cliff. `evaluate_widar.py` prints that
> in-domain → worst-cross-domain gap explicitly at the end of its run.

Why cross-orientation tends to be hardest: although BVP is computed in a
body-frame, the rotation that removes facing direction is imperfect (it relies on
a discrete orientation label and `nearest`-interpolated grid rotation — see
chunk 11 §3), so some orientation-dependent residue survives into the velocity
grid. Cross-position is gentler because torso location mostly changes the bistatic
geometry that BVP already inverts away.

## 4. Reading `figures/spatial_motion.png`

The spatial-motion figure is the chunk-12 milestone and the tier-2 analog of
chunk 6's 3-panel figure. Where chunk 6 showed *time vs. predicted activity* over
a raw-CSI stream, BVP lets us show something raw CSI never could: the **spatial
shape of the motion itself**. For six representative gestures it draws a two-row
block:

- **Row 1 — six BVP frames** evenly sampled across the gesture (20×20 each), each
  the raw per-frame motion energy over the body-frame velocity plane (±2 m/s, x
  horizontal, y vertical, origin at center = "not moving"). A bright blob top-right
  means "moving in +x,+y right now". Frames are drawn with `nearest` interpolation
  — **raw model input, no smoothing** (same raw-output-first honesty as chunk 6).
- **Row 2 — the integrated motion trajectory.** Each frame's energy-weighted
  centroid is a velocity; integrating velocity over time (`position = Σ v·dt`,
  `dt = 1/10 s`) reconstructs the path the hand traced. Green circle = start, red
  square = end, arrows mark direction of travel. A loop gesture (Draw-O) closes on
  itself; a linear gesture (Slide, Push&Pull) traces an out-and-back line; a
  zig-zag traces a zig-zag.

Each block is titled with the **ground truth**, the model's **predicted class**
(with confidence), and the result. **Misclassifications are left visible and
flagged** — a red `MISS`, red metadata, and a red border around the whole block —
so failures are honest, never cherry-picked or hidden. The sample drawn per
gesture is deterministic in `--seed`.

```bash
python scripts/spatial_viz.py                                   # uses the first runs/best_bvp_*.pt
python scripts/spatial_viz.py --checkpoint runs/best_bvp_cross_user.pt --room 1
```

A note on honesty (carried from chunk 6): the trajectory is computed from the
**raw non-negative energy**, independent of the model, and no interpolation or
smoothing is applied to the BVP frames. If we ever add frame smoothing for
legibility, that is a deliberate change to raise with the project owner first.

## 5. Where to go from here — tier-3 pose estimation

This project has climbed a representation ladder: **raw CSI** (UT-HAR / NTU-Fi,
chunks 1–9, environment-bound) → **BVP** (Widar3.0, chunks 10–12,
environment-invariant *gesture* recognition). The natural tier-3 step is from
recognizing *which gesture* to estimating *body pose / motion in 3-D*:

- **[MM-Fi](https://ntu-aiot-lab.github.io/mm-fi)** — a multi-modal (WiFi CSI +
  LiDAR + mmWave + RGB-D) dataset with 2-D/3-D human pose annotations. WiFi-to-pose
  regression is the headline task; the other modalities give supervision and a
  ceiling to compare against.
- **[Person-in-WiFi-3D](https://aiotgroup.github.io/Person-in-WiFi-3D/)** — 3-D
  human pose estimation from commodity WiFi, with multi-person scenes. The closest
  successor to the "WiFi as a sensor of the body" thesis BVP started.

Both move the output space from a 22-way classifier to a structured regression
(joint coordinates), which changes the model head (keypoint/heatmap regression),
the loss (MPJPE / PCK rather than cross-entropy), and the evaluation (pose error,
not accuracy). The BVP intuition carries over: a body-frame motion representation
is a strong prior for where the limbs are and how they move.

## Artifacts

| Path | What |
|---|---|
| `src/models/bvp_cnn_rnn.py` | `BVPCNNRNN` — CNN-RNN over `(B,T,20,20)` (~326k params) |
| `src/train_widar.py` | per-split trainer → `runs/best_bvp_<split>.pt` |
| `src/evaluate_widar.py` | per-split eval + `figures/widar_domain_results.png` |
| `scripts/spatial_viz.py` | spatial-motion figure → `figures/spatial_motion.png` |
| `figures/widar_domain_results.png` | 2×2 confusion-matrix grid across the four splits |
| `figures/spatial_motion.png` | six-gesture spatial-motion milestone figure |
| `figures/widar/<split>_{metrics.json,predictions.csv}` | per-split metrics + per-sample predictions |
| `runs/best_bvp_<split>.pt` | the four trained checkpoints (each carries its `split_config`) |
