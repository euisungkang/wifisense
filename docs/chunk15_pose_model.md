# Chunk 15 — CSI → 3D-pose regression: model, loss & training (Phase 3)

Chunks 13–14 stood up the MM-Fi loader, the 3D-skeleton renderer, and the
`(CSI window, 3D pose)` Dataset with cross-subject / cross-environment splits —
and **proved the pairs are temporally aligned** before any training. This chunk
trains the model that turns a WiFi-CSI window into a 3D pose.

> **This is the project's first regression model.** Every model before it
> (`BiLSTM`, `BVPCNNRNN`) emits class logits and is scored on **accuracy**. This
> one emits continuous 3D joint coordinates — a `(17, 3)` pose — and is scored on
> **MPJPE**, a distance in millimetres. No softmax, no class axis, no accuracy.

New pieces:

- [`src/models/csi_pose_net.py`](../src/models/csi_pose_net.py) — `CSIPoseNet`,
  a small CNN encoder + MLP regression head (registered as `csi_pose_net`).
- [`src/train_pose.py`](../src/train_pose.py) — the regression trainer: L1/MSE +
  bone-length loss, early-stopping on val MPJPE, skeleton-convergence snapshots.
  Pipeline stage 19 (`train_pose`).

---

## 1. Architecture rationale

The MM-Fi paper's WiFi-pose baseline is a **small CNN regressor** over the
`(3, 114, 10)` amplitude map — not a Transformer. We deliberately mirror that and
keep it modest: get a simple per-window regressor converging and producing
recognizable skeletons *first*, before reaching for attention or multi-frame
temporal models.

**Input.** A sample is a centered CSI window `(W, 3, 114, 10)` = `W` frames × 3
antennas × 114 subcarriers × 10 packets. With the benchmark-faithful default
`window_size = 1` that is one `(3, 114, 10)` frame → one pose, exactly MM-Fi's
published WiFi→pose protocol.

**Encoder → head.** Fold the window's frames into the channel axis and run a 2-D
CNN over the (subcarrier × packet) plane, then flatten and regress the joint
coordinates with a 2-layer MLP:

```
input            (B, 1, 3, 114, 10)
fold W into ch   (B, 3, 114, 10)        in_channels = W * 3
conv block 1     (B,  32, 57, 5)        3x3, BN, ReLU, MaxPool2
conv block 2     (B,  64, 28, 2)        3x3, BN, ReLU, MaxPool2
conv block 3     (B, 128,  4, 2)        3x3, BN, ReLU, AdaptiveAvgPool(4,2)
flatten          (B, 1024)
FC + ReLU + drop (B, 256)
FC (head)        (B, 51)                = n_joints * 3
reshape          (B, 17, 3)
```

- **Why fold the window into channels.** At `W = 1` this is a no-op (3 antenna
  channels); for larger odd windows the extra frames simply add input channels,
  so the same architecture handles multi-frame experiments without structural
  change. The adaptive pool fixes the flattened width regardless of the exact
  conv arithmetic.
- **Why a plain linear head (no output activation).** Joint coordinates are
  unbounded real numbers (metres), so the head must be linear. The output is a
  **root-relative** pose because the Dataset's target is root-centered (pelvis at
  the origin) — the network regresses *posture*, not the person's absolute
  location in the room.
- **Size.** ~0.37 M parameters — the same modest band as the other models in this
  repo, and far from a Transformer. Per the brief: **ask before adding
  architectural complexity** (attention, temporal models). The simple version is
  the baseline these would have to beat.

---

## 2. Loss design

```
loss = coord_loss + bone_weight * bone_loss
```

| term | what | default |
|------|------|---------|
| `coord_loss` | `--loss l1` (default) or `mse` on the root-centered joint coordinates (metres) | L1 |
| `bone_loss` | mean absolute error between predicted and GT **bone lengths** over the MM-Fi kinematic tree | weight **0.1** |

- **Why L1 by default.** L1 (mean absolute error) on coordinates is robust to the
  occasional badly-placed joint, which otherwise dominates an MSE gradient. It is
  what directly drives MPJPE (itself an L2-per-joint, averaged) down. `--loss mse`
  is available for comparison.
- **The bone-length regularizer is a soft anatomical prior.** Coordinate loss
  alone can produce a skeleton that matches joints "on average" but has
  rubber-band limbs (a forearm that stretches frame to frame). `bone_loss`
  penalizes the difference between predicted and true *bone lengths* along the 16
  kinematic-tree edges (`src/viz/skeleton.SKELETON_EDGES`), nudging the model
  toward limbs of the right size.
- **Weighting (`bone_weight = 0.1`).** Bone lengths and coordinate errors are both
  in metres, so a weight of 0.1 lets the bone term contribute ~10% the scale of an
  L1 coordinate error — enough to regularize, not enough to fight the primary
  objective. Set `--bone-weight 0` to train on coordinates alone (a useful
  ablation). It is a regularizer, **not** the metric — MPJPE ignores it entirely.

---

## 3. MPJPE — what it means and how to read it

**MPJPE = Mean Per-Joint Position Error.** For one pose it is the average
Euclidean distance between each predicted joint and its ground-truth position,
after both are root-centered:

```
MPJPE(pose) = mean_j ‖ pred_j − gt_j ‖₂        # averaged over the 17 joints
```

reported over the dataset as the mean of that per-pose value. We report it in
**millimetres**; **lower is better** (this inverts the classifiers'
higher-is-better val-accuracy convention, so early stopping here minimizes).

- **Why it's translation-free.** The Dataset already centers every pose on the
  pelvis (root), so MPJPE measures *posture* error, not where the person is in the
  room. With the default `pose_scale=None` the target stays in metres, so
  `MPJPE_mm = mean(‖pred − gt‖) × 1000` directly — no un-normalization needed.
  (`train_pose.py` keeps `pose_scale=None` precisely so MPJPE stays metric; a
  scaled target would need denormalizing first, per
  `docs/chunk14_pose_pipeline.md`.)
- **How to read a number.** ~50 mm is a joint off by a finger-width; ~150 mm is a
  hand-span; ~300 mm is a forearm. WiFi pose is inherently coarse — **do not
  expect camera-quality skeletons.** Early epochs land in the low **hundreds of
  mm**; the MM-Fi paper's WiFi-only MPJPE is the target ceiling, not camera-grade
  single-digit-centimetre accuracy.

---

## 4. Watching the skeleton converge

Every `--progress-every` epochs (default 5, plus epoch 1 and every new best), the
trainer renders the model's current prediction against ground truth for a **fixed**
validation sample to `runs/<timestamp>/progress/epoch_NNN.png`. Because the sample
is fixed, the snapshots form a watchable sequence: a near-random red skeleton at
epoch 1 visibly collapsing onto the green ground-truth pose as MPJPE drops. This
is both a debug tool (a model that "trains" but predicts garbage shows it here
immediately — usually a sign of misalignment, which chunk 14's verification is
designed to rule out) and exactly the milestone output Phase 3 has been building
toward.

---

## 5. Training results

Setup: `--split cross_subject` on the downloaded MM-Fi **E01** environment
(subjects S01–S10; the default held-out S05/S10 live in E01), `window_size = 1`,
L1 + 0.1·bone loss, Adam lr 1e-3, batch 64, early stopping on val MPJPE
(patience 12). CPU-only (~35 s/epoch over the 64 k-frame train partition).

| split | held out | train frames | val frames | best val MPJPE | @ epoch | epochs run |
|-------|----------|-------------:|-----------:|---------------:|--------:|-----------:|
| cross_subject | S05, S10 | 64,152 | 16,038 | **119.0 mm** | 6 | 18 (early-stop) |

Reading the run: val MPJPE drops fast (125.7 mm → ~120 mm in the first ~6 epochs)
then **plateaus around 120 mm while train MPJPE keeps falling** (to ~96 mm by
epoch 18) — textbook mild overfitting on a single environment's 8 training
subjects, which is exactly why we early-stop on val MPJPE rather than training to
convergence. The per-sample progress snapshots tell the same story visually: by
epoch ~15 the red predicted skeleton is an upright, correctly-articulated body
tracking the green ground truth (~100 mm on that fixed sample), not a camera-grade
match but unmistakably a person in the right pose.

Read the curves in `runs/<ts>/training_curves.png` (train/val loss + val MPJPE)
and the convergence snapshots in `runs/<ts>/progress/`.

> **Scope caveat (honest).** These numbers are on **E01 alone** (2 held-out
> subjects), not the full 40-subject MM-Fi corpus, because only E01 is downloaded
> (`docs/chunk13_mmfi_setup.md`). They demonstrate the pipeline end-to-end and a
> recognizable cross-subject skeleton; they are **not** a faithful reproduction of
> the paper's cross-subject protocol over all environments. A second, smaller
> caveat: for cross_subject the validation set *is* the held-out partition and the
> checkpoint is selected on it, so the reported MPJPE is mildly optimistic as an
> estimate of truly-unseen performance (standard for this MM-Fi-style setup, but
> worth stating).

---

## 6. Honest comparison to the MM-Fi WiFi-only benchmark

The MM-Fi paper reports WiFi-CSI pose estimation MPJPE in the **low hundreds of
millimetres** — far coarser than the same paper's camera/LiDAR/mmWave modalities,
because amplitude-only CSI carries much less geometric information than depth or
point clouds. That is the regime to expect and the ceiling to aim at; this
chunk's simple per-frame CNN is intended to land *in that ballpark*, not beat it.
Our **119 mm** cross-subject result on E01 sits squarely in that order of
magnitude — encouraging confirmation that the pipeline produces a real WiFi pose
regressor — but the comparison stops at "same ballpark", not "matched", for the
reasons below.

Where our setup differs from the paper (all of which make a direct number-to-number
comparison unfair, in roughly decreasing order of impact):

1. **Data scale.** We train on E01 only (10 subjects, 2 held out) vs the paper's
   40 subjects across 4 environments. Less data and a narrower domain.
2. **Single-frame input.** `window_size = 1` matches the benchmark's per-frame
   protocol, but the paper's stronger results use temporal context; we keep it
   single-frame on purpose (ask before changing).
3. **Architecture.** A deliberately small CNN baseline, not a tuned or
   temporally-aware network.

The honest takeaway: this is a **working, recognizable** WiFi→3D-pose regressor in
the right error regime, built simple-first. Closing the gap to the paper's best
WiFi numbers is future work — more environments downloaded, temporal windows, and
a stronger encoder — and should be approached one lever at a time so each change's
effect on MPJPE is legible.

---

## 7. Pipeline

Added as stage **19** (`train_pose`) in `run_pipeline.sh`. Like the other heavy
trainers it is **excluded from the no-arg default** (so a plain reproduction run
never retrains), and like the other MM-Fi stages it **self-skips** when the
dataset is absent or `runs/best_pose.pt` already exists, so the default run
doesn't break for users who haven't downloaded MM-Fi:

```bash
./run_pipeline.sh train_pose        # train the pose regressor explicitly
```
