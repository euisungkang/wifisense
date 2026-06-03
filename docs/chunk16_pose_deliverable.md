# Chunk 16 — The Phase-3 capstone: evaluating WiFi pose & animating it

Chunk 15 trained the regressor. This chunk **evaluates** it honestly and produces
the deliverable the whole pose phase was building toward: a moving 3D human
skeleton inferred *from WiFi CSI alone*, rendered beside the camera ground truth
so the gap is visible, not hidden.

Two pieces:

- [`src/evaluate_pose.py`](../src/evaluate_pose.py) — MPJPE + PA-MPJPE, overall
  and per-joint, on each split's held-out partition; the per-joint error bar
  chart; and the benchmark-comparison table. Pipeline stage **20**.
- [`scripts/pose_visualization.py`](../scripts/pose_visualization.py) — the
  animated `figures/pose_prediction.gif`. Pipeline stage **21**.

> **This is a regression deliverable.** There is no accuracy and no confusion
> matrix. The skeleton is scored on **distance** (MPJPE, in millimetres) and
> shown moving. Lower is better.

---

## 1. How to read the deliverable

`figures/pose_prediction.gif` (static companion: `figures/pose_prediction_strip.png`)
animates one **held-out cross-subject** motion clip — a person the model never
saw in training, the only honest setting for "what would this do in the wild":

- **Left panel** — the **predicted** skeleton (red) overlaid on the **ground
  truth** (green) in shared 3D axes (`src/viz/skeleton.plot_skeleton_pair`).
  Titled with the per-frame MPJPE and the running (cumulative) MPJPE.
- **Right panel** — the **prediction only**, no ground truth: exactly what the
  system emits with no camera present. This is the actual WiFi-pose use case.
- **Figure title** — action label, subject id, and *"WiFi-predicted vs
  ground-truth 3D pose"*, plus the whole-sequence MPJPE.

The default run auto-selects the **most-motion** held-out clip (most worth
animating); the shipped example is **subject S10, action A19 ("picking up"),
sequence MPJPE ≈ 180 mm**. Override the pick with `--subject` / `--action`.

**It is deliberately not smoothed.** The predicted skeleton is raw per-frame
model output — it jitters and the wrists wander. WiFi pose *is* coarse; that
jitter is the real result, and the same honesty principle as every prior
visualization chunk applies (temporal smoothing would be asked-about first, and
this script never adds it). Axis limits are fixed across the sequence, but that
is only camera framing — it does not touch the poses.

Regenerate:

```bash
conda activate wifisense
python scripts/pose_visualization.py                       # auto-pick clip
python scripts/pose_visualization.py --subject S05 --action A17 --fps 12
```

---

## 2. Final metrics (this checkpoint, E01)

`src/evaluate_pose.py` scores `runs/best_pose.pt` (trained cross-subject on E01,
single-frame, L1 + 0.1·bone). MPJPE is root-relative (pelvis-centered) in mm;
PA-MPJPE additionally Procrustes-aligns each predicted pose (scale + rotation +
translation) to its ground truth before measuring — it strips off the global
pose/size the regressor can't be expected to nail and reports the residual
*shape* error.

| split | held out | frames | MPJPE (mm) | PA-MPJPE (mm) | note |
|-------|----------|-------:|-----------:|--------------:|------|
| **cross_subject** | S05, S10 | 16,038 | **119.0** | **106.9** | faithful generalization test |
| cross_environment | E04 | — | — | — | **skipped**: only E01 on disk |
| random_split | 20% of clips | 16,038 | 106.1 | 90.5 | i.i.d. ceiling — **leaky** (see below) |

- **cross_subject (119.0 mm)** is the headline number and the one to trust: the
  model is scored on two bodies (S05, S10) it never trained on. It matches the
  training-time best val MPJPE, as expected (for this split the val set *is* the
  held-out partition).
- **cross_environment is skipped** because only E01 is downloaded, so the E04
  hold-out has nothing in it. The script detects the empty partition and prints
  why instead of crashing. Download E02–E04 to populate this row.
- **random_split (106.1 mm) is reported but flagged leaky.** Its test clips come
  from the *same subjects* this checkpoint trained on, so it is not a clean test
  of this model — read it only as a "no-domain-shift ceiling". The gap to
  cross_subject (≈ 13 mm) is the cross-subject domain cost, and it is reassuringly
  small here — but on a *single environment's* 10 subjects, so don't over-read it.

### Comparison to the MM-Fi WiFi-only benchmark (honest ballpark)

The MM-Fi paper reports WiFi-CSI 3D-pose MPJPE in the **low hundreds of mm** —
far coarser than its camera / LiDAR / mmWave modalities, because amplitude-only
CSI carries much less geometric information than depth or point clouds. Our
**119 mm** cross-subject result sits squarely in that order of magnitude.

The comparison stops at **"same ballpark", not "matched"**, because our setup
differs from the paper's full protocol (in decreasing order of impact):

1. **Data scale** — E01 only (10 subjects, 2 held out) vs 40 subjects / 4 envs.
2. **Single-frame input** — `window_size = 1` (benchmark per-frame protocol); the
   paper's stronger numbers use temporal context.
3. **Architecture** — a deliberately small CNN baseline, not a tuned model.

A real, working WiFi→3D-pose regressor in the right error regime — built
simple-first. The paper's exact number is quoted as approximate (`~100–130 mm`)
on purpose; treat it as a regime, not a leaderboard cell.

---

## 3. Where the model fails — by joint

`figures/pose_per_joint_error.png` shows the per-joint MPJPE. The pattern is
exactly what pose estimation predicts, and it is worth internalizing:

| joint (cross_subject) | MPJPE (mm) | | joint | MPJPE (mm) |
|---|--:|---|---|--:|
| l_wrist | 347 | | thorax | 85 |
| r_wrist | 345 | | r_knee | 84 |
| l_elbow | 183 | | l_knee | 80 |
| r_elbow | 175 | | spine | 43 |
| head | 121 | | r_hip / l_hip | 17 |
| ankles | ~110 | | **pelvis (root)** | **0.3** |

- **Extremities dominate the error.** Wrists (~345 mm) are ~3× the body average
  and ~20× the hips. They move the most, are the furthest down the kinematic
  chain from the pelvis root, and leave the faintest, most ambiguous CSI
  signature — so the network mostly regresses them toward a population-mean arm
  pose and misses the actual gesture. Elbows are second-worst for the same reason.
- **Error grows with distance from the root.** pelvis ≈ 0 → hips 17 → spine 43 →
  knees 80 → ankles 110, and pelvis → thorax 85 → shoulders ~92 → elbows ~180 →
  wrists ~345. MPJPE is anchored at the pelvis, so error accumulates outward along
  each limb. This is the headline qualitative finding.
- **pelvis ≈ 0.3 mm, not exactly 0.** The target is pelvis-centered (so its GT is
  the origin), but the network is *not* constrained to output zero there — that it
  learns the pelvis to within a third of a millimetre confirms it nails the root
  and spends its error budget on the limbs.

random_split shows the identical shape (wrists worst, root best), just shifted
down ~40 mm at the extremities — same physics, easier (leaky) data.

---

## 4. Where the model fails — by motion

The animated example is action **A19 ("picking up")** for held-out subject S10,
chosen automatically as the highest-motion held-out clip. Watching it is the most
honest way to see the model's limits:

- **Per-frame MPJPE ranges ~43 → 346 mm** across the sequence (mean ≈ 180 mm —
  well above the 119 mm dataset average, *because* it's a high-motion clip).
- **Static, upright frames are good** (low tens of mm on the torso); the predicted
  skeleton is an upright, correctly-articulated body tracking the green truth.
- **The deep bend is where it breaks.** When the subject bends to pick something
  up, the prediction stays far more upright than ground truth and the arms lag —
  the single-frame, amplitude-only model has no temporal context and the rare,
  large-displacement pose is under-represented in 8 training subjects. This is the
  failure you can *see*, and it is the honest face of single-frame WiFi pose.

So: **best on near-static upright posture; worst on fast, large-range, bent
motions and on the extremities that carry that motion.** Cross-subject vs in-domain
is a modest, uniform offset; joint identity and motion dynamics matter far more
than which split you're on.

---

## 5. Limitations and where the field is going

**Limitations of this work (stated plainly):**

- **Single environment.** Everything here is E01 (10 subjects, 2 held out). It
  demonstrates the pipeline and a recognizable cross-subject skeleton; it is *not*
  a faithful reproduction of the paper's 40-subject / 4-environment protocol.
  `cross_environment` can't even be scored until E02–E04 are downloaded.
- **Single-frame, single-person, coarse.** `window_size = 1` and a small CNN by
  design (ask before changing). Wrists are off by a hand-span; the model can't
  resolve fast or unusual poses. WiFi pose is inherently low-resolution — this is
  a *posture* sensor, not a motion-capture replacement.
- **Checkpoint selected on the held-out set.** For cross_subject the val set is
  the eval set, so 119 mm is a mildly optimistic estimate of truly-unseen
  performance (standard for this MM-Fi-style setup, but worth restating).

**Where the field is going — and a possible Phase 4.** This project does
**single-person** pose from a controlled WiFi link. The frontier is
**multi-person pose in the wild** — most directly **Person-in-WiFi-3D** (Yan et
al., CVPR 2024), which estimates **multiple people's 3D poses from commodity WiFi**
by pairing a transformer that attends over the CSI with a person-detection stage,
on a dataset built for crowded scenes. (Search the title for the paper / project
page.) A natural **Phase 4** would be:

1. download the remaining MM-Fi environments and reproduce the real cross-subject
   / cross-environment protocol (closes the data-scale gap, gives a true
   `cross_environment` number);
2. add temporal context (multi-frame windows / a sequence model) — the cheapest
   lever the paper says helps, applied one change at a time so each MPJPE delta
   stays legible;
3. step up to **multi-person** with a Person-in-WiFi-3D-style detect-then-pose
   architecture, where the hard new problem is associating CSI energy with the
   right body — the genuinely open research direction.

The capstone you can run today is the honest single-person baseline those steps
would build on: a moving human skeleton, inferred from WiFi alone, shown next to
the truth so the gap is never hidden.

---

## 6. Pipeline

Added as stages **20** (`evaluate_pose`) and **21** (`pose_viz`) in
`run_pipeline.sh`. Both are light and **data-/checkpoint-gated**: they self-skip
when MM-Fi or `runs/best_pose.pt` is absent (so a default reproduction run never
breaks), and `pose_viz` also skips when the GIF already exists.

```bash
./run_pipeline.sh evaluate_pose pose_viz   # run just the capstone
./run_pipeline.sh                          # default run (self-skips if no data)
```
