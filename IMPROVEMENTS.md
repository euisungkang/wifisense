# IMPROVEMENTS.md — raising WiFi-pose precision

A prioritized, evidence-grounded roadmap for lowering the CSI→3D-pose model's
error below the current baseline. Each item states **what**, **why (evidence)**,
**effort**, **risk**, and **expected impact**, so we can pick levers knowingly
and pull them one at a time (so each change's effect on MPJPE stays legible).

> Companion to [`docs/chunk16_pose_deliverable.md`](docs/chunk16_pose_deliverable.md)
> (the eval that motivates this) and [`docs/chunk15_pose_model.md`](docs/chunk15_pose_model.md)
> (the model/loss/training being improved).

---

## Where we are (baseline)

`runs/best_pose.pt` — `CSIPoseNet`, single-frame (`window_size = 1`), L1 + 0.1·bone,
trained cross-subject on **E01 only**:

| metric | value |
|---|---|
| cross_subject MPJPE | **119.0 mm** (PA-MPJPE 106.9) |
| random_split MPJPE (leaky) | 106.1 mm (PA 90.5) |
| cross_environment | un-scorable (only E01 on disk) |

## What the diagnosis says (the three facts that drive this roadmap)

1. **Overfitting / data starvation.** Train MPJPE fell to ~96 mm while val
   plateaued ~120 mm and early-stopped at epoch 6 — the model has capacity to
   spare and not enough data. → improvements that add data or regularize.
2. **Error lives in the extremities.** Per-joint MPJPE: wrists ~345 mm, elbows
   ~180 mm; everything pelvis-inward is already <90 mm (pelvis ≈ 0.3 mm). →
   improvements that specifically discipline the arms.
3. **It breaks on dynamic / bent poses.** The animated A19 ("picking up") clip
   ranges 43→346 mm per frame; a single amplitude-only frame can't disambiguate
   fast limb positions. → improvements that add temporal context.

---

## Prioritized improvements

Ordered by expected impact-per-effort. **A → B → C** is the recommended sequence.

### A. More data — download E02–E04  ⭐ active, do first

- **What.** Train on the full 4-environment MM-Fi corpus (40 subjects) instead of
  E01's 10. Unblocks a real `cross_environment` number too.
- **Why (evidence).** Directly addresses fact #1: 119 mm is on 8 training
  subjects in one room. ~4× the subjects and a 2nd/3rd/4th environment is the
  single change most likely to move MPJPE, and it's the honest fix for the
  overfitting.
- **Effort.** Low code (none — the split builders already partition whatever is
  on disk; `cross_environment(E04)` starts working the moment E02–E04 exist).
  The cost is **bandwidth/storage**: see the sizing below.
- **Risk.** None to the code. Just disk + download time.
- **Expected impact.** High — the biggest single lever; closes the gap between
  "pipeline demo on E01" and "faithful cross-subject / cross-environment numbers".

**Download sizing (the catch: per-env zips bundle ALL five modalities):**

| Env | Download (zip) | On disk after extracting `wifi-csi` + `ground_truth` |
|---|--:|--:|
| E02.zip | 19.5 GB | ~4.3 GB |
| E03.zip | 17.6 GB | ~4.3 GB |
| E04.zip | 19.5 GB | ~4.3 GB |
| **E02–E04** | **~56.6 GB** | **~12.9 GB** (all 4 envs ≈ 17 GB) |

~57 GB of bandwidth to keep ~13 GB — there's no way to pull a single modality out
of the zip server-side. Delete each zip right after extracting to cap peak disk
at ~24 GB. (Sizes from the documented "MMFi Dataset Split" per-env zips, used for
E01 and cross-checked against E01's actual 4.3 GB footprint.)

**Steps** (file IDs from `docs/chunk13_mmfi_setup.md` §6; needs `gdown`):

```bash
conda activate wifisense
# E02 (repeat for E03, E04 with their IDs)
gdown 1oIPGmsjDlzQsnTDVzIhYRQq-3BHTxQ8o -O data/raw/mmfi_E02.zip   # E02
gdown 1WjfPToIpi1a0cRYBvIr2yZoq_2jQpPQq -O data/raw/mmfi_E03.zip   # E03
gdown 1-XTwxO0ymJ1AtI5HsOOjD-XTrIHKPaA1 -O data/raw/mmfi_E04.zip   # E04

# Extract ONLY the two modalities we use (keep the zip OUT of data/raw/mmfi/),
# then VERIFY before deleting the zip (never rm after a bad extract).
for E in E02 E03 E04; do
  UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q -o data/raw/mmfi_${E}.zip \
      "${E}/*/*/wifi-csi/*" "${E}/*/*/ground_truth.npy" -d data/raw/mmfi/
  # 80190 = 10 subjects × 27 actions × 297 frames; 270 ground_truth.npy.
  [ "$(find data/raw/mmfi/${E} -path '*/wifi-csi/*.mat' | wc -l)" = "80190" ] \
    && [ "$(find data/raw/mmfi/${E} -name ground_truth.npy | wc -l)" = "270" ] \
    && rm -v data/raw/mmfi_${E}.zip \
    || echo "${E}: extraction INCOMPLETE — keeping zip; re-extract before deleting."
done
```

**Then re-run the pipeline (numbers update automatically):**

```bash
rm runs/best_pose.pt                       # let train_pose retrain on the full corpus
./run_pipeline.sh train_pose               # cross_subject over E01–E04
rm figures/pose_prediction.gif             # let the deliverable re-render
./run_pipeline.sh evaluate_pose pose_viz   # cross_environment row now populates
```

> After this lands, update the metrics tables in `docs/chunk16_pose_deliverable.md`
> and `docs/chunk15_pose_model.md` with the full-corpus numbers and drop the
> "E01 only" caveat.

---

### B. Temporal context — multi-frame input / sequence model  (needs sign-off)

- **What.** Feed a short window of CSI frames (`window_size > 1`) and/or add a
  temporal model (GRU/TCN over frames) so the network uses motion to place limbs.
- **Why (evidence).** Directly addresses fact #3 (the dynamic-pose failures). A
  single frame has no velocity information; the wrists are exactly where motion
  cues would help most (fact #2).
- **Effort.** Medium. `CSIPoseNet` already folds the window into channels, so
  `window_size > 1` works today; a true temporal encoder is more work.
- **Risk.** **Convention:** changing `window_size` breaks comparability with the
  MM-Fi per-frame benchmark — **ask the owner first**, and report it as a
  separate "temporal" variant / checkpoint rather than overwriting the benchmark
  number. Larger windows also raise compute and can leak across clip edges (the
  Dataset already clamps within-clip, so this is handled).
- **Expected impact.** High per-change — likely the largest accuracy gain after
  data, especially on dynamic actions.

---

### C. Output representation + loss — discipline the arms  (cheap, no convention change)

- **What.** (i) Per-joint loss **weighting** — upweight wrists/elbows so the
  optimizer stops over-spending on already-good torso joints. (ii) Left/right
  **symmetry** loss (cheap anatomical prior). (iii) Optionally regress **bone
  rotations** + forward-kinematics instead of raw coordinates, which structurally
  enforces bone lengths (the current bone term is only a soft penalty).
- **Why (evidence).** Directly addresses fact #2 — error is concentrated in the
  extremities, and the current loss treats all joints equally.
- **Effort.** Low for (i)/(ii) (a few lines in `pose_loss`, `src/train_pose.py`);
  higher for (iii) (a new output head + FK layer).
- **Risk.** Low. Self-contained; no `window_size`/benchmark change. Over-weighting
  wrists too hard can hurt torso joints — sweep the weight.
- **Expected impact.** Moderate — targeted at the worst joints; best paired with B.

---

### D. Regularization & augmentation — attack the overfitting directly

- **What.** CSI-side augmentation (subcarrier/antenna dropout, additive noise,
  time masking), weight decay, an LR schedule (cosine), and try
  `--csi-normalize zscore` (currently `none`).
- **Why (evidence).** Fact #1 (train ≪ val). Augmentation manufactures the data
  variety that downloading more environments (A) provides naturally.
- **Effort.** Low–medium (augmentation transforms in `pose_preprocess` + flags).
- **Risk.** Low. Too-aggressive augmentation can underfit — introduce gradually.
- **Expected impact.** Moderate; complements A (and matters more if A is deferred).

---

### E. Architecture & training hygiene — do last

- **What.** A slightly deeper/wider encoder; multi-seed ensembling; and a **proper
  held-out test set distinct from the early-stop val** (today cross_subject
  selects the checkpoint on its eval set, so 119 mm is mildly optimistic).
- **Why.** Diminishing returns until A–D are exhausted (the model isn't
  capacity-bound — fact #1). The clean-test fix is about honest *measurement*, not
  precision per se.
- **Effort.** Medium. **Convention:** architectural complexity (attention,
  bigger models) is ask-the-owner-first per the project brief.
- **Risk.** Low–medium. Easy to overfit a bigger model on E01; pairs best with A.
- **Expected impact.** Small–moderate; mostly a finishing pass.

---

## What does NOT improve precision (deliberately excluded)

- **Temporal smoothing of the predicted skeleton.** Makes the GIF look calmer but
  barely moves true MPJPE, and it would violate the project's honesty principle
  (raw model output; ask before smoothing). Keep it out of the metric path.

---

## Tracking

| Step | Status |
|---|---|
| A — download E02–E04 + retrain | **active** |
| B — temporal context | pending (needs `window_size` sign-off) |
| C — per-joint/symmetry loss | pending |
| D — augmentation + regularization | pending |
| E — architecture + clean test split | pending |
