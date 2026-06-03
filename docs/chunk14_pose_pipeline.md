# Chunk 14 — MM-Fi pose pipeline: CSI windows ↔ 3D-keypoint targets (Phase 3)

Chunk 13 stood up the MM-Fi loader and the 3D-skeleton renderer. This chunk
builds the **preprocessing + PyTorch Dataset** that pairs WiFi-CSI inputs with 3D
pose targets for training, plus a verification script that proves those pairs are
temporally aligned before a single epoch runs.

> **Still regression, not classification.** The target is continuous 3D joint
> coordinates `(17, 3)`; the metric is MPJPE (a distance in metres), not accuracy.
> See `docs/chunk13_mmfi_setup.md` for the dataset and `src/viz/skeleton.py` for
> the joint table / kinematic tree.

Deliverables:

- [`src/data/pose_preprocess.py`](../src/data/pose_preprocess.py) — pure
  CSI/keypoint transforms (windowing, root-centering, (de)normalization).
- [`src/data/mmfi_pose_dataset.py`](../src/data/mmfi_pose_dataset.py) — the
  `MMFiPoseDataset` and the three canonical split builders.
- [`scripts/verify_pose_pairs.py`](../scripts/verify_pose_pairs.py) — alignment
  checks + `figures/pose_pair_check.png`. Pipeline stage 18.

---

## 1. Windowing strategy (and why)

A pose is labeled **per frame**. We pair each labeled frame with a short,
**center-aligned** window of CSI frames:

```
CSI frames:   ... [ t-h ... t-1   t   t+1 ... t+h ] ...
                          \_______ ↑ _______/
target pose:                      pose at frame t   (the CENTER frame)
```

- **The target is the pose at the window's center frame** — never an end of the
  window. The network sees symmetric past/future context around the frame it must
  predict. The core index logic is `pose_preprocess.window_frame_indices`.
- **Window length is `window_size` (must be odd so a true center exists).** The
  project default is **`window_size = 1`** — one `(3, 114, 10)` frame → one
  `(17, 3)` pose. That is exactly MM-Fi's published WiFi→pose protocol
  (`data_unit: frame` in `vendor/MMFi/config.yaml`), so our numbers stay
  comparable to the paper. Larger odd windows (3, 5, …) are supported for
  experiments — **ask the project owner before changing the default**, since it
  breaks comparability with published results.
- A window stacks its frames on a new leading axis →
  **`(window_size, 3, 114, 10)`** (so `(1, 3, 114, 10)` at the default).
- **Edge clamping at clip boundaries.** Near the start/end of a clip the window
  would run past frame 0 or 296; those out-of-range positions are replaced by the
  nearest valid frame (edge replication), so the window is always full length and
  **never crosses into another `(subject, action)` clip**. Crossing clips would
  silently pair CSI with a different person's/action's pose — see §4.

### CSI value handling

MM-Fi WiFi is **amplitude only** (`CSIamp`) — there is no phase in this dataset.
The official reader already imputes NaN/inf and **min-max normalizes each frame to
`[0, 1]`**. So:

- `csi_amplitude` is a guarded pass-through (it would take `|z|` only if ever
  handed complex CSI from elsewhere).
- `normalize_csi` defaults to **`"none"`** to preserve the reader's `[0, 1]`
  scaling (benchmark-faithful). `"zscore"` / `"minmax"` are available
  (per-sample) if a model trains better with them.

---

## 2. Keypoint normalization math

### Coordinate conventions (get these wrong → scrambled skeletons)

MM-Fi keypoints are `(17, 3)` float32, **metres, camera frame**:

| axis | direction |
|------|-----------|
| x | right |
| y | **down** |
| z | forward (away from camera) |

Joint **0 is the pelvis (root)**, Human3.6M ordering. `pose_preprocess` only
**translates/scales** coordinates — it never reorders joints — so these axis
semantics are preserved. (The `(x, z, -y)` "body-upright" flip is **display-only**
and lives in `src/viz/skeleton.py`; do not bake it into targets.)

### Forward (absolute → training target)

`normalize_pose(kp, root=0, scale=None)`:

1. **Center on the root:** `offset = kp[root]`, `centered = kp − offset`. After
   this the pelvis sits at the origin, so the model predicts *posture* without
   having to also regress the person's absolute location in the room.
2. **Optional scale:**
   - `scale=None` (default, **recommended**): no scaling. Target stays in
     **metres**, so MPJPE is directly meaningful.
   - `scale="rms"`: divide by the pose's own RMS joint-to-root distance
     (`pose_scale`) → size/distance-invariant, unitless target.
   - `scale=<float>`: divide by a fixed constant (consistent unit across samples).

It returns `(norm_kp, info)` with `info = {"offset": (1,3), "scale": float,
"root": int}` — everything needed to invert it.

### Un-normalization recipe (for MPJPE + viz)

```
absolute = norm_kp * scale + offset          # pose_preprocess.denormalize_pose
```

- Apply it to predictions (and use absolute GT) **before** computing MPJPE or
  plotting with `src/viz/skeleton.py`. MPJPE is in metres — if you trained on a
  scaled target you **must** un-normalize first, or the error is in the wrong
  units.
- `MMFiPoseDataset.get_pair(i)` returns `keypoints_abs` (the original absolute
  pose) alongside `offset`/`scale`, and `MMFiPoseDataset.recover_absolute(...)`
  wraps `denormalize_pose` for convenience.
- With the defaults (`root=0, scale=None`), un-normalization is just
  `pred + offset` and MPJPE on the centered poses already equals the metric MPJPE
  (translation cancels in per-joint distances).

---

## 3. Split semantics

`mmfi_pose_dataset` exposes three builders, each returning `(train_ds, test_ds)`
that share one underlying lazy `MMFiSubset` and the same Dataset kwargs
(`window_size`, `csi_normalize`, `pose_root`, `pose_scale`, `cache`):

| builder | held out | what it measures |
|---------|----------|------------------|
| `cross_subject(test_subjects=…)` | whole subjects | **the headline test** — pose a body never seen in training |
| `cross_environment(test_envs=…)` | whole environments (E01..E04) | robustness to a new room/multipath layout |
| `random_split(test_ratio=…, seed=…)` | random clips | i.i.d. baseline (no domain shift) |

- **Defaults match MM-Fi protocols** for comparability:
  `cross_subject` → `DEFAULT_TEST_SUBJECTS = [S05,S10,…,S40]` (every 5th subject,
  the paper's cross-subject val set); `cross_environment` → `DEFAULT_TEST_ENVS =
  [E04]` (the cross-scene protocol). Override either to hold out a custom set.
- **All splitting is at the CLIP level** (`(scene, subject, action)`), never
  mid-clip. This is essential: adjacent frames — and the frames inside one window
  — must not straddle the train/test boundary, or the score is inflated by
  leakage. (Note MM-Fi's own `random_split` actually splits by *subject*; our
  `random_split` splits by clip to stay an honest i.i.d. baseline while still
  preventing window leakage.)
- **Partial downloads:** every builder loads all on-disk frames and partitions
  them itself, so `cross_subject` works on **E01 alone** (its default held-out
  S05/S10 live in E01). `cross_environment(test_envs=['E04'])` needs E04 on disk
  and otherwise raises a clear "empty TEST partition" error.

```python
from src.data.mmfi_pose_dataset import cross_subject
train_ds, test_ds = cross_subject(window_size=1)      # benchmark-exact
csi, kp = train_ds[0]    # csi (1,3,114,10) float32, kp (17,3) float32, root at origin
```

---

## 4. Temporal-alignment gotchas

Misalignment — pairing a CSI window with the wrong frame's pose — produces a
model that *trains* (loss goes down) but predicts garbage, with no error to
explain it. The traps, and how this code avoids each:

1. **Window crossing a clip boundary.** `window_frame_indices` is given the frame
   count of the *single clip* the center belongs to and clamps to it, so a window
   near a clip edge replicates that clip's first/last frame instead of bleeding
   into the neighbouring clip. Clips are grouped by `(scene, subject, action)`.
2. **Frames not in temporal order.** The loader yields frames per clip, but we
   **sort each clip by its frame `idx`** in `_group_clips`, so list position ==
   time. If you ever swap loaders, re-check this sort.
3. **Targeting the wrong frame of the window.** The target is always
   `members[center]`, the middle of the window (`window_size // 2`), not its
   first/last frame. `verify_pose_pairs.py` asserts
   `window_frame_idx[mid] == center_idx` for every sampled pair.
4. **Off-by-one between CSI and pose indexing.** The frame-mode loader's `idx`
   field is the same index used to slice `ground_truth.npy` (`(297,17,3)`) inside
   the official reader, so CSI frame `idx` and pose `idx` refer to the same
   instant. `verify_pose_pairs.py` confirms this **independently** by re-reading
   the clip in `data_unit='sequence'` mode and checking that the sequence's
   `[center_idx]` pose **and** CSI frame equal what the frame-mode dataset
   produced.

### The verification script

`python scripts/verify_pose_pairs.py [--split cross_subject] [--n 5]
[--window-size 1]` runs both checks on a handful of random pairs and writes
**`figures/pose_pair_check.png`** — each CSI amplitude window (antenna-averaged
subcarrier × packet heatmap) beside the GT skeleton it is paired with. It exits
non-zero if any pair fails, so it can gate training. **Run it (and look at the
figure) before training** — alignment bugs are far cheaper to catch here than
after a model has "converged" to nonsense.

---

## 5. Pipeline

Added as stage **18** (`verify_pose_pairs`) in `run_pipeline.sh`; like the other
MM-Fi stage it self-skips when the dataset is absent so the default reproduction
run doesn't break for users who haven't downloaded it.
