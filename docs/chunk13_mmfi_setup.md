# Chunk 13 — MM-Fi setup: WiFi-CSI → 3D human-pose estimation (Phase 3)

This chunk opens **Phase 3** of the project and introduces a **new task type**.

> **Regression, not classification.** Chunks 1–12 were *classification*: predict
> a discrete activity/gesture label from CSI (UT-HAR, NTU-Fi, Widar3.0 BVP). The
> body never appeared — only class indices. Pose estimation is *supervised
> regression*: from WiFi CSI, predict **continuous 3D joint coordinates**
> (17 joints × (x, y, z) metres). There is no class head and no cross-entropy;
> the loss/metric is a **distance** between predicted and ground-truth joints
> (MPJPE — see `vendor/MMFi/mmfi_lib/evaluate.py`). This chunk only stands up the
> data + visualization; training the regressor comes later.

This chunk delivers:

- [`src/data/mmfi_loader.py`](../src/data/mmfi_loader.py) — a thin wrapper over the
  official MM-Fi toolbox exposing `load_mmfi(...)` with paired CSI + 3D pose.
- [`src/viz/skeleton.py`](../src/viz/skeleton.py) — `plot_skeleton_3d` /
  `plot_skeleton_pair`, the new 3D-skeleton rendering primitive.
- [`scripts/explore_mmfi.py`](../scripts/explore_mmfi.py) — census + the first
  real-pose figure, `figures/mmfi_gt_skeletons.png`.
- `run_pipeline.sh` stage `explore_mmfi` (stage 17).

---

## 1. Dataset overview

**MM-Fi** (Yang et al., *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for
Versatile Wireless Sensing*, **NeurIPS 2023 Datasets & Benchmarks**) is the first
multi-modal, non-intrusive 4D human-pose dataset built for wireless sensing.

| Property      | Value |
|---------------|-------|
| Frames        | ~320k synchronized |
| Subjects      | 40 (grouped into 4 environments) |
| Environments  | E01–E04 (each = 10 subjects) |
| Actions       | 27 (`A01`–`A27`), daily + rehabilitation |
| Frames/clip   | 297 per (subject, action) |
| Modalities    | RGB, infra1/2, depth, LiDAR, mmWave, **WiFi-CSI** |
| Annotations   | 2D & **3D pose (17 keypoints)**, 3D position, dense pose, action |

- Project page: <https://ntu-aiot-lab.github.io/mm-fi>
- Paper: <https://arxiv.org/abs/2305.10345>
- Toolbox (vendored at `vendor/MMFi/`): <https://github.com/ybhbingo/MMFi_dataset>

**We use exactly two modalities**: `wifi-csi` as input, 3D pose as target. RGB /
depth / LiDAR / mmWave are ignored except for occasional sanity checks. (The
official loader imports OpenCV for the image modalities; since we never touch
them, `mmfi_loader` stubs `cv2` so you don't need OpenCV installed — only
`pyyaml`, `scipy`, `numpy`, `torch`, `matplotlib`.)

### Subject ↔ environment map (fixed)

| Env | Subjects |
|-----|----------|
| E01 | S01–S10  |
| E02 | S11–S20  |
| E03 | S21–S30  |
| E04 | S31–S40  |

---

## 2. License & citation

MM-Fi is released for **academic, non-commercial research** (see the dataset's
terms on the project page / Google Drive; RGB images are anonymized and
distributed separately). Cite:

```bibtex
@inproceedings{yang2023mm,
  title={MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing},
  author={Yang, Jianfei and Huang, He and Zhou, Yunjiao and Chen, Xinyan and Xu, Yuecong and Yuan, Shenghai and Zou, Han and Lu, Chris Xiaoxuan and Xie, Lihua},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023},
  url={https://openreview.net/forum?id=1uAsASS1th}
}
```

The vendored toolbox keeps its upstream license at `vendor/MMFi/` (cloned from
the official repo).

---

## 3. The subset we use

### WiFi CSI (input)

Per **frame**, the on-disk `frameNNN.mat` holds `CSIamp` of shape **`(3, 114, 10)`**:

- **3** receiver antennas,
- **114** subcarriers (5 GHz, 40 MHz bandwidth),
- **10** packets within the ~100 ms frame window.

It is *amplitude only*. The official reader imputes NaN/inf and min-max
normalizes each frame to `[0, 1]` — our loader returns it as-is from that reader.

Per **sequence** (a whole clip): `(297, 3, 114, 10)`.

### 3D pose (target)

`ground_truth.npy` per (subject, action) is `(297, 17, 3)` — 297 frames, **17
joints**, `(x, y, z)` in **metres** in the camera coordinate frame. Per frame the
target is `(17, 3)`. The keypoints were extracted from the camera modalities (a
ResNet-based 2D estimator lifted to 3D) and quality-checked via PCKh@0.5 against
manual annotation; occlusion-heavy actions can have noisier GT.

> 17 joints follow the **Human3.6M** ordering (MM-Fi paper §3). The exact index
> table and kinematic tree are in §5.

---

## 4. Loader API (`src/data/mmfi_loader.py`)

```python
from src.data.mmfi_loader import load_mmfi

subset = load_mmfi(
    modality="wifi-csi",        # the only modality this project uses
    split="all",                # "train" | "val" | "all"
    protocol="protocol3",       # protocol1=daily, 2=rehab, 3=all 27 actions
    split_strategy="random_split",  # | "cross_scene_split" | "cross_subject_split" | "manual_split"
    data_unit="frame",          # "frame" -> per-frame pairs; "sequence" -> 297-frame clips
    data_root=None,             # defaults to data/raw/mmfi/
)

len(subset)                     # number of samples
sample = subset[0]              # arrays loaded on demand
sample["csi"]                   # np.ndarray (3, 114, 10)   [frame unit]
sample["keypoints"]             # np.ndarray (17, 3) float32 [frame unit]
sample["subject"], sample["scene"], sample["action"], sample["idx"]

subset.metadata                 # list[dict] census, NO array loads
```

Design notes:

- **Reuses the official loader** (`vendor/MMFi/mmfi_lib/mmfi.py`) for the dir
  walk, file decoding, NaN/inf cleanup, protocol/split logic, and train/val
  partitioning. We only normalize key names (`input_wifi-csi`→`csi`,
  `output`→`keypoints`) and make it robust to partial downloads.
- **Partial-download safe.** `decode_config` always spans all 40 subjects × the
  protocol's actions; the official frame-mode loader then `stat`s every file and
  crashes on the first missing one. Our wrapper prunes the train/val "data_form"
  to the subjects *and* actions actually present on disk, so having **only E01**
  extracted works fine. (Caveat: with only E01 present, `cross_scene_split`
  trains on E01–E03 and validates on E04 — so `split="val"` may be empty.
  Use `split="all"` or `random_split` when developing on a single environment.)
- **Lazy.** Nothing is materialized up front; safe on the full ~320k-frame corpus.
- **Fails loudly** with a pointer to this doc if `data/raw/mmfi/` is missing.

The `MPJPE` / `PA-MPJPE` evaluation metric ports live in
`vendor/MMFi/mmfi_lib/evaluate.py` (`calulate_error`) for when we train.

---

## 5. Kinematic tree (Human3.6M 17-joint skeleton)

Defined once in [`src/viz/skeleton.py`](../src/viz/skeleton.py) as `JOINT_NAMES`
and `SKELETON_EDGES`, and reused everywhere.

### Joint-index table

| idx | joint          | idx | joint            |
|----:|----------------|----:|------------------|
| 0   | pelvis (root)  | 9   | neck / nose      |
| 1   | r_hip          | 10  | head (top)       |
| 2   | r_knee         | 11  | l_shoulder       |
| 3   | r_ankle        | 12  | l_elbow          |
| 4   | l_hip          | 13  | l_wrist          |
| 5   | l_knee         | 14  | r_shoulder       |
| 6   | l_ankle        | 15  | r_elbow          |
| 7   | spine (mid)    | 16  | r_wrist          |
| 8   | thorax (neck base) | | |

### Bones (parent → child), grouped by limb

```
right leg : 0-1, 1-2, 2-3
left  leg : 0-4, 4-5, 5-6
spine     : 0-7, 7-8, 8-9, 9-10
left  arm : 8-11, 11-12, 12-13
right arm : 8-14, 14-15, 15-16          (16 bones total)
```

```
                   10  head
                    |
                    9  neck/nose
                    |
   13-12-11------ 8(thorax) ------14-15-16
                    |
                    7  spine
                    |
                    0  pelvis
                   / \
                  4   1        (L / R hip)
                  |   |
                  5   2        (L / R knee)
                  |   |
                  6   3        (L / R ankle)
```

**Display frame.** MM-Fi coords are camera-frame metres (y points *down*).
`plot_skeleton_3d` remaps to `(x, z, -y)` so "up" on the page is up on the body;
pass `raw_axes=True` to plot the unmodified `(x, y, z)`. Left-side limbs are
tinted lighter and right-side darker for orientation. `plot_skeleton_pair(gt,
pred)` overlays two skeletons in one set of axes and titles them with the MPJPE
in millimetres — the primitive for later GT-vs-prediction comparison.

---

## 6. Download & subset instructions

> The dataset is **large** and is **NOT** downloaded by `run_pipeline.sh`.
> Storage/bandwidth are real — **start with one environment (E01)** and ask the
> project owner before pulling all four.

### Step 1 — toolbox (already vendored)

```bash
git clone https://github.com/ybhbingo/MMFi_dataset.git vendor/MMFi
```

(Already present in this repo at `vendor/MMFi/`.)

### Step 2 — download the data

The dataset lives on the official **Google Drive** (or Baidu Netdisk), linked
from the MM-Fi README / project page:

- Top folder: <https://drive.google.com/drive/folders/1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz>
  — contains `MMFi_Dataset.zip` (the full 77 GB archive) **and** a subfolder
  **"MMFi Dataset Split"** with handy **per-environment** zips:

  | File | Size (≈) | Google Drive file ID |
  |------|---------:|----------------------|
  | E01.zip | 20.8 GB | `1ExV3AQeHstQ3Z1VBFOC0z1BDtTD5nZ1B` |
  | E02.zip | 19.5 GB | `1oIPGmsjDlzQsnTDVzIhYRQq-3BHTxQ8o` |
  | E03.zip | 17.6 GB | `1WjfPToIpi1a0cRYBvIr2yZoq_2jQpPQq` |
  | E04.zip | 19.5 GB | `1-XTwxO0ymJ1AtI5HsOOjD-XTrIHKPaA1` |

Each `Exx.zip` is **one environment** (10 subjects × 27 actions, **all** five
modalities). `E01.zip` alone is enough to develop the whole pose pipeline —
**ask the owner before pulling more than one environment** (storage/bandwidth).

**Download just E01** (needs `gdown`; `pip install gdown` in the env):

```bash
conda activate wifisense
gdown 1ExV3AQeHstQ3Z1VBFOC0z1BDtTD5nZ1B -O data/raw/mmfi_E01.zip
```

> Note the destination is `data/raw/mmfi_E01.zip`, **not** inside
> `data/raw/mmfi/`. The official loader treats every entry in the dataset root as
> an environment folder, so a stray `.zip` sitting in `data/raw/mmfi/` makes it
> crash with `NotADirectoryError`. Keep the zip out of the dataset root.

**Extract only the modalities we use** (`wifi-csi` + `ground_truth.npy`,
≈4.3 GB) instead of the full ≈25.6 GB — the other modalities are not needed:

```bash
# UNZIP_DISABLE_ZIPBOMB_DETECTION bypasses a FALSE-POSITIVE "zip bomb" guard in
# Info-ZIP's unzip that misfires on this large many-file archive. The archive is
# fine (verify with `zipinfo data/raw/mmfi_E01.zip | tail -1`).
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q -o data/raw/mmfi_E01.zip \
    "E01/*/*/wifi-csi/*" "E01/*/*/ground_truth.npy" -d data/raw/mmfi/
```

To instead keep **all** modalities (for RGB/depth sanity checks), drop the
include-patterns: `... unzip -q -o data/raw/mmfi_E01.zip -d data/raw/mmfi/`.

After extraction the tree under `data/raw/mmfi/` is:

```
data/raw/mmfi/                 # dataset root — ONLY Exx folders here
|-- E01/
|   |-- S01/
|   |   |-- A01/
|   |   |   |-- wifi-csi/   frame001.mat ... frame297.mat
|   |   |   |-- ground_truth.npy
|   |   |   |-- (rgb/ depth/ mmwave/ lidar/ — only if you kept them)
|   |   |-- A02/ ... A27/
|   |-- S02/ ... S10/
|-- E02/ ... E04/           # optional — add later, ask first
```

Sanity check: `find data/raw/mmfi/E01 -name '*.mat' | wc -l` → **80190**
(10 × 27 × 297) and `... -name ground_truth.npy | wc -l` → **270** (10 × 27).

`data/raw/` is git-ignored; the data lives only on your machine. Once extracted
you may delete `data/raw/mmfi_E01.zip` to reclaim ~21 GB (re-download to recover
the other modalities), or keep it to extract them locally later.

> You can keep the data anywhere and point the loader/script at it with
> `load_mmfi(..., data_root=...)` or `python scripts/explore_mmfi.py
> --data-root /path/to/MMFi_Dataset`.

### Step 3 — explore + render ground-truth poses

```bash
conda activate wifisense
python scripts/explore_mmfi.py            # or: ./run_pipeline.sh explore_mmfi
```

Prints per-env/subject/action counts and CSI/keypoint shapes & ranges, then
writes **`figures/mmfi_gt_skeletons.png`** — a grid of 8 ground-truth skeletons
across actions. This is the first time the project visualizes real human poses;
it's what the WiFi-CSI regressor will later try to predict.
