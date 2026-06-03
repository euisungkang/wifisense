# WiFi CSI Human Activity Detection

Detect human activities from WiFi Channel State Information (CSI) using deep
learning. End deliverable: a 3-panel visualization for continuous captures
(CSI spectrogram, predicted-activity probabilities, ground-truth labels).

## Directory Layout

```
wifi/
├── data/
│   └── raw/
│       ├── ut_har/              # Primary dataset
│       │   └── UT_HAR/
│       │       ├── data/        # X_train/val/test.csv (numpy binary, not text CSV)
│       │       └── label/       # y_train/val/test.csv (numpy binary)
│       ├── ntu_fi_har/          # Secondary dataset (generalization check)
│       │   └── NTU-Fi_HAR/
│       │       ├── train_amp/   # .mat files, 6 activity subdirs
│       │       └── test_amp/
│       └── widar3/              # Widar3.0 BVP (chunk 10, cross-domain gestures)
│           └── bvp/BVP/         # <date>-VS/[6-link/]<userN>/*.mat (20x20xT)
├── src/
│   ├── data/                    # loader.py, preprocess.py, widar_loader.py,
│   │                            #   bvp_preprocess.py, widar_dataset.py
│   ├── models/                  # bilstm.py, bvp_cnn_rnn.py + build_model registry
│   ├── viz/                     # csi_plots.py
│   ├── train.py                 # UT-HAR/NTU training loop (`python -m src.train`)
│   ├── train_widar.py           # Widar3.0 BVP per-split trainer (chunk 12)
│   ├── evaluate.py              # checkpoint evaluation (`python -m src.evaluate`)
│   └── evaluate_widar.py        # 4-split BVP eval + domain-results grid (chunk 12)
├── scripts/
│   ├── explore_data.py          # dataset characterization + sample dumps
│   ├── explore_widar.py         # Widar3.0 BVP characterization (chunk 10)
│   ├── visualize_widar_bvp.py   # BVP example grid -> figures/widar_bvp_examples.png
│   ├── bvp_pipeline_demo.py     # raw->normalized->padded + motion trajectory (chunk 11)
│   ├── spatial_viz.py           # spatial-motion milestone figure (chunk 12)
│   ├── preprocess_data.py       # raw -> data/processed/ut_har/ut_har.npz
│   ├── visualize_classes.py     # per-class CSI sanity grids
│   ├── sweep.py                 # multi-seed robustness sweep over one config
│   ├── build_continuous_capture.py  # stitch test clips into one stream
│   ├── final_visualization.py   # 3-panel milestone figure
│   └── diagnose_accuracy.py     # decompose the continuous-capture accuracy gap
├── runs/                        # per-run outputs; best_bilstm.pt = stable checkpoint
├── figures/                     # confusion matrix, prediction CSVs, plots
├── docs/                        # per-chunk write-ups (see docs/pipeline.md)
├── vendor/
│   └── SenseFi/                 # Reference library (cloned from GitHub)
├── run_pipeline.sh              # Ordered runner for the whole chain (see docs/pipeline.md)
├── verify_setup.py              # Environment & data sanity check
└── README.md
```

## Project Documentation

The project is built in numbered **chunks**; each leaves a focused write-up under
[`docs/`](docs/). A reader landing here can follow the whole arc — raw-CSI
activity recognition, the domain-shift problem, the move to Widar3.0 BVP and
environment-invariant gesture recognition, and then the shift from
classification to **3D human-pose estimation (regression)** with MM-Fi —
through these:

| Chunk | Doc | What it covers |
|---|---|---|
| 1–6 | [`docs/pipeline.md`](docs/pipeline.md), [`docs/preprocessing.md`](docs/preprocessing.md), [`docs/visualization.md`](docs/visualization.md), [`docs/streaming_inference.md`](docs/streaming_inference.md) | UT-HAR data → preprocessing → BiLSTM training → the 3-panel continuous-capture figure |
| 7 | [`docs/diagnostics.md`](docs/diagnostics.md) | Diagnosing the continuous-capture accuracy gap (model vs. boundary vs. edge effects) |
| 8 | [`docs/chunk8_postprocessing.md`](docs/chunk8_postprocessing.md) | Temporal post-processing (moving average / majority vote / HMM) over the prediction stream |
| 9 | [`docs/chunk9_domain_shift.md`](docs/chunk9_domain_shift.md) | UT-HAR ↔ NTU-Fi domain shift: raw-CSI models collapse across domains |
| 10 | [`docs/chunk10_widar_bvp.md`](docs/chunk10_widar_bvp.md) | Widar3.0 BVP: the dataset, loader, and label conventions |
| 11 | [`docs/chunk11_bvp_pipeline.md`](docs/chunk11_bvp_pipeline.md) | BVP preprocessing transforms, the `Dataset`, the four cross-domain splits, optional CSI→BVP derivation |
| 12 | [`docs/chunk12_spatial_visualization.md`](docs/chunk12_spatial_visualization.md) | BVP CNN-RNN gesture model, cross-domain evaluation vs. chunk 9, and the spatial-motion figure |
| 13 | [`docs/chunk13_mmfi_setup.md`](docs/chunk13_mmfi_setup.md) | **Phase 3 — pose estimation (regression):** MM-Fi WiFi-CSI → 3D-pose loader, the 3D-skeleton primitive, and the first ground-truth human-pose figure |

Supporting references: [`docs/datasets.md`](docs/datasets.md) /
[`docs/data_summary.md`](docs/data_summary.md) (dataset details),
[`docs/class_visual_separability.md`](docs/class_visual_separability.md), and the
deeper notes under [`notes/`](notes/) (`widar_data.md`, `domain_shift.md`,
`class_mapping.md`, …).

## Datasets

| Dataset | Source | CSI Shape | Classes | Train | Test |
|---------|--------|-----------|---------|-------|------|
| UT-HAR | Intel 5300 | 1 x 250 x 90 | 7 (lie down, fall, walk, pickup, run, sit down, stand up) | 3977 | 500 (+496 val) |
| NTU-Fi HAR | NTU-Fi | 3 x 114 x 500 | 6 (box, circle, clean, fall, run, walk) | 936 | 264 |
| Widar3.0 BVP | Tsinghua (Intel 5300 ×6 Rx) | T x 20 x 20 (BVP, T varies) | 22 gestures (Push&Pull, Sweep, Clap, Slide, Draw-{O,N,Zigzag,…}, Draw-0…9) | 43,658 instances (17 users · 8 positions · 5 orientations · 3 rooms) | — |

Widar3.0 (chunk 10) is a different kind of input: not raw CSI but **BVP
(Body-coordinate Velocity Profile)** — an environment-invariant 2-D velocity
representation. See [`docs/chunk10_widar_bvp.md`](docs/chunk10_widar_bvp.md) and
[`notes/widar_data.md`](notes/widar_data.md). The model-ready preprocessing
pipeline, cross-domain splits, and the optional CSI→BVP re-derivation are
documented in [`docs/chunk11_bvp_pipeline.md`](docs/chunk11_bvp_pipeline.md).

### Downloading the raw data

The raw datasets are **not tracked in git** (multi-GB). Both come from the
[SenseFi / WiFi-CSI-Sensing-Benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)
project, which hosts the processed CSI data on
[Google Drive](https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt?usp=sharing).
This project only uses the `UT_HAR` and `NTU-Fi_HAR` subfolders from that drive.

The easiest way to pull them from the command line is [`gdown`](https://github.com/wkentaro/gdown):

```bash
pip install gdown

# 1. Download the whole SenseFi "Data" folder into a temp dir
gdown --folder https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt -O /tmp/sensefi_data

# 2. Place the two datasets this project uses into data/raw/
mkdir -p data/raw/ut_har data/raw/ntu_fi_har
cp -r /tmp/sensefi_data/Data/UT_HAR      data/raw/ut_har/UT_HAR
cp -r /tmp/sensefi_data/Data/NTU-Fi_HAR  data/raw/ntu_fi_har/NTU-Fi_HAR

# 3. (optional) drop the unused datasets (Widar, NTU-Fi-HumanID) to save space
rm -rf /tmp/sensefi_data
```

> `gdown --folder` pulls everything in the drive (including the SenseFi
> Widar variant, which is a *different*, fixed-size repackaging we don't use —
> see below). To grab only the two folders, open the Drive link in a browser and
> download `UT_HAR` and `NTU-Fi_HAR` individually instead.

After copying, the layout must match the [Directory Layout](#directory-layout)
above. Verify with:

```bash
conda activate wifisense
python verify_setup.py
```

### Downloading Widar3.0 BVP (chunk 10)

The Widar3.0 **BVP** portion (~400 MB) comes straight from the official Tsinghua
Cloud share, *not* from the SenseFi drive (SenseFi ships a fixed 22×20×20
repackaging; we want the genuine variable-T BVP). It's gitignored like the rest.

```bash
mkdir -p data/raw/widar3/bvp
BASE="https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/files/?p="

# BVP volumes (~400 MB, 43.7k .mat files) + the dataset spec + extraction code
curl -L "${BASE}/BVP/BVP.zip&dl=1"            -o data/raw/widar3/BVP.zip
curl -L "${BASE}/README.pdf&dl=1"             -o data/raw/widar3/README.pdf
curl -L "${BASE}/BVPExtractionCode.zip&dl=1"  -o data/raw/widar3/BVPExtractionCode.zip

unzip -q data/raw/widar3/BVP.zip -d data/raw/widar3/bvp   # -> data/raw/widar3/bvp/BVP/...
```

Then explore it:

```bash
python scripts/explore_widar.py            # counts, T/value stats, label conventions
python scripts/visualize_widar_bvp.py      # figures/widar_bvp_examples.png
# or: ./run_pipeline.sh widar
```

The model-ready pipeline (chunk 11) lives in `src/data/bvp_preprocess.py`
(`normalize_bvp` / `pad_or_truncate` / `augment_bvp`) and
`src/data/widar_dataset.py` (`WidarBVPDataset` plus the four canonical
cross-domain splits — `cross_user`, `cross_position`, `cross_orientation`,
`in_domain` — each returning `(train_ds, test_ds)` with augmentation forced off
on test):

```python
from src.data import cross_user, make_dataloader
train_ds, test_ds = cross_user(test_users=[3], gesture="Push&Pull", augment=True)
loader = make_dataloader(train_ds, batch_size=32)  # yields (B, T, 20, 20), labels
```

```bash
python scripts/bvp_pipeline_demo.py        # figures/bvp_pipeline_demo.png
# or: ./run_pipeline.sh bvp
```

Optionally, re-derive a BVP from raw CSI to verify the physics (chunk 11,
educational — needs the toolkit's sample data shipped in `BVPExtractionCode`):

```bash
python -m src.data.csi_to_bvp             # figures/csi_to_bvp_check.png + metrics
```

## Environment

- **Conda env:** `wifisense` (Python 3.10)
- **ML:** PyTorch (CPU-only for now), torchvision
- **Data/viz:** numpy, scipy, scikit-learn, matplotlib, seaborn, pandas
- **Utilities:** pywavelets, einops, tqdm

```bash
conda activate wifisense
python verify_setup.py
```

## Reproducing the pipeline

`run_pipeline.sh` is an ordered runner for the whole chain (verify → explore →
preprocess → visualize → train → sweep → evaluate → capture → finalviz →
diagnose → postprocess → preprocess_ntu → train_ntu → domainshift → widar → bvp →
train_widar → evaluate_widar → spatial). It activates the `wifisense` env itself.
The middle (chunk 9) characterizes the UT-HAR ↔ NTU-Fi domain gap
([`docs/chunk9_domain_shift.md`](docs/chunk9_domain_shift.md)); the Widar3.0 tail
(chunks 10–12) moves to environment-invariant BVP gesture recognition
([`docs/chunk12_spatial_visualization.md`](docs/chunk12_spatial_visualization.md)).
Heavy trainers (`train`, `sweep`, `train_ntu`, `train_widar`) are excluded from
the no-arg default; `evaluate_widar` and `spatial` self-skip when their outputs
exist or no BVP checkpoint has been trained yet, so reruns stay cheap.

```bash
./run_pipeline.sh                 # safe reproduction (uses the frozen checkpoint; skips train/sweep)
./run_pipeline.sh all             # every stage, including the heavy train + sweep
./run_pipeline.sh preprocess evaluate diagnose   # only these, in this order
./run_pipeline.sh list            # show the stage list
```

Per-stage variables are documented inline in each `stage_<name>()` function.
Full reference: [`docs/pipeline.md`](docs/pipeline.md).

## Stack

- **Model:** BiLSTM (starting architecture, from SenseFi)
- **Reference code:** `vendor/SenseFi/` — used for model definitions and data
  loading patterns, not installed as a package
- **Framework:** PyTorch

## Training & Evaluation

All commands run from the repo root with the env active (`conda activate
wifisense`). The three splits have three distinct jobs: **train** fits weights,
**val** drives every decision (hyperparameters, early stopping, model
selection), and **test** is touched exactly once at the very end for an
unbiased number. Tuning against test silently leaks and inflates results.

The workflow runs in three phases:

**1. Hyperparameter search** — fix a seed, vary the config, compare val acc:

```bash
python -m src.train --seed 42 --lr 1e-3 --hidden 64 --dropout 0.3
python -m src.train --seed 42 --lr 5e-4 --hidden 128 --dropout 0.5
# compare best_val_acc across runs/<timestamp>/metrics.json
```

Each run writes `runs/<timestamp>/`: `best.pt` (best-val checkpoint),
`metrics.json` (per-epoch history), `training_curves.png`.

**2. Robustness sweep** — take the winning config, run it across several seeds
to confirm the result is stable (not a lucky init). `sweep.py` shells out to
`src.train` once per seed and reports val accuracy as `mean ± std`:

```bash
python scripts/sweep.py --seeds 42 43 44 45 46 \
    --lr 1e-3 --hidden 64 --layers 2 --dropout 0.3 --epochs 80
```

Runs land in `runs/sweep_<timestamp>/seed_<s>/`; a `summary.json` records the
aggregate and names the best-by-val run. Tight spread (std < ~1–2%) → proceed.

**3. Freeze & evaluate once** — promote the best-by-val checkpoint to the
stable path chunk 6 expects, then evaluate on test a single time:

```bash
cp runs/sweep_<timestamp>/seed_<best>/best.pt runs/best_bilstm.pt   # or: sweep.py --promote
python -m src.evaluate --split test                                 # defaults to runs/best_bilstm.pt
```

Evaluation prints overall accuracy, macro F1 and per-class precision/recall/F1,
and writes `figures/confusion_matrix.png`, `figures/predictions_test.csv`, and
`figures/eval_metrics_test.json`.

**What "good enough" looks like** before moving on: val acc ≈ 93–95%+ (SenseFi
reports ~95% for this model), low seed variance, test acc in the same ballpark
as val, and a confusion matrix whose errors fall between *similar* activities
rather than a whole class collapsing to zero recall.
