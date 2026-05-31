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
│       └── ntu_fi_har/          # Secondary dataset (generalization check)
│           └── NTU-Fi_HAR/
│               ├── train_amp/   # .mat files, 6 activity subdirs
│               └── test_amp/
├── src/
│   ├── data/                    # loader.py, preprocess.py
│   ├── models/                  # bilstm.py + build_model registry
│   ├── viz/                     # csi_plots.py
│   ├── train.py                 # training loop (run via `python -m src.train`)
│   └── evaluate.py              # checkpoint evaluation (`python -m src.evaluate`)
├── scripts/
│   ├── explore_data.py          # dataset characterization + sample dumps
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

## Datasets

| Dataset | Source | CSI Shape | Classes | Train | Test |
|---------|--------|-----------|---------|-------|------|
| UT-HAR | Intel 5300 | 1 x 250 x 90 | 7 (lie down, fall, walk, pickup, run, sit down, stand up) | 3977 | 500 (+496 val) |
| NTU-Fi HAR | NTU-Fi | 3 x 114 x 500 | 6 (box, circle, clean, fall, run, walk) | 936 | 264 |

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

> `gdown --folder` pulls everything in the drive (including the large Widar set
> you don't need here). To grab only the two folders, open the Drive link in a
> browser and download `UT_HAR` and `NTU-Fi_HAR` individually instead.

After copying, the layout must match the [Directory Layout](#directory-layout)
above. Verify with:

```bash
conda activate wifisense
python verify_setup.py
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
diagnose → postprocess → preprocess_ntu → train_ntu → domainshift). It activates
the `wifisense` env itself. The tail (chunk 9) characterizes the UT-HAR ↔ NTU-Fi
domain gap — see [`docs/chunk9_domain_shift.md`](docs/chunk9_domain_shift.md).

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
