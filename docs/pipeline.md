# Pipeline Orchestrator (`run_pipeline.sh`)

`run_pipeline.sh` at the repo root is a single ordered runner for every chunk
built so far. It exists so the whole chain is reproducible from one command
without remembering each script's invocation. It is a personal convenience
script: not every argument is exposed as a flag — each stage documents its
tunable variables inline, and you expose more per-stage as a chunk needs them.

## Usage

```bash
./run_pipeline.sh                 # run DEFAULT_STAGES (safe reproduction)
./run_pipeline.sh all             # run every stage in order (incl. heavy ones)
./run_pipeline.sh preprocess evaluate diagnose   # run only these, in this order
./run_pipeline.sh list            # print the stage list and exit
```

The script activates the `wifisense` conda env automatically (override with
`CONDA_ENV=other ./run_pipeline.sh`) and runs from the repo root regardless of
where it's invoked.

## Stages (dependency order)

| Stage | Chunk | Runs | Key outputs |
|---|---|---|---|
| `verify` | setup | `verify_setup.py` | env / data sanity |
| `explore` | 1 | `scripts/explore_data.py` | `data/samples/`, dataset stats |
| `preprocess` | 2 | `scripts/preprocess_data.py` | `data/processed/ut_har/ut_har.npz` |
| `visualize` | 3 | `scripts/visualize_classes.py` | `figures/class_grid.png`, `doppler_grid.png` |
| `train` | 4 | `python -m src.train` | `runs/<ts>/best.pt` (**heavy**) |
| `sweep` | 4 | `scripts/sweep.py` | multi-seed runs; promotes `runs/best_bilstm.pt` (**heavy**) |
| `evaluate` | 5 | `python -m src.evaluate` | confusion matrix, predictions CSV, metrics JSON |
| `capture` | 6 | `scripts/build_continuous_capture.py` | `data/continuous/synthetic_capture.npz` |
| `finalviz` | 6 | `scripts/final_visualization.py` | `figures/final_visualization.png` |
| `diagnose` | 7 | `scripts/diagnose_accuracy.py` | diagnostics figures + summary (see `diagnostics.md`) |
| `postprocess` | 8 | `scripts/compare_postprocessing.py` | `notes/postprocessing.md`, smoothed figure |
| `preprocess_ntu` | 9 | `scripts/preprocess_ntu_fi.py` | `data/processed/ntu_fi/ntu_fi.npz` |
| `train_ntu` | 9 | `scripts/sweep.py` | multi-seed NTU runs; promotes `runs/best_bilstm_ntu.pt` (**heavy**) |
| `domainshift` | 9 | `cross_dataset_eval.py` + `domain_shift_matrix.py` | `figures/domain_shift_matrix.png`, cross metrics (see `chunk9_domain_shift.md`) |
| `widar` | 10 | `scripts/explore_widar.py` + `scripts/visualize_widar_bvp.py` | Widar3.0 BVP stats (stdout), `figures/widar_bvp_examples.png` (see `chunk10_widar_bvp.md`) |

## Defaults and the heavy stages

A no-arg run executes `DEFAULT_STAGES`:

```
verify → explore → preprocess → visualize → evaluate → capture → finalviz →
diagnose → postprocess → preprocess_ntu → domainshift → widar
```

> **`widar` needs the Widar3.0 BVP download** (`data/raw/widar3/bvp/`, gitignored,
> ~400 MB). Like the other data stages it errors out if the dataset isn't on
> disk — see `README.md` for the download command.

This is the **safe reproduction**: it uses the existing frozen checkpoints
(`runs/best_bilstm.pt`, `runs/best_bilstm_ntu.pt`) and deliberately **skips
`train`, `sweep`, and `train_ntu`** so a routine run neither burns CPU
retraining nor overwrites a checkpoint. Run those explicitly when you actually
want to (re)train:

```bash
./run_pipeline.sh train      # one UT-HAR config
./run_pipeline.sh sweep      # multi-seed UT-HAR; promotes the winner if SWEEP_PROMOTE=1
./run_pipeline.sh train_ntu  # multi-seed NTU-Fi; promotes runs/best_bilstm_ntu.pt
./run_pipeline.sh all        # the full chain including all three
```

## Per-stage variables

Each stage's tunable arguments are declared as `local` variables at the top of
its `stage_<name>()` function, with a comment on what each means — edit them
there. Example (`stage_capture`):

```bash
local CAPTURE_N=8          # number of clips stitched end-to-end
local CAPTURE_SEED=0       # selection RNG seed (round-robin across classes)
local CAPTURE_SPLIT="test" # which split to draw clips from
```

> **`window_size` / `stride`.** These are surfaced in `finalviz` and
> `diagnose` but carry an `(ASK before changing)` note. Per the project brief
> they are easy levers that can mask root causes — consult the owner before
> moving them (see `diagnostics.md`).

## Extending

To add a chunk: write a `stage_<name>()` function (banner + inline variables +
the command), append its name to `ALL_STAGES` in dependency order, and add it
to `DEFAULT_STAGES` if it should run by default. Update the table above when the
work lands.
