# Domain Shift: UT-HAR ↔ NTU-Fi Cross-Dataset Evaluation

Chunks 1–8 built a strong UT-HAR pipeline (92% in-domain test accuracy). But
UT-HAR was captured in **one** environment with **one** hardware setup — high
in-distribution accuracy says nothing about whether the model works in a
different room. Cross-environment failure (**domain shift**) is the dominant
open problem in WiFi sensing.

This chunk **characterizes** that gap honestly using NTU-Fi HAR (downloaded in
chunk 1, unused since) as a free second domain. We are *not* trying to fix it —
domain adaptation is its own field. The point is to produce the honest 2×2
number that motivates the move to Widar3.0 / BVP in chunk 10.

Companion write-ups: [`notes/domain_shift.md`](../notes/domain_shift.md) (the
one-page interpretation) and [`notes/class_mapping.md`](../notes/class_mapping.md)
(how the two label sets are aligned).

## Reproduce

```bash
conda activate wifisense
./run_pipeline.sh preprocess_ntu        # NTU-Fi → data/processed/ntu_fi/ntu_fi.npz
./run_pipeline.sh train_ntu             # (HEAVY) trains runs/best_bilstm_ntu.pt
./run_pipeline.sh domainshift           # cross-evals + figures/domain_shift_matrix.png
```

`domainshift` alone reproduces every number from the frozen checkpoints
(`runs/best_bilstm.pt`, `runs/best_bilstm_ntu.pt`), so it lives in the default
no-arg pipeline; `train_ntu` is heavy and excluded by default.

## Two problems before any number is meaningful

**1. Incompatible shapes.** UT-HAR is `(250 time, 90 feat)` (90 = 30 subcarriers
× 3 RX); NTU-Fi is `(342 feat, 2000 time)` (342 = 114 subcarriers × 3 antennas).
The UT-HAR BiLSTM has `input_size=90` baked in, so NTU-Fi *cannot* be fed to it
as-is. We chose a **common representation**: bilinearly resize every NTU-Fi
spectrogram `(342, 2000) → (90, 250)`, transpose to UT-HAR's `(250, 90)` layout,
and run the *identical* UT-HAR preprocessing pipeline (amplitude → hampel →
median → per-sample z-score). One `input_size=90` architecture then runs in both
cross-eval directions. Cost: NTU-Fi is never seen at native resolution (noted as
a caveat in the interpretation). See `scripts/preprocess_ntu_fi.py`.

**2. Partial class overlap.** UT-HAR `{lie_down, fall, walk, pickup, run,
sit_down, stand_up}` and NTU-Fi `{box, circle, clean, fall, run, walk}` share
only **`fall`, `run`, `walk`**. Cross-domain accuracy is computed over target
test samples whose true class is shared; predictions that leak to non-shared
classes count as wrong and show up in the confusion off-diagonal. Full rationale
in `notes/class_mapping.md`.

## Recipe fidelity

The NTU-Fi model uses the **identical chunk-5 recipe** — BiLSTM (hidden 64, 2
layers, dropout 0.3, bidirectional), Adam @ 1e-3, batch 64, ≤80 epochs, patience
15, multi-seed sweep over 42–46 promoting best-by-val. Two unavoidable,
data-dictated deviations, both confirmed with the owner before building:

- `input_size` 90 (same) but `num_classes` 6 instead of 7;
- NTU-Fi ships no val split, so a **stratified 10%** val set is carved from the
  936 train samples (seeded) to drive the same early-stopping rule. The
  264-sample test set is untouched.

`src/train.py` now reads `class_names` from the `.npz` (UT-HAR fallback) so
checkpoints self-describe; `scripts/sweep.py` gained `--data` and `--promote-to`
so the same sweep harness trains either dataset.

## Results

|                  | tested on UT-HAR | tested on NTU-Fi |
|------------------|:----------------:|:----------------:|
| **UT-HAR-trained** | **92.0%** (in-domain, 7-class) | **42.4%** (zero-shot, shared) |
| **NTU-Fi-trained** | **1.6%** (zero-shot, shared) | **98.5%** (in-domain, 6-class) |

*Diagonal = full in-domain test accuracy; off-diagonal = zero-shot on the 3
shared classes (chance 33.3%).* Figure: `figures/domain_shift_matrix.png` (2×2
matrix + both cross-domain confusion matrices).

Both models are excellent in-domain and **collapse** out-of-domain:

- **UT-HAR → NTU-Fi (42.4%)** — barely above chance. The model funnels NTU-Fi
  locomotion into `run`: `walk` → `run` 44/44 (0% recall), `fall` → `run` 32/44.
- **NTU-Fi → UT-HAR (1.6%)** — *below* chance. 95% of inputs route to `clean`
  (an NTU-Fi-only class); the model has overfit its own signal manifold so
  tightly that UT-HAR inputs don't even land on the shared classes.

The asymmetry (42% vs 1.6%) reflects each model's "attractor" class (`run` vs
`clean`) and whether it happens to be shared — neither is real generalization.

## Why (short version) and what's next

CSI encodes the *room's* multipath as much as the motion; chipsets differ in
subcarrier sensitivity; even matched labels are different physical gestures.
Raw-CSI models therefore memorize their capture environment. The principled fix
is a **domain-invariant representation** — chiefly **BVP** (Body-coordinate
Velocity Profile), which Widar3.0 is built around and which factors out the
TX/RX geometry and static multipath. Full reasoning and the menu of mitigation
techniques are in [`notes/domain_shift.md`](../notes/domain_shift.md). **Chunk 10
moves to Widar3.0 and BVP.**

## Artifacts

| Path | What |
|---|---|
| `scripts/preprocess_ntu_fi.py` | NTU-Fi → common `(250,90)` npz + resize figure |
| `data/processed/ntu_fi/ntu_fi.npz` | processed NTU-Fi (train/val/test + class_names) |
| `runs/best_bilstm_ntu.pt` | NTU-Fi-trained BiLSTM (chunk-5 recipe) |
| `scripts/cross_dataset_eval.py` | generic zero-shot cross-domain evaluator |
| `scripts/domain_shift_matrix.py` | assembles the 2×2 results figure |
| `figures/domain_shift_matrix.png` | the headline figure |
| `figures/cross_*_{confusion.png,metrics.json}` | per-direction cross results |
| `figures/ntu/eval_metrics_test.json` | NTU-Fi in-domain metrics |
| `notes/{domain_shift,class_mapping}.md` | interpretation + class alignment |
