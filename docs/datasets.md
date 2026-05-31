# Data Acquisition: UT-HAR & NTU-Fi HAR

`verify_setup.py` is a **sanity check**, not training code. It confirms three
things before you build anything:

1. PyTorch works and sees your GPU.
2. Both datasets are on disk in the expected folders.
3. Each loads into the exact tensor shape the model will expect.

It loads one sample from each dataset and prints its shape / dtype / value
range. Think of it as "does the plumbing work before I turn on the water."

## What CSI actually is (the common thread)

Both datasets are **WiFi Channel State Information (CSI)**. When a WiFi signal
travels from transmitter to receiver, a person moving in the room bends,
reflects, and absorbs it. CSI measures how the signal was distorted across many
**subcarriers** (WiFi splits its channel into ~30–114 narrow frequency bins)
over **time**. A walking person distorts the signal differently than a falling
person — so a sequence of CSI readings is a "fingerprint" of an activity. Both
datasets are just CSI recordings labeled with what the person was doing.

## UT-HAR (primary dataset)

- **Hardware:** Intel 5300 NIC (a classic CSI research card).
- **7 activities:** lie down, fall, walk, pickup, run, sit down, stand up
  (labels `0`–`6`).
- **Shape `(250, 90)` per sample:** 250 time steps × 90 values. The 90 = 30
  subcarriers × 3 antenna pairs, flattened.
- **Format gotcha:** files are named `.csv` but are actually **numpy binary
  dumps** — that's why the script uses `np.load`, not a CSV reader.
- **Pre-split** into train (3977) / val (496) / test (500), as plain arrays
  `X_*` (data) and `y_*` (labels).

A real sample (sample 0, label `0` = lie down), first 3 time steps × 6
subcarriers:

```
[[7.217 4.206 1.902 7.584 4.912 7.217]
 [1.402 4.897 4.637 2.24  6.526 4.315]
 [1.897 2.153 5.392 5.163 6.781 8.088]]
```

These are processed CSI amplitudes (already normalized-ish, ranges roughly -10
to +30). Each row = one moment in time; each column = one subcarrier.

## NTU-Fi HAR (secondary / generalization-check dataset)

- **Different hardware/setup** → used to check whether a model trained on
  UT-HAR generalizes, not as the main training set.
- **6 activities:** box, circle, clean, fall, run, walk.
- **Storage is totally different:** one MATLAB `.mat` file *per sample*, sorted
  into one folder per activity. The **folder name is the label**
  (`train_amp/box/box1.mat`). 936 train / 264 test files.
- **Raw shape `(342, 2000)`:** 342 subcarriers (3 antennas × 114) × 2000
  packets (time). Higher resolution than UT-HAR.
- The script then **downsamples** (every 4th packet → 500) and reshapes to
  `(3, 114, 500)` = 3 antennas × 114 subcarriers × 500 time steps.

A real sample (`box`), first 3 subcarriers × 6 packets:

```
[[42.49  44.341 44.091 43.14  45.392 45.485]
 [43.259 45.145 44.413 43.552 45.975 46.079]
 [43.644 45.362 44.792 44.359 46.162 46.299]]
```

Note these are **raw amplitudes** (all positive, 0–54), unlike UT-HAR's
pre-processed values. Here each row = one subcarrier, each column = one moment
in time (axes are transposed relative to UT-HAR).

> **Used for real in chunk 9.** NTU-Fi finally earns its keep as a second
> *domain* for the cross-dataset / domain-shift study. To make one BiLSTM run on
> both datasets, `scripts/preprocess_ntu_fi.py` resizes each `(342, 2000)` sample
> to UT-HAR's `(250, 90)` and runs the identical UT-HAR pipeline. See
> [`chunk9_domain_shift.md`](chunk9_domain_shift.md).

## The key contrast to internalize

| | UT-HAR | NTU-Fi |
|---|---|---|
| One sample = | a row in a big array | a whole `.mat` file |
| Label comes from | `y_*` array | folder name |
| Layout | (time, subcarrier) | (subcarrier, time) |
| State | pre-processed & pre-split | raw, needs reshaping |

So "Data Acquisition" here just means: get both datasets on disk, then prove
each one loads into a clean numeric tensor of known shape. The two datasets
deliberately differ in format and hardware so downstream code has to normalize
them into a consistent shape — which is exactly the reshaping at the end of
each check function in `verify_setup.py`.

## Widar3.0 BVP (chunk 10 — a different *kind* of input)

UT-HAR and NTU-Fi are both **raw CSI**: the amplitude fingerprint of a specific
room. Widar3.0 brings a fundamentally different input — **BVP (Body-coordinate
Velocity Profile)**, a *derived, environment-invariant* feature. It is not CSI
amplitude at all; it's an estimate of **how fast the body is moving and in which
direction**, in the person's own coordinate frame, with the room/radios/facing
factored out. This is the answer to the cross-domain collapse measured in
[`chunk9_domain_shift.md`](chunk9_domain_shift.md).

- **Hardware:** 1 Tx + **6 Intel 5300 receivers** (Tsinghua, MobiSys '19).
- **Per-sample shape `(T, 20, 20)`:** a 20×20 velocity grid (`v_x` × `v_y`, both
  ±2 m/s) per timestep, T timesteps at 10 Hz. **T varies** by gesture — so the
  loader returns a *list* of arrays, not one stacked tensor.
- **Storage:** one `.mat` per instance (`velocity_spectrum_ro`), nested under
  `<date>-VS/[6-link/]<userN>/`. The filename `<userN>-a-b-c-d-…` encodes
  gesture / position / orientation / repetition — but the gesture id `a` is
  **date-dependent** (see below).
- **Scale:** 43,658 instances; 22 gestures, 17 users, 8 positions, 5
  orientations, 3 rooms — explicitly built for cross-user / -position /
  -orientation / -room evaluation.

**The label gotcha that bites everyone:** gesture id `a` means different gestures
on different collection dates (id `4` is "Slide" on 2018-11-09 but "Draw-O" on
2018-11-15), and on three dates it differs per user. `src/data/widar_loader.py`
encodes the README's per-date tables and resolves every file to a name; filter by
name, never by raw id.

This is a *load-and-explore* chunk — no preprocessing or modeling. Details:
[`chunk10_widar_bvp.md`](chunk10_widar_bvp.md); BVP physics and invariance:
[`../notes/widar_data.md`](../notes/widar_data.md).
