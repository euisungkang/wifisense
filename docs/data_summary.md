# Data Summary

Findings from `scripts/explore_data.py`, run 2026-05-29.

## UT-HAR

| Split | Samples |
|-------|---------|
| train | 3,977   |
| val   | 496     |
| test  | 500     |

**Shape:** `(N, 250, 90)` — 250 time steps × 90 features (30 subcarriers × 3 RX antennas, flattened).

**Value range:** −10.67 to +30.54, mean ≈ 17.65, std ≈ 5.90.
Values are real-valued, NOT complex. They include negative values, so these are
not raw amplitudes — the dataset has been pre-processed (likely some form of
amplitude extraction + signal processing from the Intel 5300 CSI Tool). NOT
normalized to [0,1] or z-score. No NaN or Inf.

**Class imbalance (train):**

| Label | Class     | Count | Pct   |
|-------|-----------|-------|-------|
| 0     | lie_down  | 525   | 13.2% |
| 1     | fall      | 354   | 8.9%  |
| 2     | walk      | 1,172 | 29.5% |
| 3     | pickup    | 396   | 10.0% |
| 4     | run       | 967   | 24.3% |
| 5     | sit_down  | 320   | 8.0%  |
| 6     | stand_up  | 243   | 6.1%  |

**Walk and run together make up 54% of training data.** Max/min class ratio is
4.8× (walk vs stand_up). The val and test splits have the same proportional
imbalance, so it's stratified — not a random artifact. This will need
class-weighted loss or oversampling during training.

**File format gotcha:** Files are `.csv` extension but are numpy binary dumps
(loaded with `np.load`, not a CSV parser).

## NTU-Fi HAR

| Split | Samples |
|-------|---------|
| train | 936     |
| test  | 264     |

**Shape:** `(N, 342, 2000)` raw, where 342 = 3 antennas × 114 subcarriers,
2000 = time packets. SenseFi downsamples to every 4th packet and reshapes to
`(3, 114, 500)`, but our raw loader returns the original `(342, 2000)`.

**Value range:** 0.00 to 57.20, mean ≈ 42.30, std ≈ 4.98.
All positive, real-valued — these are raw CSI amplitudes (dB scale, not
linear). NOT normalized. No NaN or Inf.

**Perfectly balanced:** 156 samples/class (train), 44 samples/class (test).

| Label | Class  |
|-------|--------|
| 0     | box    |
| 1     | circle |
| 2     | clean  |
| 3     | fall   |
| 4     | run    |
| 5     | walk   |

**Memory:** Loading full train split as a stacked tensor uses ~2.5 GB (float32).
The `NTUFiHARDataset` class loads lazily instead.

## Surprising / Notable

1. **UT-HAR class imbalance is significant** — walk alone is 29.5% of data.
   Must address during training.
2. **UT-HAR values go negative** — these are NOT raw CSI amplitudes but some
   pre-processed representation. The SenseFi benchmark then min-max normalizes
   on top of this, which may wash out per-sample variation.
3. **NTU-Fi has zeros** (min = 0.00) — some subcarrier/time entries have zero
   amplitude. Worth checking if these are null subcarriers or actual signal
   dropouts.
4. **SenseFi's hardcoded normalization constants** for NTU-Fi
   (`(x - 42.3199) / 4.9802`) match our computed mean/std almost exactly
   (42.30 / 4.98), confirming they were derived from the training set.
5. **No NaN/Inf in either dataset** — clean data, no missing values.
6. **The two datasets have very different scales** — UT-HAR ≈ [−11, 31],
   NTU-Fi ≈ [0, 57]. Any cross-dataset work needs separate normalization.
