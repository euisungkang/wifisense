# Chunk 11 — Widar3.0 BVP: preprocessing pipeline & cross-domain splits

Chunk 10 downloaded the BVP set and built a loader that returns a *list* of
variable-length `(T, 20, 20)` volumes plus metadata. That is not yet trainable:
a model needs fixed-shape batches, a normalization scheme, augmentation, and —
crucially for Widar3.0 — splits that actually test cross-domain generalization.
This chunk adds all of that. Chunk 12 trains and visualizes on top of it.

Three new pieces:

- [`src/data/bvp_preprocess.py`](../src/data/bvp_preprocess.py) — pure,
  composable transforms (same conventions as `src/data/preprocess.py`).
- [`src/data/widar_dataset.py`](../src/data/widar_dataset.py) — a lazy PyTorch
  `Dataset` plus the four canonical cross-domain split builders.
- [`src/data/csi_to_bvp.py`](../src/data/csi_to_bvp.py) — *optional, educational*
  re-derivation of one BVP from raw CSI, to verify the physics.

The demo figure is `figures/bvp_pipeline_demo.png`
([`scripts/bvp_pipeline_demo.py`](../scripts/bvp_pipeline_demo.py)); the
derivation check is `figures/csi_to_bvp_check.png` (`python -m
src.data.csi_to_bvp`).

## 1. Preprocessing transforms and their rationale

All three operate on one sample `(T, 20, 20)`, return `float32`, and never mutate
their input. In a training pipeline `WidarBVPDataset` applies them in the order
**augment → normalize → pad/truncate** (see below for why that order).

### `normalize_bvp(x, mode='per_sample' | 'global')`

On disk each 20×20 frame is ~L1-normalized (cells sum to ~1), so raw magnitudes
are tiny (~1e-3) and nearly constant across samples — poorly conditioned input.
We z-score to zero-mean/unit-variance. `mode` only changes the *scope* of the
statistics:

- `per_sample` — mean/std from this volume alone. Simple, leak-free, and the
  sensible default since BVP is already environment-normalized.
- `global` — uses a precomputed `{mean, std}` (see `compute_global_stats`,
  estimated from the **training** split only). Use when you want one fixed input
  scale across the whole corpus.

Z-scoring sacrifices non-negativity, which is exactly why augmentation (whose
noise clip and physical interpretation assume an energy map) must run *before*
it.

### `pad_or_truncate(x, target_T)` — why pad vs. truncate this way

Gestures have variable duration (`T` ranges 9–28 across the corpus, median 17),
but a batch needs one fixed `T`. The policy is **anchor at the start, fix length
at the tail**:

- `T < target_T`: **append zero frames.** After per-sample z-scoring zero ≈ the
  volume mean, so padded frames read as "no motion" rather than as a spurious
  signal. Appending (not prepending) keeps the gesture onset at `t=0`, so every
  sample shares a temporal origin.
- `T > target_T`: **keep the first `target_T` frames.** With `target_T` set to a
  high percentile of the corpus (default 32 > max observed 28), truncation never
  fires on normal data and, when it does, only drops the tail of the longest
  gestures.

Why not center-crop or symmetric-pad? Both break the shared `t=0` origin, which
matters for the sequence models in chunk 12. Anchoring at the start is the
simplest policy that keeps every sample temporally aligned; the cost (a long
gesture losing its tail) is rare by construction.

### `augment_bvp(x, rng, ...)` — when (and when not) to augment

**Training only — never at eval time.** It is stochastic; applying it to
held-out data would corrupt the measurement. `WidarBVPDataset` enforces this:
the split builders force `augment=False` on the test dataset regardless of what
you pass. Three perturbations, each independently disable-able, all in raw
non-negative energy space:

- **Random temporal crop** — keep a contiguous window covering a random fraction
  in `[1 - temporal_crop, 1]` of the frames. Simulates slightly different
  segmentation. Changes `T`, hence the downstream `pad_or_truncate`.
- **Horizontal flip** (`v_x → -v_x`) — valid *only* for left/right-symmetric
  gestures (e.g. Push&Pull), where a leftward and rightward execution are the
  same class. Disable for direction-defined gestures.
- **Gaussian noise** (clipped to ≥ 0) — mild jitter; `noise_std` is in raw BVP
  units (frames sum to ~1, peak cells ~0.15, so 0.01 is gentle).

## 2. Dataset and the four cross-domain splits

`WidarBVPDataset` loads one `.mat` per `__getitem__` (the corpus is ~44k samples,
far too large to hold in memory), applies the transform chain, and yields
`(tensor (target_T, 20, 20), label int)`. Labels come from a shared gesture-name
→ int map so train and test encode the same gesture identically.

The reason this module exists at all is the **Widar3.0 evaluation protocol**: the
dataset's purpose is cross-domain generalization, so we expose four split
builders, each returning `(train_ds, test_ds)`. The gap between `in_domain` and
any cross-domain split *is* the domain gap BVP is meant to close.

```python
from src.data import cross_user, cross_position, cross_orientation, in_domain, make_dataloader

# Leave-users-out: train on everyone except users 3 & 5, test on them.
train, test = cross_user(test_users=[3, 5], augment=True)

# Leave-positions-out: hold out torso locations 6–8; fix one gesture to scope it.
train, test = cross_position(test_positions=[6, 7, 8], gesture="Push&Pull")

# Leave-orientations-out: the strongest test of body-frame invariance.
train, test = cross_orientation(test_orientations=[5], room=1)

# In-domain (i.i.d. random split) — the easy baseline, no domain shift.
train, test = in_domain(test_frac=0.2, gesture=["Slide", "Sweep"])

loader = make_dataloader(train, batch_size=32)   # yields (B, T, 20, 20), labels
```

Split semantics:

| Builder | Held out | Question it answers |
|---|---|---|
| `cross_user(test_users=[…])` | whole users | Generalize to **people** never trained on? |
| `cross_position(test_positions=[…])` | torso locations 1–8 | Generalize to **room locations** never seen? |
| `cross_orientation(test_orientations=[…])` | face orientations 1–5 | Is the representation truly **facing-invariant**? |
| `in_domain(test_frac=…)` | random fraction | Baseline accuracy with **no** domain shift. |

All four accept the same optional filters (`gesture`, `user`, `position`,
`orientation`, `date`, `room` — except the axis being split on) to scope the
corpus first, plus dataset kwargs (`target_T`, `normalize`, `augment`,
`augment_kwargs`, `seed`, `global_stats_max_samples`). Reproducibility note: with
`augment=True`, use `num_workers=0` for a deterministic augmentation stream (each
DataLoader worker otherwise copies the same RNG state).

## 3. (Optional) Re-deriving a BVP from raw CSI — the physics, verified

`src/data/csi_to_bvp.py` reimplements the official CSI → Doppler → velocity
extractor in Python for the one sample shipped inside the toolkit
(`BVPExtractionCode/.../Data/userA-1-1-1-1-r{1..6}.dat`) and compares against the
toolkit's own output (`.../BVP/user-user-1-1-1-1-1-…-L0.mat`, shape `(20,20,14)`).
The goal is understanding, not a bit-exact match.

### The math (Widar3.0 §4)

1. **CSI → per-receiver Doppler (DFS).** Per receiver: read raw 802.11n CSI
   (Linux CSI Tool log format), pick a reference antenna (WiDance), amplitude-
   adjust (IndoTrack), **conjugate-multiply** against the reference to cancel the
   carrier-frequency offset, band-pass to 2–60 Hz (kill the static 0 Hz room
   component and >60 Hz noise), reduce subcarriers with PCA, then take an STFT to
   get energy vs. Doppler over time. → a `(121, T)` map per receiver, F spanning
   [−60, +60] Hz.

2. **Forward physics velocity → Doppler (Eq. 3).** A body part at velocity
   `v = (v_x, v_y)` produces on receiver *i* a Doppler shift

   ```
   f_i = (1/λ) · a_i · v,     a_i = (p−p_tx)/‖p−p_tx‖ + (p−p_rx,i)/‖p−p_rx,i‖
   ```

   with `λ = c / 5.825 GHz ≈ 5.15 cm`. `a_i` (`get_A_matrix`) is the bistatic
   geometry vector — *one velocity yields six different Doppler shifts*, one per
   link. Discretizing the 20×20 velocity grid gives a sparse 0/1 operator `G`
   (the code's `VDM`) mapping a velocity distribution to per-receiver Doppler.

3. **Inverse problem Doppler → velocity (the BVP).** Find the non-negative 20×20
   `P` whose six predicted Doppler spectra match the six measured ones. The
   official extractor minimizes an **Earth-Mover's-Distance** loss plus an **L0
   sparsity** term under non-negativity, via `fmincon` SQP. We instead solve the
   **same linear system** `G·vec(P) = d`, `P ≥ 0`, by non-negative least squares
   (`scipy.optimize.nnls`) — stable, no tuning, and faithful to the inversion
   while deliberately dropping the EMD/sparsity refinement.

4. **Body-frame rotation.** Rotate `P` by the torso orientation
   (`get_rotated_spectrum` → `scipy.ndimage.rotate`) so the result is in the
   person's own frame.

### Result and documented discrepancies

On the sample (position 1, orientation −90°) the derivation produces 12 frames
(official 14), with **mean per-frame cosine ≈ 0.34** and **mean peak-velocity
error ≈ 0.32 m/s** (~1.6 velocity bins). The time-aggregated energy and the
dominant motion direction line up well; per-frame agreement is modest. Looking at
`figures/csi_to_bvp_check.png`, the derived BVP is **diffuse with axis-aligned
streaks** while the official is **compact**. The discrepancies, in rough order of
impact:

- **No sparsity / EMD (biggest).** NNLS minimizes an L2 residual with no
  regularizer, so energy spreads across all velocity cells that map to a
  plausible Doppler — hence the streaks. The official L0 term forces the compact,
  sparse blobs. This is the deliberate simplification.
- **STFT vs. `tfrsp`.** We use `scipy.signal.stft` (Gaussian window, 1 Hz bins
  via `nfft = fs`, hop = 100 → 10 Hz frames) where the toolkit uses TFTB's
  `tfrsp`. Different time-frequency kernels give different Doppler maps.
- **Frame count / segmentation.** Our STFT hop yields 12 frames vs. the toolkit's
  14 segment-means; boundary handling differs slightly.
- **Minor numerical paths.** PCA component sign/phase ambiguity, causal IIR
  filtering edge effects, and `nearest` rotation interpolation all contribute at
  the margins.

The takeaway: the **physics inversion is correct** (the geometry operator and the
linear solve recover the right dominant velocities); the gap to the official file
is the EMD + sparsity post-processing, which is exactly the part the paper flags
as numerically delicate and which we chose not to reproduce.

