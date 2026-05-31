# UT-HAR ↔ NTU-Fi class mapping

Cross-dataset evaluation only makes sense for activities that *both* datasets
contain. The two label sets overlap only partially, so we align on the shared
classes by **name** and treat everything else as unmatched.

## Label sets

| Dataset | Classes |
|---------|---------|
| UT-HAR (7) | `lie_down`, `fall`, `walk`, `pickup`, `run`, `sit_down`, `stand_up` |
| NTU-Fi (6) | `box`, `circle`, `clean`, `fall`, `run`, `walk` |

## Alignment

| UT-HAR class | NTU-Fi class | Status |
|--------------|--------------|--------|
| `fall`  | `fall`  | ✅ shared |
| `run`   | `run`   | ✅ shared |
| `walk`  | `walk`  | ✅ shared |
| `lie_down`  | —   | ❌ UT-HAR only |
| `pickup`    | —   | ❌ UT-HAR only |
| `sit_down`  | —   | ❌ UT-HAR only |
| `stand_up`  | —   | ❌ UT-HAR only |
| —   | `box`    | ❌ NTU-Fi only |
| —   | `circle` | ❌ NTU-Fi only |
| —   | `clean`  | ❌ NTU-Fi only |

**Shared classes: `fall`, `run`, `walk` (3).**

## How unmatched classes are handled

Cross-domain accuracy is computed **only over test samples whose true label is
one of the three shared classes**. Samples belonging to a source-only or
target-only class are *excluded* from the cross-domain metric, because:

- A target-only class (e.g. NTU-Fi `box`) has no correct label the source
  model could ever emit — the UT-HAR model has no `box` output neuron, so it
  would always be wrong. Scoring it would conflate "can't represent the class"
  with "domain shift," which is what we're trying to isolate.
- A source-only class (e.g. UT-HAR `sit_down`) simply never appears as a true
  label in the target test set, so it contributes nothing to the metric.

Note the asymmetry that remains and is *kept* on purpose: the model can still
**predict** a non-shared class for a shared-class input (e.g. predict
`sit_down` for a true NTU-Fi `walk`). Those predictions count as wrong and show
up in the off-diagonal of the confusion matrix — that mis-routing is exactly
the domain-shift signal we want to see. So:

- **Rows** of the cross-domain confusion matrix = the 3 shared *true* classes
  (after filtering the target test set to them).
- **Columns** = *all* source classes the model can output (7 for UT-HAR, 6 for
  NTU-Fi), so leakage into non-shared predictions is visible.

## Semantic caveats (the mapping is looser than the names suggest)

Even the "shared" classes are not guaranteed to mean the same motion:

- **`fall`** — UT-HAR falls are a person collapsing to the floor; NTU-Fi `fall`
  is a deliberate, repeated falling-down gesture in a controlled rig. Different
  duration, velocity profile, and body orientation.
- **`run` / `walk`** — captured with different room geometry, transmitter–
  receiver placement, sampling, and subjects. The gross Doppler signature is
  similar but the multipath context differs.

This semantic drift is part of the domain gap (see `notes/domain_shift.md`),
not just a chipset/environment artifact.

## Shape note

The two datasets also have incompatible tensor shapes (UT-HAR `(250, 90)` vs
NTU-Fi `(342, 2000)`). To let one BiLSTM architecture run in both directions,
NTU-Fi is bilinearly resized to UT-HAR's `(250, 90)` and put through the
identical UT-HAR preprocessing pipeline (see `scripts/preprocess_ntu_fi.py`).
This is a representation choice, not a class-mapping choice, but it matters for
interpreting the numbers: the cross-domain model never sees NTU-Fi at native
resolution.
