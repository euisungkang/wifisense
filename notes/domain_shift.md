# Domain shift: UT-HAR ↔ NTU-Fi

**Goal of this chunk:** measure, not fix, how a WiFi-sensing model trained in
one environment fails in another. We aligned the two datasets on their shared
classes `{fall, run, walk}` (see `notes/class_mapping.md`), put both into a
common `(250, 90)` representation, and ran every train/test combination.

## What the numbers say

|                  | tested on UT-HAR | tested on NTU-Fi |
|------------------|:---------------:|:----------------:|
| **UT-HAR-trained** | **92.0%** (in-domain, 7-class) | **42.4%** (zero-shot, shared) |
| **NTU-Fi-trained** | **1.6%** (zero-shot, shared) | **98.5%** (in-domain, 6-class) |

*Diagonal = full in-domain test accuracy. Off-diagonal = zero-shot accuracy on
the 3 shared classes only (chance = 33.3%).*

Both models are excellent **in-domain** (92% / 98.5%). Both **collapse across
domains**:

- **UT-HAR → NTU-Fi: 42.4%**, barely above chance. The model funnels almost
  everything into `run`: NTU-Fi `walk` → `run` 44/44 (0% recall), `fall` → `run`
  32/44. It only "succeeds" because `run` is one of the three shared answers.
- **NTU-Fi → UT-HAR: 1.6%**, *below* chance. The model routes 95% of inputs to
  `clean` — an NTU-Fi-only class. UT-HAR `walk`/`run`/`fall` all look like the
  NTU-Fi "wiping" gesture to it. The model has learned NTU-Fi's specific
  signal manifold so tightly that out-of-domain inputs don't even land on the
  shared classes.

The asymmetry (42% vs 1.6%) is itself informative: each model has a "default
attractor" class (`run` for UT-HAR, `clean` for NTU-Fi) that swallows
unfamiliar inputs. Whether that attractor happens to be a shared class decides
whether zero-shot accuracy looks mediocre or catastrophic. Neither is real
generalization — the in-domain → cross-domain drop is ~50 and ~97 points.

## Why this happens

1. **Multipath fingerprints are environment-specific.** CSI encodes how the
   signal bounces off walls, furniture, and bodies *in that room*. The model
   learns the room as much as the motion; a new room rewrites every reflection
   path, so the learned features no longer apply.
2. **Different chipsets sense different things.** UT-HAR is Intel 5300
   (30 subcarriers × 3 RX = 90); NTU-Fi is an Atheros-class radio
   (114 subcarriers × 3 = 342). Subcarrier spacing, frequency response, and
   amplitude scaling differ, so "the same" activity produces differently shaped
   CSI. Our bilinear resize to a common `(250, 90)` grid forces a comparison but
   cannot undo this — it actually adds a resampling mismatch on top.
3. **Class semantics drift.** Even matched labels aren't the same motion: a
   UT-HAR `fall` (person collapsing) differs from a NTU-Fi `fall` (deliberate
   repeated gesture) in duration, velocity, and orientation. The label agrees;
   the physics doesn't.
4. **Distinct sampling, geometry, and subjects.** Packet rate, TX–RX placement,
   and who performed the activity all shift the distribution. The 98.5%
   in-domain / 1.6% cross result shows how much capacity goes into fitting these
   nuisance factors rather than the activity itself.

## Techniques that address it (not applied here)

- **Domain adaptation** — align source and target feature distributions:
  adversarial methods (DANN), discrepancy minimization (MMD/CORAL), or
  self-supervised target adaptation. Needs (unlabeled) target data.
- **Few-shot fine-tuning** — a handful of labeled target samples recovers much
  of the loss; cheap but requires per-environment labels.
- **Augmentation / environment simulation** — perturb CSI (subcarrier dropout,
  phase noise, synthetic multipath) so the model can't overfit one room.
- **Domain-invariant representations** — the most principled fix: transform CSI
  into a feature that *is* environment- and link-independent before
  classification. **BVP (Body-coordinate Velocity Profile)**, the representation
  **Widar3.0** is built around, is the canonical example: it estimates the
  velocity components of the body in a body-centric coordinate frame, factoring
  out the transmitter/receiver geometry and the room's static multipath. A model
  trained on BVP transfers across rooms and orientations far better than one
  trained on raw CSI — because the input no longer carries the room's
  fingerprint.

## Takeaway → next chunk

Raw-CSI models memorize their capture environment. High in-domain accuracy says
nothing about deployment in a new room. The path forward is a representation
that discards environment-specific structure — which is exactly why **chunk 10
moves to Widar3.0 and BVP.**

## Reproduce

```bash
conda activate wifisense
python scripts/preprocess_ntu_fi.py                       # NTU-Fi -> (250,90)
python scripts/sweep.py --seeds 42 43 44 45 46 \
    --data data/processed/ntu_fi/ntu_fi.npz \
    --promote --promote-to runs/best_bilstm_ntu.pt --tag ntu
python -m src.evaluate --checkpoint runs/best_bilstm_ntu.pt \
    --data data/processed/ntu_fi/ntu_fi.npz --split test --out-dir figures/ntu
python scripts/cross_dataset_eval.py --checkpoint runs/best_bilstm.pt \
    --data data/processed/ntu_fi/ntu_fi.npz --tag uthar_on_ntu
python scripts/cross_dataset_eval.py --checkpoint runs/best_bilstm_ntu.pt \
    --data data/processed/ut_har/ut_har.npz --tag ntu_on_uthar
python scripts/domain_shift_matrix.py                     # figures/domain_shift_matrix.png
```
