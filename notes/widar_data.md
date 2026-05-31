# Widar3.0 BVP — what the data *means*

This note explains the **Body-coordinate Velocity Profile (BVP)** in my own
words, grounded in the Widar3.0 paper (Zheng et al., MobiSys '19) and the
official extraction code shipped with the dataset
(`data/raw/widar3/BVPExtractionCode/.../Doppler2VelocityMapping/`). It is the
conceptual foundation for chunks 11–12, where we actually model this data.

The one-line version: **a BVP frame is a 20×20 picture of how fast the body is
moving, and in which direction, expressed in the person's own coordinate
frame — with the room, the radios, and the body's facing direction all
factored out.** That last clause is the entire point.

## The file

Each `.mat` is one gesture instance: a `velocity_spectrum_ro` array of shape
`(20, 20, T)`, float, non-negative.

- The two `20` axes are a **velocity grid**. The first is velocity along the
  body's x-axis (`v_x`), the second along the body's y-axis (`v_y`). Both span
  **[−2, +2] m/s** in 20 bins, so the resolution is **0.2 m/s per bin**
  (`velocity_bin = ((1:20) − 10)/10 × 2`, i.e. −1.8 … +2.0 m/s). Our loader
  transposes to `(T, 20, 20)` so the leading axis is time.
- `T` is **time at 10 Hz**. In this dataset T runs ~7–26 frames (≈0.7–2.6 s) —
  one frame per 100 ms STFT segment of the underlying Doppler signal. T is *not*
  fixed; it is the gesture's duration, which is why the loader returns a list of
  variable-length arrays rather than one stacked tensor.

So a single instance is a **short movie** (T frames) of a **2-D velocity
distribution** (20×20).

## What a single cell means

The value at grid cell `(v_x, v_y)` is, roughly, **how much of the moving body
is travelling at that velocity right now**. The paper frames it as the
distribution of body-part velocities; the code recovers it as a non-negative
weight per velocity bin. A bright spot at `(+1.0, 0)` means "a chunk of the body
is moving at 1 m/s in the +x body direction at this instant." Each frame is
≈L1-normalized (the cells sum to ~1), so a frame is an *energy distribution*
over velocities, not an absolute power — absolute amplitude has been deliberately
thrown away (see invariance below).

Read across time, a gesture traces a **path through this velocity plane**:
- **Push&Pull** — energy oscillates along one axis (push out → +v, pull back → −v).
- **Slide** — a sustained streak along a diagonal (steady translation).
- **Draw-O / Draw-N** — the bright spot loops / zig-zags through the plane,
  echoing the hand's actual trajectory.

`figures/widar_bvp_examples.png` shows exactly this: 6 gestures × 4 timesteps,
and the velocity signatures are visibly distinct. This is the first point in the
project where the WiFi data *looks* like motion in space.

## How it's computed (and why each step buys invariance)

The pipeline is **CSI → Doppler → velocity**, solved as an inverse problem:

1. **CSI → per-receiver Doppler spectrum (DFS).** For each of the **6 receivers**,
   an STFT of the (denoised, conjugate-multiplied) CSI gives a Doppler Frequency
   Shift spectrum: how much signal energy is shifted by each frequency in
   [−60, +60] Hz over time. Movement toward a link ⇒ positive shift, away ⇒
   negative. This is the `6 × 121 × T` DFS product. Doppler is a property of
   *motion*; a static room contributes a 0 Hz (DC) component that is discarded,
   so the static multipath fingerprint of the room is already largely gone here.

2. **The forward physics: velocity → Doppler.** A body part moving with velocity
   `v = (v_x, v_y)` produces, on receiver *i*, a Doppler shift
   `f_i = (1/λ) · a_i · v`, where `λ ≈ 5.15 cm` (5.825 GHz) and `a_i` is a
   geometric coefficient set by the Tx–target–Rx layout (`get_A_matrix`). The
   key fact: **one velocity produces six different Doppler shifts**, one per
   receiver, because each link projects the velocity vector differently. The
   code precomputes this as the velocity→Doppler mapping `VDM`.

3. **The inverse problem: Doppler → velocity (the actual BVP).** Given the six
   measured Doppler spectra, solve for the 20×20 velocity distribution `P` whose
   predicted six-receiver Doppler response best matches the measurements. The
   extractor (`DVM_main.m` + `DVM_target_func.m`) minimises, with `fmincon`
   (SQP), an **Earth-Mover's-Distance** loss between predicted and measured
   Doppler across all receivers, plus a **sparsity** regulariser (L0 by default,
   `λ=1e-7`), under a **non-negativity** constraint. In plain terms: *find the
   sparsest, non-negative set of body velocities that simultaneously explains
   what all six receivers heard.* Using six links jointly is what pins down a
   2-D velocity (a single link would be ambiguous).

4. **Cross-receiver path-loss normalisation.** Before the solve, each receiver's
   spectrum is rescaled to a common total (`...× sum(rx1)/sum(rxi)`). This
   discards absolute amplitude — which depends on distance and attenuation, i.e.
   on the environment — and keeps only the *shape* of the Doppler distribution.

5. **Rotation into body coordinates.** `get_rotated_spectrum(..., torso_ori)`
   rotates the recovered velocity plane by the person's facing direction (one of
   −90…+90°). After this, "push away from the chest" maps to the same place in
   the grid **no matter which way the person was facing the radios.** This is the
   `_ro` ("rotated") in the variable name, and it is what makes the profile
   *body-coordinate*.

## Why BVP is environment-invariant (the payoff)

Raw CSI (chunks 1–9) bakes in the room: it encodes how the signal bounced off
*these* walls and furniture, through *this* Tx/Rx geometry, for *this* person's
orientation. A model trained on it learns the room as much as the motion — which
is exactly the cross-domain collapse measured in
[`domain_shift.md`](domain_shift.md) (92% in-domain → ~42% / ~2% cross-domain).

BVP removes those nuisance factors by construction:

| Nuisance factor | Removed by |
|---|---|
| Static room multipath (walls, furniture) | Doppler keeps only *moving* reflectors; DC discarded |
| Tx/Rx placement & link geometry | the `A`/`VDM` mapping + joint 6-receiver inversion |
| Absolute signal strength / distance | cross-receiver path-loss normalisation |
| Person's facing direction | rotation into body coordinates (`_ro`) |
| Where the person stands in the room | velocity is relative to the body, not the room |

What's left is the **kinematics of the gesture itself** — speed and direction of
body motion in a personal frame. The same gesture, performed by a different user,
in a different room, at a different position and orientation, yields a similar
BVP. That is why Widar3.0 reports near-constant accuracy across users / rooms /
orientations on BVP, and why this representation is the field's reference point
for *zero-effort cross-domain* gesture recognition.

## Caveats to carry into chunk 11

- **Variable T.** Any model must handle variable-length sequences (pad/mask,
  RNN, temporal pooling, or resample) — there is no fixed time dimension.
- **Gesture-id ambiguity.** The integer gesture id in the filename means
  different gestures on different collection dates (and, for 3 dates, different
  users). Always use the resolved name from the loader, never the raw id. See
  the per-date tables printed by `scripts/explore_widar.py` and encoded in
  `src/data/widar_loader.py:GESTURE_MAP`.
- **Not perfectly invariant.** BVP is an *estimate* from a noisy, regularised
  inverse problem; residual domain effects remain. It is far better than raw CSI,
  not magic.
- **It's already a derived feature.** Unlike UT-HAR/NTU-Fi raw CSI, BVP has had
  heavy physics-based processing applied upstream. We are consuming the dataset's
  pre-computed BVP, not recomputing it from CSI (that needs the raw `.dat` +
  MATLAB extractor).
