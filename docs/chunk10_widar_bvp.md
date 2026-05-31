# Chunk 10 — Widar3.0 BVP: download, load, explore

The pivot chunk. Chunks 1–9 lived on **raw CSI** (UT-HAR, NTU-Fi) and ended by
measuring how badly raw-CSI models collapse across environments
([`chunk9_domain_shift.md`](chunk9_domain_shift.md)). The diagnosis pointed at
the fix: a representation that discards environment-specific structure. Chunk 10
brings in that representation — **Widar3.0's Body-coordinate Velocity Profile
(BVP)** — and does *only* the data work: download, a loader, and an exploration
report. No modeling yet.

For what BVP *means* (axes, magnitude, why it's environment-invariant), see the
companion note [`../notes/widar_data.md`](../notes/widar_data.md). This doc is
the chunk's engineering record.

## What was downloaded

Source: the official Widar3.0 distribution on Tsinghua Cloud (linked from
`http://tns.thss.tsinghua.edu.cn/widar3.0/`). We grabbed the **BVP portion only**
— the pre-computed velocity profiles — and skipped the multi-GB raw CSI `.dat`
files, which we don't need to consume the BVP.

| File | Size | Into |
|---|---|---|
| `BVP.zip` | ~403 MB → 43,658 `.mat` files | `data/raw/widar3/bvp/BVP/` |
| `README.pdf` | dataset spec (format, gesture/position tables, bug list) | `data/raw/widar3/` |
| `BVPExtractionCode.zip` | MATLAB extractor (authoritative algorithm) | `data/raw/widar3/BVPExtractionCode/` |

The whole BVP set is only ~400 MB, so we took all of it rather than a
development subset. Everything under `data/raw/` is gitignored.

## The data, as it actually is

- **43,658 BVP instances**, each a `.mat` holding one variable
  `velocity_spectrum_ro`, shape **(20, 20, T)**, float64, non-negative.
- **20×20 velocity grid**, both axes spanning [−2, +2] m/s (0.2 m/s/bin):
  axis 0 = `v_x`, axis 1 = `v_y` in the body frame.
- **T = timesteps at 10 Hz**, ranging ~7–26 (≈0.7–2.6 s). **T varies per
  sample** — there is no fixed time dimension.
- Each frame is ≈L1-normalized (cells sum to ~1): an *energy distribution* over
  velocity, not absolute power.
- **22 distinct gestures**, **17 users**, **8 torso positions** (1–5 dominant,
  6–8 only in a few captures), **5 face orientations** (balanced ~20% each),
  **3 rooms** (classroom 73% / hall 20% / office 7%).

### The gesture-id trap

Filenames are `<userN>-a-b-c-d-<suffix>.mat` (`a`=gesture, `b`=position,
`c`=orientation, `d`=repetition). **`a` is not a global label**: Widar3.0 was
collected over 14 days and the meaning of gesture id `a` depends on the
collection **date**, and for three dates on the **user** (e.g. id `4` = "Slide"
on 2018-11-09 but "Draw-O" on 2018-11-15). The README's per-date tables are
transcribed verbatim into `src/data/widar_loader.py:GESTURE_MAP`, and the loader
resolves every sample to a human-readable name. Resolving all 43,658 files yields
exactly **22 gesture names with zero "unknown"** — a clean check that the mapping
is right. **Always filter/label by name, never by raw id.**

## Deliverables

- **`src/data/widar_loader.py`** — the loader.
  - `load_widar_bvp(user, gesture, position, orientation, date, room, limit)
    → (X, metadata)`. `X` is a **list** of `(T, 20, 20)` float32 arrays (a list,
    not a stacked tensor, because T varies); `metadata` is a list of per-sample
    dicts (user/gesture/position/orientation/room/date/…) for cross-domain
    splits. Each filter takes a scalar, an iterable, or None; `gesture` matches
    the resolved name.
  - `index_widar_bvp(...)` — same filters, returns metadata only (no array
    loads) for fast counting/splitting.
  - `parse_bvp_filename(path)`, `load_bvp_file(path)`, and the label tables
    (`GESTURE_MAP`, `ROOM_BY_DATE`, `POSITION_COORDS_M`, `ORIENTATION_DEG`).
  - Loads **only** — no padding, truncation, or extra normalization. It reorders
    the on-disk `(20, 20, T)` to `(T, 20, 20)` and casts to float32; nothing else.
- **`scripts/explore_widar.py`** — counts by gesture/user/position/orientation/
  room/date, BVP shape & value statistics (T min/max/mean, value range,
  per-frame energy sum), and a printout of the class names + per-date label
  conventions. Shape stats run over a random `--sample` of files (default 3000)
  for speed; `--sample 0` scans all.
- **`figures/widar_bvp_examples.png`** (`scripts/visualize_widar_bvp.py`) — 6
  gestures × 4 timesteps, each a 20×20 BVP heatmap (`v_x` →, `v_y` ↑, ±2 m/s).
  The velocity signatures are visibly distinct and trace each gesture's motion.
  **This is the first project visual where the WiFi data looks spatial.**
- **`notes/widar_data.md`** — BVP semantics and invariance, grounded in the paper
  and the shipped extraction code.

## Reproduce

```bash
conda activate wifisense

# 1. Download BVP (see README.md "Widar3.0 BVP" for the exact curl commands) into
#    data/raw/widar3/ and unzip BVP.zip to data/raw/widar3/bvp/.

# 2. Explore + figure
python scripts/explore_widar.py            # add --sample 0 to scan every file
python scripts/visualize_widar_bvp.py      # -> figures/widar_bvp_examples.png

# or via the orchestrator:
./run_pipeline.sh widar
```

## Notes / gotchas for later

- The README lists 7 empty/corrupt raw `.dat` files; those are a *CSI*-side issue
  and don't affect the BVP `.mat` set we use.
- BVP is a *derived* feature (heavy physics-based processing upstream), unlike the
  raw-CSI datasets. We consume the dataset's pre-computed BVP; we do not recompute
  it from CSI.
- The dataset card says "16 users"; the BVP release on disk actually contains
  **17** (user1–17). Reported here as observed.
