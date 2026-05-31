#!/usr/bin/env python3
"""Characterize the Widar3.0 BVP dataset.

Prints, for the extracted BVP files under data/raw/widar3/bvp/:
  * sample counts by gesture / user / position / orientation / room / date,
  * BVP shape statistics (T = #timesteps: min/max/mean; value ranges; per-frame
    energy sum), computed over a random sample of files for speed,
  * the class names and the (date-dependent) label conventions.

Counts come from the fast filename index (no arrays loaded). The shape/value
statistics require opening files, so by default they are computed over a random
``--sample`` of files; pass ``--sample 0`` to scan every file (slow).

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/explore_widar.py [--sample N] [--seed S]

Runnable directly (not via ``-m``): it prepends the repo root to sys.path
itself so the ``src`` package imports resolve.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.widar_loader import (  # noqa: E402
    BVP_SAMPLE_RATE_HZ,
    GESTURE_MAP,
    ORIENTATION_DEG,
    POSITION_COORDS_M,
    ROOM_BY_DATE,
    VELOCITY_RANGE_MPS,
    index_widar_bvp,
    load_bvp_file,
)


def print_header(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def print_counts(index: list[dict], key: str, fmt=str) -> None:
    counts = Counter(fmt(m[key]) for m in index)
    print(f"\n  Counts by {key}:")
    total = sum(counts.values())
    for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"    {str(name):>22s}: {count:6d}  ({100.0 * count / total:5.1f}%)")
    print(f"    {'(distinct)':>22s}: {len(counts):6d}")


def describe_shapes(index: list[dict], n_sample: int, seed: int) -> None:
    """Open a (sampled) subset of files and report T and value statistics."""
    rng = np.random.default_rng(seed)
    if n_sample and n_sample < len(index):
        chosen = rng.choice(len(index), size=n_sample, replace=False)
        subset = [index[i] for i in chosen]
        scope = f"random sample of {n_sample} / {len(index)} files"
    else:
        subset = index
        scope = f"all {len(index)} files"

    Ts, gmin, gmax, frame_sums = [], np.inf, -np.inf, []
    spatial = set()
    for md in subset:
        arr = load_bvp_file(md["path"])  # (T, 20, 20)
        Ts.append(arr.shape[0])
        spatial.add(arr.shape[1:])
        gmin, gmax = min(gmin, float(arr.min())), max(gmax, float(arr.max()))
        frame_sums.append(arr.reshape(arr.shape[0], -1).sum(axis=1))
    Ts = np.array(Ts)
    fs = np.concatenate(frame_sums)

    print(f"\n  BVP shape statistics ({scope}):")
    print(f"    spatial dims (per timestep): {sorted(spatial)}")
    print(f"    T (timesteps): min={Ts.min()}  max={Ts.max()}  "
          f"mean={Ts.mean():.1f}  median={int(np.median(Ts))}")
    print(f"    duration @ {BVP_SAMPLE_RATE_HZ}Hz: "
          f"{Ts.min() / BVP_SAMPLE_RATE_HZ:.1f}s – "
          f"{Ts.max() / BVP_SAMPLE_RATE_HZ:.1f}s")
    print(f"    value range: [{gmin:.4g}, {gmax:.4g}]  (non-negative energy)")
    print(f"    per-frame energy sum: min={fs.min():.3f}  max={fs.max():.3f}  "
          f"mean={fs.mean():.3f}  (~1 => each frame L1-normalized)")


def print_conventions() -> None:
    print_header("Class names & label conventions")
    print(f"""
  Each BVP file is a 20x20xT volume named <userN>-a-b-c-d-<suffix>.mat:
    a = gesture id     b = torso location (position 1-8)
    c = face orient.   d = repetition
  Velocity grid: 20x20, both axes span {VELOCITY_RANGE_MPS} m/s; T @ {BVP_SAMPLE_RATE_HZ}Hz.

  IMPORTANT: gesture id 'a' is NOT global — its meaning depends on the
  collection date (and, for 3 dates, the user). The loader resolves the
  human-readable name per (date, user, id). Per-date gesture tables:""")
    for date in sorted(GESTURE_MAP):
        entry = GESTURE_MAP[date]
        room = ROOM_BY_DATE.get(date, "?")
        if any(isinstance(v, dict) for v in entry.values()):
            print(f"\n    {date} (room {room}) — per-user:")
            for u, m in sorted(entry.items()):
                pairs = ", ".join(f"{k}:{v}" for k, v in sorted(m.items()))
                print(f"      user{u}: {pairs}")
        else:
            pairs = ", ".join(f"{k}:{v}" for k, v in sorted(entry.items()))
            print(f"\n    {date} (room {room}): {pairs}")

    print("\n  Torso locations (id: x,y metres; Tx at origin):")
    for pid, (x, y) in sorted(POSITION_COORDS_M.items()):
        print(f"    {pid}: ({x}, {y})")
    print("\n  Face orientations (id: degrees; 0 = facing Tx):")
    print("    " + ", ".join(f"{k}:{v}°" for k, v in sorted(ORIENTATION_DEG.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=3000,
                    help="files to open for shape/value stats (0 = all; slow)")
    ap.add_argument("--seed", type=int, default=42, help="sampling RNG seed")
    args = ap.parse_args()

    print_header("Widar3.0 BVP Dataset")
    index = index_widar_bvp()
    print(f"\n  Total BVP samples: {len(index)}")

    print_counts(index, "gesture")
    print_counts(index, "user", fmt=lambda u: f"user{u}")
    print_counts(index, "position")
    print_counts(index, "orientation",
                 fmt=lambda o: f"{o} ({ORIENTATION_DEG.get(o, '?')}°)")
    print_counts(index, "room")
    print_counts(index, "date")

    print_header("BVP shape / value statistics")
    describe_shapes(index, args.sample, args.seed)

    print_conventions()

    print_header("Done")


if __name__ == "__main__":
    main()
