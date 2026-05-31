"""Loader for the Widar3.0 Body-coordinate Velocity Profile (BVP) dataset.

Widar3.0 (Tsinghua, MobiSys '19) is a WiFi gesture dataset captured with
1 transmitter + 6 Intel 5300 receivers, designed for cross-domain evaluation.
Its signature representation is the **BVP**: a 20x20xT volume where each
timestep is a 20x20 grid of motion energy over a 2-D body-frame velocity space
(x-velocity on one axis, y-velocity on the other). BVP is far more
environment-invariant than raw CSI because the room geometry and link
configuration have been factored out. See `notes/widar_data.md` for the physics.

This module loads BVP **only** — no preprocessing (no padding/truncating T, no
normalization beyond what the dataset already applies). Each on-disk file is a
MATLAB struct with one variable, ``velocity_spectrum_ro``, shape (20, 20, T),
float64, non-negative, each timeslice ~L1-normalized.

On-disk layout (from BVP.zip, see data/raw/widar3/README.pdf)::

    data/raw/widar3/bvp/BVP/<date>-VS/[6-link/]<userN>/<userN>-a-b-c-d-<suffix>.mat

where ``a`` = gesture id, ``b`` = torso location (position 1-8), ``c`` = face
orientation (1-5), ``d`` = repetition. Most dates nest users under ``6-link/``;
the 20181130 date places ``userN/`` directly under the date folder. The loader
handles both via a recursive glob.

**The crucial gotcha: gesture ids are NOT globally consistent.** Widar3.0 was
collected over 14 days and the meaning of gesture id ``a`` depends on the
collection date (and, for three dates, on the user). For example id ``4`` is
"Slide" on 2018-11-09 but "Draw-O" on 2018-11-15. The README's per-date tables
are transcribed into GESTURE_MAP below, and every sample's metadata carries the
resolved human-readable ``gesture`` name. Filter by name, not by raw id.

Public API:
    parse_bvp_filename(path)  -> metadata dict for one file (no array load)
    index_widar_bvp(...)      -> list[metadata dict], fast, no array loads
    load_bvp_file(path)       -> np.ndarray (T, 20, 20)
    load_widar_bvp(user=None, gesture=None, position=None, orientation=None,
                   date=None, room=None, limit=None) -> (X, metadata)
        X        : list of np.ndarray, each (T, 20, 20); T varies per sample.
        metadata : list of dicts aligned with X (one per sample).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WIDAR_BVP_DIR = PROJECT_ROOT / "data" / "raw" / "widar3" / "bvp" / "BVP"

# The MATLAB variable holding the 20x20xT volume in every BVP .mat file.
BVP_VAR = "velocity_spectrum_ro"

# Physical extent of the velocity grid, from data/raw/widar3/README.pdf:
# both axes span [-2, +2] m/s across 20 bins; timesteps are sampled at 10 Hz.
VELOCITY_RANGE_MPS = (-2.0, 2.0)
GRID_SIZE = 20
BVP_SAMPLE_RATE_HZ = 10

# Filename: <userN>-<a>-<b>-<c>-<d>-<suffix>.mat
#   a = gesture id, b = torso location, c = face orientation, d = repetition.
_FNAME_RE = re.compile(r"^user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-.*\.mat$")

# ---------------------------------------------------------------------------
# Label conventions (transcribed verbatim from the Widar3.0 README tables).
#
# gesture id -> name, keyed by collection date. Three dates differ per user, so
# their values are nested {user_id: {gesture_id: name}}. Names follow the
# README's "Gesture Description" section; (H)/(V) = horizontal/vertical plane.
# ---------------------------------------------------------------------------
_SIX = {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
        5: "Draw-Zigzag(V)", 6: "Draw-N(V)"}
_DIGITS = {1: "Draw-1", 2: "Draw-2", 3: "Draw-3", 4: "Draw-4", 5: "Draw-5",
           6: "Draw-6", 7: "Draw-7", 8: "Draw-8", 9: "Draw-9", 10: "Draw-0"}
_DRAW_O_V = {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Draw-O(V)",
             5: "Draw-Zigzag(V)", 6: "Draw-N(V)"}
_HORIZ6 = {1: "Slide", 2: "Draw-O(H)", 3: "Draw-Zigzag(H)", 4: "Draw-N(H)",
           5: "Draw-Triangle(H)", 6: "Draw-Rectangle(H)"}
_PSC_DRAW_NH = {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Draw-O(H)",
                5: "Draw-Zigzag(H)", 6: "Draw-N(H)"}
_NINE = {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide", 5: "Draw-O(H)",
         6: "Draw-Zigzag(H)", 7: "Draw-N(H)", 8: "Draw-Triangle(H)",
         9: "Draw-Rectangle(H)"}

GESTURE_MAP: dict[str, dict] = {
    "20181109": _SIX,
    "20181112": _DIGITS,
    "20181115": _DRAW_O_V,
    "20181117": _DRAW_O_V,
    "20181118": _DRAW_O_V,
    "20181121": _HORIZ6,
    "20181127": _HORIZ6,
    "20181128": _PSC_DRAW_NH,
    "20181130": _NINE,
    "20181204": _NINE,
    # per-user differences below
    "20181205": {
        2: {1: "Draw-O(H)", 2: "Draw-Zigzag(H)", 3: "Draw-N(H)",
            4: "Draw-Triangle(H)", 5: "Draw-Rectangle(H)"},
        3: _HORIZ6,
    },
    "20181208": {
        2: {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide"},
        3: {1: "Push&Pull", 2: "Sweep", 3: "Clap"},
    },
    "20181209": {
        2: {1: "Push&Pull"},
        6: {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
            5: "Draw-O(H)", 6: "Draw-Zigzag(H)"},
    },
    "20181211": {1: "Push&Pull", 2: "Sweep", 3: "Clap", 4: "Slide",
                 5: "Draw-O(H)", 6: "Draw-Zigzag(H)"},
}

# Capture room per date (README "Floor Plan"): 1=Classroom, 2=Hall, 3=Office.
ROOM_BY_DATE: dict[str, int] = {
    "20181109": 1, "20181112": 1, "20181115": 1, "20181121": 1, "20181130": 1,
    "20181117": 2, "20181118": 2, "20181127": 2, "20181128": 2, "20181204": 2,
    "20181205": 2, "20181208": 2, "20181209": 2,
    "20181211": 3,
}

# Torso-location id -> (x, y) metres, and face-orientation id -> degrees,
# from the README "Device Deployment" table. Tx is at (0, 0); orientation 3
# (facing the Tx, 0 deg) is the reference.
POSITION_COORDS_M: dict[int, tuple[float, float]] = {
    1: (1.365, 0.455), 2: (0.455, 0.455), 3: (0.455, 1.365),
    4: (1.365, 1.365), 5: (0.910, 0.910), 6: (2.275, 1.365),
    7: (2.275, 2.275), 8: (1.365, 2.275),
}
ORIENTATION_DEG: dict[int, int] = {1: -90, 2: -45, 3: 0, 4: 45, 5: 90}


def gesture_name(date: str, user: int, gesture_id: int) -> str:
    """Resolve a gesture id to its human-readable name for a given date/user.

    Returns ``"unknown(<date>:<id>)"`` if the (date, user, id) combination is
    not covered by the README tables, rather than raising — a few stray ids
    exist on disk and we prefer to surface them in metadata over crashing.
    """
    entry = GESTURE_MAP.get(date)
    if entry is None:
        return f"unknown({date}:{gesture_id})"
    # Per-user dates: keys are user ids (ints); otherwise keys are gesture ids.
    if entry and all(isinstance(k, int) for k in entry) and \
            any(isinstance(v, dict) for v in entry.values()):
        entry = entry.get(user, {})
    name = entry.get(gesture_id)
    return name if name is not None else f"unknown({date}:{gesture_id})"


def parse_bvp_filename(path: Path | str) -> dict:
    """Parse one BVP file path into a metadata dict (no array is loaded).

    Returns keys: ``path`` (str), ``date`` (str, YYYYMMDD), ``room`` (int),
    ``user`` (int), ``gesture_id`` (int), ``gesture`` (str, resolved name),
    ``position`` (int, torso location 1-8), ``orientation`` (int, 1-5),
    ``repetition`` (int).

    Raises ValueError if the filename does not match the expected pattern.
    """
    path = Path(path)
    m = _FNAME_RE.match(path.name)
    if m is None:
        raise ValueError(f"unexpected BVP filename: {path.name}")
    user, gid, pos, ori, rep = (int(g) for g in m.groups())

    # Date is the "<date>-VS" ancestor directory (e.g. "20181109-VS").
    date = next(
        (p.name[:-3] for p in path.parents if p.name.endswith("-VS")),
        "unknown",
    )
    return {
        "path": str(path),
        "date": date,
        "room": ROOM_BY_DATE.get(date, -1),
        "user": user,
        "gesture_id": gid,
        "gesture": gesture_name(date, user, gid),
        "position": pos,
        "orientation": ori,
        "repetition": rep,
    }


def _as_set(value) -> set | None:
    """Normalize a filter argument (scalar, iterable, or None) to a set/None."""
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return {value}
    return set(value)


def index_widar_bvp(
    user=None,
    gesture=None,
    position=None,
    orientation=None,
    date=None,
    room=None,
) -> list[dict]:
    """Index BVP files matching the given filters WITHOUT loading any arrays.

    Each filter accepts a scalar, an iterable of values, or None (no filter).
    ``gesture`` matches the resolved human-readable name (e.g. "Push&Pull"),
    not the raw id. Returns a list of metadata dicts (see parse_bvp_filename),
    sorted by file path for determinism.
    """
    if not WIDAR_BVP_DIR.exists():
        raise FileNotFoundError(
            f"Widar3.0 BVP not found at {WIDAR_BVP_DIR}. "
            "Download BVP.zip and extract it there (see README.md)."
        )
    f_user, f_ges = _as_set(user), _as_set(gesture)
    f_pos, f_ori = _as_set(position), _as_set(orientation)
    f_date, f_room = _as_set(date), _as_set(room)

    out: list[dict] = []
    for p in sorted(WIDAR_BVP_DIR.rglob("*.mat")):
        try:
            md = parse_bvp_filename(p)
        except ValueError:
            continue
        if f_user is not None and md["user"] not in f_user:
            continue
        if f_ges is not None and md["gesture"] not in f_ges:
            continue
        if f_pos is not None and md["position"] not in f_pos:
            continue
        if f_ori is not None and md["orientation"] not in f_ori:
            continue
        if f_date is not None and md["date"] not in f_date:
            continue
        if f_room is not None and md["room"] not in f_room:
            continue
        out.append(md)
    return out


def load_bvp_file(path: Path | str) -> np.ndarray:
    """Load one BVP .mat file as a float32 array of shape (T, 20, 20).

    On disk the volume is (20, 20, T) = (vx, vy, time); this transposes it to
    (time, vx, vy) so the leading axis is time and each (20, 20) slice is a
    single-timestep velocity grid (x-velocity along axis 0, y along axis 1).
    """
    vol = sio.loadmat(str(path))[BVP_VAR]  # (20, 20, T)
    return np.ascontiguousarray(np.transpose(vol, (2, 0, 1)), dtype=np.float32)


def load_widar_bvp(
    user=None,
    gesture=None,
    position=None,
    orientation=None,
    date=None,
    room=None,
    limit=None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Load Widar3.0 BVP samples matching the given filters.

    Args:
        user/gesture/position/orientation/date/room: filters; each accepts a
            scalar, an iterable, or None (no filter). ``gesture`` matches the
            resolved name (e.g. "Sweep"), ``position`` is the torso-location id
            (1-8), ``orientation`` the face-orientation id (1-5).
        limit: if set, return at most this many samples (after filtering, in
            sorted path order) — handy for quick development loads.

    Returns:
        X: list of float32 arrays, each shape (T, 20, 20). T varies per sample
           (gesture duration at 10 Hz), so this is a list, not a stacked tensor.
        metadata: list of dicts aligned 1:1 with X (see parse_bvp_filename),
           giving user/gesture/position/orientation/etc. for cross-domain splits.
    """
    index = index_widar_bvp(
        user=user, gesture=gesture, position=position,
        orientation=orientation, date=date, room=room,
    )
    if limit is not None:
        index = index[:limit]
    X = [load_bvp_file(md["path"]) for md in index]
    return X, index
