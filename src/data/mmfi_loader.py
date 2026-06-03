"""Loader for the MM-Fi WiFi-CSI → 3D-pose subset (Phase 3: pose estimation).

MM-Fi (Yang et al., NeurIPS 2023 Datasets & Benchmarks) is the first multi-modal
non-intrusive 4D human pose dataset: ~320k synchronized frames, 40 subjects, 4
environments (E01–E04), 27 actions, five modalities. **This project uses only
two of them**: WiFi CSI as the *input* and the 3D pose keypoints as the
*regression target*. RGB / depth / LiDAR / mmWave are ignored except for sanity
checks.

This is a NEW task type for the project. Chunks 1–12 were CLASSIFICATION
(activity / gesture labels). Pose estimation is supervised REGRESSION from WiFi
CSI to continuous 3D joint coordinates — there is no class head; the loss is a
distance between predicted and ground-truth joint positions (see MPJPE in
``vendor/MMFi/mmfi_lib/evaluate.py``).

Why a wrapper rather than a fresh parser
----------------------------------------
The official toolbox (``vendor/MMFi/mmfi_lib/mmfi.py``) already handles the
directory walk, the per-modality file decoding, the NaN/inf cleanup of CSI, the
protocol/split logic, and the train/val partitioning used in the paper. We reuse
it verbatim and only:
  * stub out ``cv2`` (the official module imports OpenCV at top level for the
    depth/RGB code paths we never touch — see ``_install_cv2_stub``),
  * prune the train/val "data_form" down to the subjects actually present on
    disk, so a partial download (e.g. ONLY E01) works without the official
    loader crashing on ``os.path.getsize`` of missing files,
  * normalize each sample's keys to ``{'csi', 'keypoints', 'subject', 'scene',
    'action', 'idx'}`` so downstream code does not carry the awkward
    ``'input_wifi-csi'`` / ``'output'`` names.

Data shapes (per the MM-Fi paper, §3 and confirmed against the loader)
----------------------------------------------------------------------
* CSI tensor, ``data_unit='frame'``:    ``(3, 114, 10)``  float64 in [0, 1]
      3 receiver antennas × 114 subcarriers (5 GHz, 40 MHz) × 10 packets / 100 ms.
      Amplitude only (``CSIamp``); NaN/inf already imputed and the frame is
      min-max normalized by the official reader.
* CSI tensor, ``data_unit='sequence'``: ``(297, 3, 114, 10)`` — 297 frames/action.
* Keypoints, ``data_unit='frame'``:     ``(17, 3)``  float32, metres, camera frame.
      17 joints in the Human3.6M ordering (see ``src/viz/skeleton.py`` for the
      joint-index table and kinematic tree).
* Keypoints, ``data_unit='sequence'``:  ``(297, 17, 3)``.

On-disk layout expected under ``data/raw/mmfi/`` (the ``DATASET_ROOT``)::

    data/raw/mmfi/
    |-- E01/
    |   |-- S01/
    |   |   |-- A01/
    |   |   |   |-- wifi-csi/  frame001.mat ... frame297.mat   (CSIamp)
    |   |   |   |-- ground_truth.npy        (297, 17, 3)
    |   |   |   |-- (rgb/ mmwave/ ... — ignored)
    |   |   |-- A02/ ... A27/
    |   |-- S02/ ... S10/
    |-- E02/ ... E04/                       (optional; download as needed)

Subject → environment mapping (fixed by the dataset):
    E01 = S01–S10,  E02 = S11–S20,  E03 = S21–S30,  E04 = S31–S40.

See ``docs/chunk13_mmfi_setup.md`` for download / subset instructions and the
license/citation. The dataset is NOT downloaded by ``run_pipeline.sh``; this
loader raises a helpful error if it is missing.

Public API
----------
    load_mmfi(modality='wifi-csi', split='train', ...) -> MMFiSubset
        A lazy, indexable view over the requested partition. Iterating / indexing
        yields normalized sample dicts (arrays loaded on demand).
    MMFiSubset.metadata -> list[dict]   # subject/scene/action/idx, NO array load
    available_scenes(data_root=None) -> list[str]   # which Exx are on disk
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MMFI_ROOT = PROJECT_ROOT / "data" / "raw" / "mmfi"
VENDOR_DIR = PROJECT_ROOT / "vendor" / "MMFi"
SETUP_DOC = "docs/chunk13_mmfi_setup.md"

# Dataset facts (from the MM-Fi paper / official loader).
CSI_FRAME_SHAPE = (3, 114, 10)   # antennas x subcarriers x packets, per 100 ms
NUM_KEYPOINTS = 17               # Human3.6M ordering; tree in src/viz/skeleton.py
KEYPOINT_DIM = 3                 # (x, y, z) metres, camera frame
FRAMES_PER_SEQUENCE = 297        # frames per (subject, action) clip

# Subject -> environment, fixed by the dataset collection.
SCENE_OF_SUBJECT = {
    **{f"S{i:02d}": "E01" for i in range(1, 11)},
    **{f"S{i:02d}": "E02" for i in range(11, 21)},
    **{f"S{i:02d}": "E03" for i in range(21, 31)},
    **{f"S{i:02d}": "E04" for i in range(31, 41)},
}

_VALID_STRATEGIES = (
    "random_split",
    "cross_scene_split",
    "cross_subject_split",
    "manual_split",
)


def _install_cv2_stub() -> None:
    """Register a dummy ``cv2`` module so the official loader imports cleanly.

    ``vendor/MMFi/mmfi_lib/mmfi.py`` does ``import cv2`` at module top for the
    depth/RGB readers. We only ever request ``modality='wifi-csi'``, so those
    code paths never run. Rather than force every user to install OpenCV just to
    read CSI, we install a stub that imports fine but raises a clear error if any
    cv2 attribute is actually used (which would only happen for an image
    modality we explicitly do not support here).
    """
    if "cv2" in sys.modules:
        return

    class _Cv2Stub(types.ModuleType):
        def __getattr__(self, name):  # pragma: no cover - defensive
            raise RuntimeError(
                f"cv2.{name} was called, but OpenCV is intentionally stubbed in "
                "mmfi_loader (we only use the 'wifi-csi' modality). Install "
                "opencv-python if you need an image modality."
            )

    sys.modules["cv2"] = _Cv2Stub("cv2")


def _import_official():
    """Import the vendored MM-Fi toolbox, returning its useful symbols."""
    _install_cv2_stub()
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    try:
        from mmfi_lib.mmfi import MMFi_Database, MMFi_Dataset, decode_config
    except ImportError as e:  # pragma: no cover - setup error
        raise ImportError(
            f"Could not import the MM-Fi toolbox from {VENDOR_DIR}. Clone it with:\n"
            "  git clone https://github.com/ybhbingo/MMFi_dataset.git vendor/MMFi\n"
            f"(original error: {e})"
        ) from e
    return MMFi_Database, MMFi_Dataset, decode_config


def _check_data_present(data_root: Path) -> None:
    """Fail loudly, with a pointer to the setup doc, if the data is missing."""
    if not data_root.exists():
        raise FileNotFoundError(
            f"MM-Fi data not found at {data_root}.\n"
            f"This dataset is NOT downloaded by run_pipeline.sh. See {SETUP_DOC} "
            "for the Google Drive link and subset instructions. Start with ONE "
            "environment (E01) extracted to data/raw/mmfi/E01/."
        )
    scenes = [p.name for p in sorted(data_root.iterdir())
              if p.is_dir() and p.name.startswith("E")]
    if not scenes:
        raise FileNotFoundError(
            f"MM-Fi root {data_root} exists but contains no environment folders "
            f"(expected E01 ... E04). See {SETUP_DOC}."
        )


def _build_config(modality, protocol, split_strategy, data_unit, ratio, random_seed):
    """Assemble the dict the official ``decode_config`` expects.

    We start from the vendored ``config.yaml`` (it carries the paper's exact
    cross-subject / manual subject lists) and override the knobs we expose.
    Falls back to a minimal embedded config if the yaml is unavailable.
    """
    base: dict = {}
    cfg_path = VENDOR_DIR / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path) as fd:
                base = yaml.safe_load(fd) or {}
        except Exception:
            base = {}
    base.update({
        "modality": modality,
        "protocol": protocol,
        "data_unit": data_unit,
        "split_to_use": split_strategy,
        "init_rand_seed": random_seed,
    })
    base.setdefault("random_split", {})
    base["random_split"]["ratio"] = ratio
    base["random_split"]["random_seed"] = random_seed
    return base


def available_scenes(data_root: str | Path | None = None) -> list[str]:
    """Return the environment folders (E01 ...) actually present on disk."""
    data_root = Path(data_root) if data_root else MMFI_ROOT
    if not data_root.exists():
        return []
    return [p.name for p in sorted(data_root.iterdir())
            if p.is_dir() and p.name.startswith("E")]


class MMFiSubset:
    """A lazy, indexable view over one MM-Fi partition (train, val, or all).

    Wraps the official ``MMFi_Dataset`` and normalizes each yielded sample to::

        {
            'csi':       np.ndarray,  # (3, 114, 10) frame / (297, 3, 114, 10) seq
            'keypoints': np.ndarray,  # (17, 3) frame / (297, 17, 3) seq, float32
            'subject':   str,         # e.g. 'S01'
            'scene':     str,         # e.g. 'E01'
            'action':    str,         # e.g. 'A03'
            'idx':       int | None,  # frame index within the clip (frame unit)
        }

    Arrays are read from disk on ``__getitem__`` — nothing is materialized up
    front, so this is safe even on the full ~320k-frame corpus. Use ``.metadata``
    for a cheap (no array load) census of what the subset contains.
    """

    def __init__(self, datasets, modality: str, split: str):
        # ``datasets`` is a list of underlying MMFi_Dataset objects to chain.
        self._datasets = [d for d in datasets if len(d) > 0]
        self._modality = modality
        self.split = split
        # Cumulative lengths for index routing across the chained datasets.
        self._cum = []
        total = 0
        for d in self._datasets:
            total += len(d)
            self._cum.append(total)
        self._len = total

    def __len__(self) -> int:
        return self._len

    def _route(self, i: int):
        if i < 0:
            i += self._len
        if not 0 <= i < self._len:
            raise IndexError(i)
        for d_idx, end in enumerate(self._cum):
            if i < end:
                start = self._cum[d_idx - 1] if d_idx else 0
                return self._datasets[d_idx], i - start
        raise IndexError(i)  # pragma: no cover

    @staticmethod
    def _normalize(sample: dict, modality: str) -> dict:
        kp = sample["output"]
        kp = kp.numpy() if hasattr(kp, "numpy") else np.asarray(kp)
        out = {
            "csi": np.asarray(sample[f"input_{modality}"]),
            "keypoints": kp.astype(np.float32),
            "subject": sample["subject"],
            "scene": sample["scene"],
            "action": sample["action"],
            "idx": sample.get("idx"),
        }
        return out

    def __getitem__(self, i: int) -> dict:
        ds, local = self._route(i)
        return self._normalize(ds[local], self._modality)

    def __iter__(self):
        for i in range(self._len):
            yield self[i]

    @property
    def metadata(self) -> list[dict]:
        """Per-sample metadata with NO array loads (subject/scene/action/idx)."""
        meta = []
        for d in self._datasets:
            for item in d.data_list:
                meta.append({
                    "subject": item["subject"],
                    "scene": item["scene"],
                    "action": item["action"],
                    "idx": item.get("idx"),
                })
        return meta


def load_mmfi(
    modality: str = "wifi-csi",
    split: str = "train",
    *,
    protocol: str = "protocol3",
    split_strategy: str = "random_split",
    data_unit: str = "frame",
    data_root: str | Path | None = None,
    ratio: float = 0.8,
    random_seed: int = 0,
    limit: int | None = None,
) -> MMFiSubset:
    """Load an MM-Fi partition exposing paired CSI inputs and 3D-pose targets.

    Parameters
    ----------
    modality : str
        Input modality. Defaults to ``'wifi-csi'`` — the only one this project
        uses. (The official loader also supports rgb/depth/lidar/mmwave, but
        those need OpenCV/etc. and are out of scope here.)
    split : {'train', 'val', 'all'}
        Which partition to return. ``'all'`` chains train + val.
    protocol : {'protocol1', 'protocol2', 'protocol3'}
        Action subset: 1 = daily, 2 = rehabilitation, 3 = all 27 actions.
    split_strategy : {'random_split', 'cross_scene_split', 'cross_subject_split',
                      'manual_split'}
        How subjects/actions are partitioned into train vs. val — the three
        paper protocols plus manual. Cross-domain generalization (the point of
        WiFi sensing) is exercised by ``cross_scene_split`` / ``cross_subject_split``.
    data_unit : {'frame', 'sequence'}
        ``'frame'`` → one (CSI, pose) pair per frame; ``'sequence'`` → a whole
        297-frame clip per sample. See module docstring for resulting shapes.
    data_root : path, optional
        Dataset root (the folder holding E01 ...). Defaults to
        ``data/raw/mmfi/``.
    ratio, random_seed : float, int
        Train fraction and seed for ``random_split`` (ignored otherwise).
    limit : int, optional
        If set, truncate the returned subset to at most this many samples
        (handy for smoke tests on a partial download).

    Returns
    -------
    MMFiSubset
        Lazy, indexable view; ``len()`` and ``[i]`` work, ``[i]`` returns a
        normalized sample dict. See ``MMFiSubset`` for the dict schema.

    Notes
    -----
    Robust to PARTIAL downloads: the train/val "data_form" produced by the
    official ``decode_config`` (which always spans all 40 subjects) is pruned to
    the subjects actually present on disk, so having only E01 extracted is fine.
    """
    if split not in ("train", "val", "all"):
        raise ValueError(f"split must be 'train', 'val', or 'all'; got {split!r}")
    if split_strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"split_strategy must be one of {_VALID_STRATEGIES}; got {split_strategy!r}"
        )

    data_root = Path(data_root) if data_root else MMFI_ROOT
    _check_data_present(data_root)

    MMFi_Database, MMFi_Dataset, decode_config = _import_official()
    database = MMFi_Database(str(data_root))
    available = set(database.subjects.keys())

    config = _build_config(
        modality, protocol, split_strategy, data_unit, ratio, random_seed
    )
    dataset_config = decode_config(config)

    def _prune(form: dict) -> dict:
        """Drop subjects/actions not present on disk (partial-download safety).

        ``decode_config`` always spans all 40 subjects × the protocol's actions,
        but the official frame-mode loader stats every file and crashes on a
        missing one. We intersect with what is actually on disk: subjects from
        ``database.subjects``, and each subject's actions from its own folder.
        """
        out = {}
        for subj, acts in form.items():
            if subj not in available:
                continue
            on_disk = set(database.subjects[subj].keys())
            keep = [a for a in acts if a in on_disk]
            if keep:
                out[subj] = keep
        return out

    parts = ["train", "val"] if split == "all" else [
        "train" if split == "train" else "val"
    ]
    datasets = []
    for p in parts:
        pruned = _prune(dataset_config[f"{p}_dataset"]["data_form"])
        if pruned:
            datasets.append(MMFi_Dataset(database, data_unit, modality, p, pruned))

    if not datasets:
        raise RuntimeError(
            f"No samples for split={split!r} with the subjects present on disk "
            f"({sorted(available)}). With only E01 downloaded, the "
            f"{split_strategy!r} strategy may place all on-disk subjects in the "
            "other partition — try split='all' or a different split_strategy. "
            f"See {SETUP_DOC}."
        )

    subset = MMFiSubset(datasets, modality, split)

    if limit is not None and limit < len(subset):
        # Cheaply truncate by trimming the underlying data_lists.
        remaining = limit
        kept = []
        for d in subset._datasets:
            if remaining <= 0:
                break
            if len(d) <= remaining:
                kept.append(d)
                remaining -= len(d)
            else:
                d.data_list = d.data_list[:remaining]
                kept.append(d)
                remaining = 0
        subset = MMFiSubset(kept, modality, split)

    return subset
