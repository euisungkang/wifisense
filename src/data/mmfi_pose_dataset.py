"""PyTorch Dataset + canonical MM-Fi splits for WiFi-CSI → 3D-pose regression.

This sits on top of ``mmfi_loader`` (the lazy, partial-download-safe view over
the MM-Fi corpus) and ``pose_preprocess`` (the pure CSI/keypoint transforms). It
turns the per-frame ``{csi, keypoints, ...}`` samples into model-ready
``(csi_window, keypoints_3d)`` pairs and exposes the three benchmark splits.

Why a window-index layer
------------------------
``mmfi_loader`` in ``data_unit='frame'`` mode yields one frame per sample, each
carrying its ``idx`` (0..296) within its ``(subject, action)`` clip. To build a
*centered temporal window* (``pose_preprocess.window_frame_indices``) we must
know each frame's neighbours **within the same clip and only that clip** — a
window may never straddle two clips, or CSI would be paired with the wrong pose.
So on construction we group the underlying samples into clips, sort each clip by
frame ``idx``, and build a flat list of ``(clip, center_position)`` window items.
``__getitem__`` then reads the ``window_size`` frames around the center
(edge-clamped at clip boundaries) and the pose **at the center frame**.

With the project's benchmark-faithful default ``window_size=1`` a "window" is a
single frame, exactly matching MM-Fi's published WiFi→pose protocol; larger odd
windows are supported for experiments (ask before deviating — see
``docs/chunk14_pose_pipeline.md``).

The three split builders
------------------------
    cross_subject(test_subjects=[...])      # leave-subjects-out — THE headline test
    cross_environment(test_envs=[...])      # leave-environments-out (E01..E04)
    random_split(test_ratio=..., seed=...)  # i.i.d. baseline, split by clip

Each returns ``(train_ds, test_ds)``. Splitting is done at the **clip** level
(never mid-clip) so temporally adjacent frames — and the frames inside a single
window — never leak across the train/test boundary. Cross-subject is the
generalization number that matters for WiFi pose: can the model pose a body it
never trained on?
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .mmfi_loader import load_mmfi
from .pose_preprocess import (
    ROOT_JOINT,
    csi_amplitude,
    denormalize_pose,
    normalize_csi,
    normalize_pose,
    window_frame_indices,
)

__all__ = [
    "MMFiPoseDataset",
    "cross_subject",
    "cross_environment",
    "random_split",
    "DEFAULT_TEST_SUBJECTS",
    "DEFAULT_TEST_ENVS",
]

# MM-Fi paper's cross-subject held-out subjects (config.yaml cross_subject_split
# val list): every 5th subject. Using these keeps cross-subject numbers
# comparable to published results.
DEFAULT_TEST_SUBJECTS = ["S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"]
# MM-Fi cross-scene protocol holds out environment E04 (config.yaml).
DEFAULT_TEST_ENVS = ["E04"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MMFiPoseDataset(Dataset):
    """Lazy ``(csi_window, keypoints_3d)`` pairs over a set of MM-Fi clips.

    Construct via the split helpers (``cross_subject`` / ``cross_environment`` /
    ``random_split``) rather than directly — they wire up a shared underlying
    ``MMFiSubset`` and hand each side its own list of clip keys.

    Args:
        subset: a frame-unit ``MMFiSubset`` (from ``load_mmfi(data_unit='frame')``).
            Shared between the train and test datasets of a split.
        clips: ordered ``{clip_key: [global_sample_index, ...]}`` where each list
            is the subset indices of that clip's frames, sorted by frame idx.
            Only the clips belonging to THIS partition are passed in.
        window_size: number of CSI frames per window (odd; center is the target
            frame). Default 1 = MM-Fi benchmark per-frame protocol.
        csi_normalize: ``"none"`` (default; keep the reader's [0,1] min-max),
            ``"zscore"`` or ``"minmax"`` — forwarded to ``normalize_csi``.
        pose_root: root joint to center poses on (pelvis = 0).
        pose_scale: ``None`` (metres, recommended), ``"rms"``, or a float — see
            ``pose_preprocess.normalize_pose``.
        cache: if True, memoize decoded CSI frames by global index so overlapping
            windows (window_size > 1) don't re-read the same .mat. Requires
            ``num_workers=0`` to share the cache. Harmless/unused at window_size=1.

    ``__getitem__`` returns ``(csi_window, keypoints)`` as float32 tensors of
    shape ``(window_size, 3, 114, 10)`` and ``(17, 3)``. The keypoints are the
    root-centered (optionally scaled) target. Use ``get_pair`` for an unpacked
    dict (raw CSI for display + absolute GT pose + the un-normalization offset).
    """

    def __init__(
        self,
        subset,
        clips: "dict[tuple, list[int]]",
        *,
        window_size: int = 1,
        csi_normalize: str = "none",
        pose_root: int = ROOT_JOINT,
        pose_scale: float | str | None = None,
        cache: bool = False,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1; got {window_size}")
        if window_size % 2 == 0:
            # Even windows have no true center frame; the target would be
            # off-center. Force odd so the pose sits in the middle of its window.
            raise ValueError(
                f"window_size must be odd so the target frame is centered; got {window_size}"
            )
        self.subset = subset
        self.clips = clips
        self.window_size = window_size
        self.csi_normalize = csi_normalize
        self.pose_root = pose_root
        self.pose_scale = pose_scale
        self.cache = cache
        self._frame_cache: dict[int, np.ndarray] = {}

        # Flatten clips into a list of (clip_key, position_in_clip) window items.
        # Every frame of every clip is a valid window center.
        self._items: list[tuple] = []
        for key, members in clips.items():
            for pos in range(len(members)):
                self._items.append((key, pos))

    def __len__(self) -> int:
        return len(self._items)

    @property
    def clip_keys(self) -> list[tuple]:
        """The ``(scene, subject, action)`` keys in this partition (sorted)."""
        return list(self.clips.keys())

    def _read_csi(self, global_index: int) -> np.ndarray:
        """Decode one CSI frame ``(3, 114, 10)`` float32, with optional cache."""
        if self.cache and global_index in self._frame_cache:
            return self._frame_cache[global_index]
        csi = csi_amplitude(self.subset[global_index]["csi"])  # (3, 114, 10)
        if self.cache:
            self._frame_cache[global_index] = csi
        return csi

    def _build(self, key: tuple, pos: int) -> dict:
        """Assemble the full record (arrays + provenance) for one window item.

        Centralizes the alignment logic used by both ``__getitem__`` and
        ``get_pair`` so they can never drift out of sync.
        """
        members = self.clips[key]  # global indices, sorted by frame idx
        n_frames = len(members)
        win_pos = window_frame_indices(pos, n_frames, self.window_size)

        # CSI window: stack the W frames on a new leading axis.
        frames = [self._read_csi(members[p]) for p in win_pos]
        csi_window = np.stack(frames, axis=0)  # (W, 3, 114, 10)
        csi_window = normalize_csi(csi_window, mode=self.csi_normalize)

        # Pose target: the pose AT THE CENTER FRAME (never an end of the window).
        center_global = members[pos]
        center_sample = self.subset[center_global]
        kp_abs = np.asarray(center_sample["keypoints"], dtype=np.float32)  # (17, 3)
        kp_norm, info = normalize_pose(
            kp_abs, root=self.pose_root, scale=self.pose_scale
        )

        return {
            "csi_window": csi_window,
            "keypoints": kp_norm,
            "keypoints_abs": kp_abs,
            "offset": info["offset"],
            "scale": info["scale"],
            "subject": center_sample["subject"],
            "scene": center_sample["scene"],
            "action": center_sample["action"],
            "center_idx": center_sample.get("idx"),
            "window_frame_idx": [
                self.subset[members[p]].get("idx") for p in win_pos
            ],
        }

    def __getitem__(self, i: int) -> "tuple[torch.Tensor, torch.Tensor]":
        key, pos = self._items[i]
        rec = self._build(key, pos)
        csi = torch.from_numpy(np.ascontiguousarray(rec["csi_window"], dtype=np.float32))
        kp = torch.from_numpy(np.ascontiguousarray(rec["keypoints"], dtype=np.float32))
        return csi, kp

    def get_pair(self, i: int) -> dict:
        """Full record for sample ``i`` — for verification / visualization.

        Returns a dict with the model input ``csi_window`` (W, 3, 114, 10) and
        normalized target ``keypoints`` (17, 3), PLUS the pieces needed to check
        alignment and draw the skeleton in metric space:
            ``keypoints_abs`` (absolute camera-frame pose, metres),
            ``offset`` / ``scale`` (the un-normalization recipe:
                ``absolute = keypoints * scale + offset``; see ``recover_absolute``),
            ``subject`` / ``scene`` / ``action`` / ``center_idx`` (provenance),
            ``window_frame_idx`` (the clip-frame index of each frame actually used
                — confirms the window is centered on ``center_idx`` and stays in-clip).
        """
        key, pos = self._items[i]
        return self._build(key, pos)

    @staticmethod
    def recover_absolute(keypoints, offset, scale: float = 1.0) -> np.ndarray:
        """Convenience wrapper around ``pose_preprocess.denormalize_pose``."""
        return denormalize_pose(keypoints, offset, scale)


# ---------------------------------------------------------------------------
# Clip grouping + split builders
# ---------------------------------------------------------------------------


def _group_clips(subset) -> "dict[tuple, list[int]]":
    """Group a frame-unit ``MMFiSubset`` into clips, sorted by frame idx.

    Returns an ordered ``{(scene, subject, action): [global_index, ...]}`` whose
    per-clip lists are sorted by the in-clip frame index, so list position ==
    temporal order (what ``window_frame_indices`` assumes). ``subset.metadata``
    is index-aligned with ``subset[i]``, so we can read keys cheaply (no arrays).
    """
    meta = subset.metadata
    groups: "dict[tuple, list[tuple[int, int]]]" = {}
    for gi, m in enumerate(meta):
        key = (m["scene"], m["subject"], m["action"])
        groups.setdefault(key, []).append((gi, m["idx"] if m["idx"] is not None else gi))
    clips: "dict[tuple, list[int]]" = {}
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda t: t[1])  # sort by frame idx
        clips[key] = [gi for gi, _ in members]
    return clips


def _load_all(protocol: str, data_root, limit: int | None):
    """Load every on-disk frame as one subset (split-agnostic), then we partition.

    Uses ``split='all'`` + ``random_split`` purely to materialize a view over all
    available subjects/scenes; the actual train/test partition is done by us at
    the clip level so the requested ``test_subjects`` / ``test_envs`` are honored
    exactly (independent of the official config's fixed lists).
    """
    return load_mmfi(
        modality="wifi-csi",
        split="all",
        protocol=protocol,
        split_strategy="random_split",
        data_unit="frame",
        data_root=data_root,
        limit=limit,
    )


def _split_by(clips, predicate):
    """Partition clip keys into (train, test) dicts by a key→bool predicate."""
    train = {k: v for k, v in clips.items() if not predicate(k)}
    test = {k: v for k, v in clips.items() if predicate(k)}
    return train, test


def _build_pair(train_clips, test_clips, subset, ds_kwargs, *, what: str):
    """Construct the (train_ds, test_ds) pair, failing loudly on an empty side."""
    if not train_clips:
        raise ValueError(
            f"{what}: empty TRAIN partition. Check the held-out set and that the "
            "remaining subjects/scenes are actually downloaded."
        )
    if not test_clips:
        raise ValueError(
            f"{what}: empty TEST partition — none of the held-out subjects/scenes "
            "are present on disk. With only E01 downloaded, e.g. "
            "cross_environment(test_envs=['E04']) has nothing to test on; download "
            "the held-out environment or choose held-out subjects within E01."
        )
    train_ds = MMFiPoseDataset(subset, train_clips, **ds_kwargs)
    test_ds = MMFiPoseDataset(subset, test_clips, **ds_kwargs)
    return train_ds, test_ds


def cross_subject(
    test_subjects: "list[str] | None" = None,
    *,
    protocol: str = "protocol3",
    data_root=None,
    limit: int | None = None,
    **ds_kwargs,
) -> "tuple[MMFiPoseDataset, MMFiPoseDataset]":
    """Leave-subjects-out split — the headline WiFi-pose generalization test.

    Train on every subject NOT in ``test_subjects``; test on those held-out
    subjects. The model is scored on bodies it never saw during training, which
    is the question that matters for non-intrusive sensing. Defaults to the
    MM-Fi paper's held-out set (``DEFAULT_TEST_SUBJECTS``) for comparability.

    ``ds_kwargs`` (window_size, csi_normalize, pose_root, pose_scale, cache) are
    forwarded identically to both the train and test ``MMFiPoseDataset``.
    """
    held = set(test_subjects if test_subjects is not None else DEFAULT_TEST_SUBJECTS)
    subset = _load_all(protocol, data_root, limit)
    clips = _group_clips(subset)
    train_clips, test_clips = _split_by(clips, lambda k: k[1] in held)  # k[1] = subject
    return _build_pair(
        train_clips, test_clips, subset, ds_kwargs, what="cross_subject"
    )


def cross_environment(
    test_envs: "list[str] | None" = None,
    *,
    protocol: str = "protocol3",
    data_root=None,
    limit: int | None = None,
    **ds_kwargs,
) -> "tuple[MMFiPoseDataset, MMFiPoseDataset]":
    """Leave-environments-out split: test on the held-out environments (E01..E04).

    Probes robustness to a new room/multipath layout — the environment changes,
    the subjects in it change too (each env is a distinct 10-subject group, see
    ``SCENE_OF_SUBJECT``). Defaults to holding out E04 (``DEFAULT_TEST_ENVS``),
    matching MM-Fi's cross-scene protocol. Needs more than one environment on
    disk; raises a clear error otherwise.
    """
    held = set(test_envs if test_envs is not None else DEFAULT_TEST_ENVS)
    subset = _load_all(protocol, data_root, limit)
    clips = _group_clips(subset)
    train_clips, test_clips = _split_by(clips, lambda k: k[0] in held)  # k[0] = scene
    return _build_pair(
        train_clips, test_clips, subset, ds_kwargs, what="cross_environment"
    )


def random_split(
    *,
    test_ratio: float = 0.2,
    seed: int = 0,
    protocol: str = "protocol3",
    data_root=None,
    limit: int | None = None,
    **ds_kwargs,
) -> "tuple[MMFiPoseDataset, MMFiPoseDataset]":
    """I.i.d. baseline split — the easy, no-domain-shift reference.

    Clips (NOT individual frames) are shuffled and ``test_ratio`` of them held
    out, so train and test come from the same subjects/environments. Splitting by
    clip — never mid-clip — prevents temporally adjacent frames (and the frames
    inside one window) from leaking across the boundary, which would inflate the
    score. The gap between this and ``cross_subject`` is the domain gap.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1); got {test_ratio}")
    subset = _load_all(protocol, data_root, limit)
    clips = _group_clips(subset)
    keys = list(clips.keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    n_test = max(1, int(round(len(keys) * test_ratio)))
    test_keys = {keys[i] for i in perm[:n_test]}
    train_clips = {k: v for k, v in clips.items() if k not in test_keys}
    test_clips = {k: v for k, v in clips.items() if k in test_keys}
    return _build_pair(
        train_clips, test_clips, subset, ds_kwargs, what="random_split"
    )
