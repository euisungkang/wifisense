"""PyTorch Dataset and cross-domain split helpers for Widar3.0 BVP.

This sits on top of ``widar_loader`` (which only indexes/loads raw .mat volumes)
and ``bvp_preprocess`` (the pure transforms). ``WidarBVPDataset`` lazily loads
one .mat per ``__getitem__`` — the corpus is ~44k samples, far too large to hold
in memory — and applies, in order:

    augment_bvp (train only, raw energy space) → normalize_bvp → pad_or_truncate

yielding a fixed-shape ``(target_T, 20, 20)`` float32 tensor plus an integer
gesture label.

The reason this module exists separately from a plain Dataset is the **Widar3.0
evaluation protocol**: the dataset's whole purpose is *cross-domain*
generalization, so we expose four canonical split builders —

    cross_user(test_users=...)             leave-users-out
    cross_position(test_positions=...)     leave-torso-locations-out
    cross_orientation(test_orientations=...) leave-face-orientations-out
    in_domain(test_frac=...)               i.i.d. random split (the easy baseline)

Each returns ``(train_ds, test_ds)`` sharing one label map, with augmentation on
for train and forced off for test. Reporting in-domain alongside one or more
cross-domain numbers is how you quantify the domain gap BVP is meant to close.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .bvp_preprocess import augment_bvp, normalize_bvp, pad_or_truncate
from .widar_loader import _as_set, index_widar_bvp, load_bvp_file

__all__ = [
    "WidarBVPDataset",
    "build_label_map",
    "compute_global_stats",
    "cross_user",
    "cross_position",
    "cross_orientation",
    "in_domain",
]


# ---------------------------------------------------------------------------
# Label encoding and global normalization statistics
# ---------------------------------------------------------------------------


def build_label_map(metadata: list[dict]) -> dict[str, int]:
    """Map the gesture names present in *metadata* to contiguous int labels.

    Sorted by name for determinism. Built from the union of train+test metadata
    by the split helpers so both splits encode the same gesture to the same id.
    """
    names = sorted({m["gesture"] for m in metadata})
    return {name: i for i, name in enumerate(names)}


def compute_global_stats(
    metadata: list[dict],
    max_samples: int | None = 500,
    seed: int = 0,
) -> dict:
    """Estimate global mean/std of raw BVP cell values for ``mode='global'``.

    Streams over the (optionally subsampled) samples accumulating sum and sum of
    squares across every cell of every frame, so memory stays flat regardless of
    how many files are scanned.

    Args:
        metadata: samples to estimate from (typically the *training* split).
        max_samples: cap the number of files loaded (a random subset); None
            uses all. 500 is plenty for a stable two-moment estimate.
        seed: controls which subset is drawn.

    Returns:
        dict with float ``"mean"`` and ``"std"``.
    """
    if not metadata:
        raise ValueError("cannot compute global stats from empty metadata")
    md = metadata
    if max_samples is not None and len(md) > max_samples:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(md), size=max_samples, replace=False)
        md = [metadata[i] for i in sel]

    total = 0
    s = 0.0
    ss = 0.0
    for m in md:
        try:
            x = load_bvp_file(m["path"]).astype(np.float64)
        except Exception:
            continue  # skip truncated/corrupt files (see WidarBVPDataset._try_load)
        if x.size == 0:
            continue
        total += x.size
        s += x.sum()
        ss += np.square(x).sum()
    mean = s / total
    var = max(ss / total - mean * mean, 0.0)
    return {"mean": float(mean), "std": float(np.sqrt(var))}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WidarBVPDataset(Dataset):
    """Lazy PyTorch Dataset over Widar3.0 BVP samples.

    Args:
        metadata: list of metadata dicts (from ``index_widar_bvp``) defining the
            samples in this dataset, in order.
        label_map: gesture-name → int label (see ``build_label_map``).
        target_T: fixed number of frames every sample is padded/truncated to.
        normalize: ``"per_sample"`` or ``"global"`` (passed to ``normalize_bvp``).
        norm_stats: required when ``normalize="global"``; the global mean/std.
        augment: if True, apply ``augment_bvp`` before normalization. Set False
            for any eval/test split.
        augment_kwargs: extra kwargs forwarded to ``augment_bvp``.
        seed: seeds this dataset's augmentation Generator. For fully
            reproducible augmentation use ``num_workers=0`` (DataLoader workers
            each copy the same Generator state).
        cache: if True, memoize each sample's raw ``(T, 20, 20)`` volume in
            memory on first access. The ``.mat`` IO over ~44k tiny files is the
            dominant cost of a CPU training epoch; caching the *raw* (pre-augment)
            volume removes it on every epoch after the first while leaving the
            stochastic augment/normalize/pad chain to run fresh each ``__getitem__``.
            Raw volumes average ~27 KB, so the full corpus is ~1.2 GB. Requires
            ``num_workers=0`` to share the cache (workers don't share memory).

    ``__getitem__`` returns ``(tensor (target_T, 20, 20), label int)``.
    """

    def __init__(
        self,
        metadata: list[dict],
        label_map: dict[str, int],
        target_T: int = 32,
        normalize: str = "per_sample",
        norm_stats: dict | None = None,
        augment: bool = False,
        augment_kwargs: dict | None = None,
        seed: int = 0,
        cache: bool = False,
    ) -> None:
        if normalize == "global" and norm_stats is None:
            raise ValueError("norm_stats required when normalize='global'")
        self.metadata = list(metadata)
        self.label_map = dict(label_map)
        self.target_T = target_T
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.augment = augment
        self.augment_kwargs = dict(augment_kwargs or {})
        self._rng = np.random.default_rng(seed)
        self.cache = cache
        self._raw_cache: dict[int, np.ndarray] = {}

    @property
    def classes(self) -> list[str]:
        """Gesture names ordered by their integer label."""
        return [name for name, _ in sorted(self.label_map.items(), key=lambda kv: kv[1])]

    def __len__(self) -> int:
        return len(self.metadata)

    def _try_load(self, idx: int) -> np.ndarray | None:
        """Load one raw volume, or ``None`` if the file is unusable.

        A small fraction of the on-disk corpus is **truncated/corrupt** or has
        zero timesteps; ``scipy.io.loadmat`` raises on the former and an empty
        ``T`` poisons normalization on the latter. Both are treated as a miss so
        ``__getitem__`` can substitute a valid neighbour rather than crash the
        whole DataLoader. Bad indices are memoized (as ``False`` in the cache) so
        we never re-open them.
        """
        if self.cache:
            cached = self._raw_cache.get(idx, None)
            if cached is False:  # known-bad, don't retry
                return None
            if cached is not None:
                return cached
        try:
            x = load_bvp_file(self.metadata[idx]["path"])  # (T, 20, 20) raw energy
        except Exception:
            x = None
        if x is None or x.shape[0] == 0:
            if self.cache:
                self._raw_cache[idx] = False  # type: ignore[assignment]
            return None
        if self.cache:
            self._raw_cache[idx] = x
        return x

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Resolve to the first usable sample at/after idx (wrapping), using that
        # sample's own label so a substitution never mislabels. The corrupt-file
        # count is tiny, so the resulting duplication is negligible.
        n = len(self.metadata)
        x, md = None, None
        for step in range(n):
            j = (idx + step) % n
            x = self._try_load(j)
            if x is not None:
                md = self.metadata[j]
                break
        if x is None:
            raise RuntimeError("no readable BVP samples in this dataset partition")

        if self.augment:
            x = augment_bvp(x, rng=self._rng, **self.augment_kwargs)
        x = normalize_bvp(x, mode=self.normalize, stats=self.norm_stats)
        x = pad_or_truncate(x, self.target_T)
        return torch.from_numpy(x), self.label_map[md["gesture"]]


# ---------------------------------------------------------------------------
# Cross-domain split builders
# ---------------------------------------------------------------------------


def _make_split(
    train_md: list[dict],
    test_md: list[dict],
    *,
    target_T: int = 32,
    normalize: str = "per_sample",
    augment: bool = True,
    augment_kwargs: dict | None = None,
    seed: int = 0,
    global_stats_max_samples: int | None = 500,
    cache: bool = False,
) -> tuple[WidarBVPDataset, WidarBVPDataset]:
    """Build aligned train/test datasets from two metadata partitions.

    The label map spans both partitions so ids stay consistent; global stats (if
    requested) are estimated from the *train* partition only to avoid leakage;
    augmentation is on for train and always off for test.
    """
    if not train_md:
        raise ValueError("empty training split — check filters/split values")
    if not test_md:
        raise ValueError("empty test split — check the held-out split values")

    label_map = build_label_map(train_md + test_md)
    norm_stats = None
    if normalize == "global":
        norm_stats = compute_global_stats(
            train_md, max_samples=global_stats_max_samples, seed=seed
        )

    train_ds = WidarBVPDataset(
        train_md, label_map, target_T=target_T, normalize=normalize,
        norm_stats=norm_stats, augment=augment, augment_kwargs=augment_kwargs,
        seed=seed, cache=cache,
    )
    test_ds = WidarBVPDataset(
        test_md, label_map, target_T=target_T, normalize=normalize,
        norm_stats=norm_stats, augment=False, seed=seed, cache=cache,
    )
    return train_ds, test_ds


def cross_user(
    test_users,
    *,
    gesture=None,
    position=None,
    orientation=None,
    date=None,
    room=None,
    **split_kwargs,
) -> tuple[WidarBVPDataset, WidarBVPDataset]:
    """Leave-users-out split: test on ``test_users``, train on all the rest.

    The headline Widar3.0 generalization test — does the model recognize
    gestures from people it never trained on? Optional filters scope the corpus
    first (e.g. fix ``room`` so only the user varies). ``split_kwargs`` are
    forwarded to ``_make_split`` (target_T, normalize, augment, augment_kwargs,
    seed, global_stats_max_samples).
    """
    held = _as_set(test_users)
    index = index_widar_bvp(
        gesture=gesture, position=position, orientation=orientation,
        date=date, room=room,
    )
    train_md = [m for m in index if m["user"] not in held]
    test_md = [m for m in index if m["user"] in held]
    return _make_split(train_md, test_md, **split_kwargs)


def cross_position(
    test_positions,
    *,
    gesture=None,
    user=None,
    orientation=None,
    date=None,
    room=None,
    **split_kwargs,
) -> tuple[WidarBVPDataset, WidarBVPDataset]:
    """Leave-torso-locations-out split: test on ``test_positions`` (ids 1-8).

    Probes spatial generalization — gestures performed at room locations the
    model never saw during training.
    """
    held = _as_set(test_positions)
    index = index_widar_bvp(
        gesture=gesture, user=user, orientation=orientation,
        date=date, room=room,
    )
    train_md = [m for m in index if m["position"] not in held]
    test_md = [m for m in index if m["position"] in held]
    return _make_split(train_md, test_md, **split_kwargs)


def cross_orientation(
    test_orientations,
    *,
    gesture=None,
    user=None,
    position=None,
    date=None,
    room=None,
    **split_kwargs,
) -> tuple[WidarBVPDataset, WidarBVPDataset]:
    """Leave-face-orientations-out split: test on ``test_orientations`` (1-5).

    Tests whether the body-frame velocity representation truly removes the
    person's facing direction: train on some orientations, test on the others.
    """
    held = _as_set(test_orientations)
    index = index_widar_bvp(
        gesture=gesture, user=user, position=position,
        date=date, room=room,
    )
    train_md = [m for m in index if m["orientation"] not in held]
    test_md = [m for m in index if m["orientation"] in held]
    return _make_split(train_md, test_md, **split_kwargs)


def in_domain(
    *,
    test_frac: float = 0.2,
    gesture=None,
    user=None,
    position=None,
    orientation=None,
    date=None,
    room=None,
    seed: int = 0,
    **split_kwargs,
) -> tuple[WidarBVPDataset, WidarBVPDataset]:
    """Standard i.i.d. random split — the in-domain baseline (no domain shift).

    Samples are shuffled and split by ``test_frac`` regardless of user/position/
    orientation, so train and test come from the same distribution. The accuracy
    gap between this and the cross-domain splits *is* the domain gap. Filters
    scope which samples are in play; ``seed`` controls the shuffle.
    """
    index = index_widar_bvp(
        gesture=gesture, user=user, position=position,
        orientation=orientation, date=date, room=room,
    )
    if not index:
        raise ValueError("no samples match the given filters")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(index))
    n_test = int(round(len(index) * test_frac))
    test_idx = set(perm[:n_test].tolist())
    train_md = [index[i] for i in range(len(index)) if i not in test_idx]
    test_md = [index[i] for i in range(len(index)) if i in test_idx]
    return _make_split(train_md, test_md, seed=seed, **split_kwargs)
