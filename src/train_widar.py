#!/usr/bin/env python
"""Train the BVP CNN-RNN gesture classifier on Widar3.0, one model per split.

This is the Widar3.0 analog of ``src/train.py``: same Adam + cross-entropy loop,
the same early-stopping-on-val and checkpoint conventions, but over the BVP
``Dataset`` and the four canonical cross-domain splits from chunk 11
(``src/data/widar_dataset.py``). The whole point of Widar3.0 is *cross-domain*
generalization, so each split defines a *different* train/test partition and
therefore needs its *own* trained model:

    in_domain          i.i.d. random split (no domain shift) — the easy baseline
    cross_user         leave-users-out  — recognize people never trained on?
    cross_position     leave-torso-locations-out — novel room locations?
    cross_orientation  leave-face-orientations-out — truly facing-invariant?

With no ``--split`` argument this script trains **all four** in turn, writing one
stable checkpoint per split (``runs/best_bvp_<split>.pt``) plus a timestamped run
directory. ``src/evaluate_widar.py`` then loads the four checkpoints and scores
each on *its own* held-out partition to assemble ``figures/widar_domain_results.png``.

Each split's checkpoint stores its full ``split_config`` (split name, held-out
values, scoping filters, ``target_T``, ``normalize``, ``seed``, ``val_frac``) so
the evaluator reconstructs the exact same test partition without guessing.

Method (per split):
    * Build ``(train_full, _)`` from the split builder; the held-out test
      partition is *not* used here (the evaluator rebuilds it).
    * Carve a ``val_frac`` validation set out of ``train_full`` — same domain as
      train, so it drives early stopping while the true domain-shift test is held
      back. Validation has augmentation forced off.
    * Fit the CNN-RNN; checkpoint the best-val-accuracy weights.

Augmentation note: the default training augmentation disables the **horizontal
flip** (``flip_prob=0``). The flip (``v_x -> -v_x``) is only label-preserving for
left/right-symmetric gestures; across the 22-gesture set many are
direction-defined (Slide, Draw-N, the digits), so flipping would corrupt labels.
Temporal crop and mild Gaussian noise remain on.

Each run writes ``runs/<timestamp>_<split>/``:
    best.pt              — best-val-accuracy checkpoint (model-agnostic format)
    metrics.json         — per-epoch history + run summary
    training_curves.png  — loss & accuracy curves
and copies best.pt to ``runs/best_bvp_<split>.pt`` (the stable path eval expects).

Examples (from the repo root, with the project env active)::

    conda activate wifisense
    python src/train_widar.py                       # all four splits, full corpus
    python src/train_widar.py --split cross_user --test-users 3
    # quick smoke test on a tiny scoped subset:
    python src/train_widar.py --split in_domain --room 1 --max-per-gesture 30 \
        --epochs 2 --batch-size 16

Timing probe: every split prints a **FULL-RUN TIME ESTIMATE** at the end,
extrapolated from its measured per-epoch time (epoch 1 carries a one-time
file-cache fill, reported separately; steady-state epochs drive the projection).
To gauge a full run cheaply, run one split for a few epochs and read the estimate::

    python src/train_widar.py --split in_domain --epochs 4   # prints the projection

Runnable via ``-m`` or directly (it prepends the repo root to sys.path itself).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import make_dataloader  # noqa: E402
from src.data.widar_dataset import (  # noqa: E402
    WidarBVPDataset,
    cross_orientation,
    cross_position,
    cross_user,
    in_domain,
)
from src.models import build_model  # noqa: E402
from src.train import (  # noqa: E402
    evaluate_loader,
    plot_curves,
    set_seed,
    train_one_epoch,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"

# The four canonical splits and their default held-out values. Defaults are
# chosen to be well-populated (see scripts/explore_widar.py): user 3 is a large,
# broadly-covered user; position 5 is the central torso location; orientation 5
# is +90 deg. in_domain holds out a random 20%.
SPLITS = ("in_domain", "cross_user", "cross_position", "cross_orientation")
DEFAULT_TEST_USERS = [3]
DEFAULT_TEST_POSITIONS = [5]
DEFAULT_TEST_ORIENTATIONS = [5]
DEFAULT_TEST_FRAC = 0.2

# Heuristic epochs-to-early-stop used to project a full run from a partial one
# (patience-8 runs on this corpus typically converge somewhere in this band).
ASSUMED_EPOCHS = 25
ASSUMED_EPOCHS_RANGE = (15, 35)


def _fmt_dur(seconds: float) -> str:
    """Human-readable duration: '1h 23m', '12m 30s', or '45s'."""
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


def print_full_run_estimate(
    epoch_times: list[float], early_stopped: bool, this_split: str, n_splits: int
) -> None:
    """Project a full 4-split run time from one split's measured epoch times.

    Epoch 1 carries a one-time file-cache fill, so it is reported separately and
    the *steady-state* per-epoch time (mean of epochs 2+) drives the projection.
    If this split early-stopped we use its observed epoch count; otherwise (a
    deliberately short partial run) we project with a heuristic epoch band.
    """
    if not epoch_times:
        return
    first = epoch_times[0]
    steady = sum(epoch_times[1:]) / len(epoch_times[1:]) if len(epoch_times) > 1 else first
    overhead = max(first - steady, 0.0)  # the cache-fill cost paid once per split

    def split_cost(n_epochs: float) -> float:
        return overhead + steady * n_epochs

    print(f"\n{'-' * 70}")
    print("FULL-RUN TIME ESTIMATE (extrapolated from this split)")
    print(f"  measured: epoch 1 {_fmt_dur(first)} (incl. cache fill) | "
          f"steady-state ~{steady:.1f}s/epoch | {len(epoch_times)} epochs timed")
    print(f"  formula:  full ≈ n_splits × (cache_fill + per_epoch × epochs_to_stop)")
    print(f"          = {n_splits} × ({_fmt_dur(overhead)} + {steady:.1f}s × E)")
    if early_stopped:
        e = len(epoch_times)
        total = split_cost(e) * n_splits
        print(f"  this split early-stopped at E={e} epochs; if the others behave "
              f"similarly:")
        print(f"          ≈ {n_splits} × {_fmt_dur(split_cost(e))} = ~{_fmt_dur(total)} "
              f"for all {n_splits} splits")
    else:
        e = ASSUMED_EPOCHS
        lo_e, hi_e = ASSUMED_EPOCHS_RANGE
        mid = split_cost(e) * n_splits
        lo = split_cost(lo_e) * n_splits
        hi = split_cost(hi_e) * n_splits
        print(f"  partial run (didn't early-stop); assuming E≈{e} epochs to stop:")
        print(f"          ≈ {n_splits} × {_fmt_dur(split_cost(e))} = ~{_fmt_dur(mid)} "
              f"for all {n_splits} splits")
        print(f"  plausible band (E={lo_e}–{hi_e}): {_fmt_dur(lo)} – {_fmt_dur(hi)}")
    print("  note: cross_user trains on the most data, so it runs a bit longer "
          "than this split.")
    print(f"{'-' * 70}")


def _scope_filters(args: argparse.Namespace) -> dict:
    """Optional corpus-scoping filters shared by every split builder."""
    f: dict = {}
    if args.room is not None:
        f["room"] = args.room
    if args.gesture:
        f["gesture"] = args.gesture
    return f


def build_split(split: str, args: argparse.Namespace) -> tuple[WidarBVPDataset, dict]:
    """Build the *training* dataset for one split + the config to reproduce it.

    Returns ``(train_full_ds, split_config)``. ``train_full_ds`` is the full
    in-domain training partition (val is carved from it in ``run_split``); the
    held-out test partition is intentionally rebuilt later by the evaluator from
    ``split_config`` so train time never touches it.
    """
    ds_kwargs = dict(
        target_T=args.target_T,
        normalize=args.normalize,
        augment=True,
        augment_kwargs={"flip_prob": 0.0},
        seed=args.seed,
        cache=args.cache,
    )
    filters = _scope_filters(args)
    cfg: dict = {
        "split": split,
        "filters": filters,
        "max_per_gesture": args.max_per_gesture,
        "target_T": args.target_T,
        "normalize": args.normalize,
        "seed": args.seed,
        "val_frac": args.val_frac,
    }
    if split == "in_domain":
        # seed is already in ds_kwargs; in_domain consumes it as its own param.
        train_full, _ = in_domain(
            test_frac=args.test_frac, **filters, **ds_kwargs
        )
        cfg["test_frac"] = args.test_frac
    elif split == "cross_user":
        train_full, _ = cross_user(test_users=args.test_users, **filters, **ds_kwargs)
        cfg["test_users"] = list(args.test_users)
    elif split == "cross_position":
        train_full, _ = cross_position(
            test_positions=args.test_positions, **filters, **ds_kwargs
        )
        cfg["test_positions"] = list(args.test_positions)
    elif split == "cross_orientation":
        train_full, _ = cross_orientation(
            test_orientations=args.test_orientations, **filters, **ds_kwargs
        )
        cfg["test_orientations"] = list(args.test_orientations)
    else:
        raise ValueError(f"unknown split {split!r}")
    return train_full, cfg


def carve_val(
    train_full: WidarBVPDataset, val_frac: float, seed: int, cache: bool
) -> tuple[WidarBVPDataset, WidarBVPDataset]:
    """Split ``train_full`` into (train, val), stratified by gesture.

    Val shares the parent's label map and normalization stats and has
    augmentation forced off. The split is stratified so every gesture is
    represented in val (important when a split scopes to few samples per class).
    """
    md = train_full.metadata
    by_g: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(md):
        by_g[m["gesture"]].append(i)

    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    for g, idxs in by_g.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac))) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
    val_set = set(val_idx)
    train_md = [md[i] for i in range(len(md)) if i not in val_set]
    val_md = [md[i] for i in range(len(md)) if i in val_set]

    common = dict(
        label_map=train_full.label_map,
        target_T=train_full.target_T,
        normalize=train_full.normalize,
        norm_stats=train_full.norm_stats,
        seed=seed,
        cache=cache,
    )
    train_ds = WidarBVPDataset(
        train_md, augment=True, augment_kwargs={"flip_prob": 0.0}, **common
    )
    val_ds = WidarBVPDataset(val_md, augment=False, **common)
    return train_ds, val_ds


def run_split(split: str, args: argparse.Namespace, device: torch.device) -> dict:
    """Train one split end-to-end; return a small summary dict."""
    set_seed(args.seed)
    train_full, split_cfg = build_split(split, args)
    train_ds, val_ds = carve_val(train_full, args.val_frac, args.seed, args.cache)
    class_names = train_full.classes
    n_classes = len(class_names)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = make_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=g,
    )
    val_loader = make_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model_cfg = {
        "num_classes": n_classes,
        "gru_hidden": args.hidden,
        "gru_layers": args.layers,
        "dropout": args.dropout,
    }
    model = build_model("bvp_cnn_rnn", **model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    run_dir = RUNS_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    held = (
        split_cfg.get("test_users")
        or split_cfg.get("test_positions")
        or split_cfg.get("test_orientations")
        or f"{args.test_frac:.0%} random"
    )
    print(f"\n{'=' * 70}\nSPLIT: {split}   (held out: {held})")
    print(f"Device: {device} | run dir: {run_dir.name} | params: {n_params/1e3:.0f}k")
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | classes {n_classes}")

    history: list[dict] = []
    epoch_times: list[float] = []
    early_stopped = False
    best_val_acc, best_epoch, epochs_no_improve = -1.0, -1, 0
    ckpt_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        epoch_times.append(time.perf_counter() - t_epoch)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        flag = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, epochs_no_improve, flag = val_acc, epoch, 0, " *"
            torch.save({
                "model_name": "bvp_cnn_rnn",
                "model_config": model.config,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "val_acc": val_acc,
                "split_config": split_cfg,
                "args": vars(args) | {"save_dir": str(run_dir)},
            }, ckpt_path)
        else:
            epochs_no_improve += 1
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}{flag} | "
            f"{epoch_times[-1]:.1f}s"
        )
        if epochs_no_improve >= args.patience:
            print(f"Early stopping: no val-acc gain in {args.patience} epochs.")
            early_stopped = True
            break

    wall = sum(epoch_times)
    summary = {
        "split": split,
        "split_config": split_cfg,
        "model_config": model.config,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "epochs_run": history[-1]["epoch"],
        "n_params": n_params,
        "wall_seconds": wall,
        "sec_per_epoch_steady": (
            sum(epoch_times[1:]) / len(epoch_times[1:]) if len(epoch_times) > 1 else wall
        ),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_curves(history, run_dir / "training_curves.png")

    stable = RUNS_ROOT / f"best_bvp_{split}.pt"
    shutil.copy2(ckpt_path, stable)
    print(f"Best val acc {best_val_acc:.4f} @ epoch {best_epoch} -> {stable.name} "
          f"({history[-1]['epoch']} epochs in {_fmt_dur(wall)})")

    # Project the full multi-split run from this split's measured epoch times —
    # most useful when running one split as a quick partial timing probe.
    print_full_run_estimate(epoch_times, early_stopped, split, len(SPLITS))
    return {"split": split, "best_val_acc": best_val_acc, "checkpoint": str(stable),
            "wall_seconds": wall}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--split", default="all", choices=("all", *SPLITS),
        help="Which split to train ('all' trains every split in turn).",
    )
    # Held-out values per split.
    p.add_argument("--test-users", type=int, nargs="+", default=DEFAULT_TEST_USERS)
    p.add_argument("--test-positions", type=int, nargs="+", default=DEFAULT_TEST_POSITIONS)
    p.add_argument("--test-orientations", type=int, nargs="+", default=DEFAULT_TEST_ORIENTATIONS)
    p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    # Corpus scoping (applies to every split) — handy for tractable CPU runs.
    p.add_argument("--room", type=int, default=None, help="Scope to one capture room (1/2/3).")
    p.add_argument("--gesture", nargs="+", default=None, help="Scope to these gesture names.")
    p.add_argument("--max-per-gesture", type=int, default=None,
                   help="Cap samples per gesture (random subset) before splitting.")
    # Data shaping.
    p.add_argument("--target-T", type=int, default=32, dest="target_T")
    p.add_argument("--normalize", default="per_sample", choices=("per_sample", "global"))
    p.add_argument("--val-frac", type=float, default=0.15, dest="val_frac")
    p.add_argument("--no-cache", dest="cache", action="store_false",
                   help="Disable the in-memory raw-volume cache (lower RAM, slower).")
    # Optimization.
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    # Model hyperparameters.
    p.add_argument("--hidden", type=int, default=128, help="GRU hidden units per direction.")
    p.add_argument("--layers", type=int, default=1, help="Stacked GRU layers.")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def cap_index(max_per_gesture: int | None, seed: int) -> None:
    """Monkeypatch ``index_widar_bvp`` to cap samples per gesture, if requested.

    Kept as a thin wrapper so every split builder honours the cap uniformly
    without threading it through the chunk-11 API. Idempotent and deterministic
    in ``seed``, so ``evaluate_widar`` can re-apply the exact same cap to rebuild
    the identical test partition. Pass ``None`` to disable.
    """
    if max_per_gesture is None:
        return
    import src.data.widar_dataset as wd

    orig = getattr(wd, "_index_widar_bvp_uncapped", wd.index_widar_bvp)
    wd._index_widar_bvp_uncapped = orig  # so a second call doesn't nest caps
    cap = max_per_gesture

    def capped(*a, **k):
        idx = orig(*a, **k)
        by_g: dict[str, list[dict]] = defaultdict(list)
        for m in idx:
            by_g[m["gesture"]].append(m)
        rng = np.random.default_rng(seed)
        out: list[dict] = []
        for g, ms in by_g.items():
            if len(ms) > cap:
                sel = rng.choice(len(ms), size=cap, replace=False)
                ms = [ms[i] for i in sorted(sel)]
            out.extend(ms)
        out.sort(key=lambda m: m["path"])
        return out

    wd.index_widar_bvp = capped


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )
    cap_index(args.max_per_gesture, args.seed)

    splits = list(SPLITS) if args.split == "all" else [args.split]
    results = [run_split(s, args, device) for s in splits]

    total = sum(r["wall_seconds"] for r in results)
    print(f"\n{'=' * 70}\nDONE — trained {len(results)} split(s) in {_fmt_dur(total)}:")
    for r in results:
        print(f"  {r['split']:18} best val acc {r['best_val_acc']:.4f}  "
              f"({_fmt_dur(r['wall_seconds'])})  -> {Path(r['checkpoint']).name}")
    if len(results) < len(SPLITS):
        print(f"\n(ran {len(results)}/{len(SPLITS)} splits — see each split's "
              "FULL-RUN TIME ESTIMATE above to project a complete run.)")


if __name__ == "__main__":
    main()
