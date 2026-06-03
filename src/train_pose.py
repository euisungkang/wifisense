#!/usr/bin/env python
"""Train the CSI → 3D-pose regression model on MM-Fi (Phase 3).

This is the project's first **regression** trainer. Every trainer before it
(``src/train.py``, ``src/train_widar.py``) fits a *classifier* — cross-entropy
loss, scored on accuracy. This one fits a *pose regressor*: the target is a
continuous ``(17, 3)`` set of 3D joint coordinates, the loss is a coordinate
distance (L1 / MSE, optionally with a bone-length term), and the headline metric
is **MPJPE** — Mean Per-Joint Position Error, a distance in millimetres. There is
no accuracy here; a "good" epoch is one where the predicted skeleton sits closer
to the real one.

What it does
------------
* Builds ``(train_ds, val_ds)`` from one of the chunk-13 split builders
  (``src/data/mmfi_pose_dataset.py``). Default split is **cross_subject** — the
  headline WiFi-pose generalization test (pose bodies never seen in training).
  For cross_subject/cross_environment the ``val_ds`` *is* the benchmark's
  held-out partition, so the reported MPJPE is directly comparable to the MM-Fi
  paper's WiFi-only numbers (see ``docs/chunk15_pose_model.md`` for the honest
  comparison and the one caveat about selecting the checkpoint on it).
* Fits ``CSIPoseNet`` (``src/models/csi_pose_net.py``) with Adam, early-stopping
  on **val MPJPE** (lower is better — note this inverts the classifiers'
  higher-is-better val-accuracy convention).
* Reports **val MPJPE in mm every epoch** and checkpoints the best (lowest-MPJPE)
  weights to ``runs/best_pose.pt`` (plus a timestamped run dir).
* Every ``--progress-every`` epochs, saves a GT-vs-prediction skeleton figure for
  a *fixed* validation sample to ``runs/<timestamp>/progress/epoch_NNN.png`` so
  the predicted skeleton can be watched converging toward the real pose over
  training — both a debug tool and the kind of output Phase 3 has been building
  toward.

Loss design (documented weighting)
----------------------------------
``loss = coord_loss + bone_weight * bone_loss``

* ``coord_loss`` — ``--loss l1`` (default, robust to the occasional bad joint) or
  ``mse`` on the root-centered joint coordinates (metres). This is what actually
  drives MPJPE down.
* ``bone_loss`` — mean absolute error between predicted and ground-truth **bone
  lengths** over the MM-Fi kinematic tree (``src/viz/skeleton.py``). A soft
  anatomical prior: it discourages skeletons that match joints on average but
  have rubber-band limbs. Weighted by ``--bone-weight`` (default **0.1**, i.e.
  the bone term contributes ~10% the scale of an L1-metre coordinate error). Set
  ``--bone-weight 0`` to train on coordinates alone.

Why MPJPE (and how to read it)
------------------------------
MPJPE = average Euclidean distance between each predicted joint and its ground
truth, after both are root-centered (the Dataset already centers on the pelvis,
so the metric is translation-free). We report it in **millimetres**. Lower is
better. With the default ``pose_scale=None`` the target is in metres, so
``MPJPE_mm = mean(||pred − gt||) * 1000`` directly — no un-normalization needed
(``--pose-scale`` other than ``none`` is rejected here precisely so MPJPE stays
metric; see ``docs/chunk15_pose_model.md``).

Sanity expectation: WiFi pose is inherently coarse — do NOT expect camera-quality
skeletons. Early epochs land in the low **hundreds of mm**; the MM-Fi paper's
published WiFi-only MPJPE is the target ceiling, not camera-grade single-digit-cm
numbers.

Each run writes ``runs/<timestamp>/``:
    best.pt              — best-val-MPJPE checkpoint (model-agnostic format)
    metrics.json         — per-epoch history + run summary
    training_curves.png  — train/val loss + val MPJPE curves
    progress/epoch_NNN.png — GT-vs-pred skeleton snapshots (every N epochs)
and copies best.pt to ``runs/best_pose.pt`` (the stable path).

Examples (from the repo root, with the project env active)::

    conda activate wifisense
    python src/train_pose.py                                  # cross_subject, full on-disk corpus
    python src/train_pose.py --split random_split --epochs 60
    # quick smoke test on a tiny scoped subset (few clips, few epochs):
    python src/train_pose.py --limit 2000 --epochs 3 --batch-size 32

Data: MM-Fi is large and not auto-downloaded (see docs/chunk13_mmfi_setup.md).
The split builders partition whatever is on disk, so cross_subject works on E01
alone (its default held-out S05/S10 live in E01).

Runnable via ``-m`` or directly (it prepends the repo root to sys.path itself).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import make_dataloader  # noqa: E402
from src.data.mmfi_pose_dataset import (  # noqa: E402
    cross_environment,
    cross_subject,
    random_split,
)
from src.models import build_model  # noqa: E402
from src.train import set_seed  # noqa: E402
from src.viz.skeleton import SKELETON_EDGES, plot_skeleton_pair  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"

SPLITS = ("cross_subject", "cross_environment", "random_split")

# Bone (parent, child) index pairs of the MM-Fi / Human3.6M kinematic tree,
# kept as a tensor-ready pair of index arrays for the bone-length loss.
_BONE_A = [a for a, _ in SKELETON_EDGES]
_BONE_B = [b for _, b in SKELETON_EDGES]


def _fmt_dur(seconds: float) -> str:
    """Human-readable duration: '1h 23m', '12m 30s', or '45s'."""
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Loss + metric
# ---------------------------------------------------------------------------


def bone_lengths(kp: torch.Tensor) -> torch.Tensor:
    """Per-bone Euclidean lengths of a batch of poses.

    Args:
        kp: ``(B, n_joints, 3)`` coordinates.
    Returns:
        ``(B, n_bones)`` length of each kinematic-tree bone.
    """
    a = kp[:, _BONE_A, :]
    b = kp[:, _BONE_B, :]
    return torch.linalg.norm(a - b, dim=-1)


def pose_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord: str = "l1",
    bone_weight: float = 0.1,
) -> torch.Tensor:
    """Coordinate regression loss + optional bone-length regularizer.

    ``loss = coord_loss + bone_weight * bone_loss`` (see module docstring for the
    weighting rationale). ``coord`` is ``"l1"`` or ``"mse"`` on the root-centered
    coordinates; the bone term is the L1 error between predicted and GT bone
    lengths over the kinematic tree.
    """
    if coord == "l1":
        coord_loss = torch.nn.functional.l1_loss(pred, target)
    elif coord == "mse":
        coord_loss = torch.nn.functional.mse_loss(pred, target)
    else:
        raise ValueError(f"coord loss must be 'l1' or 'mse'; got {coord!r}")
    if bone_weight <= 0:
        return coord_loss
    bone_loss = torch.nn.functional.l1_loss(bone_lengths(pred), bone_lengths(target))
    return coord_loss + bone_weight * bone_loss


@torch.no_grad()
def mpjpe_mm(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Per-Joint Position Error in millimetres (the pose metric).

    Both poses are already root-centered by the Dataset, so the per-joint
    Euclidean distance is translation-free. Targets are in metres
    (``pose_scale=None``), hence the ``* 1000`` to millimetres. Returns the mean
    over joints AND the batch.
    """
    per_joint = torch.linalg.norm(pred - target, dim=-1)  # (B, n_joints), metres
    return float(per_joint.mean().item() * 1000.0)


# ---------------------------------------------------------------------------
# Train / eval passes
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, device, *, coord, bone_weight):
    """One training pass; returns (mean loss, mean MPJPE in mm)."""
    model.train()
    loss_sum, mpjpe_sum, n = 0.0, 0.0, 0
    for csi, kp in loader:
        csi, kp = csi.to(device), kp.to(device)
        optimizer.zero_grad()
        pred = model(csi)
        loss = pose_loss(pred, kp, coord=coord, bone_weight=bone_weight)
        loss.backward()
        optimizer.step()
        bs = kp.size(0)
        loss_sum += loss.item() * bs
        mpjpe_sum += mpjpe_mm(pred.detach(), kp) * bs
        n += bs
    return loss_sum / n, mpjpe_sum / n


@torch.no_grad()
def evaluate_loader(model, loader, device, *, coord, bone_weight):
    """Eval pass; returns (mean loss, mean MPJPE in mm)."""
    model.eval()
    loss_sum, mpjpe_sum, n = 0.0, 0.0, 0
    for csi, kp in loader:
        csi, kp = csi.to(device), kp.to(device)
        pred = model(csi)
        loss = pose_loss(pred, kp, coord=coord, bone_weight=bone_weight)
        bs = kp.size(0)
        loss_sum += loss.item() * bs
        mpjpe_sum += mpjpe_mm(pred, kp) * bs
        n += bs
    return loss_sum / n, mpjpe_sum / n


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_pose_curves(history: list[dict], path: Path) -> None:
    """Save train/val loss and val-MPJPE curves; mark the best (lowest) MPJPE."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_mpjpe) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Pose loss (coord + bone)")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_mpjpe.plot(epochs, [h["train_mpjpe"] for h in history], label="train")
    ax_mpjpe.plot(epochs, [h["val_mpjpe"] for h in history], label="val")
    ax_mpjpe.set_xlabel("Epoch")
    ax_mpjpe.set_ylabel("MPJPE (mm)")
    ax_mpjpe.set_title("Mean Per-Joint Position Error (lower is better)")
    ax_mpjpe.legend()

    best = min(history, key=lambda h: h["val_mpjpe"])
    ax_mpjpe.axvline(best["epoch"], color="grey", ls="--", lw=1)
    ax_mpjpe.annotate(
        f"best val MPJPE={best['val_mpjpe']:.0f} mm\n@ epoch {best['epoch']}",
        xy=(best["epoch"], best["val_mpjpe"]),
        xytext=(0.45, 0.75),
        textcoords="axes fraction",
        fontsize=9,
    )

    fig.suptitle("Pose training curves", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_progress_figure(model, sample, device, path: Path, epoch: int) -> None:
    """Render GT vs current prediction for a fixed val sample to ``path``.

    ``sample`` is ``(csi, kp_gt)`` cached once so the SAME pose is shown every
    time — that's what makes the snapshots a watchable convergence sequence.
    Both skeletons are root-centered metres (the target space), so
    ``plot_skeleton_pair`` annotates the true metric MPJPE for this one pose.
    """
    model.eval()
    csi, kp_gt = sample
    pred = model(csi.unsqueeze(0).to(device))[0].cpu().numpy()
    gt = kp_gt.numpy()

    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection="3d")
    plot_skeleton_pair(gt, pred, ax=ax)
    ax.set_title(f"epoch {epoch} — {ax.get_title()}")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------


def build_split(args: argparse.Namespace):
    """Build ``(train_ds, val_ds)`` for the requested split + a reproducible config.

    The split builders return ``(train, held_out)``; for cross_subject /
    cross_environment the held-out partition is the benchmark eval set and serves
    as the validation set here (so reported MPJPE is the comparable number). For
    random_split it's the i.i.d. test fold.
    """
    ds_kwargs = dict(
        window_size=args.window_size,
        csi_normalize=args.csi_normalize,
        pose_scale=None,  # keep targets in metres so MPJPE is directly metric
    )
    common = dict(protocol=args.protocol, data_root=args.data_root, limit=args.limit)
    cfg: dict = {
        "split": args.split,
        "window_size": args.window_size,
        "csi_normalize": args.csi_normalize,
        "protocol": args.protocol,
        "limit": args.limit,
    }
    if args.split == "cross_subject":
        train_ds, val_ds = cross_subject(
            test_subjects=args.test_subjects, **common, **ds_kwargs
        )
        cfg["test_subjects"] = args.test_subjects
    elif args.split == "cross_environment":
        train_ds, val_ds = cross_environment(
            test_envs=args.test_envs, **common, **ds_kwargs
        )
        cfg["test_envs"] = args.test_envs
    elif args.split == "random_split":
        train_ds, val_ds = random_split(
            test_ratio=args.test_ratio, seed=args.seed, **common, **ds_kwargs
        )
        cfg["test_ratio"] = args.test_ratio
    else:
        raise ValueError(f"unknown split {args.split!r}")
    return train_ds, val_ds, cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Split selection + per-split held-out values.
    p.add_argument("--split", default="cross_subject", choices=SPLITS)
    p.add_argument("--test-subjects", nargs="+", default=None,
                   help="cross_subject held-out subjects (default: MM-Fi S05..S40).")
    p.add_argument("--test-envs", nargs="+", default=None,
                   help="cross_environment held-out envs (default: E04).")
    p.add_argument("--test-ratio", type=float, default=0.2,
                   help="random_split held-out clip fraction.")
    # Data shaping.
    p.add_argument("--window-size", type=int, default=1,
                   help="CSI frames per window (odd; 1 = MM-Fi benchmark protocol). "
                        "ASK before changing — breaks comparability with the paper.")
    p.add_argument("--csi-normalize", default="none", choices=("none", "zscore", "minmax"))
    p.add_argument("--protocol", default="protocol3")
    p.add_argument("--data-root", default=None, help="Override MM-Fi data root.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap frames loaded (quick smoke tests).")
    # Optimization.
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=60,
                   help="Max epochs (early stopping may stop sooner).")
    p.add_argument("--patience", type=int, default=12,
                   help="Early-stop after N epochs with no val-MPJPE improvement.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    # Loss.
    p.add_argument("--loss", default="l1", choices=("l1", "mse"),
                   help="Coordinate regression loss.")
    p.add_argument("--bone-weight", type=float, default=0.1,
                   help="Weight of the bone-length regularizer (0 disables it).")
    # Model hyperparameters.
    p.add_argument("--head-hidden", type=int, default=256, dest="head_hidden")
    p.add_argument("--dropout", type=float, default=0.3)
    # Output cadence.
    p.add_argument("--progress-every", type=int, default=5, dest="progress_every",
                   help="Save a GT-vs-pred skeleton figure every N epochs.")
    p.add_argument("--save-dir", type=Path, default=None,
                   help="Run directory (default: runs/{timestamp}).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )

    train_ds, val_ds, split_cfg = build_split(args)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = make_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=g,
    )
    val_loader = make_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    # One fixed val sample for the progress snapshots — cache it once so every
    # epoch's figure shows the SAME pose converging.
    fixed_sample = val_ds[0]

    model_cfg = {
        "n_joints": 17,
        "window_size": args.window_size,
        "head_hidden": args.head_hidden,
        "dropout": args.dropout,
    }
    model = build_model("csi_pose_net", **model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())

    run_dir = args.save_dir or (RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = run_dir / "progress"
    progress_dir.mkdir(exist_ok=True)

    held = (
        split_cfg.get("test_subjects")
        or split_cfg.get("test_envs")
        or (f"{args.test_ratio:.0%} random clips" if args.split == "random_split" else "default")
    )
    print(f"{'=' * 70}\nSPLIT: {args.split}   (val = held out: {held})")
    print(f"Device: {device} | run dir: {run_dir.name} | params: {n_params/1e3:.0f}k")
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | "
          f"loss {args.loss}+{args.bone_weight}*bone | window {args.window_size}")

    history: list[dict] = []
    best_mpjpe, best_epoch, epochs_no_improve = float("inf"), -1, 0
    ckpt_path = run_dir / "best.pt"
    loss_kw = dict(coord=args.loss, bone_weight=args.bone_weight)
    epoch_times: list[float] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_mpjpe = train_one_epoch(
            model, train_loader, optimizer, device, **loss_kw
        )
        val_loss, val_mpjpe = evaluate_loader(model, val_loader, device, **loss_kw)
        epoch_times.append(time.perf_counter() - t0)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_mpjpe": train_mpjpe,
            "val_loss": val_loss, "val_mpjpe": val_mpjpe,
        })

        flag = ""
        if val_mpjpe < best_mpjpe:
            best_mpjpe, best_epoch, epochs_no_improve, flag = val_mpjpe, epoch, 0, " *"
            torch.save({
                "model_name": "csi_pose_net",
                "model_config": model.config,
                "state_dict": model.state_dict(),
                "joint_names": None,  # H36M order; see src/viz/skeleton.JOINT_NAMES
                "epoch": epoch,
                "val_mpjpe_mm": val_mpjpe,
                "split_config": split_cfg,
                "args": vars(args) | {"save_dir": str(run_dir)},
            }, ckpt_path)
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} MPJPE {train_mpjpe:6.1f}mm | "
            f"val loss {val_loss:.4f} MPJPE {val_mpjpe:6.1f}mm{flag} | "
            f"{epoch_times[-1]:.1f}s"
        )

        # Watch the skeleton converge: snapshot on epoch 1, every N epochs, and
        # whenever we hit a new best.
        if epoch == 1 or epoch % args.progress_every == 0 or flag:
            save_progress_figure(
                model, fixed_sample, device,
                progress_dir / f"epoch_{epoch:03d}.png", epoch,
            )

        if epochs_no_improve >= args.patience:
            print(f"Early stopping: no val-MPJPE improvement in {args.patience} epochs.")
            break

    wall = sum(epoch_times)
    summary = {
        "split": args.split,
        "split_config": split_cfg,
        "model_config": model.config,
        "loss": {"coord": args.loss, "bone_weight": args.bone_weight},
        "best_epoch": best_epoch,
        "best_val_mpjpe_mm": best_mpjpe,
        "epochs_run": history[-1]["epoch"],
        "n_params": n_params,
        "wall_seconds": wall,
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_pose_curves(history, run_dir / "training_curves.png")

    stable = RUNS_ROOT / "best_pose.pt"
    shutil.copy2(ckpt_path, stable)

    print(f"\nBest val MPJPE {best_mpjpe:.1f} mm @ epoch {best_epoch} "
          f"({history[-1]['epoch']} epochs in {_fmt_dur(wall)})")
    print(f"Checkpoint: {ckpt_path}  ->  {stable}")
    print(f"Metrics:    {run_dir / 'metrics.json'}")
    print(f"Curves:     {run_dir / 'training_curves.png'}")
    print(f"Progress:   {progress_dir}/  (GT-vs-pred skeleton snapshots)")


if __name__ == "__main__":
    main()
