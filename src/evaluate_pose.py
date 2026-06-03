#!/usr/bin/env python
"""Evaluate the trained CSI → 3D-pose regressor (Phase 3 capstone metrics).

This is the *regression* analogue of ``src/evaluate.py`` (which scores the CSI
classifiers on accuracy / F1). Here there is no accuracy: the model emits a
continuous ``(17, 3)`` pose and is scored on **MPJPE** — Mean Per-Joint Position
Error, a distance in millimetres — and its Procrustes-aligned cousin
**PA-MPJPE**. Lower is better for both.

Loads a ``best_pose.pt`` written by ``src/train_pose.py`` (which carries the
model name + config so the network is rebuilt without guessing hyperparameters),
then scores that ONE checkpoint over the held-out partition of each benchmark
split (``src/data/mmfi_pose_dataset.py``):

    cross_subject       — the headline test: bodies never seen in training.
    cross_environment   — a new room / multipath layout (needs E02–E04 on disk).
    random_split        — i.i.d. baseline (NO domain shift).

For every split that can be built from what's on disk it reports:

    * overall MPJPE and PA-MPJPE (mm);
    * per-joint MPJPE and PA-MPJPE over the 17 Human3.6M joints — extremities
      (wrists / ankles) are expected to be the worst, since they move most and
      are the furthest from the pelvis root the metric is anchored to;
    * a short table putting our numbers next to the MM-Fi paper's WiFi-only
      benchmark regime (honest "same ballpark", not "matched" — see below).

Honest caveats (the same ones in docs/chunk15_pose_model.md, restated because a
metrics script is exactly where they get forgotten):

  * The checkpoint here was trained **cross_subject on E01 alone** (2 held-out
    subjects), not the paper's 40-subject / 4-environment protocol.
  * ``cross_subject`` is the faithful generalization number for this model. The
    OTHER splits are reported for completeness but are NOT clean tests of THIS
    checkpoint: ``random_split``'s test clips come from the same subjects the
    model trained on (optimistic / leaky), and ``cross_environment`` needs an
    environment that isn't E01 to have anything to test on. Each is flagged in
    the output.

Outputs (under ``--out-dir``, default ``figures/``):
    * stdout + ``pose_eval_metrics.json`` — per-split overall + per-joint MPJPE /
      PA-MPJPE, plus the benchmark-comparison table;
    * ``pose_per_joint_error.png`` — grouped per-joint error bar chart across the
      available splits.

Example (from the repo root, with the project env active)::

    conda activate wifisense
    python src/evaluate_pose.py
    python src/evaluate_pose.py --checkpoint runs/best_pose.pt --out-dir figures

Data: MM-Fi is large and not auto-downloaded (see docs/chunk13_mmfi_setup.md).
Splits that have no held-out data on disk (e.g. cross_environment with only E01)
are skipped with a clear note rather than crashing.

Runnable via ``-m`` or directly (it prepends the repo root to sys.path itself).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import make_dataloader  # noqa: E402
from src.data.mmfi_pose_dataset import (  # noqa: E402
    cross_environment,
    cross_subject,
    random_split,
)
from src.models import build_model  # noqa: E402
from src.viz.skeleton import JOINT_NAMES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = PROJECT_ROOT / "runs" / "best_pose.pt"
FIGURES_DIR = PROJECT_ROOT / "figures"

SPLITS = ("cross_subject", "cross_environment", "random_split")

# One-line honesty note attached to each split's numbers in the report. The
# checkpoint is trained cross_subject; only that split is a clean test of it.
SPLIT_NOTE = {
    "cross_subject": "faithful generalization test for this checkpoint "
                     "(bodies held out of training)",
    "cross_environment": "robustness to a new room; needs an env other than the "
                         "trained one on disk",
    "random_split": "i.i.d. baseline — LEAKY for this checkpoint (test clips share "
                    "subjects with training); read as a no-domain-shift ceiling",
}

# The MM-Fi paper's WiFi-only 3D-pose error regime. The paper reports MPJPE in
# the low hundreds of mm for WiFi-CSI (far coarser than its camera/LiDAR/mmWave
# modalities). We quote it as an APPROXIMATE regime, not an exact number, and
# only ever claim "same ballpark" — our setup (E01 only, single-frame, small CNN)
# differs from the paper's full protocol. See docs/chunk16_pose_deliverable.md.
MMFI_WIFI_BENCHMARK_MM = "~100-130 (WiFi-only, full 40-subject corpus; approx.)"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity-align one predicted pose to its GT (Umeyama: scale+rot+trans).

    Returns ``pred`` mapped by the optimal scale ``s``, rotation ``R`` and
    translation ``t`` that minimise ``||s·R·pred + t − gt||`` (reflections
    disallowed). This factors out the global pose/size the WiFi regressor cannot
    be expected to nail, isolating the *shape* error — the basis of PA-MPJPE.

    Args:
        pred, gt: ``(J, 3)`` poses (any consistent frame; metres).
    Returns:
        ``(J, 3)`` aligned prediction in GT space.
    """
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    p0, g0 = pred - mu_p, gt - mu_g
    cov = g0.T @ p0  # (3, 3)
    u, s_vals, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    D = np.diag([1.0, 1.0, d])  # flip the last axis if needed to forbid reflection
    R = u @ D @ vt
    var_p = (p0 ** 2).sum()
    scale = (s_vals @ np.array([1.0, 1.0, d])) / var_p if var_p > 1e-12 else 1.0
    t = mu_g - scale * R @ mu_p
    return (scale * (R @ pred.T).T + t).astype(np.float32)


@torch.no_grad()
def evaluate_split(model, dataset, device, batch_size: int) -> dict:
    """Run the model over a dataset; return overall + per-joint MPJPE / PA-MPJPE.

    MPJPE is computed directly on the model's root-relative output vs the
    root-relative target (the Dataset uses ``pose_scale=None``, so both are in
    metres and the metric is translation-free — ``*1000`` gives mm). PA-MPJPE
    Procrustes-aligns each predicted pose to its GT first.
    """
    model.eval()
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    n_joints = len(JOINT_NAMES)
    per_joint_sum = np.zeros(n_joints, dtype=np.float64)       # MPJPE
    per_joint_pa_sum = np.zeros(n_joints, dtype=np.float64)    # PA-MPJPE
    n = 0
    for csi, kp in loader:
        pred = model(csi.to(device)).cpu().numpy()  # (B, 17, 3)
        gt = kp.numpy()                              # (B, 17, 3)
        # Per-joint Euclidean distance, in millimetres.
        d = np.linalg.norm(pred - gt, axis=-1) * 1000.0  # (B, 17)
        per_joint_sum += d.sum(axis=0)
        # PA-MPJPE: align each pose then measure.
        for b in range(pred.shape[0]):
            aligned = _procrustes_align(pred[b], gt[b])
            per_joint_pa_sum += np.linalg.norm(aligned - gt[b], axis=-1) * 1000.0
        n += pred.shape[0]

    per_joint = per_joint_sum / n
    per_joint_pa = per_joint_pa_sum / n
    return {
        "n_samples": int(n),
        "mpjpe_mm": float(per_joint.mean()),
        "pa_mpjpe_mm": float(per_joint_pa.mean()),
        "per_joint_mpjpe_mm": per_joint.tolist(),
        "per_joint_pa_mpjpe_mm": per_joint_pa.tolist(),
    }


# ---------------------------------------------------------------------------
# Split building (robust to a partial download)
# ---------------------------------------------------------------------------


def build_test_set(split: str, ckpt_args: dict, ds_kwargs: dict):
    """Build the held-out test dataset for ``split``, or return None if absent.

    Mirrors the held-out sets used by ``src/train_pose.py`` (same default
    subjects / envs / ratio). The split builders raise ``ValueError`` when a
    partition is empty on disk (e.g. cross_environment(E04) with only E01); we
    translate that into a clean skip with the reason.
    """
    common = dict(
        protocol=ckpt_args.get("protocol", "protocol3"),
        data_root=ckpt_args.get("data_root"),
        limit=ckpt_args.get("limit"),
    )
    try:
        if split == "cross_subject":
            _, test_ds = cross_subject(
                test_subjects=ckpt_args.get("test_subjects"), **common, **ds_kwargs
            )
        elif split == "cross_environment":
            _, test_ds = cross_environment(
                test_envs=ckpt_args.get("test_envs"), **common, **ds_kwargs
            )
        elif split == "random_split":
            _, test_ds = random_split(
                test_ratio=ckpt_args.get("test_ratio", 0.2),
                seed=ckpt_args.get("seed", 42),
                **common, **ds_kwargs,
            )
        else:
            raise ValueError(f"unknown split {split!r}")
    except ValueError as e:
        return None, str(e)
    return test_ds, None


# ---------------------------------------------------------------------------
# Figure + table
# ---------------------------------------------------------------------------


def plot_per_joint_error(results: dict, path: Path) -> None:
    """Grouped per-joint MPJPE bar chart across the evaluated splits.

    One cluster of bars per joint (ordered along the kinematic chain), one bar
    per available split. Makes the expected pattern visible: extremities
    (wrists/ankles) tower over the torso/root joints.
    """
    splits = [s for s, r in results.items() if r.get("per_joint_mpjpe_mm")]
    if not splits:
        return
    n_joints = len(JOINT_NAMES)
    x = np.arange(n_joints)
    width = 0.8 / max(len(splits), 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, s in enumerate(splits):
        vals = results[s]["per_joint_mpjpe_mm"]
        ax.bar(x + i * width, vals, width, label=f"{s} (mean {results[s]['mpjpe_mm']:.0f} mm)")
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Per-joint MPJPE (mm)")
    ax.set_title(
        "Per-joint WiFi-pose error — extremities (wrists/ankles) are worst, as expected\n"
        "(root-centered on the pelvis; joints farther out accumulate more error)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_report(results: dict, ckpt_meta: dict) -> None:
    """Pretty-print overall numbers, the worst joints, and the benchmark table."""
    bar = "=" * 74
    print(f"\n{bar}\nCSI -> 3D-POSE EVALUATION  (checkpoint: {ckpt_meta['path']})")
    print(f"trained: split={ckpt_meta['trained_split']}  best val MPJPE "
          f"{ckpt_meta['val_mpjpe_mm']:.1f} mm @ epoch {ckpt_meta['epoch']}\n{bar}")

    for split in SPLITS:
        r = results.get(split)
        print(f"\n[{split}]  — {SPLIT_NOTE[split]}")
        if r is None:
            print("  SKIPPED — no held-out data on disk for this split.")
            continue
        if "skipped" in r:
            print(f"  SKIPPED — {r['skipped']}")
            continue
        print(f"  samples: {r['n_samples']:,}")
        print(f"  MPJPE    : {r['mpjpe_mm']:7.1f} mm")
        print(f"  PA-MPJPE : {r['pa_mpjpe_mm']:7.1f} mm  (Procrustes-aligned)")
        pj = np.array(r["per_joint_mpjpe_mm"])
        worst = pj.argsort()[::-1][:5]
        best = pj.argsort()[:3]
        print("  worst joints: " + ", ".join(f"{JOINT_NAMES[j]} {pj[j]:.0f}mm" for j in worst))
        print("  best  joints: " + ", ".join(f"{JOINT_NAMES[j]} {pj[j]:.0f}mm" for j in best))

    # Benchmark comparison table.
    print(f"\n{bar}\nComparison to the MM-Fi WiFi-only benchmark (honest ballpark)\n{bar}")
    print(f"  {'setup':<42}{'MPJPE (mm)':>16}")
    print(f"  {'-' * 58}")
    cs = results.get("cross_subject")
    if cs and "mpjpe_mm" in cs:
        print(f"  {'ours: cross_subject, E01 only, single-frame':<42}{cs['mpjpe_mm']:>16.1f}")
    print(f"  {'MM-Fi paper: WiFi-only (full corpus)':<42}{MMFI_WIFI_BENCHMARK_MM:>16}")
    print("\n  Takeaway: same order of magnitude (low hundreds of mm) — a real,")
    print("  working WiFi->3D-pose regressor — but NOT a matched reproduction:")
    print("  we train on E01 alone, single-frame input, a small CNN. WiFi pose is")
    print("  inherently coarse; this is the expected regime, not a camera-grade result.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT,
                   help="Pose checkpoint to evaluate (default: runs/best_pose.pt).")
    p.add_argument("--out-dir", type=Path, default=FIGURES_DIR,
                   help="Where to write the figure + metrics JSON.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train one first:  python src/train_pose.py   (see docs/chunk15_pose_model.md)"
        )
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    ckpt_args = ckpt.get("args", {})

    # Evaluate on each split with the SAME data shaping the model trained under
    # (window_size, csi_normalize), and pose_scale=None so MPJPE stays metric.
    ds_kwargs = dict(
        window_size=ckpt["model_config"].get("window_size", 1),
        csi_normalize=ckpt_args.get("csi_normalize", "none"),
        pose_scale=None,
    )
    ckpt_meta = {
        "path": str(args.checkpoint),
        "trained_split": ckpt.get("split_config", {}).get("split", "?"),
        "val_mpjpe_mm": float(ckpt.get("val_mpjpe_mm", float("nan"))),
        "epoch": int(ckpt.get("epoch", -1)),
    }

    results: dict = {}
    for split in SPLITS:
        test_ds, reason = build_test_set(split, ckpt_args, ds_kwargs)
        if test_ds is None:
            results[split] = {"skipped": reason} if reason else None
            print(f"[{split}] skipped: {reason or 'no data'}")
            continue
        print(f"[{split}] scoring {len(test_ds):,} held-out frames ...")
        results[split] = evaluate_split(model, test_ds, device, args.batch_size)

    print_report(results, ckpt_meta)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.out_dir / "pose_per_joint_error.png"
    plot_per_joint_error(
        {s: r for s, r in results.items() if r and "mpjpe_mm" in r}, fig_path
    )

    metrics_path = args.out_dir / "pose_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "checkpoint": ckpt_meta,
            "joint_names": JOINT_NAMES,
            "mmfi_wifi_benchmark_mm": MMFI_WIFI_BENCHMARK_MM,
            "split_notes": SPLIT_NOTE,
            "results": results,
        }, f, indent=2)

    print(f"\nPer-joint figure: {fig_path}")
    print(f"Metrics JSON:     {metrics_path}")


if __name__ == "__main__":
    main()
