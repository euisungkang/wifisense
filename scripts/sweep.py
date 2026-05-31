#!/usr/bin/env python
"""Multi-seed robustness sweep for a single training config.

Phase 2 of the train workflow (see README): once a hyperparameter config is
chosen on the val set, run that *same* config across several seeds to gauge
how stable the result is, and to get a pool of checkpoints to pick the final
model from.

Each seed is just a normal ``python -m src.train`` run — this script shells
out once per seed, grouping the runs under ``runs/sweep_{timestamp}/seed_{s}/``
so they're easy to find and don't clutter the top-level ``runs/``.  After all
seeds finish it reads each run's ``metrics.json`` and reports
**best validation accuracy** as ``mean ± std``, then names the best-by-val run.

It deliberately never looks at the test split: selecting on val keeps the
single, final ``src.evaluate --split test`` peek an unbiased estimate.

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/sweep.py --seeds 42 43 44 45 46 \
        --lr 1e-3 --hidden 64 --layers 2 --dropout 0.3 --epochs 80

Then promote the winner (or pass --promote to do it automatically)::

    cp runs/sweep_<ts>/seed_<best>/best.pt runs/best_bilstm.pt
    python -m src.evaluate --split test

Runnable directly (not via ``-m``): it shells out to ``python -m src.train``
per seed via ``sys.executable``, so it needs no ``src`` import of its own.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
STABLE_CKPT = RUNS_ROOT / "best_bilstm.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
        help="Seeds to run the (fixed) config under.",
    )
    p.add_argument("--tag", default=None, help="Optional label appended to the sweep dir name.")
    p.add_argument(
        "--promote", action="store_true",
        help=f"Copy the best-by-val checkpoint to {STABLE_CKPT.relative_to(PROJECT_ROOT)}.",
    )
    p.add_argument(
        "--promote-to", type=Path, default=STABLE_CKPT,
        help="Destination for the promoted checkpoint (default: runs/best_bilstm.pt). "
        "Use e.g. runs/best_bilstm_ntu.pt when sweeping a different dataset.",
    )
    p.add_argument(
        "--data", type=Path, default=None,
        help="Preprocessed .npz path passed through to src.train "
        "(default: train.py's UT-HAR path).",
    )
    # --- passthrough training hyperparameters (the fixed config) -----------
    p.add_argument("--model", default="bilstm")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    return p.parse_args()


def build_train_cmd(args: argparse.Namespace, seed: int, save_dir: Path) -> list[str]:
    """Construct the `python -m src.train` command for one seed."""
    cmd = [
        sys.executable, "-m", "src.train",
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--hidden", str(args.hidden),
        "--layers", str(args.layers),
        "--dropout", str(args.dropout),
        "--seed", str(seed),
        "--save-dir", str(save_dir),
    ]
    if args.data is not None:
        cmd += ["--data", str(args.data)]
    return cmd


def main() -> None:
    args = parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = f"sweep_{stamp}" + (f"_{args.tag}" if args.tag else "")
    sweep_dir = RUNS_ROOT / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "patience": args.patience, "hidden": args.hidden,
        "layers": args.layers, "dropout": args.dropout,
        "data": str(args.data) if args.data is not None else None,
    }
    print(f"Sweep dir: {sweep_dir}")
    print(f"Config:    {config}")
    print(f"Seeds:     {args.seeds}\n")

    results: list[dict] = []
    for i, seed in enumerate(args.seeds, start=1):
        seed_dir = sweep_dir / f"seed_{seed}"
        print(f"\n{'=' * 70}\n=== Seed {seed}  ({i}/{len(args.seeds)})  ->  {seed_dir}\n{'=' * 70}")
        cmd = build_train_cmd(args, seed, seed_dir)
        # Inherit stdout/stderr so per-epoch progress streams live.
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if proc.returncode != 0:
            print(f"!! Seed {seed} failed (exit {proc.returncode}); skipping in aggregation.")
            results.append({"seed": seed, "ok": False})
            continue

        metrics = json.loads((seed_dir / "metrics.json").read_text())
        results.append({
            "seed": seed,
            "ok": True,
            "best_val_acc": metrics["best_val_acc"],
            "best_epoch": metrics["best_epoch"],
            "epochs_run": metrics["epochs_run"],
            "checkpoint": str(seed_dir / "best.pt"),
        })

    ok = [r for r in results if r["ok"]]
    if not ok:
        print("\nAll runs failed; nothing to aggregate.")
        sys.exit(1)

    accs = [r["best_val_acc"] for r in ok]
    mean = statistics.mean(accs)
    std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    best = max(ok, key=lambda r: r["best_val_acc"])

    # --- summary table ------------------------------------------------------
    print(f"\n{'=' * 70}\nSWEEP SUMMARY  ({len(ok)}/{len(args.seeds)} runs ok)\n{'=' * 70}")
    print(f"{'seed':>6}  {'best_val_acc':>13}  {'best_epoch':>10}  {'epochs_run':>10}")
    for r in ok:
        marker = "  <-- best" if r is best else ""
        print(f"{r['seed']:>6}  {r['best_val_acc']:>13.4f}  {r['best_epoch']:>10}  {r['epochs_run']:>10}{marker}")
    print(f"\nval acc: mean {mean:.4f} ± {std:.4f}  (min {min(accs):.4f}, max {max(accs):.4f})")
    print(f"best by val: seed {best['seed']} -> {best['checkpoint']}")

    summary = {
        "sweep_dir": str(sweep_dir),
        "config": config,
        "seeds": args.seeds,
        "results": results,
        "val_acc_mean": mean,
        "val_acc_std": std,
        "val_acc_min": min(accs),
        "val_acc_max": max(accs),
        "best_seed": best["seed"],
        "best_checkpoint": best["checkpoint"],
    }
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON -> {sweep_dir / 'summary.json'}")

    # --- optional promotion to the stable path ------------------------------
    if args.promote:
        args.promote_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best["checkpoint"], args.promote_to)
        print(f"Promoted best-by-val checkpoint -> {args.promote_to}")
        print("Next: python -m src.evaluate --split test")
    else:
        print("\nNot promoted (no --promote). To freeze the winner for chunk 6:")
        print(f"  cp {best['checkpoint']} {args.promote_to}")
        print("  python -m src.evaluate --split test")


if __name__ == "__main__":
    main()
