#!/usr/bin/env python3
"""Plot Widar3.0 BVP examples — the first view where WiFi data looks spatial.

For 6 representative gestures, draws a row of 4 evenly-spaced timesteps of one
sample's BVP. Each cell is a 20x20 heatmap of motion energy over the body-frame
velocity plane: x-velocity horizontal, y-velocity vertical, both in [-2, +2] m/s
(see notes/widar_data.md). Bright = more body reflectors moving at that velocity.

Output: figures/widar_bvp_examples.png

Run (from the repo root, with the project env active)::

    conda activate wifisense
    python scripts/visualize_widar_bvp.py

Runnable directly (not via ``-m``): it prepends the repo root to sys.path
itself so the ``src`` package imports resolve.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.widar_loader import (  # noqa: E402
    PROJECT_ROOT,
    VELOCITY_RANGE_MPS,
    load_widar_bvp,
)

# Six gestures with distinct motion signatures. All available for user1 so the
# panel holds the user fixed; we take the first matching sample per gesture.
GESTURES = ["Push&Pull", "Sweep", "Clap", "Slide", "Draw-O(H)", "Draw-N(H)"]
N_STEPS = 4  # timesteps shown per gesture


def pick_timesteps(T: int, n: int) -> np.ndarray:
    """n timestep indices evenly spaced across [0, T), endpoints included."""
    if T <= n:
        return np.arange(T)
    return np.linspace(0, T - 1, n).round().astype(int)


def main() -> None:
    vmin, vmax = VELOCITY_RANGE_MPS
    fig, axes = plt.subplots(
        len(GESTURES), N_STEPS, figsize=(N_STEPS * 2.4, len(GESTURES) * 2.4)
    )

    for row, gesture in enumerate(GESTURES):
        X, md = load_widar_bvp(user=1, gesture=gesture, limit=1)
        if not X:  # fall back to any user if user1 lacks this gesture
            idx = load_widar_bvp(gesture=gesture, limit=1)
            X, md = idx
        bvp = X[0]  # (T, 20, 20)
        meta = md[0]
        T = bvp.shape[0]
        steps = pick_timesteps(T, N_STEPS)
        # shared color scale per gesture so relative brightness is meaningful
        gmax = float(bvp.max()) or 1.0

        for col in range(N_STEPS):
            ax = axes[row, col]
            if col < len(steps):
                t = steps[col]
                # bvp[t] is (vx, vy); transpose so vx is horizontal, vy vertical,
                # and origin='lower' puts +y upward (extent maps bins to m/s).
                ax.imshow(
                    bvp[t].T, origin="lower", cmap="inferno",
                    vmin=0, vmax=gmax, extent=[vmin, vmax, vmin, vmax],
                    aspect="equal",
                )
                ax.set_title(f"t={t}/{T - 1}", fontsize=9)
            else:
                ax.axis("off")
                continue

            if col == 0:
                ax.set_ylabel(f"{gesture}\n$v_y$ (m/s)", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row == len(GESTURES) - 1:
                ax.set_xlabel("$v_x$ (m/s)", fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Widar3.0 BVP — body-frame velocity energy over time "
        "(user1; x-velocity →, y-velocity ↑, ±2 m/s)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = PROJECT_ROOT / "figures" / "widar_bvp_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
