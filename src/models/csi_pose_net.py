"""CSI → 3D-pose regression network for MM-Fi (Phase 3).

This is the project's first *regression* model: every model before it
(``BiLSTM``, ``BVPCNNRNN``) emits class logits and is scored on accuracy. This
one emits **continuous 3D joint coordinates** — a ``(17, 3)`` pose — and is
scored on MPJPE (a distance in millimetres). There is no softmax and no class
axis; the head is a linear layer whose ``n_joints * 3`` outputs are reshaped
into joint coordinates.

Input shape
-----------
A sample is a *centered CSI window* ``(W, 3, 114, 10)`` (see
``src/data/pose_preprocess.py``): ``W`` frames, each 3 antennas × 114 subcarriers
× 10 packets. With the benchmark-faithful default ``window_size = 1`` that is
``(1, 3, 114, 10)`` — one frame → one pose, matching MM-Fi's published
WiFi→pose protocol. ``forward`` accepts a batched ``(B, W, 3, 114, 10)`` (or a
single-frame ``(B, 3, 114, 10)``, treated as ``W = 1``).

Architecture (deliberately modest — get this converging before anything fancier)
--------------------------------------------------------------------------------
The MM-Fi paper's WiFi-pose baseline is a small CNN regressor over the
``(3, 114, 10)`` amplitude map, not a Transformer. We mirror that: fold the
window's frames into the channel axis and run a 2-D CNN over the
(subcarrier × packet) plane, then flatten and regress the joint coordinates with
a 2-layer MLP head.

Shape walk-through (defaults, batch ``B``, ``window_size = 1``)::

    input            (B, 1, 3, 114, 10)
    fold W into ch   (B, 3, 114, 10)        in_channels = W * 3
    conv block 1     (B,  32, 57, 5)        3x3, BN, ReLU, MaxPool2
    conv block 2     (B,  64, 28, 2)        3x3, BN, ReLU, MaxPool2
    conv block 3     (B, 128,  4, 2)        3x3, BN, ReLU, AdaptiveAvgPool(4,2)
    flatten          (B, 1024)
    FC + ReLU + drop (B, 256)
    FC (head)        (B, 51)                = n_joints * 3
    reshape          (B, 17, 3)

Parameter count at the defaults is ~0.4 M — in the same modest band as the
other models here, and far from a Transformer. The output is a *root-relative*
pose because the Dataset's target is root-centered (pelvis at the origin); the
network regresses posture, not the person's absolute location in the room.

Registered as ``"csi_pose_net"`` in ``src/models/__init__.py`` so checkpoints
self-describe and ``build_model`` can reconstruct it from a saved ``config``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CSIPoseNet"]


class CSIPoseNet(nn.Module):
    """Small CNN encoder + MLP head regressing a ``(n_joints, 3)`` pose from CSI.

    Args:
        n_joints: number of body joints to regress (17 for MM-Fi / Human3.6M).
        window_size: CSI frames per input window. Folded into the conv input
            channels (``in_channels = window_size * n_antennas``). Default 1 =
            MM-Fi benchmark per-frame protocol.
        n_antennas: CSI antenna count per frame (3 for MM-Fi).
        conv_channels: output channels of the three conv blocks (length-3).
        head_hidden: width of the hidden FC layer in the regression head.
        pool: spatial size ``(subcarrier, packet)`` the third block is adaptive-
            pooled to; the flattened feature width is ``conv_channels[2]*pool[0]*pool[1]``.
        dropout: applied on the pooled feature before the output layer.

    ``forward`` accepts ``(B, W, 3, 114, 10)`` (or ``(B, 3, 114, 10)`` for W=1)
    and returns ``(B, n_joints, 3)`` coordinates in the same (root-relative)
    space as the Dataset's target.
    """

    def __init__(
        self,
        n_joints: int = 17,
        window_size: int = 1,
        n_antennas: int = 3,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
        head_hidden: int = 256,
        pool: tuple[int, int] = (4, 2),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(conv_channels) != 3:
            raise ValueError(f"conv_channels must have length 3, got {conv_channels}")
        if len(pool) != 2:
            raise ValueError(f"pool must be a (height, width) pair, got {pool}")
        self.n_joints = n_joints
        self.window_size = window_size
        self.n_antennas = n_antennas
        self.conv_channels = tuple(conv_channels)
        self.head_hidden = head_hidden
        self.pool = tuple(pool)
        self.dropout_p = dropout

        in_ch = window_size * n_antennas
        c1, c2, c3 = conv_channels
        # 2-D CNN over the (subcarrier x packet) plane. Two MaxPools shrink the
        # 114x10 map (114->57->28 on the subcarrier axis, 10->5->2 on packets),
        # then an adaptive average pool fixes the final map at ``pool`` so the
        # flattened width is independent of the exact arithmetic above.
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 114x10 -> 57x5
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 57x5 -> 28x2
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(self.pool),  # 28x2 -> pool
        )
        self.feat_dim = c3 * self.pool[0] * self.pool[1]

        # Regression head: flatten -> hidden -> (n_joints * 3). No activation on
        # the output (coordinates are unbounded, in metres).
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_joints * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a CSI window to a ``(B, n_joints, 3)`` pose.

        Args:
            x: ``(B, W, 3, 114, 10)`` CSI window, or ``(B, 3, 114, 10)`` for W=1.
        Returns:
            ``(B, n_joints, 3)`` root-relative joint coordinates.
        """
        if x.dim() == 4:  # (B, 3, 114, 10) -> (B, 1, 3, 114, 10)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(
                f"Expected (B, W, 3, 114, 10) input, got shape {tuple(x.shape)}"
            )
        b, w, a, s, p = x.shape
        # Fold the window's frames into the channel axis: (B, W*A, S, P).
        x = x.reshape(b, w * a, s, p)
        x = self.cnn(x)
        x = x.reshape(b, self.feat_dim)
        x = self.head(x)
        return x.reshape(b, self.n_joints, 3)

    @property
    def config(self) -> dict:
        """Hyperparameters needed to reconstruct this model from a checkpoint."""
        return {
            "n_joints": self.n_joints,
            "window_size": self.window_size,
            "n_antennas": self.n_antennas,
            "conv_channels": list(self.conv_channels),
            "head_hidden": self.head_hidden,
            "pool": list(self.pool),
            "dropout": self.dropout_p,
        }
