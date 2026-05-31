"""CNN-RNN gesture classifier for Widar3.0 BVP volumes.

A BVP sample is a ``(T, 20, 20)`` volume: ``T`` timesteps, each a 20x20 grid of
motion energy over a 2-D body-frame velocity space (x-velocity x y-velocity).
The natural architecture mirrors that structure — *space first, then time*:

    per-timestep 2-D CNN   -> a spatial "what velocities are active now" feature
    GRU over the T features -> how that velocity pattern evolves through the gesture

This is the SenseFi ``Widar_CNN_GRU`` idea (``vendor/SenseFi/widar_model.py``)
brought into this project's conventions: a ``config`` property so checkpoints
self-describe and ``build_model`` can reconstruct it, an explicit ``num_classes``
(22 gestures across the corpus), and a few exposed hyperparameters. Unlike the
SenseFi reference — which fixes ``T = 22`` by folding time into the channel axis
of an MLP/CNN — this model is *time-length agnostic*: it runs the same CNN on
every one of the ``T`` frames (whatever ``T`` is) and lets the GRU consume the
resulting sequence, so the chunk-11 ``pad_or_truncate`` length is a free choice.

Shape walk-through (defaults, batch ``B``, ``target_T = 32``)::

    input            (B, 32, 20, 20)
    fold time        (B*32, 1, 20, 20)
    conv block 1     (B*32, 16, 10, 10)   3x3, BN, ReLU, MaxPool2
    conv block 2     (B*32, 32,  5,  5)   3x3, BN, ReLU, MaxPool2
    conv block 3     (B*32, 64,  2,  2)   3x3, BN, ReLU, AdaptiveAvgPool(2)
    flatten          (B*32, 256)
    unfold time      (B, 32, 256)
    BiGRU            final fwd+bwd hidden -> (B, 2*hidden)
    dropout + FC     (B, num_classes)

Parameter count at the defaults is ~0.33 M (well inside the 100k-500k target).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BVPCNNRNN"]


class BVPCNNRNN(nn.Module):
    """CNN-RNN over BVP ``(B, T, 20, 20)`` volumes.

    Each timestep's 20x20 velocity grid is encoded by a small 2-D CNN into a
    spatial feature vector; a (bi)GRU then aggregates the ``T`` vectors and the
    final hidden state is classified.

    Args:
        num_classes: number of gesture classes (22 across the full Widar corpus;
            fewer when a split scopes to a subset).
        conv_channels: output channels of the three conv blocks. Length-3 tuple.
        gru_hidden: GRU hidden units per direction.
        gru_layers: stacked GRU layers.
        bidirectional: run the GRU both ways and concat the final hidden states.
        dropout: applied between stacked GRU layers (PyTorch only wires inter-layer
            dropout when ``gru_layers > 1``) and on the pooled feature before the
            classifier head.
        pool: spatial size the third block is adaptive-pooled to (per side); the
            flattened per-timestep feature width is ``conv_channels[2] * pool**2``.

    ``forward`` accepts ``(B, T, 20, 20)`` (or ``(B, T, 1, 20, 20)``) and returns
    ``(B, num_classes)`` logits.
    """

    def __init__(
        self,
        num_classes: int = 22,
        conv_channels: tuple[int, int, int] = (16, 32, 64),
        gru_hidden: int = 128,
        gru_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pool: int = 2,
    ) -> None:
        super().__init__()
        if len(conv_channels) != 3:
            raise ValueError(f"conv_channels must have length 3, got {conv_channels}")
        self.num_classes = num_classes
        self.conv_channels = tuple(conv_channels)
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.bidirectional = bidirectional
        self.dropout_p = dropout
        self.pool = pool

        c1, c2, c3 = conv_channels
        # Per-timestep spatial encoder. Two MaxPools take 20x20 -> 10x10 -> 5x5,
        # then an adaptive average pool fixes the final map at pool x pool so the
        # flattened width is independent of the exact arithmetic above.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 20 -> 10
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 10 -> 5
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(pool),  # 5 -> pool
        )
        self.feat_dim = c3 * pool * pool

        self.gru = nn.GRU(
            input_size=self.feat_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # PyTorch ignores (and warns about) dropout when num_layers == 1.
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(gru_hidden * num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a BVP volume to class logits.

        Args:
            x: ``(B, T, 20, 20)`` or ``(B, T, 1, 20, 20)`` motion-energy volume.
        Returns:
            ``(B, num_classes)`` logits.
        """
        if x.dim() == 5:  # (B, T, 1, 20, 20) -> (B, T, 20, 20)
            x = x.squeeze(2)
        if x.dim() != 4:
            raise ValueError(f"Expected (B, T, 20, 20) input, got shape {tuple(x.shape)}")
        b, t, h, w = x.shape

        # Fold time into the batch so the CNN sees one frame at a time.
        x = x.reshape(b * t, 1, h, w)
        x = self.cnn(x)                  # (B*T, c3, pool, pool)
        x = x.reshape(b, t, self.feat_dim)  # unfold time -> sequence of features

        # hn: (num_layers * num_directions, B, gru_hidden)
        _, hn = self.gru(x)
        if self.bidirectional:
            # Last layer's forward state is hn[-2], backward is hn[-1].
            feat = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            feat = hn[-1]
        feat = self.dropout(feat)
        return self.fc(feat)

    @property
    def config(self) -> dict:
        """Hyperparameters needed to reconstruct this model from a checkpoint."""
        return {
            "num_classes": self.num_classes,
            "conv_channels": list(self.conv_channels),
            "gru_hidden": self.gru_hidden,
            "gru_layers": self.gru_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout_p,
            "pool": self.pool,
        }
