"""BiLSTM classifier for UT-HAR CSI, ported from SenseFi.

Reference implementation: ``vendor/SenseFi/UT_HAR_model.py`` →
``UT_HAR_BiLSTM`` (commit vendored under ``vendor/SenseFi``).

The SenseFi original is a single-layer bidirectional LSTM that feeds only
``ht[-1]`` — the *backward* hidden state of the last layer — into the
classifier, silently discarding the forward direction.  This port keeps the
same recurrent backbone but:

    * exposes ``hidden_size``, ``num_layers``, ``dropout`` and
      ``bidirectional`` as constructor arguments (defaults match this
      project's config: hidden=64, 2 layers, bidirectional, dropout=0.3);
    * concatenates the final forward *and* backward hidden states so the
      classifier sees the full bidirectional summary;
    * uses ``batch_first=True`` and accepts CSI shaped ``(B, T, S)`` (or
      ``(B, 1, T, S)``) directly, matching ``data/processed`` tensors.

For UT-HAR, ``input_size`` is the per-time-step feature width: 90
(30 subcarriers x 3 RX antennas).  ``T`` (250) need not be fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BiLSTM"]


class BiLSTM(nn.Module):
    """Bidirectional LSTM sequence classifier.

    Args:
        input_size: Per-time-step feature dimension (90 for UT-HAR).
        hidden_size: LSTM hidden units per direction.
        num_layers: Stacked LSTM layers.
        num_classes: Output classes (7 for UT-HAR).
        dropout: Applied between stacked LSTM layers (PyTorch only applies
            inter-layer dropout when ``num_layers > 1``) and on the pooled
            feature before the classifier head.
        bidirectional: If True, run forward + backward and concatenate the
            final hidden states (feature width ``2 * hidden_size``).
    """

    def __init__(
        self,
        input_size: int = 90,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # PyTorch ignores dropout when num_layers == 1 (and warns); guard it.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map CSI to class logits.

        Args:
            x: ``(B, T, S)`` or ``(B, 1, T, S)`` CSI amplitudes.
        Returns:
            ``(B, num_classes)`` logits.
        """
        if x.dim() == 4:  # (B, 1, T, S) -> (B, T, S)
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, S) input, got shape {tuple(x.shape)}")

        # hn: (num_layers * num_directions, B, hidden_size)
        _, (hn, _) = self.lstm(x)
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
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout_p,
            "bidirectional": self.bidirectional,
        }
