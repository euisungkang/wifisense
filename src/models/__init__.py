"""Model definitions and a small name -> constructor registry.

``build_model(name, **overrides)`` lets train.py / evaluate.py instantiate a
model from a string (argparse) and a config dict (checkpoint), keeping the
checkpoint format model-agnostic.
"""

from __future__ import annotations

import torch.nn as nn

from .bilstm import BiLSTM

__all__ = ["BiLSTM", "MODELS", "build_model"]

# Registered model name -> class.  Add new architectures here.
MODELS: dict[str, type[nn.Module]] = {
    "bilstm": BiLSTM,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name.

    Args:
        name: Key into ``MODELS`` (e.g. ``"bilstm"``).
        **kwargs: Constructor overrides / saved config.
    """
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name](**kwargs)
