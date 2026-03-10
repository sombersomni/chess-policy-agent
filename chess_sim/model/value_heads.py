"""Return value critic head: two-layer MLP on detached CLS embedding."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ReturnValueHead(nn.Module):
    """Two-layer MLP critic: d_model -> d_model//2 -> ReLU -> 1.

    Caller must detach cls_emb before passing to prevent encoder gradient
    contamination. This head performs no detach internally.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self._fc1: nn.Linear = nn.Linear(d_model, d_model // 2)
        self._relu: nn.ReLU = nn.ReLU()
        self._fc2: nn.Linear = nn.Linear(d_model // 2, 1)

    def forward(self, cls_emb: Tensor) -> Tensor:
        """Return scalar return estimate per board state.

        Args:
            cls_emb: [B, d_model] CLS embedding, caller must detach.

        Returns:
            [B, 1] scalar return estimate.
        """
        raise NotImplementedError("To be implemented")
