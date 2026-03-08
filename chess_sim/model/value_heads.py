"""ValueHeads: two linear value heads on the shared encoder CLS embedding.

V_win predicts probability of winning from the current board state.
V_surprise predicts the expected surprise score from the current state.
Both heads receive a detached CLS embedding so value gradients do not
distort the shared encoder's move-prediction representations.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from chess_sim.types import ValueHeadOutput


class ValueHeads(nn.Module):
    """Two linear value heads on the shared encoder CLS embedding.

    V_win predicts probability of winning from the current board state.
    V_surprise predicts the expected surprise score from the current
    state. Both heads receive a detached CLS embedding so value
    gradients do not distort the shared encoder representations.
    """

    def __init__(self, d_model: int) -> None:
        """Initialise win and surprise projection heads.

        Args:
            d_model: Dimensionality of the encoder CLS embedding.
        """
        super().__init__()
        self._win_head = nn.Linear(d_model, 1)
        self._surprise_head = nn.Linear(d_model, 1)

    def forward(self, cls_emb: Tensor) -> ValueHeadOutput:
        """Project detached CLS embedding through both value heads.

        IMPORTANT: cls_emb must be detached by the caller before this
        forward pass (cls_emb.detach()) so value MSE gradients do not
        flow back into the encoder.

        Args:
            cls_emb: CLS embedding tensor of shape [B, d_model].
                     Must be detached.

        Returns:
            ValueHeadOutput with v_win and v_surprise each [B, 1].
        """
        raise NotImplementedError("To be implemented")
