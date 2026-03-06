"""PredictionHeads: maps CLS embedding to two square-prediction logit tensors.

Two independent Linear(256, 64) layers predict:
  1. Player's source square (which piece moves)
  2. Player's target square (where the piece moves)

Each ply is always predicted from the side-to-move's perspective.
Piece identity is implicit: given the predicted source square and the known board
state, P(piece | src_sq, board) = 1 — no separate piece-type head is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.model.embedding import D_MODEL
from chess_sim.protocols import Predictable
from chess_sim.types import PredictionOutput

N_SQUARES: int = 64
N_HEADS: int = 2


class PredictionHeads(nn.Module):
    """Two independent linear heads mapping CLS embedding to square logits.

    Implements the Predictable protocol.

    Both heads share the same architecture (Linear(256, 64)) but have
    independent parameters — no weight sharing between heads.

    Attributes:
        heads: nn.ModuleList of 2 x nn.Linear(256, 64).
               Order: [src_sq, tgt_sq].

    Example:
        >>> ph = PredictionHeads()
        >>> out = ph.predict(cls_embedding)  # cls_embedding: [B, 256]
        >>> out.src_sq_logits.shape
        torch.Size([4, 64])
    """

    def __init__(self) -> None:
        """Initialize two independent Linear(256, 64) prediction heads.

        Uses nn.ModuleList for DRY construction. Index mapping:
          0 -> src_sq_head, 1 -> tgt_sq_head.

        Example:
            >>> ph = PredictionHeads()
            >>> len(ph.heads)
            2
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Linear(D_MODEL, N_SQUARES) for _ in range(N_HEADS)]
        )

    def predict(self, cls_embedding: Tensor) -> PredictionOutput:
        """Apply both heads to the CLS embedding and return logit tensors.

        Args:
            cls_embedding: torch.float [B, 256]. Global board representation.

        Returns:
            PredictionOutput with two tensors each of shape [B, 64].

        Example:
            >>> ph = PredictionHeads()
            >>> preds = ph.predict(torch.randn(4, 256))
            >>> preds.tgt_sq_logits.shape
            torch.Size([4, 64])
        """
        logits = [h(cls_embedding) for h in self.heads]
        return PredictionOutput(
            src_sq_logits=logits[0],
            tgt_sq_logits=logits[1],
        )

    def forward(self, cls_embedding: Tensor) -> PredictionOutput:
        """nn.Module forward — delegates to predict().

        Args:
            cls_embedding: torch.float [B, 256].

        Returns:
            PredictionOutput namedtuple.
        """
        return self.predict(cls_embedding)
