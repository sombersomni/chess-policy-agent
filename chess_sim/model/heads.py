"""PredictionHeads: maps CLS embedding to two square-prediction logit tensors.

Two independent Linear(256, 64) layers predict:
  1. Player's source square (which piece moves)
  2. Player's target square (where the piece moves)

Each ply is always predicted from the side-to-move's perspective.
Piece identity is implicit: given the predicted source square and the known board
state, P(piece | src_sq, board) = 1 — no separate piece-type head is needed.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from chess_sim.config import ModelConfig
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

    def __init__(
        self, model_cfg: ModelConfig | None = None
    ) -> None:
        """Initialize two independent linear prediction heads.

        Falls back to d_model=256 when model_cfg is None.

        Args:
            model_cfg: Optional ModelConfig. When None, uses d_model=256.

        Example:
            >>> ph = PredictionHeads()
            >>> ph = PredictionHeads(ModelConfig(d_model=128))
            >>> len(ph.heads)
            2
        """
        super().__init__()
        d_model = model_cfg.d_model if model_cfg else 256
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, N_SQUARES) for _ in range(N_HEADS)]
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
