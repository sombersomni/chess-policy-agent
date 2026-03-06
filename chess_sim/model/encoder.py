"""ChessEncoder: full transformer forward pass from token indices to embeddings.

Owns an EmbeddingLayer and a 6-layer TransformerEncoder. After encoding, the
output is split: index 0 becomes cls_embedding, indices 1:65 become square_embeddings.

Hyperparameters (from design doc):
  d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.model.embedding import D_MODEL, EmbeddingLayer
from chess_sim.protocols import Encodable
from chess_sim.types import EncoderOutput

N_HEADS: int = 8
N_LAYERS: int = 6
DIM_FEEDFORWARD: int = 1024
DROPOUT: float = 0.1


class ChessEncoder(nn.Module):
    """Runs the BERT-style transformer encoder over a 65-token board sequence.

    Implements the Encodable protocol.

    Internal structure:
      embedding:   EmbeddingLayer
      transformer: nn.TransformerEncoder (6 layers, 8 heads, batch_first=True)

    The CLS token at position 0 accumulates global board context through
    bidirectional self-attention across all 65 positions.

    Example:
        >>> enc = ChessEncoder()
        >>> out = enc.encode(board_tokens, color_tokens)
        >>> out.cls_embedding.shape
        torch.Size([4, 256])
    """

    def __init__(self) -> None:
        """Initialize EmbeddingLayer and TransformerEncoder with design-doc hyperparams.

        TransformerEncoderLayer config: d_model=256, nhead=8, dim_feedforward=1024,
        dropout=0.1, batch_first=True. Stacked 6 layers.

        Example:
            >>> enc = ChessEncoder()
        """
        super().__init__()
        self.embedding = EmbeddingLayer()
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=N_LAYERS,
            enable_nested_tensor=False,
        )

    def encode(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        activity_tokens: Tensor,
    ) -> EncoderOutput:
        """Encode a batch of board states into CLS and per-square embeddings.

        Calls EmbeddingLayer with all four streams, passes through
        TransformerEncoder, then splits: index 0 -> cls, 1:65 -> squares.

        Args:
            board_tokens: torch.long [B, 65]. Piece-type indices.
            color_tokens: torch.long [B, 65]. Color indices.
            activity_tokens: torch.long [B, 65]. Activity scores 0-8.

        Returns:
            EncoderOutput(cls_embedding=[B, 256],
                          square_embeddings=[B, 64, 256]).

        Example:
            >>> out = enc.encode(bt, ct, at)
            >>> out.square_embeddings.shape
            torch.Size([4, 64, 256])
        """
        x = self.embedding(
            board_tokens, color_tokens, activity_tokens
        )
        encoded = self.transformer(x)
        return EncoderOutput(
            cls_embedding=encoded[:, 0, :],
            square_embeddings=encoded[:, 1:, :],
        )

    def forward(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        activity_tokens: Tensor,
    ) -> EncoderOutput:
        """nn.Module forward -- delegates to encode().

        Args:
            board_tokens: torch.long [B, 65].
            color_tokens: torch.long [B, 65].
            activity_tokens: torch.long [B, 65].

        Returns:
            EncoderOutput namedtuple.
        """
        return self.encode(
            board_tokens, color_tokens, activity_tokens
        )
