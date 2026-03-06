"""EmbeddingLayer: composes piece, color, and square embeddings into token vectors.

Three independent embedding tables are summed element-wise then passed through
LayerNorm and Dropout. This gives the model independent learned representations
for piece type, piece ownership, and board position.

Vocabulary sizes:
  piece_emb:  8  (CLS=0, EMPTY=1, PAWN=2, KNIGHT=3, BISHOP=4, ROOK=5, QUEEN=6, KING=7)
  color_emb:  3  (EMPTY=0, PLAYER=1, OPPONENT=2)
  square_emb: 65 (index 0=CLS position, 1..64=squares a1..h8)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.protocols import Embeddable

D_MODEL: int = 256
PIECE_VOCAB_SIZE: int = 8
COLOR_VOCAB_SIZE: int = 3
SQUARE_VOCAB_SIZE: int = 65  # CLS + 64 squares
DROPOUT: float = 0.1


class EmbeddingLayer(nn.Module):
    """Composes piece + color + square embeddings into per-token vectors.

    Implements the Embeddable protocol.

    Internal structure:
      piece_emb:  nn.Embedding(8, 256)
      color_emb:  nn.Embedding(3, 256)
      square_emb: nn.Embedding(65, 256)
      layer_norm: nn.LayerNorm(256)
      dropout:    nn.Dropout(0.1)

    Output = LayerNorm(Dropout(piece_emb + color_emb + square_emb)).

    Example:
        >>> layer = EmbeddingLayer()
        >>> out = layer.embed(board_tokens, color_tokens)  # [B, 65, 256]
    """

    def __init__(self) -> None:
        """Initialize embedding tables, LayerNorm, and Dropout.

        All three embedding tables are independently learned. Square embeddings
        may optionally be initialized with 2D coordinate encodings as a geometric
        prior — this is handled at initialization time by the implementation.

        Example:
            >>> layer = EmbeddingLayer()
        """
        super().__init__()
        raise NotImplementedError("To be implemented")

    def embed(self, board_tokens: Tensor, color_tokens: Tensor) -> Tensor:
        """Compose three embedding streams into a single [B, 65, 256] tensor.

        Automatically constructs the square index tensor (0..64) from the
        batch dimension and sequence length. Sums piece + color + square embeddings,
        applies LayerNorm, then Dropout.

        Args:
            board_tokens: torch.long [B, 65]. Piece-type indices per position.
            color_tokens: torch.long [B, 65]. Color indices per position.

        Returns:
            torch.float [B, 65, 256]. Normalized, dropped-out token embeddings.

        Example:
            >>> layer = EmbeddingLayer()
            >>> out = layer.embed(torch.zeros(4, 65, dtype=torch.long),
            ...                   torch.zeros(4, 65, dtype=torch.long))
            >>> out.shape
            torch.Size([4, 65, 256])
        """
        raise NotImplementedError("To be implemented")
        return torch.zeros(board_tokens.size(0), 65, D_MODEL)

    def forward(self, board_tokens: Tensor, color_tokens: Tensor) -> Tensor:
        """nn.Module forward — delegates to embed().

        Args:
            board_tokens: torch.long [B, 65].
            color_tokens: torch.long [B, 65].

        Returns:
            torch.float [B, 65, 256].
        """
        return self.embed(board_tokens, color_tokens)
