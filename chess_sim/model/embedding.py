"""EmbeddingLayer: composes piece, color, square, and trajectory embeddings.

Four independent embedding tables are summed element-wise then passed through
LayerNorm and Dropout. This gives the model independent learned representations
for piece type, piece ownership, board position, and last-move trajectory.

Vocabulary sizes:
  piece_emb:      8  (CLS=0, EMPTY=1, PAWN=2, KNIGHT=3, BISHOP=4, ROOK=5, QUEEN=6, KING=7)
  color_emb:      3  (EMPTY=0, PLAYER=1, OPPONENT=2)
  square_emb:     65 (index 0=CLS position, 1..64=squares a1..h8)
  trajectory_emb: 5  (0=none, 1=player prev src, 2=player prev tgt,
                      3=opp prev src, 4=opp prev tgt)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.protocols import Embeddable

D_MODEL: int = 256
PIECE_VOCAB_SIZE: int = 8
COLOR_VOCAB_SIZE: int = 3
SQUARE_VOCAB_SIZE: int = 65  # CLS + 64 squares
TRAJECTORY_VOCAB_SIZE: int = 5  # 0=none, 1=pl_src, 2=pl_tgt, 3=opp_src, 4=opp_tgt
DROPOUT: float = 0.1

# Role features: [value, mobility, linearity, diagonality, can_jump, royalty, freq_prior]
_ROLE_FEATURES: list[list[float]] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # CLS
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # EMPTY
    [0.1, 0.1, 0.5, 0.0, 0.0, 0.0, 0.9],  # PAWN
    [0.3, 0.4, 0.0, 0.0, 1.0, 0.0, 0.4],  # KNIGHT
    [0.3, 0.6, 0.0, 1.0, 0.0, 0.0, 0.3],  # BISHOP
    [0.5, 0.7, 1.0, 0.0, 0.0, 0.0, 0.3],  # ROOK
    [0.9, 1.0, 1.0, 1.0, 0.0, 0.0, 0.1],  # QUEEN
    [1.0, 0.2, 0.5, 0.5, 0.0, 1.0, 0.1],  # KING
]

# Scale factor for piece role init (matches default Embedding init std).
_ROLE_SCALE: float = 0.02


class EmbeddingLayer(nn.Module):
    """Composes piece + color + square + activity embeddings.

    Implements the Embeddable protocol.

    Internal structure:
      piece_emb:      nn.Embedding(8, 256)  — role-feature init
      color_emb:      nn.Embedding(3, 256)
      square_emb:     nn.Embedding(65, 256) — geometric sin/cos init
      trajectory_emb: nn.Embedding(5, 256)
      layer_norm:     nn.LayerNorm(256)
      dropout:        nn.Dropout(0.1)

    Output = LayerNorm(Dropout(piece + color + square + trajectory)).

    Example:
        >>> layer = EmbeddingLayer()
        >>> out = layer.embed(bt, ct, tt)  # [B, 65, 256]
    """

    def __init__(self) -> None:
        """Initialize four embedding tables, LayerNorm, Dropout.

        Calls _init_square_emb() and _init_piece_emb() to apply
        geometric and role-feature priors respectively.

        Example:
            >>> layer = EmbeddingLayer()
        """
        super().__init__()
        self.piece_emb = nn.Embedding(
            PIECE_VOCAB_SIZE, D_MODEL
        )
        self.color_emb = nn.Embedding(
            COLOR_VOCAB_SIZE, D_MODEL
        )
        self.square_emb = nn.Embedding(
            SQUARE_VOCAB_SIZE, D_MODEL
        )
        self.trajectory_emb = nn.Embedding(
            TRAJECTORY_VOCAB_SIZE, D_MODEL
        )
        self.layer_norm = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self._init_square_emb()
        self._init_piece_emb()

    def _init_square_emb(self) -> None:
        """Initialize square_emb with 2D sin/cos geometric encoding.

        Index 0 (CLS): all zeros.
        For s=1..64: rank=(s-1)//8, file=(s-1)%8.
        For d=0..255: cycle through 4 patterns based on d%4.
        """
        weights = torch.zeros(SQUARE_VOCAB_SIZE, D_MODEL)
        for s in range(1, SQUARE_VOCAB_SIZE):
            rank = (s - 1) // 8
            file = (s - 1) % 8
            for d in range(D_MODEL):
                freq = 10000.0 ** (d / D_MODEL)
                pattern = d % 4
                if pattern == 0:
                    weights[s, d] = math.sin(rank / freq)
                elif pattern == 1:
                    weights[s, d] = math.cos(rank / freq)
                elif pattern == 2:
                    weights[s, d] = math.sin(file / freq)
                else:
                    weights[s, d] = math.cos(file / freq)
        with torch.no_grad():
            self.square_emb.weight.copy_(weights)

    def _init_piece_emb(self) -> None:
        """Initialize piece_emb with tiled role-feature vectors.

        Each 7-element role vector is tiled to fill 256 dims,
        then scaled by 0.02.
        """
        weights = torch.zeros(PIECE_VOCAB_SIZE, D_MODEL)
        for i, features in enumerate(_ROLE_FEATURES):
            ft = torch.tensor(features, dtype=torch.float)
            # Tile to fill D_MODEL dims
            reps = math.ceil(D_MODEL / len(features))
            tiled = ft.repeat(reps)[:D_MODEL]
            weights[i] = tiled * _ROLE_SCALE
        with torch.no_grad():
            self.piece_emb.weight.copy_(weights)

    def embed(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        trajectory_tokens: Tensor,
    ) -> Tensor:
        """Compose four embedding streams into [B, 65, 256].

        Sums piece + color + square + trajectory embeddings,
        applies LayerNorm, then Dropout.

        Args:
            board_tokens: torch.long [B, 65].
            color_tokens: torch.long [B, 65].
            trajectory_tokens: torch.long [B, 65]. Values 0-4.

        Returns:
            torch.float [B, 65, 256].

        Example:
            >>> layer = EmbeddingLayer()
            >>> z = torch.zeros(4, 65, dtype=torch.long)
            >>> out = layer.embed(z, z, z)
            >>> out.shape
            torch.Size([4, 65, 256])
        """
        B, S = board_tokens.size()
        sq_idx = torch.arange(
            S, device=board_tokens.device
        ).unsqueeze(0).expand(B, S)
        x = (
            self.piece_emb(board_tokens)
            + self.color_emb(color_tokens)
            + self.square_emb(sq_idx)
            + self.trajectory_emb(trajectory_tokens)
        )
        return self.dropout(self.layer_norm(x))

    def forward(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        trajectory_tokens: Tensor,
    ) -> Tensor:
        """nn.Module forward -- delegates to embed().

        Args:
            board_tokens: torch.long [B, 65].
            color_tokens: torch.long [B, 65].
            trajectory_tokens: torch.long [B, 65]. Values 0-4.

        Returns:
            torch.float [B, 65, 256].
        """
        return self.embed(
            board_tokens, color_tokens, trajectory_tokens
        )
