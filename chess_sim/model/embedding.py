"""EmbeddingLayer: composes piece, color, square, trajectory, and src embeddings.

Five independent embedding tables are summed element-wise then passed through
LayerNorm and Dropout. The fifth stream (src_square_emb) is a broadcast
conditioning signal — identical for all 65 positions within a sample —
that tells the model which piece the agent is considering moving.

Vocabulary sizes:
  piece_emb:      8  (CLS=0, EMPTY=1, PAWN=2, KNIGHT=3, BISHOP=4, ROOK=5, QUEEN=6, KING=7)
  color_emb:      3  (EMPTY=0, PLAYER=1, OPPONENT=2)
  square_emb:     65 (index 0=CLS position, 1..64=squares a1..h8)
  trajectory_emb: 5  (0=none, 1=player prev loc, 2=player curr loc,
                      3=opp prev loc, 4=opp curr loc)
  src_square_emb: 65 (0=no conditioning, 1..64=selected from-square)
                     initialized to zero — no-op until trained.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

D_MODEL: int = 256
PIECE_VOCAB_SIZE: int = 9
INVALID_EMPTY_IDX: int = 8
PIECE_TYPE_COND_VOCAB_SIZE: int = 8  # 0=no cond, 1-7=piece type
COLOR_VOCAB_SIZE: int = 3
SQUARE_VOCAB_SIZE: int = 65  # CLS + 64 squares
TRAJECTORY_VOCAB_SIZE: int = 5  # 0=none, 1=pl_prev, 2=pl_curr, 3=opp_prev, 4=opp_curr
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
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # INVALID_EMPTY (same as EMPTY)
]

# Scale factor for piece role init (matches default Embedding init std).
_ROLE_SCALE: float = 0.02


class EmbeddingLayer(nn.Module):
    """Composes piece + color + square + trajectory + src embeddings.

    Implements the Embeddable protocol.

    Internal structure:
      piece_emb:      nn.Embedding(9, 256)  — role-feature init (includes INVALID_EMPTY)
      color_emb:      nn.Embedding(3, 256)
      square_emb:     nn.Embedding(65, 256) — geometric sin/cos init
      trajectory_emb: nn.Embedding(5, 256)
      src_square_emb: nn.Embedding(65, 256) — zero init, broadcast signal
      piece_type_cond_emb: nn.Embedding(8, 256) — zero init, broadcast signal
      layer_norm:     nn.LayerNorm(256)
      dropout:        nn.Dropout(0.1)

    Output = LayerNorm(Dropout(piece + color + square + traj + src + cond)).

    Example:
        >>> layer = EmbeddingLayer()
        >>> z = torch.zeros(4, 65, dtype=torch.long)
        >>> s = torch.zeros(4, dtype=torch.long)
        >>> out = layer.embed(z, z, z, s)  # [B, 65, 256]
    """

    def __init__(self, d_model: int = D_MODEL) -> None:
        """Initialize five embedding tables, LayerNorm, Dropout.

        Calls _init_square_emb() and _init_piece_emb() to apply
        geometric and role-feature priors respectively.
        src_square_emb is zero-initialized (no-op by default).

        Args:
            d_model: Embedding dimension. Defaults to the module constant
                D_MODEL (256). Override via ModelConfig when training with
                a different architecture.

        Example:
            >>> layer = EmbeddingLayer()
            >>> layer = EmbeddingLayer(d_model=128)
        """
        super().__init__()
        self.d_model = d_model
        self.piece_emb = nn.Embedding(
            PIECE_VOCAB_SIZE, d_model
        )
        self.color_emb = nn.Embedding(
            COLOR_VOCAB_SIZE, d_model
        )
        self.square_emb = nn.Embedding(
            SQUARE_VOCAB_SIZE, d_model
        )
        self.trajectory_emb = nn.Embedding(
            TRAJECTORY_VOCAB_SIZE, d_model
        )
        # src_square_emb: 0=no conditioning, 1-64=selected square.
        # Zero-initialized so conditioning is a no-op until trained.
        self.src_square_emb = nn.Embedding(
            SQUARE_VOCAB_SIZE, d_model
        )
        # piece_type_cond_emb: 0=no cond, 1-7=piece type being considered.
        # Zero-initialized so conditioning is a no-op until trained.
        self.piece_type_cond_emb = nn.Embedding(
            PIECE_TYPE_COND_VOCAB_SIZE, d_model
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(DROPOUT)
        self._init_square_emb()
        self._init_piece_emb()
        nn.init.zeros_(self.src_square_emb.weight)
        nn.init.zeros_(self.piece_type_cond_emb.weight)

    def _init_square_emb(self) -> None:
        """Initialize square_emb with 2D sin/cos geometric encoding.

        Index 0 (CLS): all zeros.
        For s=1..64: rank=(s-1)//8, file=(s-1)%8.
        For d=0..d_model-1: cycle through 4 patterns based on d%4.
        """
        d = self.d_model
        weights = torch.zeros(SQUARE_VOCAB_SIZE, d)
        for s in range(1, SQUARE_VOCAB_SIZE):
            rank = (s - 1) // 8
            file = (s - 1) % 8
            for i in range(d):
                freq = 10000.0 ** (i / d)
                pattern = i % 4
                if pattern == 0:
                    weights[s, i] = math.sin(rank / freq)
                elif pattern == 1:
                    weights[s, i] = math.cos(rank / freq)
                elif pattern == 2:
                    weights[s, i] = math.sin(file / freq)
                else:
                    weights[s, i] = math.cos(file / freq)
        with torch.no_grad():
            self.square_emb.weight.copy_(weights)

    def _init_piece_emb(self) -> None:
        """Initialize piece_emb with tiled role-feature vectors.

        Each 7-element role vector is tiled to fill d_model dims,
        then scaled by 0.02.
        """
        d = self.d_model
        weights = torch.zeros(PIECE_VOCAB_SIZE, d)
        for i, features in enumerate(_ROLE_FEATURES):
            ft = torch.tensor(features, dtype=torch.float)
            reps = math.ceil(d / len(features))
            tiled = ft.repeat(reps)[:d]
            weights[i] = tiled * _ROLE_SCALE
        with torch.no_grad():
            self.piece_emb.weight.copy_(weights)

    def embed(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        trajectory_tokens: Tensor,
        src_tokens: Tensor | None = None,
        piece_type_tokens: Tensor | None = None,
    ) -> Tensor:
        """Compose six embedding streams into [B, 65, d_model].

        Sums piece + color + square + trajectory + src + piece_type_cond
        embeddings, applies LayerNorm, then Dropout. Both src_tokens and
        piece_type_tokens are broadcast across all 65 positions
        (segment-embedding style).

        Args:
            board_tokens: torch.long [B, 65].
            color_tokens: torch.long [B, 65].
            trajectory_tokens: torch.long [B, 65]. Values 0-4.
            src_tokens: torch.long [B]. Selected from-square index
                (0=no conditioning, 1-64=square). Defaults to zeros.
            piece_type_tokens: torch.long [B]. Piece type conditioning
                (0=no cond, 1-7=piece type). Defaults to zeros (no-op).

        Returns:
            torch.float [B, 65, d_model].

        Example:
            >>> layer = EmbeddingLayer()
            >>> z = torch.zeros(4, 65, dtype=torch.long)
            >>> s = torch.zeros(4, dtype=torch.long)
            >>> out = layer.embed(z, z, z, s, s)
            >>> out.shape
            torch.Size([4, 65, 256])
        """
        B, S = board_tokens.size()
        sq_idx = torch.arange(
            S, device=board_tokens.device
        ).unsqueeze(0).expand(B, S)

        if src_tokens is None:
            src_tokens = torch.zeros(
                B, dtype=torch.long,
                device=board_tokens.device,
            )
        # Broadcast src across all 65 positions: [B] -> [B, 65]
        src_expanded = src_tokens.unsqueeze(1).expand(B, S)

        if piece_type_tokens is None:
            piece_type_tokens = torch.zeros(
                B, dtype=torch.long,
                device=board_tokens.device,
            )
        # Broadcast piece_type across all 65 positions: [B] -> [B, 65]
        pt_expanded = piece_type_tokens.unsqueeze(1).expand(B, S)

        x = (
            self.piece_emb(board_tokens)
            + self.color_emb(color_tokens)
            + self.square_emb(sq_idx)
            + self.trajectory_emb(trajectory_tokens)
            + self.src_square_emb(src_expanded)
            + self.piece_type_cond_emb(pt_expanded)
        )
        return self.dropout(self.layer_norm(x))

    def forward(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        trajectory_tokens: Tensor,
        src_tokens: Tensor | None = None,
        piece_type_tokens: Tensor | None = None,
    ) -> Tensor:
        """nn.Module forward -- delegates to embed().

        Args:
            board_tokens: torch.long [B, 65].
            color_tokens: torch.long [B, 65].
            trajectory_tokens: torch.long [B, 65]. Values 0-4.
            src_tokens: torch.long [B]. Defaults to zeros.
            piece_type_tokens: torch.long [B]. Defaults to zeros.

        Returns:
            torch.float [B, 65, d_model].
        """
        return self.embed(
            board_tokens, color_tokens,
            trajectory_tokens, src_tokens,
            piece_type_tokens,
        )
