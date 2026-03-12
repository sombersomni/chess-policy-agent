"""MoveEmbedding: embeds move token sequences with positional and turn encoding.

Composes three learned embeddings summed element-wise:
  - token_emb:  move vocabulary index
  - pos_emb:    sequence position (autoregressive order)
  - color_emb:  turn ownership (0=SOS/special, 1=player, 2=opponent)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.config import DecoderConfig
from chess_sim.data.move_vocab import PAD_IDX


class MoveEmbedding(nn.Module):
    """Embeds move token sequences with learned positional and turn encoding.

    Implements the MoveEmbeddable protocol.

    Internal structure:
      token_emb: nn.Embedding(move_vocab_size, d_model, padding_idx=PAD_IDX)
      pos_emb:   nn.Embedding(max_seq_len, d_model)
      color_emb: nn.Embedding(3, d_model) — 0=special/SOS, 1=player, 2=opponent

    Output = token_emb(move_tokens) + pos_emb(positions) + color_emb(move_colors).

    Example:
        >>> emb = MoveEmbedding()
        >>> toks = torch.randint(0, 1971, (4, 20))
        >>> cols = torch.zeros(4, 20, dtype=torch.long)
        >>> out = emb(toks, cols)
        >>> out.shape
        torch.Size([4, 20, 256])
    """

    def __init__(self, decoder_cfg: DecoderConfig | None = None) -> None:
        """Initialize token, positional, and color embedding tables.

        Args:
            decoder_cfg: Optional DecoderConfig for vocab size, d_model,
                and max_seq_len. When None, uses DecoderConfig defaults.

        Example:
            >>> emb = MoveEmbedding()
            >>> emb = MoveEmbedding(DecoderConfig(d_model=128))
        """
        super().__init__()
        cfg = decoder_cfg or DecoderConfig()
        self.token_emb = nn.Embedding(
            cfg.move_vocab_size, cfg.d_model,
            padding_idx=PAD_IDX,
        )
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        # 0=special/SOS, 1=player (side-to-move), 2=opponent
        self.color_emb = nn.Embedding(3, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def embed_moves(
        self,
        move_tokens: Tensor,
        move_colors: Tensor,
    ) -> Tensor:
        """Embed a batch of move token sequences with turn ownership.

        Sums token, positional, and color embeddings per position.

        Args:
            move_tokens: LongTensor [B, T] of move vocabulary indices.
            move_colors: LongTensor [B, T]. Values: 0=SOS/special,
                1=player (side-to-move), 2=opponent.

        Returns:
            FloatTensor [B, T, d_model] of embedded move tokens.

        Example:
            >>> emb = MoveEmbedding()
            >>> toks = torch.randint(0, 1971, (4, 5))
            >>> cols = torch.zeros(4, 5, dtype=torch.long)
            >>> out = emb.embed_moves(toks, cols)
            >>> out.shape
            torch.Size([4, 5, 256])
        """
        positions = torch.arange(
            move_tokens.size(1), device=move_tokens.device
        )
        return self.dropout(
            self.token_emb(move_tokens)
            + self.pos_emb(positions)
            + self.color_emb(move_colors)
        )

    def forward(
        self,
        move_tokens: Tensor,
        move_colors: Tensor,
    ) -> Tensor:
        """nn.Module forward -- delegates to embed_moves().

        Args:
            move_tokens: LongTensor [B, T].
            move_colors: LongTensor [B, T].

        Returns:
            FloatTensor [B, T, d_model].
        """
        return self.embed_moves(move_tokens, move_colors)
