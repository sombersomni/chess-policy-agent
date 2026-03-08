"""MoveEmbedding: embeds move token sequences with positional encoding.

Composes a move-token embedding (nn.Embedding with padding_idx=0) and a
learned positional embedding (nn.Embedding over max_seq_len positions).
The two are summed element-wise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.config import DecoderConfig
from chess_sim.data.move_vocab import PAD_IDX


class MoveEmbedding(nn.Module):
    """Embeds move token sequences with learned positional encoding.

    Implements the MoveEmbeddable protocol.

    Internal structure:
      token_emb:    nn.Embedding(move_vocab_size, d_model, padding_idx=PAD_IDX)
      position_emb: nn.Embedding(max_seq_len, d_model)

    Output = token_emb(move_tokens) + position_emb(positions).

    Example:
        >>> emb = MoveEmbedding()
        >>> out = emb(torch.randint(0, 1971, (4, 20)))
        >>> out.shape
        torch.Size([4, 20, 256])
    """

    def __init__(self, decoder_cfg: DecoderConfig | None = None) -> None:
        """Initialize token and positional embedding tables.

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
        self.dropout = nn.Dropout(cfg.dropout)

    def embed_moves(self, move_tokens: Tensor) -> Tensor:
        """Embed a batch of move token sequences.

        Sums the token embedding and the positional embedding for each
        position in the sequence.

        Args:
            move_tokens: LongTensor [B, T] of move vocabulary indices.

        Returns:
            FloatTensor [B, T, d_model] of embedded move tokens.

        Example:
            >>> emb = MoveEmbedding()
            >>> out = emb.embed_moves(tokens)
            >>> out.shape
            torch.Size([4, 20, 256])
        """
        positions = torch.arange(
            move_tokens.size(1), device=move_tokens.device
        )
        return self.dropout(
            self.token_emb(move_tokens)
            + self.pos_emb(positions)
        )

    def forward(self, move_tokens: Tensor) -> Tensor:
        """nn.Module forward -- delegates to embed_moves().

        Args:
            move_tokens: LongTensor [B, T].

        Returns:
            FloatTensor [B, T, d_model].
        """
        return self.embed_moves(move_tokens)
