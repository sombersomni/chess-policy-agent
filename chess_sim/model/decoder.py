"""MoveDecoder: autoregressive transformer decoder for move generation.

Wraps nn.TransformerDecoder with causal mask generation and a final
linear projection to move vocabulary logits. Cross-attends to the
encoder's board representation (memory). Accepts move_colors [B, T]
to annotate each position as 0=SOS/special, 1=player, 2=opponent.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.config import DecoderConfig
from chess_sim.model.move_embedding import MoveEmbedding
from chess_sim.types import DecoderOutput


class MoveDecoder(nn.Module):
    """Autoregressive move decoder with cross-attention to encoder memory.

    Implements the Decodable protocol.

    Internal structure:
      move_embedding: MoveEmbedding (token + positional + color)
      transformer:    nn.TransformerDecoder (n_layers layers, n_heads heads)
      output_proj:    nn.Linear(d_model, move_vocab_size)

    Example:
        >>> dec = MoveDecoder()
        >>> memory = torch.randn(4, 65, 256)
        >>> tokens = torch.randint(0, 1971, (4, 20))
        >>> colors = torch.zeros(4, 20, dtype=torch.long)
        >>> out = dec(tokens, memory, move_colors=colors)
        >>> out.logits.shape
        torch.Size([4, 20, 1971])
    """

    def __init__(self, decoder_cfg: DecoderConfig | None = None) -> None:
        """Initialize embedding, transformer decoder, and output projection.

        Args:
            decoder_cfg: Optional DecoderConfig. When None, uses defaults
                (d_model=256, n_heads=8, n_layers=4, vocab=1971).

        Example:
            >>> dec = MoveDecoder()
            >>> dec = MoveDecoder(DecoderConfig(n_layers=2))
        """
        super().__init__()
        cfg = decoder_cfg or DecoderConfig()
        self.move_embedding = MoveEmbedding(cfg)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            layer, num_layers=cfg.n_layers,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output_proj = nn.Linear(
            cfg.d_model, cfg.move_vocab_size
        )

    def _causal_mask(self, seq_len: int) -> Tensor:
        """Generate an upper-triangular causal mask for autoregressive decoding.

        The mask prevents attention to future positions. mask[i, j] = True
        means position i cannot attend to position j (i.e., j > i is masked).

        Args:
            seq_len: Length T of the target sequence.

        Returns:
            BoolTensor [T, T]. Upper triangle (above diagonal) is True.

        Example:
            >>> dec = MoveDecoder()
            >>> mask = dec._causal_mask(4)
            >>> mask[0, 1]  # position 0 cannot see position 1
            True
            >>> mask[1, 0]  # position 1 can see position 0
            False
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1,
        )

    def decode(
        self,
        move_tokens: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
        move_colors: Optional[Tensor] = None,
    ) -> DecoderOutput:
        """Decode move tokens autoregressively using encoder memory.

        Embeds move_tokens with turn color, applies causal masking, runs
        through the transformer decoder with cross-attention to memory,
        then projects to vocabulary logits.

        Args:
            move_tokens: LongTensor [B, T] of move token indices.
            memory: FloatTensor [B, 65, d_model] from the encoder.
            tgt_key_padding_mask: Optional BoolTensor [B, T]. True = PAD.
            move_colors: Optional LongTensor [B, T]. Values: 0=SOS/special,
                1=player (side-to-move), 2=opponent. Defaults to all zeros.

        Returns:
            DecoderOutput with logits [B, T, MOVE_VOCAB_SIZE].

        Example:
            >>> colors = torch.zeros(4, 20, dtype=torch.long)
            >>> out = dec.decode(tokens, memory, move_colors=colors)
            >>> out.logits.shape
            torch.Size([4, 20, 1971])
        """
        B, T = move_tokens.shape
        if move_colors is None:
            move_colors = torch.zeros(
                B, T, dtype=torch.long, device=memory.device
            )
        tgt = self.move_embedding(move_tokens, move_colors)
        causal = self._causal_mask(T).to(memory.device)
        out = self.transformer(
            tgt, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        out = self.norm(out)
        return DecoderOutput(logits=self.output_proj(out))

    def forward(
        self,
        move_tokens: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
        move_colors: Optional[Tensor] = None,
    ) -> DecoderOutput:
        """nn.Module forward -- delegates to decode().

        Args:
            move_tokens: LongTensor [B, T].
            memory: FloatTensor [B, 65, d_model].
            tgt_key_padding_mask: Optional BoolTensor [B, T].
            move_colors: Optional LongTensor [B, T].

        Returns:
            DecoderOutput namedtuple.
        """
        return self.decode(
            move_tokens, memory,
            tgt_key_padding_mask, move_colors,
        )
