"""ChessModel: top-level encoder-decoder assembly for v2 move prediction.

Composes a ChessEncoder (board state -> memory) with a MoveDecoder
(memory + move history -> next-move logits). Provides both a training
forward pass and an inference-time predict_next_move method.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.config import DecoderConfig, ModelConfig
from chess_sim.model.decoder import MoveDecoder
from chess_sim.model.encoder import ChessEncoder


class ChessModel(nn.Module):
    """
    Top-level encoder-decoder model for chess move prediction.
    Assembles ChessEncoder and MoveDecoder into a single nn.Module.
    The encoder produces a memory tensor from board state tokens;
    the decoder cross-attends to it while autoregressively predicting
    move tokens.

    Internal structure:
      encoder: ChessEncoder — board tokens -> [B, 65, d_model] memory
      decoder: MoveDecoder  — move tokens + memory -> [B, T, V] logits

    Example:
        >>> model = ChessModel()
        >>> bt = torch.randint(0, 8, (4, 65))
        >>> ct = torch.randint(0, 3, (4, 65))
        >>> tt = torch.zeros(4, 65, dtype=torch.long)
        >>> mt = torch.randint(0, 1971, (4, 20))
        >>> logits = model(bt, ct, tt, mt)
        >>> logits.shape
        torch.Size([4, 20, 1971])
    """

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        decoder_cfg: DecoderConfig | None = None,
    ) -> None:
        """Initialize encoder and decoder sub-modules.

        Args:
            model_cfg: Optional ModelConfig for the encoder architecture.
                When None, uses ModelConfig defaults (d_model=256, 6 layers).
            decoder_cfg: Optional DecoderConfig for the decoder architecture.
                When None, uses DecoderConfig defaults (4 layers, vocab=1971).

        Example:
            >>> model = ChessModel()
            >>> model = ChessModel(ModelConfig(n_layers=4), DecoderConfig(n_layers=2))
        """
        super().__init__()
        model_cfg = model_cfg or ModelConfig()
        decoder_cfg = decoder_cfg or DecoderConfig()
        self.encoder = ChessEncoder(model_cfg)
        self.decoder = MoveDecoder(decoder_cfg)

    def forward(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        trajectory_tokens: Tensor,
        move_tokens: Tensor,
        move_pad_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Full forward pass: encode board state, then decode move sequence.

        Args:
            board_tokens: LongTensor [B, 65]. Piece-type indices.
            color_tokens: LongTensor [B, 65]. Color indices.
            trajectory_tokens: LongTensor [B, 65]. Trajectory roles 0-4.
            move_tokens: LongTensor [B, T]. Decoder input move indices.
            move_pad_mask: Optional BoolTensor [B, T]. True = PAD position.

        Returns:
            FloatTensor [B, T, MOVE_VOCAB_SIZE] of raw move logits.

        Example:
            >>> logits = model(bt, ct, tt, mt)
            >>> logits.shape
            torch.Size([4, 20, 1971])
        """
        enc_out = self.encoder.encode(
            board_tokens, color_tokens, trajectory_tokens,
        )
        # memory: [B, 65, d_model] — CLS + 64 squares
        memory = torch.cat(
            [
                enc_out.cls_embedding.unsqueeze(1),
                enc_out.square_embeddings,
            ],
            dim=1,
        )
        return self.decoder.decode(
            move_tokens, memory, move_pad_mask,
        ).logits

    def predict_next_move(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        trajectory_tokens: Tensor,
        move_history: list[str],
        legal_moves: list[str],
        temperature: float = 1.0,
    ) -> str:
        """Predict the next move given current board and move history.

        Inference-time method. Encodes the board, feeds move history
        through the decoder, applies legal-move masking and temperature
        scaling, then samples a move from the resulting distribution.

        Args:
            board_tokens: LongTensor [1, 65]. Single board state.
            color_tokens: LongTensor [1, 65]. Color indices.
            trajectory_tokens: LongTensor [1, 65]. Trajectory roles.
            move_history: List of prior UCI move strings in game order.
            legal_moves: List of legal UCI moves for the current position.
            temperature: Softmax temperature for sampling. Default 1.0.

        Returns:
            UCI move string sampled from the model's distribution.

        Example:
            >>> move = model.predict_next_move(bt, ct, tt, ["e2e4"], ["e7e5", "d7d5"])
            >>> isinstance(move, str)
            True
        """
        from chess_sim.data.move_tokenizer import MoveTokenizer

        tok = MoveTokenizer()
        # Build decoder input: SOS + move_history (drop EOS)
        move_tokens = tok.tokenize_game(
            move_history
        ).unsqueeze(0).to(board_tokens.device)
        # Drop trailing EOS for decoder input
        move_tokens = move_tokens[:, :-1]

        self.eval()
        with torch.no_grad():
            logits = self.forward(
                board_tokens, color_tokens,
                trajectory_tokens, move_tokens,
            )
        # Take logits at the last position
        next_logits = logits[0, -1, :]  # [V]
        legal_mask = tok.build_legal_mask(
            legal_moves
        ).to(next_logits.device)
        next_logits[~legal_mask] = -1e9
        probs = torch.softmax(
            next_logits / temperature, dim=-1
        )
        idx = torch.multinomial(probs, num_samples=1).item()
        return tok._vocab.decode(idx)  # type: ignore[arg-type]
