"""ChessModel: top-level encoder-decoder assembly for v2 move prediction.

Composes a ChessEncoder (board state -> memory) with a MoveDecoder
(memory + move history -> next-move logits). Provides both a training
forward pass and an inference-time predict_next_move method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from chess_sim.data.move_tokenizer import MoveTokenizer
import torch.nn as nn
from torch import Tensor

from chess_sim.config import DecoderConfig, ModelConfig
from chess_sim.model.decoder import MoveDecoder
from chess_sim.model.encoder import ChessEncoder
from chess_sim.model.value_heads import ActionConditionedValueHead

_ILLEGAL_LOGIT_FILL: float = -1e9


def _build_move_colors(
    seq_len: int,
    is_white_turn: bool,
    device: torch.device,
) -> Tensor:
    """Build a [1, seq_len] move-color tensor for the decoder.

    Position 0 (SOS) always gets 0 (special). Subsequent positions
    alternate player(1) / opponent(2) based on whose move each was.
    Even game-indices belong to white; odd to black.

    Args:
        seq_len: Total decoder input length (1 + len(move_history)).
        is_white_turn: True if white is the side-to-move.
        device: Target device for the tensor.

    Returns:
        LongTensor [1, seq_len] with values 0, 1, or 2.

    Example:
        >>> _build_move_colors(4, True, torch.device("cpu"))
        tensor([[0, 1, 2, 1]])
    """
    colors = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    for i in range(1, seq_len):
        game_idx = i - 1  # 0-indexed position in game history
        is_player_move = (game_idx % 2 == 0) == is_white_turn
        colors[0, i] = 1 if is_player_move else 2
    return colors


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
        self.value_head: ActionConditionedValueHead = (
            ActionConditionedValueHead(model_cfg.d_model)
        )

    @property
    def move_token_emb(self) -> nn.Embedding:
        """Return decoder's token embedding table (1971, d_model).

        Provides a stable reference for the Q-head caller to
        look up teacher action embeddings without exposing the
        internal module hierarchy.

        Returns:
            nn.Embedding — the decoder's move token embedding.

        Example:
            >>> model = ChessModel()
            >>> emb = model.move_token_emb
            >>> emb is model.decoder.move_embedding.token_emb
            True
        """
        return self.decoder.move_embedding.token_emb

    def forward(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        trajectory_tokens: Tensor,
        move_tokens: Tensor,
        move_pad_mask: Tensor | None = None,
        move_colors: Tensor | None = None,
        src_square: Tensor | None = None,
        piece_type: Tensor | None = None,
    ) -> Tensor:
        """Full forward pass: encode board state, then decode move sequence.

        Args:
            board_tokens: LongTensor [B, 65]. Piece-type indices.
            color_tokens: LongTensor [B, 65]. Color indices.
            trajectory_tokens: LongTensor [B, 65]. Trajectory roles 0-4.
            move_tokens: LongTensor [B, T]. Decoder input move indices.
            move_pad_mask: Optional BoolTensor [B, T]. True = PAD position.
            move_colors: Optional LongTensor [B, T]. Turn ownership per
                decoder position: 0=SOS/special, 1=player, 2=opponent.
                Defaults to all zeros (all special) when None.
            src_square: Optional LongTensor [B]. Selected from-square index
                (0=no conditioning, 1-64=square). Passed to the encoder as
                a broadcast conditioning signal for piece-conditioned encoding.
            piece_type: Optional LongTensor [B]. Piece type conditioning
                (0=no cond, 1-7=piece type). Passed to the encoder as a
                broadcast signal for candidate-piece-conditioned encoding.

        Returns:
            FloatTensor [B, T, MOVE_VOCAB_SIZE] of raw move logits.

        Example:
            >>> logits = model(bt, ct, tt, mt)
            >>> logits.shape
            torch.Size([4, 20, 1971])
        """
        enc_out = self.encoder.encode(
            board_tokens, color_tokens,
            trajectory_tokens, src_square,
            piece_type,
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
            move_tokens, memory, move_pad_mask, move_colors,
        ).logits

    def predict_next_move(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        trajectory_tokens: Tensor,
        move_history: list[str],
        legal_moves: list[str],
        is_white_turn: bool = True,
        temperature: float = 1.0,
        tokenizer: MoveTokenizer | None = None,
        src_square: int | None = None,
        piece_type: int | None = None,
    ) -> str:
        """Predict the next move given current board and move history.

        Inference-time method. Encodes the board, feeds move history
        through the decoder with turn color annotations, applies
        legal-move masking and temperature scaling, then samples a move.
        Restores the model's training mode after inference.

        When src_square is provided, the encoder is conditioned on that
        piece selection, enabling hierarchical two-stage generation:
        (1) select which piece to move, (2) call this method with that
        src_square to predict the destination.

        Args:
            board_tokens: LongTensor [1, 65]. Single board state.
            color_tokens: LongTensor [1, 65]. Color indices.
            trajectory_tokens: LongTensor [1, 65]. Trajectory roles.
            move_history: List of prior UCI move strings in game order.
            legal_moves: List of legal UCI moves for the current position.
            is_white_turn: True if white is the side-to-move. Used to
                assign player(1)/opponent(2) colors to move history.
            temperature: Softmax temperature for sampling. Default 1.0.
            tokenizer: Optional pre-built MoveTokenizer. Reuse across calls
                to avoid re-loading the vocabulary on every inference step.
            src_square: Optional int (0-63). When provided, conditions the
                encoder on this piece selection and restricts the legal mask
                to only moves originating from this square.
            piece_type: Optional int (1-6, chess.PieceType). When provided,
                conditions the encoder via piece_type_cond_emb and restricts
                the legal mask to only moves by pieces of that type via
                PieceTypeMoveLUT.

        Returns:
            UCI move string sampled from the model's distribution.

        Raises:
            ValueError: If legal_moves is empty or move_tokens is empty
                after trimming EOS.

        Example:
            >>> move = model.predict_next_move(bt, ct, tt, ["e2e4"], ["e7e5"], True)
            >>> isinstance(move, str)
            True
        """
        from chess_sim.data.move_tokenizer import MoveTokenizer

        if not legal_moves:
            raise ValueError(
                "legal_moves is empty — no legal move to sample"
            )

        tok = tokenizer or MoveTokenizer()
        # Build decoder input: SOS + move_history, then drop trailing EOS
        move_tokens = tok.tokenize_game(
            move_history
        ).unsqueeze(0).to(board_tokens.device)
        move_tokens = move_tokens[:, :-1]

        if move_tokens.shape[1] == 0:
            raise ValueError(
                "move_tokens is empty after trimming EOS — "
                "tokenize_game returned no tokens"
            )

        move_colors = _build_move_colors(
            move_tokens.shape[1],
            is_white_turn,
            board_tokens.device,
        )

        # src conditioning: convert 0-based square to 1-based vocab index
        src_tensor: Tensor | None = None
        if src_square is not None:
            src_tensor = torch.tensor(
                [src_square + 1],
                dtype=torch.long,
                device=board_tokens.device,
            )

        # piece_type conditioning: pass through as [1] tensor
        pt_tensor: Tensor | None = None
        if piece_type is not None:
            pt_tensor = torch.tensor(
                [piece_type],
                dtype=torch.long,
                device=board_tokens.device,
            )

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                logits = self.forward(
                    board_tokens, color_tokens,
                    trajectory_tokens, move_tokens,
                    move_colors=move_colors,
                    src_square=src_tensor,
                    piece_type=pt_tensor,
                )
        finally:
            self.train(was_training)

        next_logits = logits[0, -1, :]  # [V]
        legal_mask = tok.build_legal_mask(
            legal_moves
        ).to(next_logits.device)

        # Optionally narrow to moves from the selected src square
        if src_square is not None:
            from chess_sim.data.src_move_lut import SrcMoveLUT
            lut = SrcMoveLUT(device=next_logits.device)
            src_sq_t = torch.tensor(
                [src_square], dtype=torch.long,
                device=next_logits.device,
            )
            src_mask = lut.filter_legal_mask(
                legal_mask.unsqueeze(0), src_sq_t
            ).squeeze(0)
            legal_mask = src_mask

        # Optionally narrow to moves by the selected piece type
        if piece_type is not None:
            from chess_sim.data.piece_type_move_lut import (
                PieceTypeMoveLUT,
            )
            pt_lut = PieceTypeMoveLUT(
                device=next_logits.device
            )
            pt_t = torch.tensor(
                [piece_type], dtype=torch.long,
                device=next_logits.device,
            )
            pt_mask = pt_lut.filter_legal_mask(
                legal_mask.unsqueeze(0), pt_t
            ).squeeze(0)
            legal_mask = pt_mask

        next_logits[~legal_mask] = _ILLEGAL_LOGIT_FILL
        probs = torch.softmax(next_logits / temperature, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        return tok.decode(idx)  # type: ignore[arg-type]
