"""MoveTokenizer: converts UCI move strings to integer token sequences.

Implements the MoveTokenizable protocol. Uses MoveVocab internally to
map UCI strings to vocabulary indices. Provides game-level tokenization
(with SOS/EOS framing) and legal-move masking.
"""

from __future__ import annotations

import torch
from torch import Tensor

from chess_sim.data.move_vocab import EOS_IDX, MoveVocab, PAD_IDX, SOS_IDX


class MoveTokenizer:
    """Tokenizes UCI move strings into integer indices via MoveVocab.

    Implements the MoveTokenizable protocol.

    Attributes:
        vocab: MoveVocab instance for string-to-index mapping.

    Example:
        >>> tok = MoveTokenizer()
        >>> tok.tokenize_move("e2e4")
        42
        >>> tok.tokenize_game(["e2e4", "e7e5"]).shape
        torch.Size([4])
    """

    def __init__(self) -> None:
        """Initialize with a MoveVocab instance.

        Example:
            >>> tok = MoveTokenizer()
        """
        self._vocab = MoveVocab()

    def tokenize_move(self, uci: str) -> int:
        """Convert a single UCI move string to its vocabulary index.

        Args:
            uci: UCI move string, e.g. "e2e4" or "e7e8q".

        Returns:
            Integer vocabulary index.

        Raises:
            KeyError: If the move is not in the vocabulary.

        Example:
            >>> tok = MoveTokenizer()
            >>> tok.tokenize_move("e2e4")
            42
        """
        return self._vocab.encode(uci)

    def tokenize_game(self, moves: list[str]) -> Tensor:
        """Convert a list of UCI moves to a LongTensor with SOS/EOS framing.

        Prepends SOS_IDX and appends EOS_IDX to the tokenized sequence.

        Args:
            moves: List of UCI move strings in game order.

        Returns:
            LongTensor of shape [T+2] where T = len(moves).

        Example:
            >>> tok = MoveTokenizer()
            >>> out = tok.tokenize_game(["e2e4", "e7e5"])
            >>> out[0].item() == SOS_IDX
            True
            >>> out[-1].item() == EOS_IDX
            True
        """
        indices = [SOS_IDX] + [
            self._vocab.encode(m) for m in moves
        ] + [EOS_IDX]
        return torch.tensor(indices, dtype=torch.long)

    def build_legal_mask(self, legal_moves: list[str]) -> Tensor:
        """Build a boolean mask over the move vocabulary for legal moves.

        Special tokens (PAD, SOS, EOS) are always masked False.

        Args:
            legal_moves: List of legal UCI move strings for the current
                board position.

        Returns:
            BoolTensor of shape [VOCAB_SIZE]. True = legal move.

        Example:
            >>> tok = MoveTokenizer()
            >>> mask = tok.build_legal_mask(["e2e4", "d2d4"])
            >>> mask.sum().item()
            2
        """
        mask = torch.zeros(len(self._vocab), dtype=torch.bool)
        for uci in legal_moves:
            idx = self._vocab.encode(uci)
            mask[idx] = True
        # Ensure special tokens are always False.
        mask[PAD_IDX] = False
        mask[SOS_IDX] = False
        mask[EOS_IDX] = False
        return mask
