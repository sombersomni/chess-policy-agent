"""PieceTypeMoveLUT: precomputed [7, V] bool LUT for piece-type move filtering.

Maps each piece type (1-6, chess.PieceType values) to the set of move vocab
indices whose from-square contains a piece of that type on the starting board.
This is a static structural constraint derived from the move vocabulary itself,
independent of any specific board position.

Combined at inference time with the existing legal mask to produce a
piece-type-filtered mask.

Example:
    >>> lut = PieceTypeMoveLUT()
    >>> lut._lut.shape
    torch.Size([7, 1971])
"""

from __future__ import annotations

import torch
from torch import Tensor


class PieceTypeMoveLUT:
    """Precomputed piece-type-to-move LUT for per-batch mask filtering.

    Holds a static [7, V] bool tensor built once from MoveVocab. Row 0
    is unused (no piece type 0); rows 1-6 map to PAWN..KING.
    `filter_legal_mask` returns the intersection of the legal mask with
    the row corresponding to each batch item's piece type.

    Attributes:
        _lut: BoolTensor [7, V] -- row i is True for vocab indices
            whose from-square would hold piece type i on the initial
            board. Row 0 is all-False (no conditioning).

    Example:
        >>> lut = PieceTypeMoveLUT()
        >>> lut._lut.shape
        torch.Size([7, 1971])
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
    ) -> None:
        """Build the static [7, V] LUT from MoveVocab.

        Iterates over all move vocab entries, parses the UCI from-square,
        and marks each entry under the piece type that occupies that
        square on the standard starting position. Reuses
        _uci_from_square_slot from structural_mask for UCI parsing.

        Args:
            device: Target device for the LUT tensor.

        Example:
            >>> lut = PieceTypeMoveLUT(device="cpu")
            >>> lut._lut.shape
            torch.Size([7, 1971])
        """
        import chess

        from chess_sim.data.move_vocab import MoveVocab
        from chess_sim.data.structural_mask import (
            _uci_from_square_slot,
        )

        vocab = MoveVocab()
        v = len(vocab)  # 1971
        lut = torch.zeros(7, v, dtype=torch.bool)
        start_board = chess.Board()

        for idx in range(3, v):
            uci = vocab.decode(idx)
            # _uci_from_square_slot returns 1-based slot
            slot = _uci_from_square_slot(uci)
            sq = slot - 1  # convert to 0-based
            piece = start_board.piece_at(sq)
            if piece is not None:
                lut[piece.piece_type, idx] = True

        self._lut: Tensor = lut.to(device)

    def filter_legal_mask(
        self,
        legal_mask: Tensor,
        piece_types: Tensor,
    ) -> Tensor:
        """Narrow a legal mask to moves by the given piece types.

        For each batch item, ANDs the legal mask with the LUT row
        for that item's piece type, keeping only moves whose
        from-square corresponds to a piece of the requested type.

        Caller must gate on piece_type > 0 before calling; row 0
        is all-False and will zero out the entire mask.

        Args:
            legal_mask: BoolTensor [B, V] -- fully legal moves.
            piece_types: LongTensor [B] -- piece type per batch item
                (1-6, matching chess.PieceType values).

        Returns:
            BoolTensor [B, V] -- legal moves filtered to the chosen
            piece type.

        Example:
            >>> lut = PieceTypeMoveLUT()
            >>> legal = torch.ones(2, 1971, dtype=torch.bool)
            >>> pt = torch.tensor([4, 2], dtype=torch.long)
            >>> filtered = lut.filter_legal_mask(legal, pt)
            >>> filtered.shape
            torch.Size([2, 1971])
        """
        dev = legal_mask.device
        lut = self._lut.to(dev)
        per_type = lut[piece_types.to(dev)]  # [B, V]
        return legal_mask & per_type
