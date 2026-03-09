"""Shared training utilities reused across multiple trainers."""
from __future__ import annotations

import chess
import torch
from torch import Tensor

_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def material_balance(board: chess.Board) -> float:
    """Return white material minus black material in pawn units.

    Iterates all 64 squares and sums signed piece values.
    Kings are excluded (value 0).

    Args:
        board: Current chess board state.

    Returns:
        Float material balance in pawn units (positive = white
        advantage).
    """
    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = _PIECE_VALUES.get(piece.piece_type, 0)
        score += val if piece.color == chess.WHITE else -val
    return score


def l1_normalize(t: Tensor) -> Tensor:
    """L1-normalize t across plies, preserving sign.

    Returns zeros if t is all-zero to avoid spurious signal.

    Args:
        t: Tensor of shape [T].

    Returns:
        t / |t|.sum() if non-zero, else zeros_like(t).
    """
    s = t.abs().sum()
    if s < 1e-8:
        return torch.zeros_like(t)
    return t / s
