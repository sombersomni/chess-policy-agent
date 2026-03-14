"""MoveCategoryBuilder: classify a UCI move into 7 categories.

Pure python-chess, no ML dependencies. Given a UCI move string
and the board state BEFORE the move is applied, returns an
integer 0-6 identifying the move category.

Priority order: PROMOTION > CASTLE > CAPTURE_* > QUIET.

Example:
    >>> import chess
    >>> build("e2e4", chess.Board())
    0
"""
from __future__ import annotations

import chess

QUIET: int = 0
CAPTURE_PAWN: int = 1
CAPTURE_MINOR: int = 2
CAPTURE_ROOK: int = 3
CAPTURE_QUEEN: int = 4
CASTLE: int = 5
PROMOTION: int = 6

_CAPTURE_CATEGORY: dict[chess.PieceType, int] = {
    chess.PAWN: CAPTURE_PAWN,
    chess.KNIGHT: CAPTURE_MINOR,
    chess.BISHOP: CAPTURE_MINOR,
    chess.ROOK: CAPTURE_ROOK,
    chess.QUEEN: CAPTURE_QUEEN,
    chess.KING: QUIET,  # king capture shouldn't happen
}


def build(uci_move: str, board: chess.Board) -> int:
    """Classify a UCI move into one of 7 categories.

    The board must reflect the state BEFORE the move is applied.
    Priority: PROMOTION > CASTLE > CAPTURE_* > QUIET.

    Args:
        uci_move: UCI move string (e.g. "e2e4", "e7e8q").
        board: Board state before the move.

    Returns:
        Integer 0-6 indicating the move category.

    Example:
        >>> import chess
        >>> build("e1g1", chess.Board(
        ...     "r1bqkbnr/pppppppp/2n5/8/8/5N2/PPPPPPPP/"
        ...     "RNBQKB1R w KQkq - 2 2"))
        0
    """
    move = chess.Move.from_uci(uci_move)

    # Highest priority: promotion
    if move.promotion is not None:
        return PROMOTION

    # Second priority: castling
    if board.is_castling(move):
        return CASTLE

    # Third priority: captures (including en passant)
    captured = board.piece_at(move.to_square)
    if captured is not None:
        return _CAPTURE_CATEGORY.get(
            captured.piece_type, QUIET
        )

    # En passant: no piece on to_square but it's a capture
    if board.is_en_passant(move):
        return CAPTURE_PAWN

    return QUIET
