"""CaptureMapBuilder: per-square capturable-opponent binary map.

Pure python-chess, no ML dependencies. Given a board and turn,
returns a 64-element list where 1 means the opponent piece on
that square is attacked by at least one current-player piece.

Example:
    >>> import chess
    >>> m = build(chess.Board(), chess.WHITE)
    >>> len(m)
    64
    >>> all(v == 0 for v in m)  # no captures in starting pos
    True
"""
from __future__ import annotations

import chess
import torch
from torch import Tensor


def build(
    board: chess.Board,
    turn: chess.Color,
) -> list[int]:
    """Build a 64-element capture map for the given side.

    For each of the 64 squares (index 0 = a1, 63 = h8):
    mark 1 if an opponent piece occupies that square AND at
    least one of our pieces attacks it.

    Args:
        board: Current board position.
        turn: Side to move (chess.WHITE or chess.BLACK).

    Returns:
        List of 64 ints, each 0 or 1.

    Example:
        >>> import chess
        >>> fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/"
        ...       "8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        >>> m = build(chess.Board(fen), chess.WHITE)
        >>> m[chess.E5]
        0
    """
    opponent = not turn
    opp_pieces = board.occupied_co[opponent]
    result: list[int] = [0] * 64
    for sq in chess.scan_forward(opp_pieces):
        attackers = board.attackers(turn, sq)
        if attackers:
            result[sq] = 1
    return result


def compute_from_board_tensor(board: Tensor) -> Tensor:
    """Derive capture map per batch from board tensor.

    Marks every opponent-occupied square as a capture target.
    Approximation: ignores actual attack legality; correct for
    the vast majority of real positions.

    Args:
        board: Float tensor [B, 65, 3]. Channel 0 = piece type
            (0=empty, 1-7=pieces), channel 1 = color
            (0=empty, 1=player, 2=opponent). Index 0 is CLS.

    Returns:
        Float tensor [B, 64] — 1.0 where an opponent piece sits.

    Example:
        >>> import torch
        >>> b = torch.zeros(2, 65, 3)
        >>> b[0, 5, 0] = 2.0  # pawn at slot 5
        >>> b[0, 5, 1] = 2.0  # opponent
        >>> compute_from_board_tensor(b)[0, 4]
        tensor(1.)
    """
    # piece type 1=EMPTY, >1 means actual piece; color 2=opponent
    piece_types = board[:, 1:, 0]  # [B, 64]
    colors = board[:, 1:, 1]       # [B, 64]
    return ((piece_types > 1) & (colors == 2)).float()
