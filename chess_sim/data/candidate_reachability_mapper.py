"""CandidateReachabilityMapper: compute reachable empty squares for a piece type.

Given a chess.Board, a turn color, and a piece type, determines which empty
squares are reachable by any friendly piece of that type. Used to split EMPTY
tokens into VALID_EMPTY (reachable) and INVALID_EMPTY (unreachable) before
the board is tokenized for the encoder.

Example:
    >>> import chess
    >>> mask = compute_valid_empty_mask(chess.Board(), chess.WHITE, chess.ROOK)
    >>> len(mask)
    64
"""

from __future__ import annotations

import chess

# Index 8 in the board token vocabulary (INVALID_EMPTY).
INVALID_EMPTY_TOKEN: int = 8

# Index 1 in the board token vocabulary (VALID_EMPTY / EMPTY).
VALID_EMPTY_TOKEN: int = 1


class CandidateReachabilityMapper:
    """Computes which empty squares a given piece type can reach.

    Wraps compute_valid_empty_mask() for callers that prefer
    an object-oriented interface (e.g., dependency injection
    in the preprocessor).

    Example:
        >>> mapper = CandidateReachabilityMapper()
        >>> mask = mapper.compute(chess.Board(), chess.ROOK, True)
        >>> isinstance(mask, frozenset)
        True
    """

    def compute(
        self,
        board: chess.Board,
        piece_type: int,
        is_white: bool,
    ) -> frozenset[int]:
        """Return 0-based square indices reachable by friendly pieces of piece_type.

        Delegates to the module-level compute_valid_empty_mask() and
        converts the bool list to a frozenset of reachable indices.

        Args:
            board: Current board position.
            piece_type: chess.PieceType (1=PAWN .. 6=KING).
            is_white: True if computing for white pieces.

        Returns:
            Frozenset of 0-based square indices (0-63) that are empty
            and reachable by at least one friendly piece of piece_type.

        Example:
            >>> mapper = CandidateReachabilityMapper()
            >>> mapper.compute(chess.Board(), chess.KNIGHT, True)
            frozenset(...)
        """
        raise NotImplementedError("To be implemented")


def compute_valid_empty_mask(
    board: chess.Board,
    turn: chess.Color,
    piece_type: int,
) -> list[bool]:
    """Return a length-64 bool list of reachable empty squares.

    True = square is a valid empty destination for at least one
    friendly piece of piece_type. Index i = square a1+i (0-based).

    Uses pseudo-legal move generation for speed. Only considers
    moves to empty squares (captures are excluded since the target
    square is occupied).

    Args:
        board: Current board position.
        turn: chess.WHITE or chess.BLACK.
        piece_type: chess.PieceType (1=PAWN .. 6=KING).

    Returns:
        List of 64 booleans. True at index i means square i is
        empty and reachable.

    Example:
        >>> import chess
        >>> mask = compute_valid_empty_mask(
        ...     chess.Board(), chess.WHITE, chess.ROOK
        ... )
        >>> len(mask)
        64
    """
    raise NotImplementedError("To be implemented")


def build_candidate_board_tokens(
    board_tokens: list[int],
    reachable_squares: frozenset[int],
) -> list[int]:
    """Replace EMPTY tokens based on reachability.

    For each square position (indices 1-64 in the 65-element token
    list), if the token is VALID_EMPTY (1) and the square is NOT in
    reachable_squares, replace it with INVALID_EMPTY (8). Occupied
    squares and CLS (index 0) are unchanged.

    Args:
        board_tokens: Length-65 list of board tokens. Index 0 is CLS.
            Indices 1-64 correspond to squares a1-h8.
        reachable_squares: Frozenset of 0-based square indices (0-63)
            that are reachable by the candidate piece type.

    Returns:
        New length-65 list with EMPTY tokens split into VALID_EMPTY
        (reachable) and INVALID_EMPTY (unreachable).

    Example:
        >>> tokens = [0] + [1] * 64  # CLS + all EMPTY
        >>> result = build_candidate_board_tokens(
        ...     tokens, frozenset({0, 1, 2})
        ... )
        >>> result[1]  # square a1 (idx 0) reachable
        1
        >>> result[4]  # square d1 (idx 3) unreachable
        8
    """
    raise NotImplementedError("To be implemented")
