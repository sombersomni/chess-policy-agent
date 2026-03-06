"""ActivityScorer: computes per-square activity tokens from move history.

Activity tokens encode recent piece mobility as integer values 0-8.
Each move contributes 1 point to its from_square; capturing moves
contribute 1 additional point. Only the last n moves are considered.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Length of activity token sequence: CLS + 64 squares.
ACTIVITY_SEQ_LEN: int = 65
# Maximum activity value per square.
MAX_ACTIVITY: int = 8


class ActivityScorer:
    """Computes per-square activity tokens from game history.

    Implements the Scorable protocol. Never raises exceptions;
    returns all-zeros on error or edge cases.

    Example:
        >>> scorer = ActivityScorer()
        >>> tokens = scorer.score([], chess.Board())
        >>> len(tokens)
        65
    """

    def score(
        self,
        moves: list[chess.Move],
        board: chess.Board,
        n: int = 4,
    ) -> list[int]:
        """Return activity_tokens of length 65.

        Index 0 is CLS (always 0). Indices 1-64 map to
        squares a1-h8. Values are integers 0-8.

        A move contributes 1 pt to its from_square; a
        capturing move contributes 1 additional pt. Only
        the last n moves count.

        The board parameter is the current state AFTER all
        window moves have been played. We rewind via pop()
        to detect captures.

        Args:
            moves: Full move history up to (but not
                including) the current ply.
            board: Board state after all moves in the
                history have been played.
            n: Number of recent moves to consider.

        Returns:
            List of 65 ints, values clamped to [0, 8].
        """
        activity: list[int] = [0] * ACTIVITY_SEQ_LEN

        if n <= 0 or not moves:
            return activity

        window = moves[-n:]

        try:
            pairs = self._collect_pairs(window, board)
        except Exception:
            logger.debug(
                "Failed to rewind board; using fallback."
            )
            pairs = [
                (m, False) for m in window
            ]

        for move, is_capture in pairs:
            # +1 for CLS offset
            sq_idx = move.from_square + 1
            activity[sq_idx] += 1
            if is_capture:
                activity[sq_idx] += 1

        return [min(v, MAX_ACTIVITY) for v in activity]

    def _collect_pairs(
        self,
        window: list[chess.Move],
        board: chess.Board,
    ) -> list[tuple[chess.Move, bool]]:
        """Rewind board to detect captures for each move.

        Pops moves from a board copy in reverse order.
        Falls back to non-capture if the board stack is
        too short.

        Args:
            window: The last n moves to analyze.
            board: Current board state (after window).

        Returns:
            List of (move, was_capture) tuples.
        """
        board_copy = board.copy(stack=True)
        pairs: list[tuple[chess.Move, bool]] = []

        for move in reversed(window):
            if not board_copy.move_stack:
                # Stack exhausted; assume no capture.
                pairs.append((move, False))
                continue
            board_copy.pop()
            # board_copy is now pre-move state
            is_capture = (
                board_copy.piece_at(move.to_square)
                is not None
                or board_copy.is_en_passant(move)
            )
            pairs.append((move, is_capture))

        return pairs
