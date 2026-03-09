"""RLPlyParser: convert a chess.pgn.Game into RLPlyRecord list.

Applies train_color filter, encodes board/color/trajectory/prefix
tokens using existing BoardTokenizer and MoveTokenizer. RL analogue
of GameParser.
"""

from __future__ import annotations

import chess.pgn

from chess_sim.types import RLPlyRecord


class RLPlyParser:
    """Parse a PGN game into filtered RLPlyRecord instances.

    Only plies for the configured train_color are returned.
    Games with unknown result ("*") or fewer than min_moves
    return an empty list.

    Example:
        >>> parser = RLPlyParser(train_color="white")
        >>> records = parser.parse_game(game, game_id=0)
        >>> len(records)  # half the game's plies (white only)
        30
    """

    def __init__(
        self,
        train_color: str,
        min_moves: int = 5,
        max_moves: int = 512,
    ) -> None:
        """Initialize parser with filtering parameters.

        Args:
            train_color: "white" or "black" — side to keep.
            min_moves: Minimum full moves to accept a game.
            max_moves: Maximum full moves before truncation.

        Raises:
            ValueError: If train_color is not "white" or "black".

        Example:
            >>> parser = RLPlyParser("white", min_moves=10)
        """
        raise NotImplementedError("To be implemented")

    def parse_game(
        self,
        game: chess.pgn.Game,
        game_id: int,
    ) -> list[RLPlyRecord]:
        """Convert one PGN game into a list of RLPlyRecord.

        Iterates mainline moves, tokenizes board state before each
        push, builds move_prefix from prior UCI strings. Filters to
        train_color side only.

        Args:
            game: A parsed chess.pgn.Game object.
            game_id: Unique integer index for this game.

        Returns:
            List of RLPlyRecord for the configured train_color.
            Empty list for games with result "*" or < min_moves.

        Example:
            >>> records = parser.parse_game(game, game_id=42)
            >>> records[0].move_uci
            'e2e4'
        """
        raise NotImplementedError("To be implemented")
