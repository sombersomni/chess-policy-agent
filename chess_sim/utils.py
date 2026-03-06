"""Shared chess utility functions for the chess-sim pipeline."""

from __future__ import annotations

from typing import Optional

import chess
import chess.pgn


def winner_color(
    game: chess.pgn.Game,
) -> Optional[chess.Color]:
    """Return the winning color from a PGN game result, or None.

    chess.Color is bool: chess.WHITE == True,
    chess.BLACK == False.
    Draws ("1/2-1/2") and unknown results ("*") return None.

    Args:
        game: A parsed PGN game object.

    Returns:
        chess.WHITE, chess.BLACK, or None for draws/unknown.
    """
    result = game.headers.get("Result", "*")
    if result == "1-0":
        return chess.WHITE
    if result == "0-1":
        return chess.BLACK
    return None
