"""SelfPlaySource: live self-play board implementing SimSource protocol.

Holds a mutable chess.Board and applies moves from either the player or
opponent depending on whose turn it is.
"""

from __future__ import annotations

import chess

from chess_sim.env import SimSource, StepInfo  # noqa: F401


class SelfPlaySource:
    """Live self-play board implementing the SimSource protocol.

    Holds a mutable chess.Board and applies moves from either the
    player or opponent depending on whose turn it is. Implements
    SimSource structurally (no inheritance required).
    """

    def reset(self) -> chess.Board:
        """Reset board to starting position and return it.

        Returns:
            chess.Board at the starting position.
        """
        raise NotImplementedError("To be implemented")

    def step(self, move_uci: str) -> StepInfo:
        """Apply move_uci to the board and return StepInfo.

        Args:
            move_uci: UCI move string to apply.

        Returns:
            StepInfo for the applied move.
        """
        raise NotImplementedError("To be implemented")

    def legal_moves(self) -> list[str]:
        """Return all legal moves in UCI notation.

        Returns:
            List of legal UCI move strings (empty at terminal).
        """
        raise NotImplementedError("To be implemented")

    def move_history(self) -> list[str]:
        """Return all moves played so far in UCI notation.

        Returns:
            Ordered list of UCI strings from game start.
        """
        raise NotImplementedError("To be implemented")

    def is_terminal(self) -> bool:
        """Return True if game has ended (checkmate/stalemate/draw).

        Returns:
            True if no further moves can be made.
        """
        raise NotImplementedError("To be implemented")

    def current_ply(self) -> int:
        """Return the number of half-moves played so far.

        Returns:
            Current ply count (0 at start).
        """
        raise NotImplementedError("To be implemented")

    def total_plies(self) -> int:
        """Return the total number of half-moves in the game.

        Returns:
            Total ply count of the completed game.
        """
        raise NotImplementedError("To be implemented")
