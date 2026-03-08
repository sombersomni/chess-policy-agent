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

    def __init__(self) -> None:
        """Initialize with a fresh chess board."""
        self._board: chess.Board = chess.Board()
        self._move_history: list[str] = []

    def reset(self) -> chess.Board:
        """Reset board to starting position and return it.

        Returns:
            chess.Board at the starting position.
        """
        self._board = chess.Board()
        self._move_history = []
        return self._board

    def step(self, move_uci: str) -> StepInfo:
        """Apply move_uci to the board and return StepInfo.

        If the move is illegal, falls back to the first legal move.

        Args:
            move_uci: UCI move string to apply.

        Returns:
            StepInfo for the applied move.
        """
        move = chess.Move.from_uci(move_uci)
        if move not in self._board.legal_moves:
            move = next(iter(self._board.legal_moves))
            move_uci = move.uci()
        self._board.push(move)
        self._move_history.append(move_uci)
        terminated = self._board.is_game_over()
        reward = 0.0
        if terminated:
            result = self._board.result()
            if result == "1-0":
                reward = 1.0
            elif result == "0-1":
                reward = -1.0
        return StepInfo(
            board=self._board,
            reward=reward,
            terminated=terminated,
            truncated=False,
            move_uci=move_uci,
        )

    def legal_moves(self) -> list[str]:
        """Return all legal moves in UCI notation.

        Returns:
            List of legal UCI move strings (empty at terminal).
        """
        return [m.uci() for m in self._board.legal_moves]

    def move_history(self) -> list[str]:
        """Return all moves played so far in UCI notation.

        Returns:
            Ordered list of UCI strings from game start.
        """
        return list(self._move_history)

    def is_terminal(self) -> bool:
        """Return True if game has ended.

        Returns:
            True if no further moves can be made.
        """
        return self._board.is_game_over()

    def current_ply(self) -> int:
        """Return the number of half-moves played so far.

        Returns:
            Current ply count (0 at start).
        """
        return len(self._move_history)

    def total_plies(self) -> int:
        """Return the total number of half-moves in the game.

        Returns:
            Total ply count of the completed game.
        """
        return len(self._move_history)
