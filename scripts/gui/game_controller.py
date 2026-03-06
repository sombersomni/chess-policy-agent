"""GameController: loads a PGN game and checkpoint; serves ply-indexed data."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chess
import numpy as np

from scripts.evaluate import GameEvaluator, StepResult
from scripts.train_real import stream_pgn


class GameController:
    """Implements GameSource. Tkinter-free; fully testable.

    Call load() before accessing any other method.

    Attributes:
        _boards: Board snapshot before each move, length == total_plies.
        _results: StepResult per ply from GameEvaluator.
        _piece_emb: Piece embedding weight [8, 256] extracted on load.
        _current_ply: Index of the currently displayed ply.
    """

    def __init__(self) -> None:
        """Initialise with empty state; call load() before use."""
        self._boards: list[chess.Board] = []
        self._results: list[StepResult] = []
        self._piece_emb: Optional[np.ndarray] = None
        self._current_ply: int = 0

    def load(
        self,
        pgn_path: Path,
        checkpoint_path: Path,
        game_index: int = 0,
    ) -> None:
        """Load game, run evaluation, capture board snapshots.

        Args:
            pgn_path: Path to .pgn or .pgn.zst file.
            checkpoint_path: Path to .pt model checkpoint.
            game_index: Zero-based index of game in the PGN.

        Raises:
            IndexError: If game_index >= number of games in pgn_path.
        """
        raise NotImplementedError("To be implemented")

    @property
    def current_ply(self) -> int:
        """Zero-based index of the currently displayed ply."""
        return self._current_ply

    @current_ply.setter
    def current_ply(self, ply: int) -> None:
        """Set current ply; clamped to [0, total_plies - 1]."""
        raise NotImplementedError("To be implemented")

    def total_plies(self) -> int:
        """Return the number of evaluated plies."""
        raise NotImplementedError("To be implemented")

    def board_at(self, ply: int) -> chess.Board:
        """Return the board snapshot at the given ply.

        Args:
            ply: Zero-based ply index in [0, total_plies).
        """
        raise NotImplementedError("To be implemented")

    def step_result_at(self, ply: int) -> StepResult:
        """Return StepResult metrics for the given ply.

        Args:
            ply: Zero-based ply index in [0, total_plies).
        """
        raise NotImplementedError("To be implemented")

    def piece_embeddings(self) -> np.ndarray:
        """Return piece embedding weight matrix of shape (8, 256).

        Extracted from encoder.embedding.piece_emb.weight on load.
        """
        raise NotImplementedError("To be implemented")
