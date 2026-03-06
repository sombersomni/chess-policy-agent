"""GUI Protocols for the chess-sim viewer."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import chess
import numpy as np

from scripts.evaluate import StepResult


@runtime_checkable
class Renderable(Protocol):
    """Widget that redraws itself from current controller state."""

    def render(self) -> None:
        """Redraw the widget from the current controller state."""
        ...


@runtime_checkable
class Navigable(Protocol):
    """Widget that can navigate between game plies."""

    def go_to_ply(self, ply: int) -> None:
        """Jump to a specific ply index.

        Args:
            ply: Target ply index; must be in [0, total_plies).
        """
        ...

    def next_ply(self) -> None:
        """Advance one ply if not already at the last ply."""
        ...

    def prev_ply(self) -> None:
        """Step back one ply if not at ply 0."""
        ...


@runtime_checkable
class GameSource(Protocol):
    """Read-only facade over a loaded game and its evaluated metrics."""

    @property
    def current_ply(self) -> int:
        """Zero-based index of the currently displayed ply."""
        ...

    @current_ply.setter
    def current_ply(self, ply: int) -> None:
        """Set the current ply index."""
        ...

    def load(
        self,
        pgn_path: Path,
        checkpoint_path: Path,
        game_index: int = 0,
    ) -> None:
        """Load a game and checkpoint; run evaluation.

        Args:
            pgn_path: Path to .pgn or .pgn.zst file.
            checkpoint_path: Path to .pt checkpoint file.
            game_index: Zero-based game index within the PGN.
        """
        ...

    def total_plies(self) -> int:
        """Return the total number of evaluated plies."""
        ...

    def board_at(self, ply: int) -> chess.Board:
        """Return the board snapshot at the given ply.

        Args:
            ply: Zero-based ply index in [0, total_plies).

        Returns:
            Board position *before* the move at that ply was played.
        """
        ...

    def step_result_at(self, ply: int) -> StepResult:
        """Return per-ply evaluation metrics.

        Args:
            ply: Zero-based ply index in [0, total_plies).

        Returns:
            StepResult with loss, accuracy, and entropy for all four heads.
        """
        ...

    def piece_embeddings(self) -> np.ndarray:
        """Return the learned piece embedding weight matrix.

        Returns:
            Float ndarray of shape (8, 256).
            Row order: 0=CLS, 1=EMPTY, 2=PAWN, 3=KNIGHT,
                       4=BISHOP, 5=ROOK, 6=QUEEN, 7=KING.
        """
        ...
