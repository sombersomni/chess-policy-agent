"""chess_sim.env: protocols and shared types for the terminal simulation environment.

Mirrors the pattern established by scripts/gui/__init__.py but targets a
headless gymnasium-compatible loop rather than Tkinter widgets.

Protocols
---------
SimSource      -- board management (reset / step / legal_moves)
Policy         -- agent interface (select_action)
TerminalRenderable -- render board + context to stdout

NamedTuples
-----------
StepInfo       -- result of a single env step
MovePrediction -- one model prediction with probability and rank
RenderContext  -- snapshot of data the renderer needs per frame

Type alias
----------
Observation -- numpy float32 array of shape (65, 3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import chess
import numpy as np

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# Packed board state: 65 tokens × 3 channels
# channel 0 = board_tokens (piece type 0-7)
# channel 1 = color_tokens (0=empty, 1=player, 2=opponent)
# channel 2 = trajectory_tokens (0=none, 1-4 = last-move roles)
Observation = np.ndarray  # shape (65, 3), dtype float32


# ---------------------------------------------------------------------------
# NamedTuple data containers
# ---------------------------------------------------------------------------


class StepInfo(NamedTuple):
    """Result of a single env step.

    Attributes:
        board: Board state *after* the move was applied.
        reward: Sparse reward: +1.0 win, -1.0 loss, 0.0 otherwise.
        terminated: True when the game has reached a terminal state.
        truncated: True when max_plies is exceeded.
        move_uci: The UCI string of the move that was applied.
    """

    board: chess.Board
    reward: float
    terminated: bool
    truncated: bool
    move_uci: str


class MovePrediction(NamedTuple):
    """One model prediction at a given ply.

    Attributes:
        move_uci: Predicted UCI move string.
        probability: Softmax probability in [0, 1].
        rank: 1-based rank (1 = top prediction).
    """

    move_uci: str
    probability: float
    rank: int


class RenderContext(NamedTuple):
    """Snapshot of data the renderer needs for one terminal frame.

    Attributes:
        ply: Zero-based ply index that was just completed.
        move_history: Full move history up to and including this ply.
        last_move_uci: The move applied this step (empty string at reset).
        predictions: Agent predictions made *before* the move (may be empty).
        material_balance: White material minus black material in pawns.
        game_phase: "opening" | "middlegame" | "endgame" by piece count.
    """

    ply: int
    move_history: list[str]
    last_move_uci: str
    predictions: list[MovePrediction]
    material_balance: int
    game_phase: str


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SimSource(Protocol):
    """Manages board state for ChessSimEnv.

    Implementations: PGNSource (replays a game), RandomSource (random moves).
    """

    def reset(self) -> chess.Board:
        """Reset to starting position and return the initial board.

        Returns:
            chess.Board at the starting position.
        """
        ...

    def step(self, move_uci: str) -> StepInfo:
        """Apply a move and advance the game by one ply.

        For PGNSource the submitted move is ignored; the game's actual move
        is applied instead.  RandomSource applies the submitted move if legal,
        falling back to a random legal move otherwise.

        Args:
            move_uci: UCI move string selected by the caller.

        Returns:
            StepInfo for the applied move.
        """
        ...

    def legal_moves(self) -> list[str]:
        """Return UCI strings of all legal moves in the current position.

        Returns:
            List of legal UCI move strings (may be empty at terminal state).
        """
        ...

    def move_history(self) -> list[str]:
        """Return the full move history as UCI strings.

        Returns:
            Ordered list of UCI strings from the start of the game.
        """
        ...

    def is_terminal(self) -> bool:
        """Return True when the game has ended (checkmate/stalemate/draw).

        Returns:
            True if no further moves can be made.
        """
        ...

    def total_plies(self) -> int:
        """Return total number of plies in the game.

        For PGNSource this is the fixed game length.  RandomSource returns
        the configured max_plies limit.

        Returns:
            Non-negative integer.
        """
        ...

    def current_ply(self) -> int:
        """Return the zero-based ply index of the current position.

        Returns:
            Current ply (0 at start, increments after each step).
        """
        ...


@runtime_checkable
class Policy(Protocol):
    """Agent interface: maps observation + legal moves to a vocab action index.

    Implementations: ChessModelAgent (wraps ChessModel.predict_next_move).
    """

    def select_action(
        self,
        obs: Observation,
        legal_moves: list[str],
    ) -> int:
        """Choose a move index from the MoveVocab action space.

        Args:
            obs: Float32 array of shape (65, 3) — packed board observation.
            legal_moves: UCI strings of all legal moves in the current position.

        Returns:
            Integer MoveVocab index of the selected move.
        """
        ...


@runtime_checkable
class TerminalRenderable(Protocol):
    """Renders board state and game context to the terminal (stdout)."""

    def render(
        self,
        board: chess.Board,
        ctx: RenderContext,
    ) -> None:
        """Write one full terminal frame to stdout.

        Args:
            board: Current board state to display.
            ctx: Contextual information (ply, history, predictions, etc.).
        """
        ...
