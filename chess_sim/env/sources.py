"""Concrete SimSource implementations for ChessSimEnv.

PGNSource  -- replays a fixed PGN game; submitted moves are ignored.
RandomSource -- generates a random-legal-move game from the starting position.

Both expose the SimSource protocol from chess_sim.env.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn

from chess_sim.env import MovePrediction, RenderContext, SimSource, StepInfo

# Piece-value table for material balance (standard centipawn values in pawns).
_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

# Game-phase thresholds (non-king piece count on both sides combined).
_PHASE_OPENING: int = 24     # more than this = opening
_PHASE_ENDGAME: int = 12     # fewer than or equal to this = endgame


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _compute_reward(board: chess.Board) -> float:
    """Return sparse terminal reward from board's perspective.

    +1.0 if the side that just moved won, -1.0 if they lost, 0.0 otherwise.

    Args:
        board: Board state *after* the move was applied.

    Returns:
        Float reward value.
    """
    if not board.is_game_over():
        return 0.0
    result = board.result()
    if result == "1-0":
        return 1.0 if not board.turn else -1.0
    if result == "0-1":
        return -1.0 if not board.turn else 1.0
    return 0.0  # draw


def _material_balance(board: chess.Board) -> int:
    """White material minus black material in pawn units.

    Args:
        board: Current board state.

    Returns:
        Integer material balance (positive = white advantage).
    """
    balance = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = _PIECE_VALUES[piece.piece_type]
        balance += val if piece.color == chess.WHITE else -val
    return balance


def _game_phase(board: chess.Board) -> str:
    """Classify game phase by non-king piece count.

    Args:
        board: Current board state.

    Returns:
        "opening" | "middlegame" | "endgame"
    """
    count = sum(
        1
        for sq in chess.SQUARES
        if board.piece_at(sq) is not None
        and board.piece_at(sq).piece_type != chess.KING  # type: ignore[union-attr]
    )
    if count > _PHASE_OPENING:
        return "opening"
    if count <= _PHASE_ENDGAME:
        return "endgame"
    return "middlegame"


def _build_step_info(
    board: chess.Board,
    move_uci: str,
    truncated: bool = False,
) -> StepInfo:
    """Construct a StepInfo after a move has been applied.

    Args:
        board: Board *after* the move.
        move_uci: UCI string of the move that was applied.
        truncated: True when max_plies was hit.

    Returns:
        Populated StepInfo.
    """
    return StepInfo(
        board=board,
        reward=_compute_reward(board),
        terminated=board.is_game_over(),
        truncated=truncated,
        move_uci=move_uci,
    )


def _stream_pgn(pgn_path: Path) -> Iterator[chess.pgn.Game]:
    """Yield chess.pgn.Game objects from a plain or .zst PGN file.

    Args:
        pgn_path: Path to .pgn or .pgn.zst file.

    Yields:
        chess.pgn.Game objects one at a time.
    """
    if pgn_path.suffix == ".zst":
        import io

        import zstandard

        ctx = zstandard.ZstdDecompressor()
        with open(pgn_path, "rb") as fh:
            text = ctx.decompress(fh.read()).decode("utf-8", errors="replace")
        stream = io.StringIO(text)
    else:
        stream = open(pgn_path, encoding="utf-8", errors="replace")  # noqa: SIM115

    try:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            yield game
    finally:
        stream.close()


# ---------------------------------------------------------------------------
# PGNSource
# ---------------------------------------------------------------------------


class PGNSource:
    """Replays a single PGN game ply-by-ply.

    The submitted move in step() is ignored; the source always applies the
    game's actual move so the replay follows the recorded game exactly.

    Implements the SimSource protocol.

    Attributes:
        _pgn_path: Path to the PGN or PGN.zst file.
        _game_index: Zero-based index of the target game in the file.
        _game_moves: UCI strings of every move in the loaded game.
        _board: Current board state (mutated in-place during step).
        _history: UCI strings of moves applied so far.
        _ply: Zero-based current ply index.
    """

    def __init__(self, pgn_path: Path, game_index: int = 0) -> None:
        """Initialise without loading; call reset() before use.

        Args:
            pgn_path: Path to .pgn or .pgn.zst file.
            game_index: Zero-based index of the game to replay.

        Raises:
            IndexError: If game_index exceeds the number of games in the file.
        """
        self._pgn_path = pgn_path
        self._game_index = game_index
        self._game_moves: list[str] = []
        self._board: chess.Board = chess.Board()
        self._history: list[str] = []
        self._ply: int = 0

    def reset(self) -> chess.Board:
        """Load the target game and return the initial board.

        Returns:
            chess.Board at the starting position.

        Raises:
            IndexError: If game_index is out of range.
        """
        games = list(_stream_pgn(self._pgn_path))
        game = games[self._game_index]
        self._game_moves = [
            move.uci() for move in game.mainline_moves()
        ]
        self._board = game.board()
        self._history = []
        self._ply = 0
        return self._board.copy()

    def step(self, move_uci: str) -> StepInfo:  # noqa: ARG002
        """Apply the game's actual move (ignores move_uci).

        Args:
            move_uci: Ignored. The recorded game move is applied instead.

        Returns:
            StepInfo containing the board after the game's actual move.

        Raises:
            RuntimeError: If called past the end of the game.
        """
        if self._ply >= len(self._game_moves):
            raise RuntimeError("Game has ended; no more moves to replay.")
        actual_uci = self._game_moves[self._ply]
        self._board.push_uci(actual_uci)
        self._history.append(actual_uci)
        self._ply += 1
        past_end = self._ply >= len(self._game_moves)
        truncated = past_end and not self._board.is_game_over()
        return _build_step_info(self._board.copy(), actual_uci, truncated)

    def legal_moves(self) -> list[str]:
        """Return legal UCI moves for the current position.

        Returns:
            List of legal UCI strings.
        """
        return [m.uci() for m in self._board.legal_moves]

    def move_history(self) -> list[str]:
        """Return applied moves as UCI strings.

        Returns:
            Ordered list of applied UCI strings.
        """
        return list(self._history)

    def is_terminal(self) -> bool:
        """Return True when the game is over or all moves have been replayed.

        Returns:
            True if terminal.
        """
        return (
            self._board.is_game_over()
            or self._ply >= len(self._game_moves)
        )

    def total_plies(self) -> int:
        """Return the total number of moves in the loaded game.

        Returns:
            Total ply count (0 before reset() is called).
        """
        return len(self._game_moves)

    def current_ply(self) -> int:
        """Return the zero-based current ply index.

        Returns:
            Current ply.
        """
        return self._ply


# ---------------------------------------------------------------------------
# RandomSource
# ---------------------------------------------------------------------------


class RandomSource:
    """Generates a random-legal-move game from the starting position.

    step() applies the submitted move if it is legal.  If the submitted move
    is illegal (or the caller passes an empty/dummy string), a random legal
    move is chosen instead.  Seeded via reset(seed=...) for reproducibility.

    Implements the SimSource protocol.

    Attributes:
        _max_plies: Upper bound on game length before truncation.
        _rng: Random number generator (seeded on reset).
        _board: Current board state.
        _history: UCI strings applied so far.
        _ply: Zero-based current ply.
    """

    def __init__(self, max_plies: int = 200) -> None:
        """Initialise with the maximum ply limit.

        Args:
            max_plies: Maximum number of plies before the game is truncated.
        """
        self._max_plies = max_plies
        self._rng = random.Random()
        self._board: chess.Board = chess.Board()
        self._history: list[str] = []
        self._ply: int = 0

    def reset(self, seed: int | None = None) -> chess.Board:
        """Reset to starting position.

        Args:
            seed: Optional RNG seed for reproducible random games.

        Returns:
            chess.Board at the starting position.
        """
        self._rng = random.Random(seed)
        self._board = chess.Board()
        self._history = []
        self._ply = 0
        return self._board.copy()

    def step(self, move_uci: str) -> StepInfo:
        """Apply move_uci if legal; otherwise pick a random legal move.

        Args:
            move_uci: UCI move to attempt; ignored if illegal or empty.

        Returns:
            StepInfo for the applied move.

        Raises:
            RuntimeError: If called on a terminal board.
        """
        if self._board.is_game_over():
            raise RuntimeError("Game is over; call reset() to start a new game.")

        legal_ucis = [m.uci() for m in self._board.legal_moves]
        if move_uci in legal_ucis:
            chosen = move_uci
        else:
            chosen = self._rng.choice(legal_ucis)

        self._board.push_uci(chosen)
        self._history.append(chosen)
        self._ply += 1
        truncated = self._ply >= self._max_plies and not self._board.is_game_over()
        return _build_step_info(self._board.copy(), chosen, truncated)

    def legal_moves(self) -> list[str]:
        """Return legal UCI moves for the current position.

        Returns:
            List of legal UCI strings.
        """
        return [m.uci() for m in self._board.legal_moves]

    def move_history(self) -> list[str]:
        """Return applied moves as UCI strings.

        Returns:
            Ordered list of applied UCI strings.
        """
        return list(self._history)

    def is_terminal(self) -> bool:
        """Return True when the game is over or max_plies reached.

        Returns:
            True if terminal or truncated.
        """
        return self._board.is_game_over() or self._ply >= self._max_plies

    def total_plies(self) -> int:
        """Return the configured maximum ply limit.

        Returns:
            max_plies value.
        """
        return self._max_plies

    def current_ply(self) -> int:
        """Return the zero-based current ply index.

        Returns:
            Current ply.
        """
        return self._ply


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_render_context(
    source: SimSource,
    last_move_uci: str,
    predictions: list[MovePrediction] | None = None,
) -> RenderContext:
    """Build a RenderContext from a SimSource after a step.

    Args:
        source: Active SimSource (PGNSource or RandomSource).
        last_move_uci: The move that was just applied.
        predictions: Optional agent predictions for this ply.

    Returns:
        Populated RenderContext.
    """
    # We need the board to compute balance and phase — peek at it via a copy.
    # The source's internal board reflects state *after* the last step.
    # We reconstruct the board from move history for accuracy.
    board = chess.Board()
    for uci in source.move_history():
        board.push_uci(uci)

    return RenderContext(
        ply=source.current_ply(),
        move_history=source.move_history(),
        last_move_uci=last_move_uci,
        predictions=predictions or [],
        material_balance=_material_balance(board),
        game_phase=_game_phase(board),
    )


# ---------------------------------------------------------------------------
# Type alias for static checking
# ---------------------------------------------------------------------------

__all__ = [
    "PGNSource",
    "RandomSource",
    "build_render_context",
]
