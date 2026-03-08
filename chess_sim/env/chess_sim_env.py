"""ChessSimEnv: gymnasium-compatible chess simulation environment.

Wraps a SimSource (PGNSource or RandomSource) behind the standard
gymnasium.Env interface so the environment can be used by any Policy
that follows the gymnasium agent loop.

Observation space: Box(low=0, high=7, shape=(65, 3), dtype=float32)
  - channel 0: board_tokens  (piece type  0-7)
  - channel 1: color_tokens  (0=empty, 1=player, 2=opponent)
  - channel 2: trajectory_tokens (0=none, 1-4 = last-move roles)

Action space: Discrete(1971)
  - Indices map 1-to-1 with MoveVocab (0=PAD, 1=SOS, 2=EOS, 3+=UCI moves)

Illegal action: raises ValueError with the submitted and legal moves listed.
"""

from __future__ import annotations

from typing import Any

import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chess_sim.data.move_vocab import MoveVocab
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.env import (
    Observation,
    RenderContext,
    SimSource,
    TerminalRenderable,
)
from chess_sim.env.agent_adapter import _compute_trajectory_tokens
from chess_sim.env.sources import _game_phase, _material_balance

# Move vocabulary size (PAD + SOS + EOS + UCI moves).
_VOCAB_SIZE: int = 1971


class ChessSimEnv(gym.Env[Observation, int]):
    """Gymnasium environment for chess simulation.

    Wraps a SimSource that manages board state and applies moves.  The
    environment packs three token streams into a (65, 3) float32 array,
    maps integer action indices to UCI strings via MoveVocab, and calls an
    optional TerminalRenderable after every step.

    Attributes:
        metadata: Gymnasium metadata dict; render_modes=["terminal"].
        observation_space: Box (65, 3) float32.
        action_space: Discrete(1971).

    Example:
        >>> from chess_sim.env.sources import RandomSource
        >>> source = RandomSource()
        >>> env = ChessSimEnv(source)
        >>> obs, info = env.reset()
        >>> obs.shape
        (65, 3)
    """

    metadata: dict[str, Any] = {"render_modes": ["terminal"]}

    def __init__(
        self,
        source: SimSource,
        tokenizer: BoardTokenizer | None = None,
        vocab: MoveVocab | None = None,
        renderer: TerminalRenderable | None = None,
        render_mode: str | None = "terminal",
    ) -> None:
        """Initialise the environment.

        Args:
            source: SimSource implementation that manages board state.
            tokenizer: BoardTokenizer to pack board into tokens.
                Instantiated automatically when None.
            vocab: MoveVocab for action ↔ UCI mapping.
                Instantiated automatically when None.
            renderer: Optional TerminalRenderable; called by render().
                When None, render() is a no-op.
            render_mode: Must be "terminal" or None.
        """
        super().__init__()
        assert render_mode in (None, "terminal"), (
            f"Unsupported render_mode: {render_mode!r}"
        )
        self._source = source
        self._tokenizer = tokenizer or BoardTokenizer()
        self._vocab = vocab or MoveVocab()
        self._renderer = renderer
        self.render_mode = render_mode

        # Gymnasium spaces.
        self.observation_space = spaces.Box(
            low=0.0,
            high=7.0,
            shape=(65, 3),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(_VOCAB_SIZE)

        # Internal state reset on each reset() call.
        self._current_board: chess.Board = chess.Board()
        self._last_move_uci: str = ""
        self._last_predictions: list[Any] = []

    # ------------------------------------------------------------------
    # Gymnasium required methods
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment to the starting position.

        Args:
            seed: Optional RNG seed forwarded to RandomSource if applicable.
            options: Unused; accepted for API compatibility.

        Returns:
            Tuple of (observation, info_dict).
            info_dict keys: "board", "legal_moves", "ply".
        """
        super().reset(seed=seed)
        # Forward seed to source if it accepts it (RandomSource).
        try:
            board = self._source.reset(seed=seed)  # type: ignore[call-arg]
        except TypeError:
            board = self._source.reset()

        self._current_board = board
        self._last_move_uci = ""
        self._last_predictions = []

        obs = self._pack_observation(board, [])
        info = self._build_info(board)
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply an action and advance the game by one ply.

        For PGNSource, the action is decoded but the source ignores it and
        applies the game's recorded move.  For RandomSource, the decoded UCI
        is used if legal.

        Args:
            action: Integer MoveVocab index.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict).

        Raises:
            ValueError: If action is outside the vocabulary range [0, 1971).
        """
        if action < 0 or action >= _VOCAB_SIZE:
            raise ValueError(
                f"Action {action} is outside the valid range [0, {_VOCAB_SIZE})."
                f" Legal moves: {self._source.legal_moves()[:10]}..."
            )

        # Decode action; for PAD/SOS/EOS we pass an empty string so the source
        # falls back to its default behaviour (game move or random).
        try:
            move_uci = self._vocab.decode(action)
        except KeyError:
            move_uci = ""

        prior_history = self._source.move_history()
        step_info = self._source.step(move_uci)

        self._current_board = step_info.board
        self._last_move_uci = step_info.move_uci

        full_history = prior_history + [step_info.move_uci]
        obs = self._pack_observation(step_info.board, full_history)
        info = self._build_info(step_info.board)
        info["move_uci"] = step_info.move_uci

        return obs, step_info.reward, step_info.terminated, step_info.truncated, info

    def render(self) -> None:
        """Write the current board state to the terminal.

        Calls the injected TerminalRenderable with the current board and a
        RenderContext built from the source's state.  No-op when no renderer
        was provided or render_mode is None.
        """
        if self._renderer is None or self.render_mode != "terminal":
            return
        ctx = RenderContext(
            ply=self._source.current_ply(),
            move_history=self._source.move_history(),
            last_move_uci=self._last_move_uci,
            predictions=list(self._last_predictions),
            material_balance=_material_balance(self._current_board),
            game_phase=_game_phase(self._current_board),
        )
        self._renderer.render(self._current_board, ctx)

    def close(self) -> None:
        """Clean up resources (no-op for this environment)."""

    # ------------------------------------------------------------------
    # Convenience accessors (not part of the gym.Env API)
    # ------------------------------------------------------------------

    def legal_uci_moves(self) -> list[str]:
        """Return legal moves for the current position as UCI strings.

        Returns:
            List of legal UCI strings.
        """
        return self._source.legal_moves()

    def set_predictions(self, predictions: list[Any]) -> None:
        """Store agent predictions to be included in the next render.

        Call this before render() to show predictions in the terminal frame.

        Args:
            predictions: List of MovePrediction objects from ChessModelAgent.
        """
        self._last_predictions = predictions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pack_observation(
        self,
        board: chess.Board,
        move_history: list[str],
    ) -> Observation:
        """Pack board state into a (65, 3) float32 observation array.

        Channel layout:
          - [:, 0] = board_tokens (piece type 0-7)
          - [:, 1] = color_tokens (0=empty, 1=player, 2=opponent)
          - [:, 2] = trajectory_tokens (0-4 last-move roles)

        Args:
            board: chess.Board to tokenize.
            move_history: All UCI moves applied so far (for trajectory).

        Returns:
            Float32 numpy array of shape (65, 3).
        """
        tokenized = self._tokenizer.tokenize(board, board.turn)
        traj = _compute_trajectory_tokens(move_history)

        obs = np.zeros((65, 3), dtype=np.float32)
        obs[:, 0] = tokenized.board_tokens
        obs[:, 1] = tokenized.color_tokens
        obs[:, 2] = traj
        return obs

    def _build_info(self, board: chess.Board) -> dict[str, Any]:
        """Build the info dictionary returned by reset() and step().

        Args:
            board: Current board state.

        Returns:
            Dict with keys: "board", "legal_moves", "ply".
        """
        return {
            "board": board,
            "legal_moves": self._source.legal_moves(),
            "ply": self._source.current_ply(),
        }
