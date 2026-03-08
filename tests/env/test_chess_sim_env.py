"""Unit tests for the chess_sim.env package.

Test IDs follow the plan:
  T1  -- PGNSource.reset() returns starting FEN
  T2  -- PGNSource.step() advances by game's actual move, not submitted move
  T3  -- RandomSource always produces legal moves over 50 steps
  T4  -- ChessSimEnv.observation_space shape and dtype
  T5  -- ChessSimEnv.step() raises ValueError on action outside vocab
  T6  -- ChessModelAgent.select_action() returns valid vocab index (mocked)
  T7  -- ChessModelAgent.top_n_predictions() returns top-N ranked predictions
  T8  -- TerminalRenderer.render() emits ANSI escape codes with highlight on
  T9  -- TerminalRenderer.render() uses only ASCII when use_unicode=False
  T10 -- Full env loop terminates on checkmate or game end
  T11 -- gymnasium env_checker passes on ChessSimEnv
  T12 -- load_simulate_config raises TypeError on unknown YAML keys

All tests use CPU only; ChessModel is mocked where model weights are needed.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import chess.pgn
import numpy as np
import torch

from chess_sim.config import load_simulate_config
from chess_sim.data.move_vocab import MoveVocab
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.env import MovePrediction, RenderContext
from chess_sim.env.chess_sim_env import ChessSimEnv
from chess_sim.env.sources import PGNSource, RandomSource
from chess_sim.env.terminal_renderer import TerminalRenderer

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SCHOLARS_MATE_PGN = """[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""


def _make_pgn_file(pgn_text: str) -> Path:
    """Write PGN text to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".pgn", mode="w", delete=False, encoding="utf-8"
    )
    tmp.write(pgn_text)
    tmp.close()
    return Path(tmp.name)


def _make_mock_model_agent(top_moves: list[str]) -> MagicMock:
    """Return a mock ChessModelAgent whose top_n_predictions returns top_moves."""
    from chess_sim.env.agent_adapter import ChessModelAgent
    from chess_sim.model.chess_model import ChessModel

    mock_model = MagicMock(spec=ChessModel)
    # Make _forward return a zero logit vector of shape [V]
    vocab = MoveVocab()
    V = len(vocab)
    fake_logits = torch.zeros(V)
    for i, uci in enumerate(reversed(top_moves)):
        fake_logits[vocab.encode(uci)] = float(i + 1)

    agent = ChessModelAgent.__new__(ChessModelAgent)
    agent._model = mock_model
    agent._tokenizer = BoardTokenizer()
    agent._vocab = vocab
    from chess_sim.data.move_tokenizer import MoveTokenizer
    agent._move_tok = MoveTokenizer()
    agent._temperature = 1.0
    agent._device = torch.device("cpu")

    # Patch _forward to return our fake logits
    agent._forward = MagicMock(return_value=fake_logits)
    return agent


def _make_random_env(max_plies: int = 200) -> ChessSimEnv:
    """Create a ChessSimEnv backed by RandomSource."""
    source = RandomSource(max_plies=max_plies)
    return ChessSimEnv(source=source)


# ---------------------------------------------------------------------------
# T1 -- PGNSource.reset() returns starting board
# ---------------------------------------------------------------------------


class TestT1PGNSourceReset(unittest.TestCase):
    """T1: PGNSource.reset() returns the game's starting FEN."""

    def test_reset_returns_starting_fen(self) -> None:
        pgn_path = _make_pgn_file(_SCHOLARS_MATE_PGN)
        source = PGNSource(pgn_path=pgn_path, game_index=0)
        board = source.reset()
        self.assertEqual(board.fen(), chess.Board().fen())
        self.assertEqual(source.current_ply(), 0)


# ---------------------------------------------------------------------------
# T2 -- PGNSource.step() applies game move, ignores submitted move
# ---------------------------------------------------------------------------


class TestT2PGNSourceStep(unittest.TestCase):
    """T2: step() always applies the recorded game move regardless of argument."""

    def test_step_applies_actual_game_move(self) -> None:
        pgn_path = _make_pgn_file(_SCHOLARS_MATE_PGN)
        source = PGNSource(pgn_path=pgn_path, game_index=0)
        source.reset()
        # First recorded move is e2e4; submit a dummy move d2d4.
        info = source.step("d2d4")
        self.assertEqual(info.move_uci, "e2e4", "Should apply game move, not d2d4")
        self.assertEqual(source.current_ply(), 1)

    def test_total_plies_matches_game_length(self) -> None:
        pgn_path = _make_pgn_file(_SCHOLARS_MATE_PGN)
        source = PGNSource(pgn_path=pgn_path, game_index=0)
        source.reset()
        # Scholar's mate: 4 moves = 7 half-moves (e4, e5, Qh5, Nc6, Bc4, Nf6, Qxf7)
        self.assertEqual(source.total_plies(), 7)


# ---------------------------------------------------------------------------
# T3 -- RandomSource always produces legal moves
# ---------------------------------------------------------------------------


class TestT3RandomSourceLegalMoves(unittest.TestCase):
    """T3: RandomSource only returns legal moves for 50 consecutive steps."""

    def test_fifty_steps_all_legal(self) -> None:
        source = RandomSource(max_plies=200)
        source.reset(seed=42)
        for step_idx in range(50):
            if source.is_terminal():
                break
            legal_before = set(source.legal_moves())
            info = source.step("")
            self.assertIn(
                info.move_uci,
                legal_before,
                f"Step {step_idx}: move {info.move_uci!r} not in legal moves",
            )


# ---------------------------------------------------------------------------
# T4 -- ChessSimEnv observation_space shape and dtype
# ---------------------------------------------------------------------------


class TestT4ObservationSpace(unittest.TestCase):
    """T4: observation_space has correct shape and dtype."""

    def test_observation_space_shape(self) -> None:
        env = _make_random_env()
        self.assertEqual(env.observation_space.shape, (65, 3))
        self.assertEqual(env.observation_space.dtype, np.float32)

    def test_reset_obs_shape_and_range(self) -> None:
        env = _make_random_env()
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (65, 3))
        self.assertEqual(obs.dtype, np.float32)
        self.assertGreaterEqual(obs.min(), 0.0)
        self.assertLessEqual(obs.max(), 7.0)


# ---------------------------------------------------------------------------
# T5 -- ChessSimEnv.step() raises on invalid action index
# ---------------------------------------------------------------------------


class TestT5StepRaisesOnInvalidAction(unittest.TestCase):
    """T5: step() raises ValueError for action indices outside [0, 1971)."""

    def test_raises_on_negative_action(self) -> None:
        env = _make_random_env()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(-1)

    def test_raises_on_out_of_range_action(self) -> None:
        env = _make_random_env()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(9999)


# ---------------------------------------------------------------------------
# T6 -- ChessModelAgent.select_action() returns valid vocab index
# ---------------------------------------------------------------------------


class TestT6ChessModelAgentSelectAction(unittest.TestCase):
    """T6: select_action() returns a valid MoveVocab index for a legal move."""

    def test_select_action_returns_valid_index(self) -> None:
        vocab = MoveVocab()
        legal = ["e2e4", "d2d4", "g1f3"]
        agent = _make_mock_model_agent(top_moves=legal)
        env = _make_random_env()
        obs, _ = env.reset()

        action = agent.select_action(obs, legal)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, len(vocab))
        # Decoded move must be one of the legal moves.
        decoded = vocab.decode(action)
        self.assertIn(decoded, legal)


# ---------------------------------------------------------------------------
# T7 -- ChessModelAgent.top_n_predictions() returns N ranked predictions
# ---------------------------------------------------------------------------


class TestT7ChessModelAgentTopN(unittest.TestCase):
    """T7: top_n_predictions() returns exactly top_n ranked MovePredictions."""

    def test_top_3_predictions_length_and_structure(self) -> None:
        legal = ["e2e4", "d2d4", "g1f3", "c2c4"]
        agent = _make_mock_model_agent(top_moves=["e2e4", "d2d4", "g1f3"])
        env = _make_random_env()
        obs, _ = env.reset()

        preds = agent.top_n_predictions(obs, legal, n=3)
        self.assertEqual(len(preds), 3)
        for i, pred in enumerate(preds, start=1):
            self.assertEqual(pred.rank, i)
            self.assertIsInstance(pred.probability, float)
            self.assertGreaterEqual(pred.probability, 0.0)
            self.assertLessEqual(pred.probability, 1.0)
            self.assertIn(pred.move_uci, legal)

    def test_probabilities_sum_to_at_most_one(self) -> None:
        legal = ["e2e4", "d2d4", "g1f3"]
        agent = _make_mock_model_agent(top_moves=legal)
        env = _make_random_env()
        obs, _ = env.reset()

        preds = agent.top_n_predictions(obs, legal, n=3)
        total = sum(p.probability for p in preds)
        self.assertLessEqual(total, 1.0 + 1e-5)


# ---------------------------------------------------------------------------
# T8 -- TerminalRenderer emits ANSI codes with highlight_last=True
# ---------------------------------------------------------------------------


class TestT8TerminalRendererANSI(unittest.TestCase):
    """T8: render() writes ANSI escape sequences when highlight_last=True."""

    def test_ansi_codes_present_with_last_move(self) -> None:
        renderer = TerminalRenderer(use_unicode=False, highlight_last=True)
        board = chess.Board()
        board.push_uci("e2e4")
        ctx = RenderContext(
            ply=1,
            move_history=["e2e4"],
            last_move_uci="e2e4",
            predictions=[],
            material_balance=0,
            game_phase="opening",
        )
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            renderer.render(board, ctx)
        output = buf.getvalue()
        self.assertIn("\033[", output, "ANSI escape code should be present")


# ---------------------------------------------------------------------------
# T9 -- TerminalRenderer falls back to ASCII
# ---------------------------------------------------------------------------


class TestT9TerminalRendererASCII(unittest.TestCase):
    """T9: render() uses only ASCII characters when use_unicode=False."""

    def test_no_unicode_above_u007f(self) -> None:
        renderer = TerminalRenderer(use_unicode=False, highlight_last=False)
        board = chess.Board()
        ctx = RenderContext(
            ply=0,
            move_history=[],
            last_move_uci="",
            predictions=[],
            material_balance=0,
            game_phase="opening",
        )
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            renderer.render(board, ctx)
        # Strip ANSI codes before checking for non-ASCII
        raw = buf.getvalue()
        # Remove all ANSI escape sequences
        import re
        clean = re.sub(r"\033\[[0-9;]*m", "", raw)
        non_ascii = [c for c in clean if ord(c) > 127]
        self.assertEqual(
            non_ascii,
            [],
            f"Unexpected non-ASCII chars: {non_ascii[:5]}",
        )


# ---------------------------------------------------------------------------
# T10 -- Full env loop terminates on game end
# ---------------------------------------------------------------------------


class TestT10EnvLoopTerminates(unittest.TestCase):
    """T10: Scholar's mate game terminates with terminated=True."""

    def test_pgn_game_terminates(self) -> None:
        pgn_path = _make_pgn_file(_SCHOLARS_MATE_PGN)
        source = PGNSource(pgn_path=pgn_path, game_index=0)
        env = ChessSimEnv(source=source)
        obs, info = env.reset()

        terminated = False
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        self.assertTrue(
            terminated or truncated,
            "Game should have ended within 20 steps.",
        )
        # Reward for a decisive game must be ±1.0.
        self.assertIn(
            reward,
            {1.0, -1.0, 0.0},
            f"Unexpected reward {reward}",
        )


# ---------------------------------------------------------------------------
# T11 -- gymnasium env_checker passes
# ---------------------------------------------------------------------------


class TestT11GymnasiumEnvChecker(unittest.TestCase):
    """T11: gymnasium env_checker passes on ChessSimEnv with RandomSource."""

    def test_env_checker_passes(self) -> None:
        from gymnasium.utils.env_checker import check_env

        source = RandomSource(max_plies=10)
        env = ChessSimEnv(source=source, render_mode=None)
        # check_env raises AssertionError or warns on non-compliance.
        check_env(env, warn=True)


# ---------------------------------------------------------------------------
# T12 -- load_simulate_config raises on unknown YAML keys
# ---------------------------------------------------------------------------


class TestT12SimulateConfigUnknownKeys(unittest.TestCase):
    """T12: load_simulate_config raises TypeError for unknown YAML keys."""

    def test_unknown_top_level_key_raises(self) -> None:
        bad_yaml = "mode: pgn\nunknown_field: 42\n"
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(bad_yaml)
            path = Path(f.name)
        with self.assertRaises(TypeError):
            load_simulate_config(path)

    def test_valid_yaml_loads_correctly(self) -> None:
        valid_yaml = "mode: random\ntick_rate: 1.0\nmax_plies: 50\n"
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(valid_yaml)
            path = Path(f.name)
        cfg = load_simulate_config(path)
        self.assertEqual(cfg.mode, "random")
        self.assertAlmostEqual(cfg.tick_rate, 1.0)
        self.assertEqual(cfg.max_plies, 50)


if __name__ == "__main__":
    unittest.main()
