"""Tests for PGNRLTrainerV2 (outcome-only, no value head).

T1-T5: verify v2 trainer returns correct metric schema,
single-param optimizer, and scalar current_lr.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn

from chess_sim.config import PGNRLConfig
from chess_sim.training.pgn_rl_trainer_v2 import (
    PGNRLTrainerV2,
)


def _make_fools_mate() -> chess.pgn.Game:
    """Create a Fool's Mate game (4 moves, Black wins)."""
    game = chess.pgn.Game()
    game.headers["Result"] = "0-1"
    node = game
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        node = node.add_variation(
            chess.Move.from_uci(uci)
        )
    return game


def _make_scholars_mate() -> chess.pgn.Game:
    """Scholar's Mate (7 plies, White wins 1-0)."""
    game = chess.pgn.Game()
    game.headers["Result"] = "1-0"
    node = game
    for uci in [
        "e2e4", "e7e5", "f1c4", "b8c6",
        "d1h5", "g8f6", "h5f7",
    ]:
        node = node.add_variation(
            chess.Move.from_uci(uci)
        )
    return game


def _write_pgn(game: chess.pgn.Game) -> Path:
    """Write a single game to a temp PGN file."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pgn", delete=False
    )
    exporter = chess.pgn.StringExporter(headers=True)
    tmp.write(game.accept(exporter))
    tmp.flush()
    return Path(tmp.name)


class TestTrainGameMetrics(unittest.TestCase):
    """T1-T2: train_game return dict shape and values."""

    def setUp(self) -> None:
        self.cfg = PGNRLConfig()
        self.trainer = PGNRLTrainerV2(
            cfg=self.cfg,
            device="cpu",
            total_steps=100,
        )

    def test_t1_train_game_keys(self) -> None:
        """T1: train_game returns expected keys, no v1 keys."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        expected = {
            "total_loss",
            "rsbc_loss",
            "mean_reward",
            "n_games",
        }
        self.assertTrue(
            expected.issubset(metrics.keys()),
            f"Missing keys: {expected - metrics.keys()}",
        )
        # Must NOT contain v1 value-head keys
        forbidden = {
            "value_loss",
            "mean_advantage",
            "std_advantage",
        }
        overlap = forbidden & metrics.keys()
        self.assertFalse(
            overlap,
            f"Forbidden v1 keys present: {overlap}",
        )

    def test_t2_total_equals_rsbc_at_lambda1(self) -> None:
        """T2: total_loss == rsbc_loss when lambda_rsbc=1.0."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertAlmostEqual(
            metrics["total_loss"],
            metrics["rsbc_loss"],
            places=6,
            msg="total_loss should equal rsbc_loss "
            "when lambda_rsbc=1.0",
        )


class TestOptimizerAndLR(unittest.TestCase):
    """T3-T4: single param group and scalar current_lr."""

    def setUp(self) -> None:
        self.cfg = PGNRLConfig()
        self.trainer = PGNRLTrainerV2(
            cfg=self.cfg,
            device="cpu",
            total_steps=100,
        )

    def test_t3_single_param_group(self) -> None:
        """T3: Optimizer has exactly one param group."""
        n_groups = len(
            self.trainer._opt.param_groups
        )
        self.assertEqual(
            n_groups,
            1,
            f"Expected 1 param group, got {n_groups}",
        )

    def test_t4_current_lr_is_float(self) -> None:
        """T4: current_lr returns a single float, not tuple."""
        lr = self.trainer.current_lr
        self.assertIsInstance(
            lr,
            float,
            f"current_lr should be float, got {type(lr)}",
        )


class TestTrainEpochMetrics(unittest.TestCase):
    """T5: train_epoch return dict matches v2 schema."""

    def setUp(self) -> None:
        self.cfg = PGNRLConfig()
        self.trainer = PGNRLTrainerV2(
            cfg=self.cfg,
            device="cpu",
            total_steps=100,
        )

    def test_t5_train_epoch_v2_schema(self) -> None:
        """T5: train_epoch metrics have v2 keys only."""
        game = _make_fools_mate()
        pgn_path = _write_pgn(game)
        metrics = self.trainer.train_epoch(
            pgn_path, max_games=1
        )
        expected = {
            "total_loss",
            "rsbc_loss",
            "mean_reward",
            "n_games",
        }
        self.assertEqual(
            metrics.keys(),
            expected,
            f"Unexpected keys: {metrics.keys()}",
        )
        # Confirm no v1 keys leaked in
        forbidden = {
            "value_loss",
            "mean_advantage",
            "std_advantage",
        }
        overlap = forbidden & metrics.keys()
        self.assertFalse(
            overlap,
            f"Forbidden v1 keys present: {overlap}",
        )


if __name__ == "__main__":
    unittest.main()
