"""Unit tests for scripts.evaluate (TEV01-TEV14).

All tests run on CPU only.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import chess
import torch

from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.model.encoder import ChessEncoder
from chess_sim.model.heads import PredictionHeads
from chess_sim.training.trainer import Trainer
from chess_sim.types import TrainingExample
from chess_sim.utils import winner_color
from scripts.evaluate import (
    GameEvaluator,
    StepResult,
    evaluate_step,
    per_head_ce,
    shannon_entropy,
    top1_accuracy,
)
from scripts.train_real import game_to_examples
from tests.utils import make_small_pgn_fixture, parse_pgn_games

# Maximum entropy for 64 classes: log(64)
LOG_64 = math.log(64)


class TestShannonEntropy(unittest.TestCase):
    """TEV01-TEV02: Shannon entropy pure function."""

    def test_tev01_uniform_logits(self) -> None:
        """TEV01: zeros(64) -> uniform -> H ~ log(64)."""
        logits = torch.zeros(64)
        h = shannon_entropy(logits)
        self.assertAlmostEqual(h, LOG_64, places=2)

    def test_tev02_near_one_hot(self) -> None:
        """TEV02: near-one-hot logits -> H near 0."""
        logits = torch.full((64,), -100.0)
        logits[0] = 100.0
        h = shannon_entropy(logits)
        self.assertLess(h, 0.01)


class TestTop1Accuracy(unittest.TestCase):
    """TEV03-TEV04: top1_accuracy pure function."""

    def test_tev03_correct(self) -> None:
        """TEV03: returns 1 when argmax matches label."""
        logits = torch.zeros(64)
        logits[7] = 10.0
        self.assertEqual(top1_accuracy(logits, 7), 1)

    def test_tev04_wrong(self) -> None:
        """TEV04: returns 0 when argmax != label."""
        logits = torch.zeros(64)
        logits[7] = 10.0
        self.assertEqual(top1_accuracy(logits, 3), 0)


class TestPerHeadCE(unittest.TestCase):
    """TEV06-TEV07: per_head_ce pure function."""

    def test_tev07_valid_label(self) -> None:
        """TEV07: returns finite positive float for valid label."""
        logits = torch.randn(64)
        result = per_head_ce(logits, label=10)
        self.assertGreater(result, 0.0)
        self.assertTrue(math.isfinite(result))


class TestEvaluateStep(unittest.TestCase):
    """TEV08: evaluate_step integration."""

    def setUp(self) -> None:
        """Create encoder, heads, and a synthetic example."""
        torch.manual_seed(42)
        self.encoder = ChessEncoder().to("cpu")
        self.heads = PredictionHeads().to("cpu")
        self.encoder.eval()
        self.heads.eval()
        # Synthetic valid example
        self.example = TrainingExample(
            board_tokens=[0] + [1] * 64,
            color_tokens=[0] + [0] * 64,
            trajectory_tokens=[0] * 65,
            src_sq=12,
            tgt_sq=28,
        )

    def test_tev08_step_result_populated(self) -> None:
        """TEV08: evaluate_step returns StepResult with all fields."""
        result = evaluate_step(
            example=self.example,
            move_uci="e2e4",
            ply=0,
            encoder=self.encoder,
            heads=self.heads,
            device="cpu",
        )
        self.assertIsInstance(result, StepResult)
        self.assertEqual(result.ply, 0)
        self.assertEqual(result.move_uci, "e2e4")
        # All losses are finite floats
        for field in ["loss_src", "loss_tgt", "total_loss"]:
            val = getattr(result, field)
            self.assertIsInstance(val, float)
            self.assertTrue(math.isfinite(val))
        # Accuracies are 0 or 1
        for field in ["acc_src", "acc_tgt"]:
            val = getattr(result, field)
            self.assertIn(val, (0, 1))
        # Entropies are positive
        for field in ["entropy_src", "entropy_tgt", "mean_entropy"]:
            val = getattr(result, field)
            self.assertGreater(val, 0.0)


class TestGameEvaluator(unittest.TestCase):
    """TEV10-TEV11: GameEvaluator integration tests."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a checkpoint and GameEvaluator once for all tests."""
        torch.manual_seed(99)
        trainer = Trainer(device="cpu")
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._ckpt_path = Path(cls._tmpdir.name) / "test.pt"
        trainer.save_checkpoint(cls._ckpt_path)
        cls.evaluator = GameEvaluator(
            checkpoint_path=cls._ckpt_path, device="cpu"
        )
        cls.games = parse_pgn_games(make_small_pgn_fixture())

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temp directory."""
        cls._tmpdir.cleanup()

    def test_tev10_one_result_per_ply(self) -> None:
        """TEV10: evaluate_game returns one StepResult per ply."""
        game = self.games[0]
        moves = list(game.mainline_moves())
        results = self.evaluator.evaluate_game(game)
        self.assertEqual(len(results), len(moves))
        for i, r in enumerate(results):
            self.assertEqual(r.ply, i)
            self.assertIsInstance(r, StepResult)

    def test_tev11_mean_entropy_correct(self) -> None:
        """TEV11: mean_entropy == mean of 2 head entropies."""
        game = self.games[1]
        results = self.evaluator.evaluate_game(game)
        self.assertGreater(len(results), 0)
        for r in results:
            expected = (r.entropy_src + r.entropy_tgt) / 2.0
            self.assertAlmostEqual(
                r.mean_entropy, expected, places=6
            )


class TestWinnerColorHelper(unittest.TestCase):
    """TEV12: winner_color() pure function."""

    def setUp(self) -> None:
        self.games = parse_pgn_games(
            make_small_pgn_fixture()
        )

    def test_tev12a_white_win(self) -> None:
        """TEV12a: '1-0' result returns chess.WHITE."""
        self.assertEqual(
            winner_color(self.games[0]), chess.WHITE
        )

    def test_tev12b_black_win(self) -> None:
        """TEV12b: '0-1' result returns chess.BLACK."""
        self.assertEqual(
            winner_color(self.games[1]), chess.BLACK
        )

    def test_tev12c_draw_returns_none(self) -> None:
        """TEV12c: '1/2-1/2' result returns None."""
        self.assertIsNone(winner_color(self.games[2]))


class TestWinnerOnlyFiltering(unittest.TestCase):
    """TEV13: game_to_examples() with winners_only=True."""

    def setUp(self) -> None:
        self.tokenizer = BoardTokenizer()
        self.games = parse_pgn_games(
            make_small_pgn_fixture()
        )

    def test_tev13a_fewer_examples_for_decisive_game(
        self,
    ) -> None:
        """TEV13a: winners_only=True yields fewer examples."""
        game = self.games[0]  # 1-0
        all_ex = game_to_examples(
            game, self.tokenizer, winners_only=False
        )
        win_ex = game_to_examples(
            game, self.tokenizer, winners_only=True
        )
        self.assertGreater(len(all_ex), len(win_ex))
        self.assertGreater(len(win_ex), 0)

    def test_tev13b_draw_returns_empty(self) -> None:
        """TEV13b: winners_only=True on draw returns []."""
        game = self.games[2]  # 1/2-1/2
        examples = game_to_examples(
            game, self.tokenizer, winners_only=True
        )
        self.assertEqual(examples, [])

    def test_tev13c_baseline_unchanged(self) -> None:
        """TEV13c: winners_only=False returns all plies."""
        game = self.games[0]
        all_ex = game_to_examples(
            game, self.tokenizer, winners_only=False
        )
        moves = list(game.mainline_moves())
        self.assertEqual(len(all_ex), len(moves))


class TestGameEvaluatorWinnersOnly(unittest.TestCase):
    """TEV14: GameEvaluator.evaluate_game() winners_only."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(99)
        trainer = Trainer(device="cpu")
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._ckpt_path = (
            Path(cls._tmpdir.name) / "test.pt"
        )
        trainer.save_checkpoint(cls._ckpt_path)
        cls.evaluator = GameEvaluator(
            checkpoint_path=cls._ckpt_path, device="cpu"
        )
        cls.games = parse_pgn_games(
            make_small_pgn_fixture()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_tev14a_fewer_results_than_total(
        self,
    ) -> None:
        """TEV14a: winners_only=True yields fewer results."""
        game = self.games[0]  # 1-0
        all_results = self.evaluator.evaluate_game(
            game, winners_only=False
        )
        win_results = self.evaluator.evaluate_game(
            game, winners_only=True
        )
        self.assertLess(
            len(win_results), len(all_results)
        )
        self.assertGreater(len(win_results), 0)

    def test_tev14b_draw_game_returns_empty(self) -> None:
        """TEV14b: winners_only=True on draw returns []."""
        game = self.games[2]  # 1/2-1/2
        results = self.evaluator.evaluate_game(
            game, winners_only=False
        )
        self.assertGreater(len(results), 0)
        win_results = self.evaluator.evaluate_game(
            game, winners_only=True
        )
        self.assertEqual(win_results, [])

    def test_tev14c_ply_indices_are_even_for_white_winner(
        self,
    ) -> None:
        """TEV14c: ply indices reflect original game position."""
        game = self.games[0]  # 1-0: White at plies 0,2,4..
        win_results = self.evaluator.evaluate_game(
            game, winners_only=True
        )
        for r in win_results:
            self.assertEqual(r.ply % 2, 0)


if __name__ == "__main__":
    unittest.main()
