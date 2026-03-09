"""Tests for the offline RL PGN trainer pipeline.

T1-T12: config validation, PGNReplayer, PGNRLRewardComputer,
PGNRLTrainer, checkpoint, and YAML config loading.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn
import torch

from chess_sim.config import (
    PGNRLConfig,
    RLConfig,
    load_pgn_rl_config,
)
from chess_sim.training.pgn_replayer import PGNReplayer
from chess_sim.training.pgn_rl_reward_computer import (
    PGNRLRewardComputer,
)
from chess_sim.training.pgn_rl_trainer import PGNRLTrainer
from chess_sim.training.training_utils import l1_normalize


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


def _make_draw_game() -> chess.pgn.Game:
    """Short draw game (4 plies)."""
    game = chess.pgn.Game()
    game.headers["Result"] = "1/2-1/2"
    node = game
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        node = node.add_variation(
            chess.Move.from_uci(uci)
        )
    return game


class TestRLConfig(unittest.TestCase):
    """T1-T3: RLConfig validation."""

    def test_t1_default_no_error(self) -> None:
        """T1: RLConfig defaults construct without error."""
        cfg = RLConfig()
        self.assertEqual(cfg.gamma, 0.99)
        self.assertEqual(cfg.win_reward, 1.0)

    def test_t2_warmup_gt_decay_raises(self) -> None:
        """T2: warmup >= decay_start raises ValueError."""
        with self.assertRaises(ValueError):
            RLConfig(
                warmup_fraction=0.6,
                decay_start_fraction=0.5,
            )

    def test_t3_positive_loss_reward_raises(self) -> None:
        """T3: loss_reward >= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            RLConfig(loss_reward=0.5)


class TestPGNReplayer(unittest.TestCase):
    """T4-T6: PGNReplayer correctness."""

    def setUp(self) -> None:
        """Initialize replayer."""
        self.replayer = PGNReplayer()

    def test_t4_fools_mate_plies(self) -> None:
        """T4: Fool's Mate -> 4 plies, correct sides."""
        game = _make_fools_mate()
        plies = self.replayer.replay(game)
        self.assertEqual(len(plies), 4)
        # Ply 0: white (f2f3), ply 1: black (e7e5), etc.
        self.assertTrue(plies[0].is_white_ply)
        self.assertFalse(plies[1].is_white_ply)
        self.assertTrue(plies[2].is_white_ply)
        self.assertFalse(plies[3].is_white_ply)
        # Black wins: white plies are NOT winner
        self.assertFalse(plies[0].is_winner_ply)
        self.assertTrue(plies[1].is_winner_ply)
        self.assertFalse(plies[2].is_winner_ply)
        self.assertTrue(plies[3].is_winner_ply)

    def test_t5_draw_all_winners(self) -> None:
        """T5: Draw game -> all plies are winner plies."""
        game = _make_draw_game()
        plies = self.replayer.replay(game)
        self.assertEqual(len(plies), 4)
        for ply in plies:
            self.assertTrue(ply.is_winner_ply)

    def test_t6_check_detection(self) -> None:
        """T6: d8h4 in Fool's Mate gives check."""
        game = _make_fools_mate()
        plies = self.replayer.replay(game)
        # Last move d8h4 is checkmate, which is also check
        self.assertEqual(plies[3].gave_check, 1.0)
        # Earlier moves should not give check
        self.assertEqual(plies[0].gave_check, 0.0)
        self.assertEqual(plies[1].gave_check, 0.0)
        self.assertEqual(plies[2].gave_check, 0.0)


class TestPGNRLRewardComputer(unittest.TestCase):
    """T7-T8: reward computation correctness."""

    def setUp(self) -> None:
        """Initialize components."""
        self.replayer = PGNReplayer()
        self.reward_fn = PGNRLRewardComputer()

    def test_t7_temporal_direction(self) -> None:
        """T7: Last ply reward magnitude >= first ply."""
        game = _make_fools_mate()
        plies = self.replayer.replay(game)
        cfg = RLConfig(
            lambda_material=0.0, lambda_check=0.0
        )
        rewards = self.reward_fn.compute(plies, cfg)
        self.assertEqual(rewards.shape[0], 4)
        # Last ply: gamma^0 * outcome, first: gamma^3
        # Last ply magnitude >= first ply magnitude
        self.assertGreaterEqual(
            abs(rewards[-1].item()),
            abs(rewards[0].item()),
        )

    def test_t8_winner_loser_sign(self) -> None:
        """T8: Winner plies > 0, loser plies < 0."""
        game = _make_scholars_mate()
        plies = self.replayer.replay(game)
        cfg = RLConfig(
            lambda_material=0.0, lambda_check=0.0
        )
        rewards = self.reward_fn.compute(plies, cfg)
        for i, ply in enumerate(plies):
            if ply.is_winner_ply:
                self.assertGreater(
                    rewards[i].item(), 0.0,
                    f"Winner ply {i} should be positive",
                )
            else:
                self.assertLess(
                    rewards[i].item(), 0.0,
                    f"Loser ply {i} should be negative",
                )


class TestL1Normalize(unittest.TestCase):
    """T9: l1_normalize edge case."""

    def test_t9_zeros_input(self) -> None:
        """T9: l1_normalize(zeros) returns zeros."""
        result = l1_normalize(torch.zeros(5))
        self.assertTrue(
            torch.allclose(result, torch.zeros(5))
        )


class TestPGNRLTrainer(unittest.TestCase):
    """T10-T11: trainer integration tests."""

    def setUp(self) -> None:
        """Build a minimal trainer on CPU."""
        self.cfg = PGNRLConfig()
        self.trainer = PGNRLTrainer(
            cfg=self.cfg,
            device="cpu",
            total_steps=100,
        )

    def test_t10_train_game_finite_loss(self) -> None:
        """T10: train_game on Fool's Mate returns finite loss."""
        game = _make_fools_mate()
        metrics = self.trainer.train_game(game)
        self.assertIn("total_loss", metrics)
        self.assertTrue(
            torch.isfinite(
                torch.tensor(metrics["total_loss"])
            )
        )

    def test_t11_checkpoint_roundtrip(self) -> None:
        """T11: Save and load checkpoint, weights match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            self.trainer.save_checkpoint(ckpt_path)

            # Create a second trainer and load
            trainer2 = PGNRLTrainer(
                cfg=self.cfg,
                device="cpu",
                total_steps=100,
            )
            trainer2.load_checkpoint(ckpt_path)

            # Verify model weights match
            for p1, p2 in zip(
                self.trainer.model.parameters(),
                trainer2.model.parameters(),
            ):
                self.assertTrue(torch.equal(p1, p2))


class TestConfigYAML(unittest.TestCase):
    """T12: YAML config loading."""

    def test_t12_load_pgn_rl_config(self) -> None:
        """T12: load_pgn_rl_config on train_rl.yaml succeeds."""
        cfg_path = Path(
            "configs/train_rl.yaml"
        )
        cfg = load_pgn_rl_config(cfg_path)
        self.assertEqual(cfg.rl.gamma, 0.99)
        self.assertEqual(cfg.model.d_model, 128)
        self.assertEqual(cfg.decoder.n_layers, 4)
        self.assertEqual(cfg.rl.lambda_ce, 0.5)


if __name__ == "__main__":
    unittest.main()
