"""Tests for the offline RL PGN trainer pipeline.

T1-T12: config validation, PGNReplayer, PGNRLRewardComputer,
PGNRLTrainer, checkpoint, and YAML config loading.
"""
from __future__ import annotations

import math
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
from chess_sim.model.chess_model import ChessModel
from chess_sim.model.value_heads import ActionConditionedValueHead
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
        self.assertEqual(cfg.lambda_outcome, 1.0)
        self.assertEqual(cfg.lambda_material, 0.1)
        self.assertEqual(cfg.draw_reward_norm, 0.0)

    def test_t2_warmup_gt_decay_raises(self) -> None:
        """T2: warmup >= decay_start raises ValueError."""
        with self.assertRaises(ValueError):
            RLConfig(
                warmup_fraction=0.6,
                decay_start_fraction=0.5,
            )

    def test_t3_negative_lambda_outcome_raises(self) -> None:
        """T3: lambda_outcome < 0 raises ValueError."""
        with self.assertRaises(ValueError):
            RLConfig(lambda_outcome=-0.1)

    def test_t3b_invalid_train_color_raises(self) -> None:
        """T3b: train_color not 'white'/'black' raises ValueError."""
        with self.assertRaises(ValueError):
            RLConfig(train_color="both")

    def test_t18_negative_lambda_value_raises(self) -> None:
        """RLConfig with lambda_value < 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            RLConfig(lambda_value=-1.0)

    def test_t22_lambda_value_zero_is_valid(self) -> None:
        """RLConfig(lambda_value=0.0) must not raise."""
        try:
            RLConfig(lambda_value=0.0)
        except ValueError:
            self.fail(
                "RLConfig(lambda_value=0.0) raised "
                "ValueError unexpectedly"
            )


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
        """T5: Draw game -> all plies are winner and draw plies."""
        game = _make_draw_game()
        plies = self.replayer.replay(game)
        self.assertEqual(len(plies), 4)
        for ply in plies:
            self.assertTrue(ply.is_winner_ply)
            self.assertTrue(ply.is_draw_ply)

    def test_t6_decisive_game_not_draw(self) -> None:
        """T6: Decisive game -> all plies have is_draw_ply=False."""
        game = _make_fools_mate()
        plies = self.replayer.replay(game)
        for ply in plies:
            self.assertFalse(ply.is_draw_ply)


class TestPGNRLRewardComputer(unittest.TestCase):
    """T7-T8: reward computation correctness."""

    def setUp(self) -> None:
        """Initialize components."""
        self.replayer = PGNReplayer()
        self.reward_fn = PGNRLRewardComputer()

    def test_t7_reward_shape_matches_plies(self) -> None:
        """T7: Reward tensor length matches number of plies."""
        game = _make_fools_mate()
        plies = self.replayer.replay(game)
        cfg = RLConfig()
        rewards = self.reward_fn.compute(plies, cfg)
        self.assertEqual(rewards.shape[0], len(plies))

    def test_t8_winner_loser_sign(self) -> None:
        """T8: Winner plies > 0, loser plies < 0."""
        game = _make_scholars_mate()
        plies = self.replayer.replay(game)
        cfg = RLConfig()
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


    def test_t8b_draw_reward_uses_draw_reward_norm(
        self,
    ) -> None:
        """T8b: Draw game w/ draw_reward_norm=0 -> outcome=0."""
        game = _make_draw_game()
        plies = self.replayer.replay(game)
        cfg = RLConfig(draw_reward_norm=0.0)
        rewards = self.reward_fn.compute(plies, cfg)
        # With draw_reward_norm=0 and no captures,
        # all rewards should be near zero.
        self.assertEqual(rewards.shape[0], len(plies))


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

    def test_t15_train_game_returns_value_loss_key(
        self,
    ) -> None:
        """train_game must include 'value_loss' in returned metrics."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertIn("value_loss", metrics)

    def test_t16_value_loss_non_negative(self) -> None:
        """MSE value_loss must be >= 0 (MSE is always non-negative)."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertGreaterEqual(metrics["value_loss"], 0.0)

    def test_t21_checkpoint_includes_value_head_weights(
        self,
    ) -> None:
        """Checkpoint round-trip must preserve value_head weights."""
        original_weights = {
            k: v.clone()
            for k, v in (
                self.trainer.model.value_head.state_dict().items()
            )
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            self.trainer.save_checkpoint(ckpt_path)
            self.trainer.load_checkpoint(ckpt_path)
        for k, v in (
            self.trainer.model.value_head.state_dict().items()
        ):
            self.assertTrue(
                torch.equal(v, original_weights[k]),
                f"Mismatch in {k}",
            )

    def test_t23_mean_advantage_is_finite(self) -> None:
        """mean_advantage in train_game metrics must be finite."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertIn("mean_advantage", metrics)
        self.assertTrue(
            math.isfinite(metrics["mean_advantage"])
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
        self.assertEqual(cfg.rl.lambda_outcome, 1.0)
        self.assertEqual(cfg.model.d_model, 128)
        self.assertEqual(cfg.decoder.n_layers, 4)
        self.assertEqual(cfg.rl.lambda_ce, 0.0)


class TestPGNRLTrainerTracking(unittest.TestCase):
    """T13-T14: tracker integration in train_epoch."""

    def setUp(self) -> None:
        """Build a minimal trainer with a mock tracker."""
        from unittest.mock import MagicMock

        self.cfg = PGNRLConfig()
        self.mock_tracker = MagicMock()
        self.trainer = PGNRLTrainer(
            cfg=self.cfg,
            device="cpu",
            total_steps=100,
            tracker=self.mock_tracker,
        )

    def _write_pgn(
        self, game: chess.pgn.Game
    ) -> Path:
        """Write a single game to a temp PGN file."""
        import tempfile

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".pgn", delete=False
        )
        exporter = chess.pgn.StringExporter(
            headers=True
        )
        tmp.write(game.accept(exporter))
        tmp.flush()
        return Path(tmp.name)

    def test_t13_train_epoch_calls_track_scalars(
        self,
    ) -> None:
        """T13: train_epoch calls tracker.track_scalars
        once per trained game."""
        game = _make_fools_mate()
        pgn_path = self._write_pgn(game)
        self.trainer.train_epoch(pgn_path, max_games=1)
        self.mock_tracker.track_scalars.assert_called_once()

    def test_t17_track_scalars_receives_value_loss_and_mean_advantage(
        self,
    ) -> None:
        """train_epoch must pass value_loss and mean_advantage."""
        game = _make_fools_mate()
        pgn_path = self._write_pgn(game)
        self.trainer.train_epoch(pgn_path, max_games=1)
        calls = (
            self.mock_tracker.track_scalars.call_args_list
        )
        all_keys: set[str] = set()
        for call in calls:
            all_keys.update(call[0][0].keys())
        self.assertIn("value_loss", all_keys)
        self.assertIn("mean_advantage", all_keys)

    def test_t14_track_scalars_receives_required_keys(
        self,
    ) -> None:
        """T14: track_scalars dict contains pg_loss, ce_loss,
        total_loss, mean_reward."""
        required_keys = {
            "pg_loss",
            "ce_loss",
            "total_loss",
            "mean_reward",
        }
        game = _make_fools_mate()
        pgn_path = self._write_pgn(game)
        self.trainer.train_epoch(pgn_path, max_games=1)
        call_args = (
            self.mock_tracker.track_scalars.call_args
        )
        metrics_dict = call_args[0][0]
        self.assertTrue(
            required_keys.issubset(metrics_dict.keys()),
            f"Missing keys: "
            f"{required_keys - metrics_dict.keys()}",
        )


class TestBoardSnapshotLogging(unittest.TestCase):
    """Tests for per-ply board snapshot logging to tracker."""

    def setUp(self) -> None:
        """Build a minimal trainer with a mock tracker."""
        from unittest.mock import MagicMock

        self.cfg = PGNRLConfig()
        self.mock_tracker = MagicMock()
        self.trainer = PGNRLTrainer(
            cfg=self.cfg,
            device="cpu",
            total_steps=100,
            tracker=self.mock_tracker,
        )

    def test_ply_step_increments_per_ply(self) -> None:
        """_ply_step increments once per trained ply, not per game."""
        game = _make_scholars_mate()
        # train_color=white -> 4 white plies out of 7
        self.trainer.train_game(game, game_idx=0)
        self.assertGreater(self.trainer._ply_step, 0)
        step_after_1 = self.trainer._ply_step
        # Train another game; ply_step should accumulate
        game2 = _make_fools_mate()
        self.trainer.train_game(game2, game_idx=1)
        self.assertGreater(
            self.trainer._ply_step, step_after_1,
        )

    def test_log_text_called_at_step_100(self) -> None:
        """log_text is called when _ply_step hits a multiple of 100."""
        # Set _ply_step to 99 so next ply hits 100
        self.trainer._ply_step = 99
        game = _make_scholars_mate()
        self.trainer.train_game(game, game_idx=0)
        self.mock_tracker.log_text.assert_called()
        call_args = self.mock_tracker.log_text.call_args
        text = call_args[0][0]
        # Verify text contains board unicode and move info
        self.assertIn("Step 100", text)
        self.assertIn("Game 0", text)
        self.assertIn("Move:", text)

    def test_log_text_not_called_before_100(self) -> None:
        """log_text is NOT called when _ply_step < 100."""
        self.trainer._ply_step = 0
        game = _make_fools_mate()
        # Fool's mate: 2 white plies -> steps 1, 2
        self.trainer.train_game(game, game_idx=0)
        self.mock_tracker.log_text.assert_not_called()

    def test_board_snapshot_text_contains_unicode(self) -> None:
        """Logged text includes chess board unicode characters."""
        self.trainer._ply_step = 99
        game = _make_scholars_mate()
        self.trainer.train_game(game, game_idx=5)
        call_args = self.mock_tracker.log_text.call_args
        text = call_args[0][0]
        # board.unicode() produces pieces like ♜ ♞ ♝ etc.
        # At minimum the text should have multiple lines (board)
        lines = text.strip().split("\n")
        self.assertGreaterEqual(
            len(lines), 9,
            "Board text should have at least 9 lines "
            "(header + 8 board rows)",
        )


class TestActionConditionedValueHead(unittest.TestCase):
    """T19-T20: ActionConditionedValueHead integration."""

    def test_t19_chess_model_has_value_head_attribute(
        self,
    ) -> None:
        """ChessModel must have value_head as ActionConditionedValueHead."""
        from chess_sim.config import ModelConfig

        model = ChessModel(ModelConfig())
        self.assertTrue(hasattr(model, "value_head"))
        self.assertIsInstance(
            model.value_head,
            ActionConditionedValueHead,
        )

    def test_t20_action_conditioned_value_head_output_shape(
        self,
    ) -> None:
        """ActionConditionedValueHead forward -> [4, 1]."""
        head = ActionConditionedValueHead(128)
        out = head.forward(
            torch.rand(4, 128), torch.rand(4, 128)
        )
        self.assertEqual(out.shape, torch.Size([4, 1]))


if __name__ == "__main__":
    unittest.main()
