"""Tests for PGNRLTrainerV3 (RSCE: Reward-Scaled Cross-Entropy).

T1-T7:  _build_multipliers behavior (ordering, positivity, r_ref)
T8-T12: _compute_rsce_loss behavior (finite, mask, reward scaling)
T13-T15: train_game return dict shape and edge cases
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn
import torch

from chess_sim.config import PGNRLConfig
from chess_sim.training.pgn_rl_trainer_v3 import (
    PGNRLTrainerV3,
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


# ---------------------------------------------------------
# T1-T7: _build_multipliers
# ---------------------------------------------------------
class TestBuildMultipliers(unittest.TestCase):
    """Tests for PGNRLTrainerV3._build_multipliers."""

    def setUp(self) -> None:
        self.cfg = PGNRLConfig()
        self.trainer = PGNRLTrainerV3(
            cfg=self.cfg,
            device="cpu",
            total_steps=100,
        )

    def test_t1_monotone_decreasing_with_reward(
        self,
    ) -> None:
        """T1: R=[+1,0,-1] -> m[0] < m[1] < m[2]."""
        r = torch.tensor([1.0, 0.0, -1.0])
        m = self.trainer._build_multipliers(r)
        self.assertLess(m[0].item(), m[1].item())
        self.assertLess(m[1].item(), m[2].item())

    def test_t2_all_positive_for_finite_rewards(
        self,
    ) -> None:
        """T2: All m(t) > 0 for arbitrary finite rewards."""
        r = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        m = self.trainer._build_multipliers(r)
        self.assertTrue((m > 0).all())

    def test_t3_rsce_r_ref_shifts_multipliers(
        self,
    ) -> None:
        """T3: rsce_r_ref=1.0 shifts neutral to R=1.0."""
        cfg = PGNRLConfig()
        cfg.rl.rsce_r_ref = 1.0
        cfg.rl.rsbc_normalize_per_game = False
        trainer = PGNRLTrainerV3(
            cfg=cfg, device="cpu", total_steps=100
        )
        r = torch.tensor([1.0])
        m = trainer._build_multipliers(r)
        # exp(-(1.0 - 1.0)) = exp(0) = 1.0
        self.assertAlmostEqual(m[0].item(), 1.0, places=5)

    def test_t4_normalization_on_sum_equals_n(
        self,
    ) -> None:
        """T4: With normalization, sum(m) approx N."""
        r = torch.tensor([1.5, 0.3, -0.7, -1.2, 0.8])
        m = self.trainer._build_multipliers(r)
        n = r.size(0)
        self.assertAlmostEqual(
            m.sum().item(), float(n), places=4
        )

    def test_t5_normalization_off_raw_exp_values(
        self,
    ) -> None:
        """T5: Without normalization, raw exp(-R) returned."""
        cfg = PGNRLConfig()
        cfg.rl.rsbc_normalize_per_game = False
        trainer = PGNRLTrainerV3(
            cfg=cfg, device="cpu", total_steps=100
        )
        r = torch.tensor([1.0, 0.0, -1.0])
        m = trainer._build_multipliers(r)
        for i in range(3):
            expected = math.exp(-r[i].item())
            self.assertAlmostEqual(
                m[i].item(), expected, places=5
            )

    def test_t6_single_ply_no_divide_by_zero(
        self,
    ) -> None:
        """T6: Single-ply game normalizes without error."""
        r = torch.tensor([-1.0])
        m = self.trainer._build_multipliers(r)
        # With normalization: m*1/sum(m) = 1.0
        self.assertAlmostEqual(m[0].item(), 1.0, places=5)

    def test_t7_equal_rewards_equal_multipliers(
        self,
    ) -> None:
        """T7: All equal rewards -> all equal multipliers."""
        r = torch.tensor([0.5, 0.5, 0.5, 0.5])
        m = self.trainer._build_multipliers(r)
        for i in range(1, 4):
            self.assertAlmostEqual(
                m[i].item(), m[0].item(), places=5
            )


# ---------------------------------------------------------
# T8-T12: _compute_rsce_loss
# ---------------------------------------------------------
class TestComputeRSCELoss(unittest.TestCase):
    """Tests for PGNRLTrainerV3._compute_rsce_loss."""

    def setUp(self) -> None:
        self.cfg = PGNRLConfig()
        self.trainer = PGNRLTrainerV3(
            cfg=self.cfg,
            device="cpu",
            total_steps=100,
        )
        # Vocab size from the model's decoder output projection
        self.vocab_size = (
            self.trainer.model.decoder.output_proj.out_features
        )

    def test_t8_loss_is_finite_scalar(self) -> None:
        """T8: Loss is a finite scalar for valid inputs."""
        n = 3
        logits = [
            torch.randn(self.vocab_size) for _ in range(n)
        ]
        targets = [10, 20, 30]
        rewards = torch.tensor([0.5, 0.0, -0.5])
        loss = self.trainer._compute_rsce_loss(
            logits, targets, rewards
        )
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_t9_structural_mask_finite_loss(
        self,
    ) -> None:
        """T9: Structural mask with legal target -> finite."""
        cfg = PGNRLConfig()
        cfg.rl.use_structural_mask = True
        trainer = PGNRLTrainerV3(
            cfg=cfg, device="cpu", total_steps=100
        )
        vs = self.vocab_size
        logits = [torch.randn(vs)]
        # Use a small target index that won't be masked
        targets = [5]
        rewards = torch.tensor([0.0])
        # color_tokens: 1=player on square 0, rest 0
        ct = torch.zeros(65, dtype=torch.long)
        ct[1] = 1  # player piece on square 0
        loss = trainer._compute_rsce_loss(
            logits, targets, rewards, [ct]
        )
        self.assertTrue(torch.isfinite(loss))

    def test_t10_higher_reward_lower_contribution(
        self,
    ) -> None:
        """T10: Higher reward -> lower loss contribution."""
        cfg = PGNRLConfig()
        cfg.rl.rsbc_normalize_per_game = False
        trainer = PGNRLTrainerV3(
            cfg=cfg, device="cpu", total_steps=100
        )
        vs = self.vocab_size
        # Same logits and target, different rewards
        logit = torch.randn(vs)
        target = 10
        high_r = torch.tensor([2.0])
        low_r = torch.tensor([-2.0])
        loss_high = trainer._compute_rsce_loss(
            [logit.clone()], [target], high_r
        )
        loss_low = trainer._compute_rsce_loss(
            [logit.clone()], [target], low_r
        )
        # Higher reward -> smaller multiplier -> lower loss
        self.assertLess(
            loss_high.item(), loss_low.item()
        )

    def test_t11_lower_reward_higher_contribution(
        self,
    ) -> None:
        """T11: Lower reward -> higher loss contribution."""
        cfg = PGNRLConfig()
        cfg.rl.rsbc_normalize_per_game = False
        trainer = PGNRLTrainerV3(
            cfg=cfg, device="cpu", total_steps=100
        )
        vs = self.vocab_size
        logit = torch.randn(vs)
        target = 15
        neg_r = torch.tensor([-1.0])
        zero_r = torch.tensor([0.0])
        loss_neg = trainer._compute_rsce_loss(
            [logit.clone()], [target], neg_r
        )
        loss_zero = trainer._compute_rsce_loss(
            [logit.clone()], [target], zero_r
        )
        self.assertGreater(
            loss_neg.item(), loss_zero.item()
        )

    def test_t12_zero_rewards_approx_plain_ce(
        self,
    ) -> None:
        """T12: All-zero rewards -> loss approx plain mean CE."""
        cfg = PGNRLConfig()
        cfg.rl.rsbc_normalize_per_game = False
        trainer = PGNRLTrainerV3(
            cfg=cfg, device="cpu", total_steps=100
        )
        vs = self.vocab_size
        n = 4
        logits = [torch.randn(vs) for _ in range(n)]
        targets = [5, 10, 15, 20]
        rewards = torch.zeros(n)
        loss = trainer._compute_rsce_loss(
            logits, targets, rewards
        )
        # exp(-0) = 1.0, so RSCE = plain mean CE
        import torch.nn.functional as Ftest

        stacked = torch.stack(logits)
        targets_t = torch.tensor(targets, dtype=torch.long)
        plain_ce = Ftest.cross_entropy(
            stacked, targets_t, reduction="mean"
        )
        self.assertAlmostEqual(
            loss.item(), plain_ce.item(), places=4
        )


# ---------------------------------------------------------
# T13-T15: train_game integration
# ---------------------------------------------------------
class TestTrainGameV3(unittest.TestCase):
    """Tests for PGNRLTrainerV3.train_game return dict."""

    def setUp(self) -> None:
        self.cfg = PGNRLConfig()
        self.trainer = PGNRLTrainerV3(
            cfg=self.cfg,
            device="cpu",
            total_steps=100,
        )

    def test_t13_return_dict_has_rsce_loss_key(
        self,
    ) -> None:
        """T13: Return dict has 'rsce_loss', NOT 'rsbc_loss'."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertIn("rsce_loss", metrics)
        self.assertNotIn("rsbc_loss", metrics)

    def test_t14_total_equals_lambda_rsbc_times_rsce(
        self,
    ) -> None:
        """T14: total_loss == lambda_rsbc * rsce_loss."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        expected = (
            self.cfg.rl.lambda_rsbc * metrics["rsce_loss"]
        )
        self.assertAlmostEqual(
            metrics["total_loss"], expected, places=4
        )

    def test_t15_empty_game_returns_empty_dict(
        self,
    ) -> None:
        """T15: Empty/invalid game returns {}."""
        game = chess.pgn.Game()
        game.headers["Result"] = "*"
        metrics = self.trainer.train_game(game)
        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
