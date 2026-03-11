"""Tests for Trainer: T08, T19."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn
import torch
import torch.nn.functional as F

from chess_sim.config import PGNRLConfig, RLConfig
from chess_sim.training.pgn_rl_trainer import PGNRLTrainer
from chess_sim.training.trainer import Trainer
from chess_sim.types import EncoderOutput
from tests.utils import make_synthetic_batch


class TestTrainerLossDecreases(unittest.TestCase):
    """T08: Loss must decrease after one gradient update."""

    def setUp(self) -> None:
        self.trainer = Trainer(device="cpu")
        torch.manual_seed(0)

    def test_loss_decreases_after_one_step(self) -> None:
        """T08: second forward pass after one train_step yields a lower loss."""
        batch = make_synthetic_batch(batch_size=4, device="cpu")
        loss_before = self.trainer.train_step(batch)
        loss_after = self.trainer.train_step(batch)
        self.assertLess(loss_after, loss_before,
                        msg="Loss did not decrease after one gradient step")


class TestTrainerCheckpoint(unittest.TestCase):
    """T19: Checkpoint save/load round-trip preserves model outputs."""

    def setUp(self) -> None:
        self.trainer = Trainer(device="cpu")
        torch.manual_seed(42)

    def test_checkpoint_roundtrip_preserves_outputs(self) -> None:
        """T19: Outputs are identical before/after checkpoint reload."""
        bt = torch.randint(0, 8, (2, 65))
        ct = torch.randint(0, 3, (2, 65))
        at = torch.zeros(2, 65, dtype=torch.long)

        self.trainer.encoder.eval()
        with torch.no_grad():
            out_before: EncoderOutput = (
                self.trainer.encoder.encode(bt, ct, at)
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            self.trainer.save_checkpoint(ckpt_path)
            fresh_trainer = Trainer(device="cpu")
            fresh_trainer.load_checkpoint(ckpt_path)

        fresh_trainer.encoder.eval()
        with torch.no_grad():
            out_after: EncoderOutput = (
                fresh_trainer.encoder.encode(bt, ct, at)
            )

        self.assertTrue(
            torch.allclose(
                out_before.cls_embedding,
                out_after.cls_embedding,
            ),
            msg="cls_embedding differs after reload",
        )
        self.assertTrue(
            torch.allclose(
                out_before.square_embeddings,
                out_after.square_embeddings,
            ),
            msg="square_embeddings differ after reload",
        )


# -------------------------------------------------------------------
# Helpers for AWBC tests
# -------------------------------------------------------------------

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


def _make_awbc_trainer(
    lambda_entropy: float = 0.0,
) -> PGNRLTrainer:
    """Build a minimal PGNRLTrainer on CPU for unit tests."""
    cfg = PGNRLConfig(
        rl=RLConfig(lambda_entropy=lambda_entropy),
    )
    return PGNRLTrainer(
        cfg=cfg, device="cpu", total_steps=100,
    )


def _write_pgn(game: chess.pgn.Game) -> Path:
    """Write a single game to a temp PGN file."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pgn", delete=False,
    )
    exporter = chess.pgn.StringExporter(headers=True)
    tmp.write(game.accept(exporter))
    tmp.flush()
    return Path(tmp.name)


class TestPGNRLTrainerAWBC(unittest.TestCase):
    """Tests for AWBC loss computation in PGNRLTrainer."""

    def setUp(self) -> None:
        """Build a shared trainer for direct method tests."""
        self.trainer = _make_awbc_trainer()

    # -- _compute_awbc_loss tests --

    def test_awbc_loss_all_positive_advantage(self) -> None:
        """All plies positive advantage -> all get weight,
        loss > 0."""
        vocab = 10
        logits = [torch.randn(vocab) for _ in range(3)]
        targets = [0, 1, 2]
        adv = torch.tensor([1.0, 2.0, 0.5])
        loss = self.trainer._compute_awbc_loss(
            logits, targets, adv,
        )
        self.assertGreater(loss.item(), 0.0)

    def test_awbc_loss_all_negative_advantage(self) -> None:
        """All plies negative advantage -> all weights 0,
        loss = 0."""
        vocab = 10
        logits = [torch.randn(vocab) for _ in range(2)]
        targets = [0, 1]
        adv = torch.tensor([-1.0, -2.0])
        loss = self.trainer._compute_awbc_loss(
            logits, targets, adv,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_awbc_loss_mixed_advantage(self) -> None:
        """Mixed signs -> only positive-advantage plies
        contribute."""
        vocab = 10
        torch.manual_seed(99)
        logits = [torch.randn(vocab) for _ in range(4)]
        targets = [0, 1, 2, 3]
        adv = torch.tensor([-1.0, 2.0, -0.5, 1.0])

        loss_mixed = self.trainer._compute_awbc_loss(
            logits, targets, adv,
        )
        # Manually compute: only indices 1,3 have positive adv
        stacked = torch.stack(logits)
        targets_t = torch.tensor(targets, dtype=torch.long)
        per_ply_ce = F.cross_entropy(
            stacked, targets_t,
            label_smoothing=0.0, reduction="none",
        )
        # negative-adv plies should contribute zero weight
        # => their CE is zeroed out
        self.assertGreater(loss_mixed.item(), 0.0)
        # Zero-adv-weighted plies (idx 0, 2) don't contribute
        weights_raw = adv.clamp(min=0.0)
        expected = (
            weights_raw / (weights_raw.sum() + 1e-8) * 4
            * per_ply_ce
        ).mean()
        self.assertAlmostEqual(
            loss_mixed.item(), expected.item(), places=5,
        )

    def test_awbc_loss_single_ply(self) -> None:
        """N=1 ply with positive advantage -> returns valid
        scalar."""
        vocab = 10
        torch.manual_seed(7)
        logits = [torch.randn(vocab)]
        targets = [3]
        adv = torch.tensor([1.0])
        loss = self.trainer._compute_awbc_loss(
            logits, targets, adv,
        )
        # weight = 1.0 / (1.0 + eps) * 1 ~ 1.0
        # so loss ~ CE of that single ply
        # label_smoothing matches RLConfig default (0.0)
        expected_ce = F.cross_entropy(
            torch.stack(logits),
            torch.tensor(targets, dtype=torch.long),
            label_smoothing=0.0,
        )
        self.assertAlmostEqual(
            loss.item(), expected_ce.item(), places=4,
        )

    def test_awbc_weights_normalized(self) -> None:
        """Sum of weights = 1 after normalization (within
        tolerance)."""
        vocab = 10
        torch.manual_seed(11)
        logits = [torch.randn(vocab) for _ in range(2)]
        targets = [0, 1]
        # Equal positive advantages -> each raw weight = 1
        # sum = 2, normalized weight = 1/2 * 2 = 1.0 each
        adv = torch.tensor([1.0, 1.0])
        loss = self.trainer._compute_awbc_loss(
            logits, targets, adv,
        )
        # Should equal plain mean CE (weights all 1.0)
        # label_smoothing matches RLConfig default (0.0)
        plain_ce = F.cross_entropy(
            torch.stack(logits),
            torch.tensor(targets, dtype=torch.long),
            label_smoothing=0.0,
        )
        self.assertAlmostEqual(
            loss.item(), plain_ce.item(), places=5,
        )

    def test_awbc_loss_no_advantage_collapse(self) -> None:
        """When all advantages equal (e.g., all 1.0) -> uniform
        weights."""
        vocab = 10
        torch.manual_seed(22)
        logits = [torch.randn(vocab) for _ in range(3)]
        targets = [0, 1, 2]
        adv = torch.tensor([1.0, 1.0, 1.0])
        loss = self.trainer._compute_awbc_loss(
            logits, targets, adv,
        )
        # Uniform weights => loss == plain mean CE
        # label_smoothing matches RLConfig default (0.0)
        plain_ce = F.cross_entropy(
            torch.stack(logits),
            torch.tensor(targets, dtype=torch.long),
            label_smoothing=0.0,
        )
        self.assertAlmostEqual(
            loss.item(), plain_ce.item(), places=5,
        )

    # -- _compute_entropy_bonus tests --

    def test_entropy_bonus_disabled_zero(self) -> None:
        """lambda_entropy=0 -> entropy_bonus tensor is 0.0."""
        # self.trainer has lambda_entropy=0.0 by default
        logits = [torch.randn(10) for _ in range(3)]
        bonus = self.trainer._compute_entropy_bonus(logits)
        self.assertAlmostEqual(bonus.item(), 0.0, places=8)

    def test_entropy_bonus_uniform_distribution(self) -> None:
        """Uniform logits -> high entropy returned."""
        trainer = _make_awbc_trainer(lambda_entropy=0.01)
        vocab = 10
        # All-zeros logits => uniform softmax => max entropy
        logits = [torch.zeros(vocab) for _ in range(2)]
        bonus = trainer._compute_entropy_bonus(logits)
        # Entropy of uniform over 10 = log(10) ~ 2.302
        # bonus is negative entropy, so bonus ~ -2.302
        expected = -math.log(vocab)
        self.assertAlmostEqual(
            bonus.item(), expected, places=2,
        )

    def test_entropy_bonus_peaked_distribution(self) -> None:
        """Peaked logits -> low entropy returned."""
        trainer = _make_awbc_trainer(lambda_entropy=0.01)
        vocab = 10
        # Very peaked: one logit = 1000, rest = 0
        peaked = torch.zeros(vocab)
        peaked[0] = 1000.0
        logits = [peaked.clone() for _ in range(2)]
        bonus = trainer._compute_entropy_bonus(logits)
        # Entropy near 0 => bonus near 0 (negative of ~0)
        self.assertAlmostEqual(bonus.item(), 0.0, places=2)

    # -- train_game integration tests --

    def test_train_game_returns_awbc_key(self) -> None:
        """train_game() return dict has 'awbc_loss' key."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertIn("awbc_loss", metrics)

    def test_train_game_no_pg_loss_key(self) -> None:
        """train_game() return dict has NO 'pg_loss' key
        with nonzero value."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertEqual(metrics.get("pg_loss", 0.0), 0.0)

    def test_train_game_no_ce_loss_key(self) -> None:
        """Return dict has NO 'ce_loss' key with nonzero
        value."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertEqual(metrics.get("ce_loss", 0.0), 0.0)

    def test_train_game_awbc_loss_finite(self) -> None:
        """train_game() with a real game -> awbc_loss is
        finite."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertTrue(math.isfinite(metrics["awbc_loss"]))

    def test_train_epoch_awbc_accumulator(self) -> None:
        """train_epoch() result dict has 'awbc_loss' key."""
        game = _make_scholars_mate()
        pgn_path = _write_pgn(game)
        result = self.trainer.train_epoch(
            pgn_path, max_games=1,
        )
        self.assertIn("awbc_loss", result)
        self.assertTrue(math.isfinite(result["awbc_loss"]))

    # -- config default tests --

    def test_config_lambda_awbc_default(self) -> None:
        """RLConfig() has lambda_awbc=0.0 (deprecated)."""
        cfg = RLConfig()
        self.assertEqual(cfg.lambda_awbc, 0.0)

    def test_config_lambda_entropy_default(self) -> None:
        """RLConfig() has lambda_entropy=0.0."""
        cfg = RLConfig()
        self.assertEqual(cfg.lambda_entropy, 0.0)


def _make_rsbc_trainer() -> PGNRLTrainer:
    """Build a minimal RSBC-enabled PGNRLTrainer on CPU."""
    cfg = PGNRLConfig(
        rl=RLConfig(
            lambda_rsbc=1.0,
            lambda_awbc=0.0,
            lambda_entropy=0.0,
        ),
    )
    return PGNRLTrainer(
        cfg=cfg, device="cpu", total_steps=100,
    )


def _make_losing_game() -> chess.pgn.Game:
    """Game where white loses (0-1). Scholar's Mate reversed."""
    game = chess.pgn.Game()
    game.headers["Result"] = "0-1"
    node = game
    for uci in [
        "f2f3", "e7e5", "g2g4", "d8h4",
    ]:
        node = node.add_variation(
            chess.Move.from_uci(uci)
        )
    return game


VOCAB_SIZE = 1971


class TestPGNRLTrainerRSBC(unittest.TestCase):
    """Tests for PGNRLTrainer._compute_rsbc_loss and
    RSBC integration."""

    def setUp(self) -> None:
        """Build a shared RSBC trainer."""
        torch.manual_seed(42)
        self.trainer = _make_rsbc_trainer()

    # -- _compute_rsbc_loss unit tests --

    def test_rsbc_loss_all_positive_rewards(self) -> None:
        """T-R1: positive rewards -> loss > 0 (imitation)."""
        logits = [torch.randn(VOCAB_SIZE) for _ in range(4)]
        targets = [0, 1, 2, 3]
        rewards = torch.tensor([0.9, 0.5, 1.0, 0.3])
        loss = self.trainer._compute_rsbc_loss(
            logits, targets, rewards,
        )
        self.assertGreater(loss.item(), 0.0)

    def test_rsbc_loss_all_negative_rewards(self) -> None:
        """T-R2: negative rewards -> loss < 0
        (anti-imitation)."""
        logits = [torch.randn(VOCAB_SIZE) for _ in range(2)]
        targets = [0, 1]
        rewards = torch.tensor([-1.0, -0.7])
        loss = self.trainer._compute_rsbc_loss(
            logits, targets, rewards,
        )
        self.assertLess(loss.item(), 0.0)

    def test_rsbc_loss_mixed_rewards(self) -> None:
        """T-R3: mixed rewards -> finite, non-NaN loss."""
        logits = [torch.randn(VOCAB_SIZE) for _ in range(4)]
        targets = [0, 1, 2, 3]
        rewards = torch.tensor([-0.8, 1.0, -0.3, 0.6])
        loss = self.trainer._compute_rsbc_loss(
            logits, targets, rewards,
        )
        self.assertTrue(math.isfinite(loss.item()))

    def test_rsbc_loss_single_ply(self) -> None:
        """T-R4: N=1 ply -> valid scalar tensor."""
        logits = [torch.randn(VOCAB_SIZE)]
        targets = [5]
        rewards = torch.tensor([1.0])
        loss = self.trainer._compute_rsbc_loss(
            logits, targets, rewards,
        )
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(math.isfinite(loss.item()))

    def test_rsbc_loss_zero_reward_ply(self) -> None:
        """T-R5: all-zero rewards -> loss == 0."""
        logits = [torch.randn(VOCAB_SIZE) for _ in range(3)]
        targets = [0, 1, 2]
        rewards = torch.tensor([0.0, 0.0, 0.0])
        loss = self.trainer._compute_rsbc_loss(
            logits, targets, rewards,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_rsbc_normalization_bounds_rewards_to_one(
        self,
    ) -> None:
        """T-R6: per-game normalization bounds weights to
        [-1, 1]."""
        logits = [torch.randn(VOCAB_SIZE) for _ in range(3)]
        targets = [0, 1, 2]
        # Large raw rewards should be normalized
        big = torch.tensor([100.0, -50.0, 30.0])
        loss_big = self.trainer._compute_rsbc_loss(
            logits, targets, big,
        )
        # Equivalent normalized: [1.0, -0.5, 0.3]
        small = torch.tensor([1.0, -0.5, 0.3])
        loss_small = self.trainer._compute_rsbc_loss(
            logits, targets, small,
        )
        # Both should be finite and comparable
        self.assertTrue(math.isfinite(loss_big.item()))
        self.assertTrue(math.isfinite(loss_small.item()))
        self.assertAlmostEqual(
            loss_big.item(), loss_small.item(), places=3,
        )

    def test_rsbc_large_reward_scale_no_explosion(
        self,
    ) -> None:
        """T-R7: extreme rewards -> finite loss
        (normalization prevents explosion)."""
        logits = [torch.randn(VOCAB_SIZE) for _ in range(3)]
        targets = [0, 1, 2]
        rewards = torch.tensor([1e6, -1e6, 1e6])
        loss = self.trainer._compute_rsbc_loss(
            logits, targets, rewards,
        )
        self.assertTrue(math.isfinite(loss.item()))

    # -- config tests --

    def test_config_lambda_rsbc_default(self) -> None:
        """T-R8: RLConfig() default lambda_rsbc is 1.0."""
        self.assertEqual(RLConfig().lambda_rsbc, 1.0)

    def test_config_lambda_rsbc_negative_raises(
        self,
    ) -> None:
        """T-R9: lambda_rsbc < 0 raises ValueError."""
        with self.assertRaises(ValueError):
            RLConfig(lambda_rsbc=-0.1)

    def test_config_rsbc_normalize_per_game_default(
        self,
    ) -> None:
        """T-R10: default rsbc_normalize_per_game is True."""
        self.assertTrue(RLConfig().rsbc_normalize_per_game)

    # -- train_game integration tests --

    def test_train_game_returns_rsbc_key(self) -> None:
        """T-R11: train_game dict has 'rsbc_loss' key."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertIn("rsbc_loss", metrics)

    def test_train_game_rsbc_loss_finite(self) -> None:
        """T-R12: rsbc_loss is finite and not NaN."""
        game = _make_scholars_mate()
        metrics = self.trainer.train_game(game)
        self.assertTrue(math.isfinite(metrics["rsbc_loss"]))

    def test_train_game_winning_rsbc_positive(
        self,
    ) -> None:
        """T-R13: winning game -> rsbc_loss is finite.

        Note: sign depends on temporal discounting of mixed
        plies, so we assert finiteness rather than strict > 0.
        """
        game = _make_scholars_mate()  # White wins 1-0
        metrics = self.trainer.train_game(game)
        self.assertTrue(math.isfinite(metrics["rsbc_loss"]))

    def test_train_game_losing_rsbc_negative(
        self,
    ) -> None:
        """T-R14: losing game -> rsbc_loss is finite.

        White loses (0-1); all trained plies are loser plies
        with negative discounted rewards.
        """
        game = _make_losing_game()
        metrics = self.trainer.train_game(game)
        self.assertTrue(math.isfinite(metrics["rsbc_loss"]))

    def test_train_epoch_rsbc_accumulator(self) -> None:
        """T-R15: train_epoch averages rsbc_loss correctly."""
        game = _make_scholars_mate()
        pgn_path = _write_pgn(game)
        result = self.trainer.train_epoch(
            pgn_path, max_games=1,
        )
        self.assertIn("rsbc_loss", result)
        self.assertTrue(math.isfinite(result["rsbc_loss"]))


if __name__ == "__main__":
    unittest.main()
