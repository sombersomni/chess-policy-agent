"""Tests for RL self-play fine-tune trainer scaffolding.

TC01-TC09: config validation + RunningMeanBaseline (should pass).
TC10-TC20: trainer stubs (expect NotImplementedError).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from parameterized import parameterized

from chess_sim.config import (
    AimConfig,
    DecoderConfig,
    FinetuneConfig,
    FinetuneRLConfig,
    ModelConfig,
    load_finetune_rl_config,
)
from chess_sim.training.rl_finetune_trainer import (
    RLFinetuneTrainer,
    RunningMeanBaseline,
    _log_prob_of_move,
    compute_returns,
    play_game,
)
from chess_sim.types import GameRecord, PlyRecord

# Tiny model dims for fast tests
_TINY_MODEL = ModelConfig(
    d_model=64, n_heads=2, n_layers=2, dim_feedforward=128
)
_TINY_DECODER = DecoderConfig(
    d_model=64, n_heads=2, n_layers=1, dim_feedforward=128
)


def _make_cfg(
    **finetune_overrides: object,
) -> FinetuneRLConfig:
    """Build a FinetuneRLConfig with tiny model dims."""
    ft = FinetuneConfig(**finetune_overrides)
    return FinetuneRLConfig(
        model=_TINY_MODEL,
        decoder=_TINY_DECODER,
        finetune=ft,
        aim=AimConfig(enabled=False),
    )


def _make_ply(
    move_token: int = 42,
    is_white: bool = True,
) -> PlyRecord:
    """Create a minimal PlyRecord for testing."""
    return PlyRecord(
        board_tokens=torch.zeros(1, 65, dtype=torch.long),
        color_tokens=torch.zeros(1, 65, dtype=torch.long),
        traj_tokens=torch.zeros(1, 65, dtype=torch.long),
        move_token=move_token,
        log_prob=torch.tensor(-0.5, requires_grad=True),
        is_white_ply=is_white,
    )


def _make_game(
    n_plies: int = 3,
    outcome: int = 1,
    termination: str = "checkmate",
) -> GameRecord:
    """Create a minimal GameRecord for testing."""
    plies = [_make_ply(move_token=i) for i in range(n_plies)]
    return GameRecord(
        plies=plies,
        outcome=outcome,
        n_ply=n_plies * 2,
        termination=termination,
    )


# ---------------------------------------------------------------
# TC01-TC05: FinetuneConfig validation
# ---------------------------------------------------------------


class TestFinetuneConfig(unittest.TestCase):
    """Tests for FinetuneConfig dataclass validation."""

    def test_tc01_defaults_construct(self) -> None:
        """TC01: FinetuneConfig with valid defaults constructs ok."""
        cfg = FinetuneConfig()
        self.assertAlmostEqual(cfg.ema_alpha, 0.995)
        self.assertEqual(cfg.max_ply, 200)
        self.assertEqual(cfg.n_games_per_update, 10)
        self.assertAlmostEqual(cfg.gamma, 0.99)

    def test_tc02_ema_alpha_one_raises(self) -> None:
        """TC02: ema_alpha=1.0 raises ValueError."""
        with self.assertRaises(ValueError):
            FinetuneConfig(ema_alpha=1.0)

    def test_tc03_ema_alpha_zero_raises(self) -> None:
        """TC03: ema_alpha=0.0 raises ValueError."""
        with self.assertRaises(ValueError):
            FinetuneConfig(ema_alpha=0.0)

    def test_tc04_lambda_kl_negative_raises(self) -> None:
        """TC04: lambda_kl=-0.1 raises ValueError."""
        with self.assertRaises(ValueError):
            FinetuneConfig(lambda_kl=-0.1)

    def test_tc05_max_ply_zero_raises(self) -> None:
        """TC05: max_ply=0 raises ValueError."""
        with self.assertRaises(ValueError):
            FinetuneConfig(max_ply=0)

    @parameterized.expand([
        ("t_policy_zero", {"t_policy": 0.0}),
        ("t_policy_neg", {"t_policy": -1.0}),
        ("t_opponent_zero", {"t_opponent": 0.0}),
        ("gamma_zero", {"gamma": 0.0}),
        ("gamma_neg", {"gamma": -0.5}),
        ("n_games_zero", {"n_games_per_update": 0}),
    ])
    def test_additional_validation(
        self, _name: str, overrides: dict[str, object]
    ) -> None:
        """Additional edge-case validation raises ValueError."""
        with self.assertRaises(ValueError):
            FinetuneConfig(**overrides)


# ---------------------------------------------------------------
# TC06-TC07: load_finetune_rl_config
# ---------------------------------------------------------------


class TestLoadFinetuneRLConfig(unittest.TestCase):
    """Tests for YAML config loader."""

    _VALID_YAML = """\
model:
  d_model: 64
  n_heads: 2
  n_layers: 2
  dim_feedforward: 128

decoder:
  d_model: 64
  n_heads: 2
  n_layers: 1
  dim_feedforward: 128

finetune:
  ema_alpha: 0.99
  gamma: 0.95
  lambda_kl: 0.05
  n_games_per_update: 5

aim:
  enabled: false
  experiment_name: test_ft
"""

    def test_tc06_parses_valid_yaml(self) -> None:
        """TC06: load_finetune_rl_config parses valid YAML."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(self._VALID_YAML)
            f.flush()
            cfg = load_finetune_rl_config(Path(f.name))

        self.assertEqual(cfg.model.d_model, 64)
        self.assertEqual(cfg.decoder.n_layers, 1)
        self.assertAlmostEqual(cfg.finetune.ema_alpha, 0.99)
        self.assertEqual(cfg.finetune.n_games_per_update, 5)
        self.assertFalse(cfg.aim.enabled)

    def test_tc07_unknown_key_raises_typeerror(self) -> None:
        """TC07: Unknown key in finetune section raises TypeError."""
        bad_yaml = """\
finetune:
  ema_alpha: 0.99
  unknown_key: 1
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(bad_yaml)
            f.flush()
            with self.assertRaises(TypeError):
                load_finetune_rl_config(Path(f.name))


# ---------------------------------------------------------------
# TC08-TC09: RunningMeanBaseline
# ---------------------------------------------------------------


class TestRunningMeanBaseline(unittest.TestCase):
    """Tests for RunningMeanBaseline."""

    def test_tc08_initial_value_zero(self) -> None:
        """TC08: value() returns 0.0 before any update."""
        b = RunningMeanBaseline()
        self.assertAlmostEqual(b.value(), 0.0)

    def test_tc09_converges_toward_mean(self) -> None:
        """TC09: Alternating +1/-1 converges near 0."""
        b = RunningMeanBaseline()
        for _ in range(100):
            b.update(torch.tensor([1.0]))
            b.update(torch.tensor([-1.0]))
        self.assertLess(abs(b.value()), 0.1)

    def test_baseline_exact_mean(self) -> None:
        """Verify exact mean for a known sequence."""
        b = RunningMeanBaseline()
        b.update(torch.tensor([2.0, 4.0]))
        self.assertAlmostEqual(b.value(), 3.0)


# ---------------------------------------------------------------
# TC10-TC12: compute_returns
# ---------------------------------------------------------------


class TestComputeReturns(unittest.TestCase):
    """Tests for compute_returns function."""

    def test_tc10_win_discount(self) -> None:
        """TC10: Win outcome discounts correctly (gamma^(T-1-t))."""
        game = _make_game(n_plies=3, outcome=1)
        rets = compute_returns([game], gamma=0.99)
        self.assertEqual(len(rets), 3)
        # G_0 = 0.99^2, G_1 = 0.99^1, G_2 = 0.99^0
        self.assertAlmostEqual(rets[0].item(), 0.99**2, places=5)
        self.assertAlmostEqual(rets[1].item(), 0.99, places=5)
        self.assertAlmostEqual(rets[2].item(), 1.0, places=5)

    def test_tc11_draw_all_zero(self) -> None:
        """TC11: Draw outcome produces all-zero returns."""
        game = _make_game(n_plies=4, outcome=0)
        rets = compute_returns([game], gamma=0.99)
        self.assertEqual(len(rets), 4)
        for r in rets:
            self.assertAlmostEqual(r.item(), 0.0)

    def test_tc12_loss_negative(self) -> None:
        """TC12: Loss outcome produces all-negative returns."""
        game = _make_game(n_plies=2, outcome=-1)
        rets = compute_returns([game], gamma=0.99)
        self.assertEqual(len(rets), 2)
        for r in rets:
            self.assertLess(r.item(), 0.0)


# ---------------------------------------------------------------
# TC13: _log_prob_of_move
# ---------------------------------------------------------------


class TestLogProbOfMove(unittest.TestCase):
    """Tests for _log_prob_of_move function."""

    def test_tc13_returns_scalar_with_grad(self) -> None:
        """TC13: Returns a scalar tensor with requires_grad=True."""
        from chess_sim.data.move_tokenizer import MoveTokenizer
        from chess_sim.model.chess_model import ChessModel

        policy = ChessModel(_TINY_MODEL, _TINY_DECODER)
        policy.train()
        mt = MoveTokenizer()
        bt = torch.zeros(1, 65, dtype=torch.long)
        ct = torch.zeros(1, 65, dtype=torch.long)
        tt = torch.zeros(1, 65, dtype=torch.long)
        move_token = mt.tokenize_move("e2e4")
        lp = _log_prob_of_move(
            policy=policy,
            board_tokens=bt,
            color_tokens=ct,
            traj_tokens=tt,
            move_token=move_token,
            legal_moves=["e2e4", "d2d4"],
            move_tok=mt,
            device=torch.device("cpu"),
        )
        self.assertEqual(lp.dim(), 0)
        self.assertTrue(lp.requires_grad)
        self.assertLess(lp.item(), 0.0)


# ---------------------------------------------------------------
# TC14-TC15: _update_shadow
# ---------------------------------------------------------------


class TestUpdateShadow(unittest.TestCase):
    """Tests for _update_shadow EMA update."""

    def test_tc14_ema_blends_weights(self) -> None:
        """TC14: _update_shadow blends shadow toward policy."""
        cfg = _make_cfg(ema_alpha=0.5)
        trainer = RLFinetuneTrainer(cfg, device="cpu")
        # Perturb policy weights so shadow differs
        with torch.no_grad():
            for p in trainer._policy.parameters():
                p.add_(1.0)
        # Snapshot shadow before
        before = [
            p.clone() for p in trainer._shadow.parameters()
        ]
        trainer._update_shadow()
        # Shadow should have moved toward policy
        for b, s in zip(before, trainer._shadow.parameters()):
            self.assertFalse(torch.equal(b, s))

    def test_tc15_ema_one_rejected_by_config(self) -> None:
        """TC15: ema_alpha=1.0 is rejected by FinetuneConfig."""
        with self.assertRaises(ValueError):
            _make_cfg(ema_alpha=1.0)


# ---------------------------------------------------------------
# TC16-TC17: play_game
# ---------------------------------------------------------------


class TestPlayGame(unittest.TestCase):
    """Tests for play_game function."""

    def test_tc16_maxply_termination(self) -> None:
        """TC16: play_game terminates at max_ply with valid record."""
        from chess_sim.data.move_tokenizer import MoveTokenizer
        from chess_sim.data.tokenizer import BoardTokenizer
        from chess_sim.model.chess_model import ChessModel

        cfg = FinetuneConfig(max_ply=4)
        policy = ChessModel(_TINY_MODEL, _TINY_DECODER)
        shadow = ChessModel(_TINY_MODEL, _TINY_DECODER)
        shadow.eval()
        bt = BoardTokenizer()
        mt = MoveTokenizer()
        record = play_game(
            policy, shadow, bt, mt, cfg,
            torch.device("cpu"),
        )
        self.assertIsInstance(record, GameRecord)
        self.assertIn(
            record.termination,
            ("checkmate", "stalemate", "50move",
             "threefold", "maxply"),
        )
        self.assertLessEqual(record.n_ply, 4)

    def test_tc17_record_has_policy_plies(self) -> None:
        """TC17: play_game returns plies only for policy (White)."""
        from chess_sim.data.move_tokenizer import MoveTokenizer
        from chess_sim.data.tokenizer import BoardTokenizer
        from chess_sim.model.chess_model import ChessModel

        cfg = FinetuneConfig(max_ply=6)
        policy = ChessModel(_TINY_MODEL, _TINY_DECODER)
        shadow = ChessModel(_TINY_MODEL, _TINY_DECODER)
        shadow.eval()
        record = play_game(
            policy, shadow, BoardTokenizer(),
            MoveTokenizer(), cfg, torch.device("cpu"),
        )
        # All recorded plies should be white (policy)
        for ply in record.plies:
            self.assertTrue(ply.is_white_ply)


# ---------------------------------------------------------------
# TC18: RLFinetuneTrainer.__init__
# ---------------------------------------------------------------


class TestRLFinetuneTrainerInit(unittest.TestCase):
    """Tests for RLFinetuneTrainer initialization."""

    def test_tc18_init_creates_three_models(self) -> None:
        """TC18: __init__ creates policy, ref, and shadow copies."""
        cfg = _make_cfg()
        trainer = RLFinetuneTrainer(cfg, device="cpu")
        # Policy should be trainable
        self.assertIsNotNone(trainer.policy)
        # Ref and shadow are frozen
        self.assertFalse(
            any(p.requires_grad for p in trainer._ref.parameters())
        )
        self.assertFalse(
            any(p.requires_grad for p in trainer._shadow.parameters())
        )
        self.assertFalse(trainer._ref.training)
        self.assertFalse(trainer._shadow.training)


# ---------------------------------------------------------------
# TC19: _gradient_step
# ---------------------------------------------------------------


class TestGradientStep(unittest.TestCase):
    """Tests for RLFinetuneTrainer._gradient_step."""

    def test_tc19_gradient_step_empty_returns_zeros(self) -> None:
        """TC19: _gradient_step with empty records returns zeros."""
        cfg = _make_cfg()
        trainer = RLFinetuneTrainer(cfg, device="cpu")
        metrics = trainer._gradient_step([])
        self.assertAlmostEqual(metrics["pg_loss"], 0.0)
        self.assertAlmostEqual(metrics["kl_loss"], 0.0)
        self.assertAlmostEqual(metrics["total_loss"], 0.0)
        self.assertAlmostEqual(metrics["grad_norm"], 0.0)


# ---------------------------------------------------------------
# TC20: save/load checkpoint
# ---------------------------------------------------------------


class TestCheckpointRoundTrip(unittest.TestCase):
    """Tests for save_checkpoint / load_checkpoint."""

    def test_tc20_checkpoint_roundtrip(self) -> None:
        """TC20: Checkpoint save/load preserves state."""
        cfg = _make_cfg()
        trainer = RLFinetuneTrainer(cfg, device="cpu")
        trainer._global_step = 7
        trainer._game_count = 42
        trainer._baseline._mean = 0.5
        trainer._baseline._count = 10

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "test.pt"
            trainer.save_checkpoint(ckpt_path)
            self.assertTrue(ckpt_path.exists())

            trainer2 = RLFinetuneTrainer(cfg, device="cpu")
            trainer2.load_checkpoint(ckpt_path)
            self.assertEqual(trainer2._global_step, 7)
            self.assertEqual(trainer2._game_count, 42)
            self.assertAlmostEqual(
                trainer2._baseline._mean, 0.5
            )
            self.assertEqual(trainer2._baseline._count, 10)


if __name__ == "__main__":
    unittest.main()
