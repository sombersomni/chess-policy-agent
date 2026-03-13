"""Tests for RL self-play fine-tune trainer scaffolding.

TC01-TC09: config validation + RunningMeanBaseline (should pass).
TC10-TC20: trainer stubs (expect NotImplementedError).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

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
# TC10-TC12: compute_returns (stub — expect NotImplementedError)
# ---------------------------------------------------------------


class TestComputeReturns(unittest.TestCase):
    """Tests for compute_returns function."""

    def test_tc10_win_discount(self) -> None:
        """TC10: Win outcome discounts correctly (gamma^(T-1-t))."""
        game = _make_game(n_plies=3, outcome=1)
        with self.assertRaises(NotImplementedError):
            compute_returns([game], gamma=0.99)

    def test_tc11_draw_all_zero(self) -> None:
        """TC11: Draw outcome produces all-zero returns."""
        game = _make_game(n_plies=4, outcome=0)
        with self.assertRaises(NotImplementedError):
            compute_returns([game], gamma=0.99)

    def test_tc12_loss_negative(self) -> None:
        """TC12: Loss outcome produces all-negative returns."""
        game = _make_game(n_plies=2, outcome=-1)
        with self.assertRaises(NotImplementedError):
            compute_returns([game], gamma=0.99)


# ---------------------------------------------------------------
# TC13: _log_prob_of_move (stub)
# ---------------------------------------------------------------


class TestLogProbOfMove(unittest.TestCase):
    """Tests for _log_prob_of_move function."""

    def test_tc13_returns_scalar_with_grad(self) -> None:
        """TC13: _log_prob_of_move raises NotImplementedError."""
        bt = torch.zeros(1, 65, dtype=torch.long)
        ct = torch.zeros(1, 65, dtype=torch.long)
        tt = torch.zeros(1, 65, dtype=torch.long)
        move_tok = MagicMock()
        with self.assertRaises(NotImplementedError):
            _log_prob_of_move(
                policy=MagicMock(),
                board_tokens=bt,
                color_tokens=ct,
                traj_tokens=tt,
                move_token=42,
                legal_moves=["e2e4"],
                move_tok=move_tok,
                device=torch.device("cpu"),
            )


# ---------------------------------------------------------------
# TC14-TC15: _update_shadow (stub)
# ---------------------------------------------------------------


class TestUpdateShadow(unittest.TestCase):
    """Tests for _update_shadow EMA update."""

    def test_tc14_ema_zero_copies_policy(self) -> None:
        """TC14: _update_shadow raises NotImplementedError."""
        cfg = _make_cfg()
        with self.assertRaises(NotImplementedError):
            RLFinetuneTrainer(cfg, device="cpu")

    def test_tc15_ema_one_rejected_by_config(self) -> None:
        """TC15: ema_alpha=1.0 is rejected by FinetuneConfig."""
        with self.assertRaises(ValueError):
            _make_cfg(ema_alpha=1.0)


# ---------------------------------------------------------------
# TC16-TC17: play_game (stub)
# ---------------------------------------------------------------


class TestPlayGame(unittest.TestCase):
    """Tests for play_game function."""

    def test_tc16_maxply_termination(self) -> None:
        """TC16: play_game raises NotImplementedError."""
        cfg = FinetuneConfig(max_ply=10)
        with self.assertRaises(NotImplementedError):
            play_game(
                policy=MagicMock(),
                shadow=MagicMock(),
                board_tok=MagicMock(),
                move_tok=MagicMock(),
                cfg=cfg,
                device=torch.device("cpu"),
            )

    def test_tc17_checkmate_termination(self) -> None:
        """TC17: play_game with checkmate raises NIE."""
        cfg = FinetuneConfig()
        with self.assertRaises(NotImplementedError):
            play_game(
                policy=MagicMock(),
                shadow=MagicMock(),
                board_tok=MagicMock(),
                move_tok=MagicMock(),
                cfg=cfg,
                device=torch.device("cpu"),
            )


# ---------------------------------------------------------------
# TC18: RLFinetuneTrainer.__init__
# ---------------------------------------------------------------


class TestRLFinetuneTrainerInit(unittest.TestCase):
    """Tests for RLFinetuneTrainer initialization."""

    def test_tc18_init_raises_nie(self) -> None:
        """TC18: __init__ raises NotImplementedError (stub)."""
        cfg = _make_cfg()
        with self.assertRaises(NotImplementedError):
            RLFinetuneTrainer(cfg, device="cpu")


# ---------------------------------------------------------------
# TC19: _gradient_step
# ---------------------------------------------------------------


class TestGradientStep(unittest.TestCase):
    """Tests for RLFinetuneTrainer._gradient_step."""

    def test_tc19_gradient_step_raises_nie(self) -> None:
        """TC19: _gradient_step raises NIE because __init__ does."""
        cfg = _make_cfg()
        with self.assertRaises(NotImplementedError):
            RLFinetuneTrainer(cfg, device="cpu")


# ---------------------------------------------------------------
# TC20: save/load checkpoint
# ---------------------------------------------------------------


class TestCheckpointRoundTrip(unittest.TestCase):
    """Tests for save_checkpoint / load_checkpoint."""

    def test_tc20_checkpoint_roundtrip_raises_nie(self) -> None:
        """TC20: Checkpoint round-trip blocked by __init__ NIE."""
        cfg = _make_cfg()
        with self.assertRaises(NotImplementedError):
            RLFinetuneTrainer(cfg, device="cpu")


if __name__ == "__main__":
    unittest.main()
