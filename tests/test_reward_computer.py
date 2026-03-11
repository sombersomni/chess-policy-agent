"""Tests for composite reward computation (PGNRLRewardComputer).

T-CR1 through T-CR10: verify the composite formula
R(t) = lambda_outcome * sign_outcome(t)
     + lambda_material * material_delta(t)
"""
from __future__ import annotations

import unittest

import torch

from chess_sim.config import RLConfig
from chess_sim.training.pgn_rl_reward_computer import (
    PGNRLRewardComputer,
)
from chess_sim.types import OfflinePlyTuple


def _make_ply(
    is_winner: bool,
    is_draw: bool = False,
    material_delta: float = 0.0,
) -> OfflinePlyTuple:
    """Build a minimal OfflinePlyTuple for reward tests."""
    dummy = torch.zeros(65, dtype=torch.long)
    prefix = torch.zeros(1, dtype=torch.long)
    return OfflinePlyTuple(
        board_tokens=dummy,
        color_tokens=dummy,
        traj_tokens=dummy,
        move_prefix=prefix,
        move_uci="e2e4",
        is_winner_ply=is_winner,
        is_white_ply=True,
        is_draw_ply=is_draw,
        material_delta=material_delta,
    )


def _make_cfg(
    lambda_outcome: float = 1.0,
    lambda_material: float = 0.0,
    draw_reward_norm: float = 0.0,
) -> RLConfig:
    """Build RLConfig with custom reward weights."""
    return RLConfig(
        lambda_outcome=lambda_outcome,
        lambda_material=lambda_material,
        draw_reward_norm=draw_reward_norm,
    )


class TestCompositeRewardComputer(unittest.TestCase):
    """Tests for PGNRLRewardComputer composite formula."""

    def setUp(self) -> None:
        """Initialize reward computer."""
        self.reward_fn = PGNRLRewardComputer()

    def test_non_capture_winner_ply(self) -> None:
        """T-CR1: Winner ply, no capture -> R = lambda_outcome * 1.0."""
        plies = [_make_ply(is_winner=True)]
        cfg = _make_cfg(lambda_material=0.0)
        r = self.reward_fn.compute(plies, cfg)
        self.assertAlmostEqual(r[0].item(), 1.0, places=5)

    def test_non_capture_loser_ply(self) -> None:
        """T-CR2: Loser ply, no capture -> R = lambda_outcome * -1.0."""
        plies = [_make_ply(is_winner=False)]
        cfg = _make_cfg(lambda_material=0.0)
        r = self.reward_fn.compute(plies, cfg)
        self.assertAlmostEqual(r[0].item(), -1.0, places=5)

    def test_winner_ply_with_queen_capture(self) -> None:
        """T-CR3: Winner + material_delta=9.0, lambda_material=0.1 -> R=1.9."""
        plies = [_make_ply(is_winner=True, material_delta=9.0)]
        cfg = _make_cfg(
            lambda_outcome=1.0, lambda_material=0.1,
        )
        r = self.reward_fn.compute(plies, cfg)
        self.assertAlmostEqual(r[0].item(), 1.9, places=5)

    def test_draw_ply_neutral(self) -> None:
        """T-CR4: draw_reward_norm=0.0, no capture -> R = 0.0."""
        plies = [_make_ply(is_winner=True, is_draw=True)]
        cfg = _make_cfg(draw_reward_norm=0.0, lambda_material=0.0)
        r = self.reward_fn.compute(plies, cfg)
        self.assertAlmostEqual(r[0].item(), 0.0, places=5)

    def test_draw_ply_with_capture(self) -> None:
        """T-CR5: Draw + material_delta=3.0, lambda_material=0.1 -> R=0.3."""
        plies = [
            _make_ply(
                is_winner=True, is_draw=True,
                material_delta=3.0,
            ),
        ]
        cfg = _make_cfg(
            draw_reward_norm=0.0, lambda_material=0.1,
        )
        r = self.reward_fn.compute(plies, cfg)
        self.assertAlmostEqual(r[0].item(), 0.3, places=5)

    def test_all_non_capture_winning_game(self) -> None:
        """T-CR6: 3 winner plies, no captures -> all rewards = 1.0."""
        plies = [_make_ply(is_winner=True) for _ in range(3)]
        cfg = _make_cfg(lambda_material=0.0)
        r = self.reward_fn.compute(plies, cfg)
        for i in range(3):
            self.assertAlmostEqual(
                r[i].item(), 1.0, places=5,
            )

    def test_queen_capture_ply_has_highest_reward(self) -> None:
        """T-CR7: Ply with queen capture gets highest reward."""
        plies = [
            _make_ply(is_winner=True, material_delta=0.0),
            _make_ply(is_winner=True, material_delta=9.0),
            _make_ply(is_winner=True, material_delta=0.0),
        ]
        cfg = _make_cfg(
            lambda_outcome=1.0, lambda_material=0.1,
        )
        r = self.reward_fn.compute(plies, cfg)
        self.assertGreater(r[1].item(), r[0].item())
        self.assertGreater(r[1].item(), r[2].item())

    def test_empty_ply_list(self) -> None:
        """T-CR8: Empty list -> zeros(0)."""
        cfg = _make_cfg()
        r = self.reward_fn.compute([], cfg)
        self.assertEqual(r.shape, torch.Size([0]))

    def test_negative_material_delta(self) -> None:
        """Loser ply + opponent took rook -> R = -1.5."""
        plies = [
            _make_ply(is_winner=False, material_delta=-5.0),
        ]
        cfg = _make_cfg(
            lambda_outcome=1.0, lambda_material=0.1,
        )
        r = self.reward_fn.compute(plies, cfg)
        self.assertAlmostEqual(r[0].item(), -1.5, places=5)

    def test_zero_lambda_material_ignores_delta(self) -> None:
        """lambda_material=0.0 -> R purely from outcome sign."""
        plies = [
            _make_ply(is_winner=True, material_delta=9.0),
        ]
        cfg = _make_cfg(lambda_material=0.0)
        r = self.reward_fn.compute(plies, cfg)
        self.assertAlmostEqual(r[0].item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
