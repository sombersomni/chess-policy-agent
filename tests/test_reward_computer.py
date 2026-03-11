"""Tests for composite reward computation (PGNRLRewardComputer).

T-CR1 through T-CR10: verify the composite formula
R(t) = lambda_outcome * sign_outcome(t)
     + lambda_material * material_delta(t)
"""
from __future__ import annotations

import unittest

from chess_sim.config import RLConfig  # noqa: F401
from chess_sim.training.pgn_rl_reward_computer import (
    PGNRLRewardComputer,
)


class TestCompositeRewardComputer(unittest.TestCase):
    """Tests for PGNRLRewardComputer composite formula."""

    def setUp(self) -> None:
        """Initialize reward computer."""
        self.reward_fn = PGNRLRewardComputer()

    def test_non_capture_winner_ply(self) -> None:
        """T-CR1: Winner ply, no capture -> R = lambda_outcome * 1.0."""
        raise NotImplementedError("To be implemented")

    def test_non_capture_loser_ply(self) -> None:
        """T-CR2: Loser ply, no capture -> R = lambda_outcome * -1.0."""
        raise NotImplementedError("To be implemented")

    def test_winner_ply_with_queen_capture(self) -> None:
        """T-CR3: Winner + material_delta=9.0, lambda_material=0.1 -> R=1.9."""
        raise NotImplementedError("To be implemented")

    def test_draw_ply_neutral(self) -> None:
        """T-CR4: draw_reward_norm=0.0, no capture -> R = 0.0."""
        raise NotImplementedError("To be implemented")

    def test_draw_ply_with_capture(self) -> None:
        """T-CR5: Draw + material_delta=3.0, lambda_material=0.1 -> R=0.3."""
        raise NotImplementedError("To be implemented")

    def test_all_non_capture_winning_game(self) -> None:
        """T-CR6: Uniform reward after RSBC normalization -> r_hat=1.0."""
        raise NotImplementedError("To be implemented")

    def test_queen_capture_ply_has_highest_reward(self) -> None:
        """T-CR7: Ply with queen capture gets highest r_hat."""
        raise NotImplementedError("To be implemented")

    def test_empty_ply_list(self) -> None:
        """T-CR8: Empty list -> zeros(0)."""
        raise NotImplementedError("To be implemented")

    def test_negative_material_delta(self) -> None:
        """Loser ply + opponent took rook -> R = -1.0 + (-5.0 * 0.1) = -1.5."""
        raise NotImplementedError("To be implemented")

    def test_zero_lambda_material_ignores_delta(self) -> None:
        """lambda_material=0.0 -> R purely from outcome sign."""
        raise NotImplementedError("To be implemented")


if __name__ == "__main__":
    unittest.main()
