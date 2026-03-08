"""Unit tests for Phase 2 self-play RL components (TC01-TC15).

All tests run on CPU only (per project convention).
Test method bodies raise NotImplementedError — they define the
behavioural contract for implementation developers to fill in.
"""

from __future__ import annotations

import unittest

import torch  # noqa: F401
import torch.nn as nn  # noqa: F401

from chess_sim.config import Phase2Config  # noqa: F401
from chess_sim.env.self_play_source import SelfPlaySource  # noqa: F401
from chess_sim.model.value_heads import ValueHeads  # noqa: F401
from chess_sim.training.ema_updater import EMAUpdater  # noqa: F401
from chess_sim.training.episode_recorder import EpisodeRecorder  # noqa: F401
from chess_sim.training.reward_computer import RewardComputer  # noqa: F401
from chess_sim.types import EpisodeRecord, PlyTuple, ValueHeadOutput  # noqa: F401


class TestPhase2SelfPlay(unittest.TestCase):
    """Unit tests for Phase 2 self-play RL components (TC01-TC15)."""

    def setUp(self) -> None:
        """Set up shared fixtures for all test cases."""
        self.device = "cpu"
        self.d_model = 128

    def test_TC01_ema_update_correctness(self) -> None:
        """TC01: EMA update at alpha=0.995 yields correct blend."""
        raise NotImplementedError

    def test_TC02_ema_leaves_player_unchanged(self) -> None:
        """TC02: EMA step does not modify player parameters."""
        raise NotImplementedError

    def test_TC03_surprise_certain_correct(self) -> None:
        """TC03: Surprise -- certain correct: H=0.1, c=+1, s=+1."""
        raise NotImplementedError

    def test_TC04_surprise_uncertain_wrong_winning(self) -> None:
        """TC04: Surprise -- uncertain wrong winning: H=2.0, c=-1, s=+1."""
        raise NotImplementedError

    def test_TC05_surprise_draw_collapses_to_zero(self) -> None:
        """TC05: Surprise -- draw (reward_sign=0) gives all-zero."""
        raise NotImplementedError

    def test_TC06_reward_tensor_shape_and_values(self) -> None:
        """TC06: Reward tensor shape [10] for 10 player plies."""
        raise NotImplementedError

    def test_TC07_reward_loss_trajectory_sign_flip(self) -> None:
        """TC07: Loss trajectory high-H correct is less negative."""
        raise NotImplementedError

    def test_TC08_value_heads_forward_shape(self) -> None:
        """TC08: ValueHeads.forward returns [4,1] v_win, v_surprise."""
        raise NotImplementedError

    def test_TC09_phase2_config_invalid_alpha(self) -> None:
        """TC09: Phase2Config raises ValueError when ema_alpha=1.0."""
        raise NotImplementedError

    def test_TC10_phase2_config_invalid_gamma(self) -> None:
        """TC10: Phase2Config raises ValueError when gamma=0.0."""
        raise NotImplementedError

    def test_TC11_episode_recorder_finalize(self) -> None:
        """TC11: EpisodeRecorder.finalize with 5 plies returns record."""
        raise NotImplementedError

    def test_TC12_self_play_source_terminal_detection(self) -> None:
        """TC12: SelfPlaySource.is_terminal() True on checkmate."""
        raise NotImplementedError

    def test_TC13_player_only_ply_filtering(self) -> None:
        """TC13: RewardComputer.compute returns len 10 for 20 plies."""
        raise NotImplementedError

    def test_TC14_temporal_discount_monotonicity(self) -> None:
        """TC14: R[0] > R[9] for gamma=0.99, outcome=+1, 10 plies."""
        raise NotImplementedError

    def test_TC15_self_play_loop_no_param_collision(self) -> None:
        """TC15: SelfPlayLoop.run(1) updates player, EMA for opp."""
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
