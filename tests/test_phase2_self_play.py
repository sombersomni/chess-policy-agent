"""Unit tests for Phase 2 self-play RL components (TC01-TC15).

All tests run on CPU only (per project convention).
Test method bodies raise NotImplementedError -- they define the
behavioural contract for implementation developers to fill in.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from chess_sim.config import Phase2Config
from chess_sim.env.self_play_source import SelfPlaySource
from chess_sim.model.value_heads import ValueHeads
from chess_sim.training.ema_updater import EMAUpdater
from chess_sim.training.episode_recorder import EpisodeRecorder
from chess_sim.training.reward_computer import RewardComputer
from chess_sim.types import EpisodeRecord, PlyTuple, ValueHeadOutput  # noqa: F401


class TestPhase2SelfPlay(unittest.TestCase):
    """Unit tests for Phase 2 self-play RL components (TC01-TC15)."""

    def setUp(self) -> None:
        """Set up shared fixtures for all test cases."""
        self.device = "cpu"
        self.d_model = 128

    def test_TC01_ema_update_correctness(self) -> None:
        """TC01: EMA update at alpha=0.995 yields correct blend."""
        model_a = nn.Linear(4, 4)
        model_b = nn.Linear(4, 4)
        initial_b = {
            k: v.clone()
            for k, v in model_b.state_dict().items()
        }
        updater = EMAUpdater(alpha=0.995)
        updater.step(model_a, model_b)
        for (name, p_b), (_, p_a) in zip(
            model_b.named_parameters(),
            model_a.named_parameters(),
        ):
            expected = (
                0.995 * initial_b[name] + 0.005 * p_a.data
            )
            self.assertTrue(
                torch.allclose(
                    p_b.data, expected, atol=1e-6
                ),
                name,
            )

    def test_TC02_ema_leaves_player_unchanged(self) -> None:
        """TC02: EMA step does not modify player parameters."""
        model_a = nn.Linear(4, 4)
        model_b = nn.Linear(4, 4)
        before = {
            k: v.clone()
            for k, v in model_a.state_dict().items()
        }
        EMAUpdater(alpha=0.995).step(model_a, model_b)
        for name, p in model_a.named_parameters():
            self.assertTrue(
                torch.equal(p.data, before[name]), name
            )

    def test_TC03_surprise_certain_correct(self) -> None:
        """TC03: Surprise -- certain correct: H=0.1, c=+1, s=+1."""
        rc = RewardComputer()
        result = rc._surprise(
            torch.tensor([0.1]),
            torch.tensor([1.0]),
            1.0,
        )
        self.assertAlmostEqual(
            result.item(), 0.1, places=5
        )

    def test_TC04_surprise_uncertain_wrong_winning(
        self,
    ) -> None:
        """TC04: Surprise -- uncertain wrong winning."""
        rc = RewardComputer()
        result = rc._surprise(
            torch.tensor([2.0]),
            torch.tensor([-1.0]),
            1.0,
        )
        self.assertAlmostEqual(
            result.item(), -2.0, places=5
        )

    def test_TC05_surprise_draw_collapses_to_zero(
        self,
    ) -> None:
        """TC05: Surprise -- draw (reward_sign=0) all-zero."""
        rc = RewardComputer()
        result = rc._surprise(
            torch.tensor([1.5, 0.3, 2.1]),
            torch.tensor([1.0, -1.0, 1.0]),
            0.0,
        )
        self.assertTrue(torch.all(result == 0.0))

    def test_TC06_reward_tensor_shape_and_values(
        self,
    ) -> None:
        """TC06: Reward tensor shape [10] for 10 player plies."""
        cfg = Phase2Config(
            ema_alpha=0.99,
            gamma=0.99,
            lambda_surprise=0.5,
        )
        dummy_t = torch.zeros(65, dtype=torch.long)
        dummy_probs = torch.zeros(1971)
        dummy_probs[0] = 1.0
        plies = [
            PlyTuple(
                board_tokens=dummy_t,
                color_tokens=dummy_t,
                traj_tokens=dummy_t,
                move_prefix=dummy_t[:1],
                log_prob=torch.tensor(-0.5),
                probs=dummy_probs,
                entropy=0.5,
                move_uci="e2e4",
                is_player_ply=True,
            )
            for _ in range(10)
        ]
        record = EpisodeRecord(
            plies=plies, outcome=1.0, total_plies=10
        )
        rewards = RewardComputer().compute(record, cfg)
        self.assertEqual(rewards.shape[0], 10)

    def test_TC07_reward_loss_trajectory_sign_flip(
        self,
    ) -> None:
        """TC07: Loss trajectory high-H correct is less negative."""
        from chess_sim.data.move_tokenizer import (
            MoveTokenizer,
        )

        cfg = Phase2Config(
            ema_alpha=0.99,
            gamma=0.99,
            lambda_surprise=1.0,
        )
        dummy_t = torch.zeros(65, dtype=torch.long)
        # Set argmax of probs to match e2e4 vocab idx
        tok = MoveTokenizer()
        e2e4_idx = tok.tokenize_move("e2e4")
        dummy_probs = torch.zeros(1971)
        dummy_probs[e2e4_idx] = 1.0
        ply = PlyTuple(
            board_tokens=dummy_t,
            color_tokens=dummy_t,
            traj_tokens=dummy_t,
            move_prefix=dummy_t[:1],
            log_prob=torch.tensor(-0.5),
            probs=dummy_probs,
            entropy=1.0,
            move_uci="e2e4",
            is_player_ply=True,
        )
        record = EpisodeRecord(
            plies=[ply], outcome=-1.0, total_plies=1
        )
        rewards = RewardComputer().compute(record, cfg)
        # temporal = -1.0 * 0.99^0 = -1.0
        # correct=+1 flipped to -1 (loss sign flip),
        # reward_sign=-1: surprise = 1.0*(-1)*(-1) = 1.0
        # R = -1.0 + 1.0 * 1.0 = 0.0 > -1.0
        self.assertGreater(rewards[0].item(), -1.0)

    def test_TC08_value_heads_forward_shape(self) -> None:
        """TC08: ValueHeads.forward returns [4,1] v_win, v_surprise."""
        vh = ValueHeads(d_model=128)
        cls_emb = torch.randn(4, 128).detach()
        out = vh(cls_emb)
        self.assertEqual(out.v_win.shape, (4, 1))
        self.assertEqual(out.v_surprise.shape, (4, 1))
        self.assertTrue(
            torch.all(torch.isfinite(out.v_win))
        )
        self.assertTrue(
            torch.all(torch.isfinite(out.v_surprise))
        )

    def test_TC09_phase2_config_invalid_alpha(
        self,
    ) -> None:
        """TC09: Phase2Config raises ValueError for alpha=1.0."""
        with self.assertRaises(ValueError):
            Phase2Config(ema_alpha=1.0)

    def test_TC10_phase2_config_invalid_gamma(
        self,
    ) -> None:
        """TC10: Phase2Config raises ValueError for gamma=0.0."""
        with self.assertRaises(ValueError):
            Phase2Config(gamma=0.0)

    def test_TC11_episode_recorder_finalize(self) -> None:
        """TC11: EpisodeRecorder.finalize with 5 plies."""
        recorder = EpisodeRecorder()
        dummy_t = torch.zeros(65, dtype=torch.long)
        dummy_prob = torch.ones(10) / 10.0
        for i in range(5):
            ply = PlyTuple(
                board_tokens=dummy_t,
                color_tokens=dummy_t,
                traj_tokens=dummy_t,
                move_prefix=dummy_t[:1],
                log_prob=torch.tensor(-1.0),
                probs=dummy_prob,
                entropy=float(i + 1),
                move_uci="e2e4",
                is_player_ply=(i % 2 == 0),
            )
            recorder.record(ply)
        record = recorder.finalize(outcome=1.0)
        self.assertEqual(len(record.plies), 5)
        self.assertAlmostEqual(record.outcome, 1.0)
        player_entropies = [
            p.entropy
            for p in record.plies
            if p.is_player_ply
        ]
        self.assertAlmostEqual(
            sum(player_entropies), 1.0, places=5
        )

    def test_TC12_self_play_source_terminal_detection(
        self,
    ) -> None:
        """TC12: SelfPlaySource.is_terminal() True on checkmate."""
        source = SelfPlaySource()
        source.reset()
        for move in [
            "e2e4", "e7e5", "d1h5",
            "b8c6", "f1c4", "a7a6", "h5f7",
        ]:
            source.step(move)
        self.assertTrue(source.is_terminal())
        self.assertEqual(source.legal_moves(), [])

    def test_TC13_player_only_ply_filtering(self) -> None:
        """TC13: RewardComputer.compute returns len 10 for 20 plies."""
        cfg = Phase2Config(
            ema_alpha=0.99,
            gamma=0.99,
            lambda_surprise=0.5,
        )
        dummy_t = torch.zeros(65, dtype=torch.long)
        dummy_probs = torch.zeros(1971)
        dummy_probs[0] = 1.0
        plies = [
            PlyTuple(
                board_tokens=dummy_t,
                color_tokens=dummy_t,
                traj_tokens=dummy_t,
                move_prefix=dummy_t[:1],
                log_prob=torch.tensor(-1.0),
                probs=dummy_probs,
                entropy=0.5,
                move_uci="e2e4",
                is_player_ply=(i % 2 == 0),
            )
            for i in range(20)
        ]
        record = EpisodeRecord(
            plies=plies, outcome=1.0, total_plies=20
        )
        rewards = RewardComputer().compute(record, cfg)
        self.assertEqual(rewards.shape[0], 10)

    def test_TC14_temporal_discount_monotonicity(
        self,
    ) -> None:
        """TC14: R[0] > R[9] for gamma=0.99, outcome=+1."""
        temporal = RewardComputer()._temporal_advantage(
            outcome=1.0, T=10, gamma=0.99
        )
        self.assertEqual(temporal.shape[0], 10)
        self.assertGreater(
            temporal[0].item(), temporal[9].item()
        )

    def test_TC15_self_play_loop_no_param_collision(
        self,
    ) -> None:
        """TC15: SelfPlayLoop.run(1) updates player, EMA for opp."""
        from chess_sim.config import ChessModelV2Config
        from chess_sim.model.chess_model import ChessModel
        from chess_sim.training.self_play_loop import (
            SelfPlayLoop,
        )

        v2_cfg = ChessModelV2Config()
        player = ChessModel(v2_cfg.model, v2_cfg.decoder)
        cfg = Phase2Config(
            ema_alpha=0.9, max_episode_steps=5
        )
        before_player = {
            k: v.clone()
            for k, v in player.state_dict().items()
        }
        loop = SelfPlayLoop(
            player=player, cfg=cfg, device="cpu"
        )
        opp_before = {
            k: v.clone()
            for k, v in loop._opponent.state_dict().items()
        }
        loop.run(1)
        player_changed = any(
            not torch.equal(
                player.state_dict()[k], before_player[k]
            )
            for k in before_player
        )
        self.assertTrue(
            player_changed,
            "Player parameters should change after run(1)",
        )
        alpha = cfg.ema_alpha
        for name in opp_before:
            p_opp_after = loop._opponent.state_dict()[
                name
            ]
            p_opp_init = opp_before[name]
            p_player_after = player.state_dict()[name]
            expected = (
                alpha * p_opp_init
                + (1 - alpha) * p_player_after
            )
            self.assertTrue(
                torch.allclose(
                    p_opp_after, expected, atol=1e-5
                ),
                f"EMA invariant violated for {name}",
            )


if __name__ == "__main__":
    unittest.main()
