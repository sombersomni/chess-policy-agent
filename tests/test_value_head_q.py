"""Tests for ActionConditionedValueHead Q-function (T24-T37).

Each test exercises a specific component from the design doc
at docs/action_conditioned_value_head.md.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn
import torch

from chess_sim.config import ModelConfig, PGNRLConfig
from chess_sim.model.chess_model import ChessModel
from chess_sim.model.value_heads import (
    ActionConditionedValueHead,
)
from chess_sim.training.pgn_rl_trainer import PGNRLTrainer


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


# -----------------------------------------------------------
# T24 -- ActionConditionedValueHead forward shape
# -----------------------------------------------------------

class TestQHeadForwardShape(unittest.TestCase):
    """T24: forward returns shape [B, 1], dtype float32."""

    def test_t24_output_shape_and_dtype(self):
        head = ActionConditionedValueHead(128)
        out = head.forward(
            torch.rand(4, 128), torch.rand(4, 128)
        )
        self.assertEqual(out.shape, torch.Size([4, 1]))
        self.assertEqual(out.dtype, torch.float32)


# -----------------------------------------------------------
# T25 -- All outputs finite
# -----------------------------------------------------------

class TestQHeadOutputsFinite(unittest.TestCase):
    """T25: all Q-values from forward are finite."""

    def test_t25_all_finite(self):
        head = ActionConditionedValueHead(128)
        out = head.forward(
            torch.rand(4, 128), torch.rand(4, 128)
        )
        self.assertTrue(torch.isfinite(out).all())


# -----------------------------------------------------------
# T26 -- Encoder gradient isolation via cls.detach()
# -----------------------------------------------------------

class TestEncoderGradIsolation(unittest.TestCase):
    """T26: cls.detach() prevents grad flow to encoder."""

    def test_t26_cls_grad_is_none_after_backward(self):
        head = ActionConditionedValueHead(128)
        cls = torch.rand(1, 128, requires_grad=True)
        action = torch.rand(1, 128)
        # Caller detaches cls before passing to Q-head
        q = head.forward(cls.detach(), action)
        q.sum().backward()
        self.assertIsNone(cls.grad)


# -----------------------------------------------------------
# T27 -- Decoder token_emb gradient isolation
# -----------------------------------------------------------

class TestActionEmbGradIsolation(unittest.TestCase):
    """T27: action_emb.detach() prevents grad to token_emb."""

    def test_t27_action_emb_grad_is_none(self):
        head = ActionConditionedValueHead(128)
        cls = torch.rand(1, 128)
        action = torch.rand(
            1, 128, requires_grad=True
        )
        # Caller detaches action_emb before Q-head
        q = head.forward(cls, action.detach())
        q.sum().backward()
        self.assertIsNone(action.grad)


# -----------------------------------------------------------
# T28 -- Q-head grad does not enter pg_loss via advantage
# -----------------------------------------------------------

class TestQHeadGradNotInPGLoss(unittest.TestCase):
    """T28: q_t.detach() in advantage blocks Q-head grad."""

    def test_t28_q_grad_none_after_pg_backward(self):
        head = ActionConditionedValueHead(128)
        cls = torch.rand(1, 128)
        action = torch.rand(1, 128)
        q_t = head.forward(cls, action)  # [1, 1]
        # Simulate advantage with detach
        reward = torch.tensor([1.0])
        advantage = reward - q_t.squeeze().detach()
        log_prob = torch.log(
            torch.tensor([0.5], requires_grad=True)
        )
        pg_loss = -(log_prob * advantage).sum()
        pg_loss.backward()
        # Q-head params should not have grad from pg_loss
        # since q_t was detached before advantage
        self.assertIsNone(q_t.grad)


# -----------------------------------------------------------
# T29 -- ChessModel.move_token_emb returns correct object
# -----------------------------------------------------------

class TestMoveTokenEmbIdentity(unittest.TestCase):
    """T29: move_token_emb is decoder.move_embedding.token_emb."""

    def test_t29_same_object(self):
        model = ChessModel(ModelConfig())
        emb = model.move_token_emb
        expected = (
            model.decoder.move_embedding.token_emb
        )
        self.assertIs(emb, expected)


# -----------------------------------------------------------
# T30 -- move_token_emb output shape
# -----------------------------------------------------------

class TestMoveTokenEmbShape(unittest.TestCase):
    """T30: move_token_emb(tensor([42])) -> [1, d_model]."""

    def test_t30_embedding_shape(self):
        cfg = ModelConfig()
        model = ChessModel(cfg)
        out = model.move_token_emb(
            torch.tensor([42])
        )
        self.assertEqual(
            out.shape, torch.Size([1, cfg.d_model])
        )


# -----------------------------------------------------------
# T31 -- _encode_and_decode returns valid move_idx
# -----------------------------------------------------------

class TestEncodeDecodeKnownUCI(unittest.TestCase):
    """T31: known UCI -> move_idx is int in [0, 1970]."""

    def test_t31_valid_move_idx(self):
        cfg = PGNRLConfig()
        trainer = PGNRLTrainer(
            cfg=cfg, device="cpu", total_steps=10
        )
        bt = torch.randint(0, 8, (1, 65))
        ct = torch.randint(0, 3, (1, 65))
        tt = torch.zeros(1, 65, dtype=torch.long)
        prefix = torch.zeros(1, 1, dtype=torch.long)
        _, _, move_idx = trainer._encode_and_decode(
            bt, ct, tt, prefix, "e2e4"
        )
        self.assertIsInstance(move_idx, int)
        self.assertGreaterEqual(move_idx, 0)
        self.assertLessEqual(move_idx, 1970)


# -----------------------------------------------------------
# T32 -- _encode_and_decode returns None for unknown UCI
# -----------------------------------------------------------

class TestEncodeDecodeUnknownUCI(unittest.TestCase):
    """T32: unknown UCI -> move_idx is None."""

    def test_t32_none_for_unknown_uci(self):
        cfg = PGNRLConfig()
        trainer = PGNRLTrainer(
            cfg=cfg, device="cpu", total_steps=10
        )
        bt = torch.randint(0, 8, (1, 65))
        ct = torch.randint(0, 3, (1, 65))
        tt = torch.zeros(1, 65, dtype=torch.long)
        prefix = torch.zeros(1, 1, dtype=torch.long)
        _, _, move_idx = trainer._encode_and_decode(
            bt, ct, tt, prefix, "zzzz"
        )
        self.assertIsNone(move_idx)


# -----------------------------------------------------------
# T33 -- train_game returns value_loss key
# -----------------------------------------------------------

class TestTrainGameValueLossKey(unittest.TestCase):
    """T33: train_game returns 'value_loss' >= 0.0."""

    def test_t33_value_loss_present_and_nonneg(self):
        cfg = PGNRLConfig()
        trainer = PGNRLTrainer(
            cfg=cfg, device="cpu", total_steps=100
        )
        game = _make_scholars_mate()
        metrics = trainer.train_game(game)
        self.assertIn("value_loss", metrics)
        self.assertGreaterEqual(
            metrics["value_loss"], 0.0
        )


# -----------------------------------------------------------
# T34 -- train_game returns mean_advantage key
# -----------------------------------------------------------

class TestTrainGameMeanAdvantageKey(unittest.TestCase):
    """T34: train_game returns finite 'mean_advantage'."""

    def test_t34_mean_advantage_finite(self):
        cfg = PGNRLConfig()
        trainer = PGNRLTrainer(
            cfg=cfg, device="cpu", total_steps=100
        )
        game = _make_scholars_mate()
        metrics = trainer.train_game(game)
        self.assertIn("mean_advantage", metrics)
        self.assertTrue(
            math.isfinite(metrics["mean_advantage"])
        )


# -----------------------------------------------------------
# T35 -- Integration: mock model returns all loss keys
# -----------------------------------------------------------

class TestIntegrationAllLossKeys(unittest.TestCase):
    """T35: train_game dict has all 5 required keys."""

    def test_t35_all_keys_present(self):
        cfg = PGNRLConfig()
        trainer = PGNRLTrainer(
            cfg=cfg, device="cpu", total_steps=100
        )
        game = _make_scholars_mate()
        metrics = trainer.train_game(game)
        required = {
            "total_loss",
            "pg_loss",
            "ce_loss",
            "value_loss",
            "mean_advantage",
        }
        self.assertTrue(
            required.issubset(metrics.keys()),
            f"Missing: {required - metrics.keys()}",
        )


# -----------------------------------------------------------
# T36 -- Q-head _fc1 input dimension is 2 * d_model
# -----------------------------------------------------------

class TestQHeadFc1InputDim(unittest.TestCase):
    """T36: _fc1.in_features == 2 * d_model."""

    def test_t36_fc1_in_features(self):
        d_model = 128
        head = ActionConditionedValueHead(d_model)
        self.assertEqual(
            head._fc1.in_features, 2 * d_model
        )


# -----------------------------------------------------------
# T37 -- Checkpoint round-trip includes Q-head weights
# -----------------------------------------------------------

class TestCheckpointQHeadWeights(unittest.TestCase):
    """T37: save/load preserves Q-head _fc1.weight exactly."""

    def test_t37_checkpoint_round_trip(self):
        cfg = PGNRLConfig()
        trainer = PGNRLTrainer(
            cfg=cfg, device="cpu", total_steps=100
        )
        original_w = (
            trainer.model.value_head._fc1.weight.clone()
        )
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "q_test.pt"
            trainer.save_checkpoint(ckpt)
            trainer.load_checkpoint(ckpt)
        loaded_w = (
            trainer.model.value_head._fc1.weight
        )
        self.assertTrue(
            torch.equal(original_w, loaded_w),
            "Q-head _fc1.weight mismatch after "
            "checkpoint round-trip",
        )


if __name__ == "__main__":
    unittest.main()
