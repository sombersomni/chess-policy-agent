"""Tests for Encoder v2: trajectory tokens, 4-stream embedding, and protocols.

T26-T40: Tests for trajectory token derivation, embedding init priors,
4-stream encoder signature, gradient flow, and structural subtyping.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

from chess_sim.model.embedding import D_MODEL, EmbeddingLayer
from chess_sim.model.encoder import ChessEncoder
from chess_sim.protocols import Embeddable, Encodable
from chess_sim.types import ChessBatch, TrainingExample
from scripts.train_real import _make_trajectory_tokens
from tests.utils import make_training_examples


# ------------------------------------------------------------------
# T26-T28: _make_trajectory_tokens
# ------------------------------------------------------------------

class TestTrajectoryTokensEmpty(unittest.TestCase):
    """T26: Empty move history produces all-zero tokens."""

    def test_t26_empty_history_all_zeros(self) -> None:
        """T26: _make_trajectory_tokens([]) returns length-65 all zeros."""
        result = _make_trajectory_tokens([])
        self.assertEqual(len(result), 65)
        self.assertTrue(all(v == 0 for v in result))

    def test_t26_cls_always_zero(self) -> None:
        """T26: Index 0 (CLS) is always 0 even with moves."""
        m = chess.Move.from_uci("e2e4")
        result = _make_trajectory_tokens([m])
        self.assertEqual(result[0], 0)


class TestTrajectoryTokensOpp(unittest.TestCase):
    """T27: With 1-move history, opp squares marked at 3/4."""

    def test_t27_opp_src_marked_3(self) -> None:
        """T27: opp.from_square+1 has value 3."""
        # e2e4: from_square=12 -> idx 13
        m = chess.Move.from_uci("e2e4")
        result = _make_trajectory_tokens([m])
        self.assertEqual(result[13], 3)

    def test_t27_opp_tgt_marked_4(self) -> None:
        """T27: opp.to_square+1 has value 4."""
        # e2e4: to_square=28 -> idx 29
        m = chess.Move.from_uci("e2e4")
        result = _make_trajectory_tokens([m])
        self.assertEqual(result[29], 4)

    def test_t27_no_player_marks_with_one_move(self) -> None:
        """T27: With only 1 move in history, no player marks (1 or 2)."""
        m = chess.Move.from_uci("e2e4")
        result = _make_trajectory_tokens([m])
        self.assertNotIn(1, result)
        self.assertNotIn(2, result)


class TestTrajectoryTokensPlayer(unittest.TestCase):
    """T28: With 2-move history, player marks (1,2) also set."""

    def test_t28_player_src_marked_1(self) -> None:
        """T28: player move (history[-2]) from_square+1 has value 1."""
        # Player: e2e4 (from=12 -> idx 13)
        # Opp:    e7e5 (from=52 -> idx 53)
        pl = chess.Move.from_uci("e2e4")
        opp = chess.Move.from_uci("e7e5")
        result = _make_trajectory_tokens([pl, opp])
        self.assertEqual(result[13], 1)

    def test_t28_player_tgt_marked_2(self) -> None:
        """T28: player move to_square+1 has value 2."""
        # e2e4: to_square=28 -> idx 29
        pl = chess.Move.from_uci("e2e4")
        opp = chess.Move.from_uci("d7d5")
        result = _make_trajectory_tokens([pl, opp])
        self.assertEqual(result[29], 2)

    def test_t28_opp_marks_still_correct(self) -> None:
        """T28: Opp marks are correct alongside player marks."""
        # d7d5: from=51 -> idx 52; to=35 -> idx 36
        pl = chess.Move.from_uci("e2e4")
        opp = chess.Move.from_uci("d7d5")
        result = _make_trajectory_tokens([pl, opp])
        self.assertEqual(result[52], 3)  # opp src
        self.assertEqual(result[36], 4)  # opp tgt


# ------------------------------------------------------------------
# T29-T33: EmbeddingLayer 4-stream + init priors
# ------------------------------------------------------------------

class TestEmbeddingLayer4Stream(unittest.TestCase):
    """T29-T30: 4-stream output shape and trajectory contribution."""

    def setUp(self) -> None:
        self.layer = EmbeddingLayer()
        self.layer.eval()

    def test_t29_four_stream_output_shape(self) -> None:
        """T29: 4-stream embed output shape is (4, 65, 256)."""
        bt = torch.randint(0, 8, (4, 65))
        ct = torch.randint(0, 3, (4, 65))
        tt = torch.zeros(4, 65, dtype=torch.long)
        with torch.no_grad():
            out = self.layer.embed(bt, ct, tt)
        self.assertEqual(out.shape, (4, 65, 256))

    def test_t30_trajectory_contributes_to_output(self) -> None:
        """T30: Nonzero trajectory_tokens change the output."""
        bt = torch.randint(0, 8, (2, 65))
        ct = torch.randint(0, 3, (2, 65))
        tt_zero = torch.zeros(2, 65, dtype=torch.long)
        tt_nonzero = torch.ones(2, 65, dtype=torch.long)
        with torch.no_grad():
            out_zero = self.layer.embed(bt, ct, tt_zero)
            out_nonzero = self.layer.embed(
                bt, ct, tt_nonzero
            )
        self.assertFalse(torch.allclose(out_zero, out_nonzero))


class TestSquareGeometricInit(unittest.TestCase):
    """T31-T32: Square embedding geometric initialization."""

    def setUp(self) -> None:
        self.layer = EmbeddingLayer()

    def test_t31_adjacent_more_similar_than_distant(
        self,
    ) -> None:
        """T31: Adjacent squares more similar than distant."""
        w = self.layer.square_emb.weight.data
        # e4=sq28 -> idx 29, e5=sq36 -> idx 37
        # a1=sq0 -> idx 1, h8=sq63 -> idx 64
        e4 = w[29]
        e5 = w[37]
        a1 = w[1]
        h8 = w[64]
        cos_adjacent = F.cosine_similarity(
            e4.unsqueeze(0), e5.unsqueeze(0)
        ).item()
        cos_distant = F.cosine_similarity(
            a1.unsqueeze(0), h8.unsqueeze(0)
        ).item()
        self.assertGreater(cos_adjacent, cos_distant)

    def test_t32_cls_row_all_zeros(self) -> None:
        """T32: CLS row (index 0) is all zeros."""
        w = self.layer.square_emb.weight.data
        self.assertTrue(
            torch.allclose(
                w[0], torch.zeros(D_MODEL), atol=1e-6
            )
        )


class TestPieceRoleInit(unittest.TestCase):
    """T33: Piece role initialization."""

    def setUp(self) -> None:
        self.layer = EmbeddingLayer()

    def test_t33_queen_differs_from_pawn(self) -> None:
        """T33: Queen row != pawn row after role init."""
        w = self.layer.piece_emb.weight.data
        # QUEEN=6, PAWN=2
        dist = torch.dist(w[6], w[2]).item()
        self.assertGreater(dist, 0)


# ------------------------------------------------------------------
# T34-T35: Encoder 3-arg signature + gradient flow
# ------------------------------------------------------------------

class TestEncoder3ArgSignature(unittest.TestCase):
    """T34: Encoder accepts 3 args and produces correct shapes."""

    def setUp(self) -> None:
        self.encoder = ChessEncoder()
        self.encoder.eval()
        self.B = 4

    def test_t34_output_shapes(self) -> None:
        """T34: encode(bt, ct, tt) returns correct shapes."""
        bt = torch.randint(0, 8, (self.B, 65))
        ct = torch.randint(0, 3, (self.B, 65))
        tt = torch.zeros(self.B, 65, dtype=torch.long)
        with torch.no_grad():
            out = self.encoder.encode(bt, ct, tt)
        self.assertEqual(
            out.cls_embedding.shape, (self.B, D_MODEL)
        )
        self.assertEqual(
            out.square_embeddings.shape,
            (self.B, 64, D_MODEL),
        )


class TestTrajectoryEmbGradient(unittest.TestCase):
    """T35: Gradient flows through trajectory_emb."""

    def test_t35_trajectory_emb_receives_gradient(self) -> None:
        """T35: trajectory_emb.weight.grad is not None and not all zero."""
        encoder = ChessEncoder()
        bt = torch.randint(0, 8, (2, 65))
        ct = torch.randint(0, 3, (2, 65))
        # Use nonzero trajectory tokens to ensure gradient signal
        tt = torch.randint(1, 5, (2, 65))
        out = encoder.encode(bt, ct, tt)
        loss = out.cls_embedding.sum()
        loss.backward()
        grad = encoder.embedding.trajectory_emb.weight.grad
        self.assertIsNotNone(grad)
        self.assertFalse(grad.eq(0).all())


# ------------------------------------------------------------------
# T36-T37: ChessBatch trajectory_tokens dtype and range
# ------------------------------------------------------------------

class TestChessBatchTrajectoryTokens(unittest.TestCase):
    """T36-T37: ChessBatch trajectory_tokens properties."""

    def test_t36_trajectory_tokens_dtype_long(self) -> None:
        """T36: ChessBatch trajectory_tokens dtype is torch.long."""
        from chess_sim.data.dataset import ChessDataset
        examples = make_training_examples(3)
        ds = ChessDataset(examples)
        item = ds[0]
        self.assertEqual(
            item.trajectory_tokens.dtype, torch.long
        )

    def test_t37_trajectory_tokens_values_in_range(self) -> None:
        """T37: trajectory_tokens values are in [0, 4]."""
        from chess_sim.data.dataset import ChessDataset
        examples = make_training_examples(3)
        ds = ChessDataset(examples)
        item = ds[0]
        self.assertTrue((item.trajectory_tokens >= 0).all())
        self.assertTrue((item.trajectory_tokens <= 4).all())


# ------------------------------------------------------------------
# T38: evaluate_step accepts trajectory_tokens
# ------------------------------------------------------------------

class TestEvaluateStepTrajectory(unittest.TestCase):
    """T38: evaluate_step returns finite StepResult with trajectory."""

    def test_t38_evaluate_step_with_trajectory(self) -> None:
        """T38: evaluate_step accepts trajectory_tokens in example."""
        from scripts.evaluate import StepResult, evaluate_step
        torch.manual_seed(42)
        encoder = ChessEncoder().to("cpu")
        from chess_sim.model.heads import PredictionHeads
        heads = PredictionHeads().to("cpu")
        encoder.eval()
        heads.eval()
        ex = TrainingExample(
            board_tokens=[0] + [1] * 64,
            color_tokens=[0] + [0] * 64,
            trajectory_tokens=[0] * 65,
            src_sq=12,
            tgt_sq=28,
        )
        result = evaluate_step(
            example=ex,
            move_uci="e2e4",
            ply=0,
            encoder=encoder,
            heads=heads,
            device="cpu",
        )
        self.assertIsInstance(result, StepResult)
        self.assertTrue(math.isfinite(result.total_loss))


# ------------------------------------------------------------------
# T39-T40: Structural subtyping checks
# ------------------------------------------------------------------

class TestStructuralSubtyping(unittest.TestCase):
    """T39-T40: Protocol structural subtyping checks."""

    def test_t39_embeddable_3arg(self) -> None:
        """T39: EmbeddingLayer satisfies Embeddable with 3-arg."""
        layer = EmbeddingLayer()
        self.assertIsInstance(layer, Embeddable)

    def test_t40_encodable_3arg(self) -> None:
        """T40: ChessEncoder satisfies Encodable with 3-arg."""
        encoder = ChessEncoder()
        self.assertIsInstance(encoder, Encodable)


if __name__ == "__main__":
    unittest.main()
