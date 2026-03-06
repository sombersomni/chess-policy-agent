"""Tests for PredictionHeads: T06."""

from __future__ import annotations

import unittest

import torch

from chess_sim.model.heads import N_SQUARES, PredictionHeads
from chess_sim.types import PredictionOutput


class TestPredictionHeadsOutputShape(unittest.TestCase):
    """Tests for PredictionHeads output shapes and independence (T06)."""

    def setUp(self) -> None:
        self.heads = PredictionHeads()
        self.heads.eval()
        self.batch_size = 4
        self.cls_dim = 256

    def _make_cls(self) -> torch.Tensor:
        return torch.randn(self.batch_size, self.cls_dim)

    def test_predict_returns_prediction_output(self) -> None:
        """T06: predict() returns a PredictionOutput namedtuple."""
        with torch.no_grad():
            out = self.heads.predict(self._make_cls())
        self.assertIsInstance(out, PredictionOutput)

    def test_src_sq_logits_shape(self) -> None:
        """T06: src_sq_logits shape is (B, 64)."""
        with torch.no_grad():
            out = self.heads.predict(self._make_cls())
        self.assertEqual(out.src_sq_logits.shape, (self.batch_size, N_SQUARES))

    def test_tgt_sq_logits_shape(self) -> None:
        """T06: tgt_sq_logits shape is (B, 64)."""
        with torch.no_grad():
            out = self.heads.predict(self._make_cls())
        self.assertEqual(out.tgt_sq_logits.shape, (self.batch_size, N_SQUARES))

    def test_opp_src_sq_logits_shape(self) -> None:
        """T06: opp_src_sq_logits shape is (B, 64)."""
        with torch.no_grad():
            out = self.heads.predict(self._make_cls())
        self.assertEqual(out.opp_src_sq_logits.shape, (self.batch_size, N_SQUARES))

    def test_opp_tgt_sq_logits_shape(self) -> None:
        """T06: opp_tgt_sq_logits shape is (B, 64)."""
        with torch.no_grad():
            out = self.heads.predict(self._make_cls())
        self.assertEqual(out.opp_tgt_sq_logits.shape, (self.batch_size, N_SQUARES))

    def test_heads_produce_independent_gradients(self) -> None:
        """T06: Each head's loss contributes a gradient only through its own parameters."""
        cls_emb = self._make_cls().requires_grad_(True)
        out = self.heads.predict(cls_emb)
        # Backward through only src_sq_head.
        out.src_sq_logits.sum().backward(retain_graph=True)
        src_grad = cls_emb.grad.clone()
        cls_emb.grad.zero_()
        # Backward through only tgt_sq_head.
        out.tgt_sq_logits.sum().backward()
        tgt_grad = cls_emb.grad.clone()
        # Both heads receive gradient from cls_emb, but they're independent linear layers.
        self.assertFalse(torch.allclose(src_grad, tgt_grad))


if __name__ == "__main__":
    unittest.main()
