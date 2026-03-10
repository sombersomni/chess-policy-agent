"""Tests for ChessEncoder: T05, T09."""

from __future__ import annotations

import unittest

import torch

from chess_sim.model.embedding import D_MODEL
from chess_sim.model.encoder import ChessEncoder
from chess_sim.types import EncoderOutput


class TestChessEncoderOutputShape(unittest.TestCase):
    """Tests for ChessEncoder output shapes (T05)."""

    def setUp(self) -> None:
        self.encoder = ChessEncoder()
        self.encoder.eval()
        self.batch_size = 4

    def _make_tokens(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return board, color, and activity token tensors."""
        bt = torch.randint(0, 8, (self.batch_size, 65))
        ct = torch.randint(0, 3, (self.batch_size, 65))
        at = torch.zeros(
            self.batch_size, 65, dtype=torch.long
        )
        return bt, ct, at

    def test_encode_cls_embedding_shape(self) -> None:
        """T05: cls_embedding shape is (B, 256)."""
        bt, ct, at = self._make_tokens()
        with torch.no_grad():
            out = self.encoder.encode(bt, ct, at)
        self.assertEqual(
            out.cls_embedding.shape,
            (self.batch_size, D_MODEL),
        )

    def test_encode_square_embeddings_shape(self) -> None:
        """T05: square_embeddings shape is (B, 64, 256)."""
        bt, ct, at = self._make_tokens()
        with torch.no_grad():
            out = self.encoder.encode(bt, ct, at)
        self.assertEqual(
            out.square_embeddings.shape,
            (self.batch_size, 64, D_MODEL),
        )

    def test_encode_returns_encoder_output(self) -> None:
        """T05: encode() returns an EncoderOutput namedtuple."""
        bt, ct, at = self._make_tokens()
        with torch.no_grad():
            out = self.encoder.encode(bt, ct, at)
        self.assertIsInstance(out, EncoderOutput)


class TestChessEncoderGradientFlow(unittest.TestCase):
    """Tests for gradient flow through ChessEncoder (T09)."""

    def setUp(self) -> None:
        self.encoder = ChessEncoder()

    def test_all_parameters_receive_gradients(self) -> None:
        """T09: All encoder params have non-None, non-zero grads."""
        bt = torch.randint(0, 8, (2, 65))
        ct = torch.randint(0, 3, (2, 65))
        at = torch.zeros(2, 65, dtype=torch.long)
        out = self.encoder.encode(bt, ct, at)
        loss = out.cls_embedding.sum()
        loss.backward()
        for name, param in self.encoder.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(
                    param.grad,
                    msg=f"{name} has no gradient",
                )
                self.assertFalse(
                    param.grad.eq(0).all(),
                    msg=f"{name} gradient is all zeros",
                )


if __name__ == "__main__":
    unittest.main()
