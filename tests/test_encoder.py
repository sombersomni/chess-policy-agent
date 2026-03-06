"""Tests for ChessEncoder: T05, T09."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch

from chess_sim.model.embedding import D_MODEL
from chess_sim.model.encoder import ChessEncoder
from chess_sim.types import EncoderOutput
from tests.utils import make_synthetic_batch


class TestChessEncoderOutputShape(unittest.TestCase):
    """Tests for ChessEncoder output shapes (T05)."""

    def setUp(self) -> None:
        self.encoder = ChessEncoder()
        self.encoder.eval()
        self.batch_size = 4

    def _make_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        board_tokens = torch.randint(0, 8, (self.batch_size, 65))
        color_tokens = torch.randint(0, 3, (self.batch_size, 65))
        return board_tokens, color_tokens

    def test_encode_cls_embedding_shape(self) -> None:
        """T05: cls_embedding shape is (B, 256)."""
        board_tokens, color_tokens = self._make_tokens()
        with torch.no_grad():
            out = self.encoder.encode(board_tokens, color_tokens)
        self.assertEqual(out.cls_embedding.shape, (self.batch_size, D_MODEL))

    def test_encode_square_embeddings_shape(self) -> None:
        """T05: square_embeddings shape is (B, 64, 256)."""
        board_tokens, color_tokens = self._make_tokens()
        with torch.no_grad():
            out = self.encoder.encode(board_tokens, color_tokens)
        self.assertEqual(out.square_embeddings.shape, (self.batch_size, 64, D_MODEL))

    def test_encode_returns_encoder_output(self) -> None:
        """T05: encode() returns an EncoderOutput namedtuple."""
        board_tokens, color_tokens = self._make_tokens()
        with torch.no_grad():
            out = self.encoder.encode(board_tokens, color_tokens)
        self.assertIsInstance(out, EncoderOutput)


class TestChessEncoderGradientFlow(unittest.TestCase):
    """Tests for gradient flow through ChessEncoder (T09)."""

    def setUp(self) -> None:
        self.encoder = ChessEncoder()

    def test_all_parameters_receive_gradients(self) -> None:
        """T09: After backward, all encoder parameters have non-None, non-zero gradients."""
        board_tokens = torch.randint(0, 8, (2, 65))
        color_tokens = torch.randint(0, 3, (2, 65))
        out = self.encoder.encode(board_tokens, color_tokens)
        # Use the CLS embedding sum as a scalar loss proxy.
        loss = out.cls_embedding.sum()
        loss.backward()
        for name, param in self.encoder.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(param.grad, msg=f"{name} has no gradient")
                self.assertFalse(
                    param.grad.eq(0).all(),
                    msg=f"{name} gradient is all zeros",
                )


if __name__ == "__main__":
    unittest.main()
