"""Tests for EmbeddingLayer: T04."""

from __future__ import annotations

import unittest

import torch

from chess_sim.model.embedding import D_MODEL, EmbeddingLayer


class TestEmbeddingLayerShape(unittest.TestCase):
    """Tests for EmbeddingLayer output shape and basic properties (T04)."""

    def setUp(self) -> None:
        self.layer = EmbeddingLayer()
        self.layer.eval()
        self.batch_size = 4
        self.seq_len = 65

    def _make_tokens(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return random valid board, color, and activity tokens."""
        board_tokens = torch.randint(
            0, 8, (self.batch_size, self.seq_len)
        )
        color_tokens = torch.randint(
            0, 3, (self.batch_size, self.seq_len)
        )
        activity_tokens = torch.zeros(
            self.batch_size, self.seq_len, dtype=torch.long
        )
        return board_tokens, color_tokens, activity_tokens

    def test_embed_output_shape(self) -> None:
        """T04: embed() output shape is (B, 65, 256)."""
        bt, ct, at = self._make_tokens()
        with torch.no_grad():
            out = self.layer.embed(bt, ct, at)
        self.assertEqual(
            out.shape,
            (self.batch_size, self.seq_len, D_MODEL),
        )

    def test_embed_output_dtype_float(self) -> None:
        """T04: embed() returns a float tensor (not long)."""
        bt, ct, at = self._make_tokens()
        with torch.no_grad():
            out = self.layer.embed(bt, ct, at)
        self.assertTrue(out.is_floating_point())

    def test_different_piece_types_produce_different_embeddings(self) -> None:
        """T04: Different piece-type tokens produce different output."""
        board_a = torch.zeros(1, self.seq_len, dtype=torch.long)
        board_b = torch.ones(1, self.seq_len, dtype=torch.long)
        color = torch.zeros(1, self.seq_len, dtype=torch.long)
        act = torch.zeros(1, self.seq_len, dtype=torch.long)
        with torch.no_grad():
            out_a = self.layer.embed(board_a, color, act)
            out_b = self.layer.embed(board_b, color, act)
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_forward_delegates_to_embed(self) -> None:
        """T04: forward() produces the same output as embed()."""
        bt, ct, at = self._make_tokens()
        with torch.no_grad():
            out_embed = self.layer.embed(bt, ct, at)
            out_forward = self.layer.forward(bt, ct, at)
        self.assertTrue(torch.allclose(out_embed, out_forward))


if __name__ == "__main__":
    unittest.main()
