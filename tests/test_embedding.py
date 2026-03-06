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

    def _make_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return random valid board_tokens and color_tokens."""
        board_tokens = torch.randint(0, 8, (self.batch_size, self.seq_len))
        color_tokens = torch.randint(0, 3, (self.batch_size, self.seq_len))
        return board_tokens, color_tokens

    def test_embed_output_shape(self) -> None:
        """T04: embed() output shape is (B, 65, 256)."""
        board_tokens, color_tokens = self._make_tokens()
        with torch.no_grad():
            out = self.layer.embed(board_tokens, color_tokens)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, D_MODEL))

    def test_embed_output_dtype_float(self) -> None:
        """T04: embed() returns a float tensor (not long)."""
        board_tokens, color_tokens = self._make_tokens()
        with torch.no_grad():
            out = self.layer.embed(board_tokens, color_tokens)
        self.assertTrue(out.is_floating_point())

    def test_different_piece_types_produce_different_embeddings(self) -> None:
        """T04: Different piece-type tokens at the same position produce different output."""
        board_a = torch.zeros(1, self.seq_len, dtype=torch.long)
        board_b = torch.ones(1, self.seq_len, dtype=torch.long)
        color = torch.zeros(1, self.seq_len, dtype=torch.long)
        with torch.no_grad():
            out_a = self.layer.embed(board_a, color)
            out_b = self.layer.embed(board_b, color)
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_forward_delegates_to_embed(self) -> None:
        """T04: forward() produces the same output as embed()."""
        board_tokens, color_tokens = self._make_tokens()
        with torch.no_grad():
            out_embed = self.layer.embed(board_tokens, color_tokens)
            out_forward = self.layer.forward(board_tokens, color_tokens)
        self.assertTrue(torch.allclose(out_embed, out_forward))


if __name__ == "__main__":
    unittest.main()
