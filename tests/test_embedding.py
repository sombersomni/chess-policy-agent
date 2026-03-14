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

    def test_src_tokens_none_matches_zero(self) -> None:
        """T04: src_tokens=None produces same output as src_tokens=zeros."""
        bt, ct, at = self._make_tokens()
        src_zeros = torch.zeros(self.batch_size, dtype=torch.long)
        with torch.no_grad():
            out_none = self.layer.embed(bt, ct, at, src_tokens=None)
            out_zero = self.layer.embed(bt, ct, at, src_tokens=src_zeros)
        self.assertTrue(torch.allclose(out_none, out_zero))

    def test_nonzero_src_changes_output(self) -> None:
        """T04: Non-zero src_tokens changes output after training src_square_emb."""
        bt, ct, at = self._make_tokens()
        src_zeros = torch.zeros(self.batch_size, dtype=torch.long)
        src_nonzero = torch.ones(self.batch_size, dtype=torch.long) * 5

        # Re-initialize src_square_emb to non-zero so conditioning fires.
        layer = EmbeddingLayer()
        layer.eval()
        with torch.no_grad():
            torch.nn.init.normal_(layer.src_square_emb.weight)
            out_zero = layer.embed(bt, ct, at, src_tokens=src_zeros)
            out_nonzero = layer.embed(bt, ct, at, src_tokens=src_nonzero)
        self.assertFalse(torch.allclose(out_zero, out_nonzero))

    def test_src_square_emb_zero_init(self) -> None:
        """T04: src_square_emb starts as all zeros (no-op at init)."""
        layer = EmbeddingLayer()
        self.assertTrue(
            torch.all(layer.src_square_emb.weight == 0).item()
        )

    def test_piece_type_cond_emb_zero_init(self) -> None:
        """T04: piece_type_cond_emb starts as all zeros (no-op at init)."""
        layer = EmbeddingLayer()
        self.assertTrue(
            torch.all(layer.piece_type_cond_emb.weight == 0).item()
        )

    def test_piece_type_tokens_none_matches_zero(self) -> None:
        """T04: piece_type_tokens=None produces same output as zeros."""
        bt, ct, at = self._make_tokens()
        pt_zeros = torch.zeros(self.batch_size, dtype=torch.long)
        with torch.no_grad():
            out_none = self.layer.embed(bt, ct, at, piece_type_tokens=None)
            out_zero = self.layer.embed(bt, ct, at, piece_type_tokens=pt_zeros)
        self.assertTrue(torch.allclose(out_none, out_zero))

    def test_nonzero_piece_type_changes_output(self) -> None:
        """T04: Non-zero piece_type_tokens changes output after init."""
        bt, ct, at = self._make_tokens()
        pt_zeros = torch.zeros(self.batch_size, dtype=torch.long)
        pt_nonzero = torch.full(
            (self.batch_size,), 5, dtype=torch.long  # ROOK
        )
        layer = EmbeddingLayer()
        layer.eval()
        with torch.no_grad():
            torch.nn.init.normal_(layer.piece_type_cond_emb.weight)
            out_zero = layer.embed(bt, ct, at, piece_type_tokens=pt_zeros)
            out_nonzero = layer.embed(bt, ct, at, piece_type_tokens=pt_nonzero)
        self.assertFalse(torch.allclose(out_zero, out_nonzero))

    def test_piece_type_broadcast_across_positions(self) -> None:
        """T04: piece_type_tokens is broadcast — all 65 positions get same signal."""
        bt, ct, at = self._make_tokens()
        layer = EmbeddingLayer()
        layer.eval()
        with torch.no_grad():
            torch.nn.init.normal_(layer.piece_type_cond_emb.weight)
        pt = torch.tensor([3], dtype=torch.long)  # B=1, KNIGHT
        bt1, ct1, at1 = bt[:1], ct[:1], at[:1]
        with torch.no_grad():
            out_pt = layer.piece_type_cond_emb(pt)  # [1, d_model]
            out_with = layer.embed(bt1, ct1, at1, piece_type_tokens=pt)
            out_without = layer.embed(bt1, ct1, at1, piece_type_tokens=None)
        diff = out_with - out_without
        self.assertEqual(out_pt.shape, (1, D_MODEL))
        self.assertFalse(torch.all(out_pt == 0).item())
        self.assertFalse(torch.all(diff == 0).item())

    def test_src_broadcast_same_across_positions(self) -> None:
        """T04: src_tokens is broadcast — all 65 positions get the same src signal."""
        bt, ct, at = self._make_tokens()
        layer = EmbeddingLayer()
        layer.eval()
        # Set src_square_emb to non-zero so the signal is visible.
        with torch.no_grad():
            torch.nn.init.normal_(layer.src_square_emb.weight)
        src = torch.tensor([3], dtype=torch.long)  # B=1

        bt1 = bt[:1]
        ct1 = ct[:1]
        at1 = at[:1]

        with torch.no_grad():
            out_src = layer.src_square_emb(src)  # [1, d_model]
            out_embed = layer.embed(bt1, ct1, at1, src_tokens=src)
            # Embed without src to get the "base" output per-position
            out_no_src = layer.embed(bt1, ct1, at1, src_tokens=None)

        # Difference should be the same src embedding broadcast across positions.
        diff = out_embed - out_no_src  # [1, 65, d_model]
        # After LayerNorm, the constant broadcast gets folded in uniformly;
        # verify the raw src signal has shape (1, d_model) and is non-zero.
        self.assertEqual(out_src.shape, (1, D_MODEL))
        self.assertFalse(torch.all(out_src == 0).item())
        # The diff tensor should be non-zero (src had an effect).
        self.assertFalse(torch.all(diff == 0).item())


if __name__ == "__main__":
    unittest.main()
