"""Tests for SrcMoveLUT: source-square move filtering."""

from __future__ import annotations

import unittest

import torch

from chess_sim.data.src_move_lut import SrcMoveLUT


class TestSrcMoveLUT(unittest.TestCase):
    """Tests for SrcMoveLUT initialization and filter_legal_mask."""

    def setUp(self) -> None:
        self.lut = SrcMoveLUT(device="cpu")
        self.V = 1971

    # -- LUT shape and properties --

    def test_lut_shape(self) -> None:
        """LUT is [64, 1971] bool tensor."""
        self.assertEqual(self.lut._lut.shape, (64, self.V))
        self.assertEqual(self.lut._lut.dtype, torch.bool)

    def test_lut_has_entries_per_square(self) -> None:
        """Every source square has at least one valid move in the LUT."""
        counts = self.lut._lut.long().sum(dim=1)  # [64]
        self.assertTrue((counts > 0).all().item())

    def test_lut_total_matches_vocab(self) -> None:
        """Total True entries equals number of non-special vocab moves (idx >= 3)."""
        total = self.lut._lut.long().sum().item()
        # Each move idx 3..1970 maps to exactly one src square
        self.assertEqual(total, self.V - 3)

    # -- filter_legal_mask --

    def test_filter_shape(self) -> None:
        """filter_legal_mask returns [B, V] tensor."""
        B = 4
        legal = torch.ones(B, self.V, dtype=torch.bool)
        src = torch.zeros(B, dtype=torch.long)
        out = self.lut.filter_legal_mask(legal, src)
        self.assertEqual(out.shape, (B, self.V))

    def test_filter_dtype_bool(self) -> None:
        """filter_legal_mask output is bool."""
        legal = torch.ones(2, self.V, dtype=torch.bool)
        src = torch.zeros(2, dtype=torch.long)
        out = self.lut.filter_legal_mask(legal, src)
        self.assertEqual(out.dtype, torch.bool)

    def test_filter_reduces_count(self) -> None:
        """Filtering to a specific square produces fewer true entries than all-legal."""
        legal = torch.ones(1, self.V, dtype=torch.bool)
        src = torch.tensor([4], dtype=torch.long)  # e1 (square index 4)
        out = self.lut.filter_legal_mask(legal, src)
        full_count = legal.long().sum().item()
        filtered_count = out.long().sum().item()
        self.assertLess(filtered_count, full_count)

    def test_filter_subset_of_legal(self) -> None:
        """Filtered mask is always a subset of the input legal mask."""
        legal = torch.zeros(2, self.V, dtype=torch.bool)
        # Allow only the first 100 moves.
        legal[:, :100] = True
        src = torch.tensor([0, 10], dtype=torch.long)
        out = self.lut.filter_legal_mask(legal, src)
        # Any True in output must also be True in legal.
        self.assertTrue((out & ~legal).sum().item() == 0)

    def test_filter_all_false_input(self) -> None:
        """All-False legal mask yields all-False output regardless of src."""
        legal = torch.zeros(3, self.V, dtype=torch.bool)
        src = torch.tensor([0, 1, 2], dtype=torch.long)
        out = self.lut.filter_legal_mask(legal, src)
        self.assertFalse(out.any().item())

    def test_filter_only_moves_from_src(self) -> None:
        """Filtered moves all originate from the chosen source square."""
        from chess_sim.data.move_vocab import MoveVocab
        vocab = MoveVocab()

        src_sq = 28  # e4 (0-based)
        legal = torch.ones(1, self.V, dtype=torch.bool)
        src = torch.tensor([src_sq], dtype=torch.long)
        out = self.lut.filter_legal_mask(legal, src)

        for idx in out[0].nonzero(as_tuple=False).squeeze(1).tolist():
            uci = vocab.decode(idx)
            # from-square of the UCI must be e4 (file=e=4, rank=4 → 0-based sq 28)
            from_sq = (ord(uci[0]) - ord("a")) + (int(uci[1]) - 1) * 8
            self.assertEqual(
                from_sq, src_sq,
                f"move {uci} (idx {idx}) not from sq {src_sq}",
            )

    def test_filter_batch_independent(self) -> None:
        """Each batch item is filtered independently by its own src square."""
        legal = torch.ones(2, self.V, dtype=torch.bool)
        src_a = torch.tensor([0], dtype=torch.long)
        src_b = torch.tensor([63], dtype=torch.long)
        src_both = torch.tensor([0, 63], dtype=torch.long)

        out_a = self.lut.filter_legal_mask(legal[:1], src_a)
        out_b = self.lut.filter_legal_mask(legal[1:], src_b)
        out_both = self.lut.filter_legal_mask(legal, src_both)

        self.assertTrue(torch.equal(out_both[0], out_a[0]))
        self.assertTrue(torch.equal(out_both[1], out_b[0]))


if __name__ == "__main__":
    unittest.main()
