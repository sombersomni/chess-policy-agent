"""Tests for chess_sim.functional -- reusable tensor utilities."""

from __future__ import annotations

import unittest

import torch

from chess_sim.functional import entropy_from_logits, mean_entropy


class TestEntropyFromLogits(unittest.TestCase):
    """Tests for entropy_from_logits."""

    def test_peaked_logits_near_zero(self) -> None:
        """Peaked distribution (one-hot) yields near-zero entropy."""
        logits = torch.zeros(2, 5, 100)
        logits[:, :, 0] = 1e6
        h = entropy_from_logits(logits)
        self.assertEqual(h.shape, (2, 5))
        self.assertLess(h.max().item(), 0.001)

    def test_uniform_logits_max_entropy(self) -> None:
        """Uniform distribution yields ln(V) entropy."""
        import math
        V = 50
        logits = torch.zeros(1, 3, V)
        h = entropy_from_logits(logits)
        expected = math.log(V)
        for val in h.flatten().tolist():
            self.assertAlmostEqual(val, expected, places=4)

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape minus last dim."""
        logits = torch.randn(4, 8, 200)
        h = entropy_from_logits(logits)
        self.assertEqual(h.shape, (4, 8))

    def test_1d_input(self) -> None:
        """Works with a single logit vector (V,)."""
        logits = torch.randn(100)
        h = entropy_from_logits(logits)
        self.assertEqual(h.shape, ())
        self.assertGreater(h.item(), 0.0)

    def test_non_negative(self) -> None:
        """Entropy is always non-negative."""
        logits = torch.randn(3, 7, 50)
        h = entropy_from_logits(logits)
        self.assertTrue((h >= 0).all())


class TestMeanEntropy(unittest.TestCase):
    """Tests for mean_entropy."""

    def test_all_masked_returns_zero(self) -> None:
        """Empty mask returns 0.0."""
        logits = torch.randn(2, 5, 100)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        self.assertEqual(mean_entropy(logits, mask), 0.0)

    def test_full_mask_matches_global_mean(self) -> None:
        """Full mask: result equals mean of all positions."""
        logits = torch.randn(2, 5, 100)
        mask = torch.ones(2, 5, dtype=torch.bool)
        h = mean_entropy(logits, mask)
        expected = entropy_from_logits(logits).mean().item()
        self.assertAlmostEqual(h, expected, places=5)

    def test_partial_mask(self) -> None:
        """Partial mask only averages over True positions."""
        logits = torch.randn(2, 4, 50)
        mask = torch.tensor([
            [True, True, False, False],
            [True, False, False, False],
        ])
        h = mean_entropy(logits, mask)
        all_h = entropy_from_logits(logits)
        expected = (
            all_h[0, 0] + all_h[0, 1] + all_h[1, 0]
        ).item() / 3.0
        self.assertAlmostEqual(h, expected, places=5)

    def test_returns_float(self) -> None:
        """Return type is Python float."""
        logits = torch.randn(1, 3, 20)
        mask = torch.ones(1, 3, dtype=torch.bool)
        self.assertIsInstance(
            mean_entropy(logits, mask), float
        )


if __name__ == "__main__":
    unittest.main()
