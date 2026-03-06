"""Tests for LossComputer: T07."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from chess_sim.training.loss import LossComputer
from chess_sim.types import LabelTensors, PredictionOutput
from tests.utils import make_label_tensors, make_prediction_output


class TestLossComputerCorrectness(unittest.TestCase):
    """Tests for LossComputer.compute() correctness against hand-computed CE (T07)."""

    def setUp(self) -> None:
        self.lc = LossComputer()
        self.batch_size = 4

    def test_compute_matches_hand_computed_ce(self) -> None:
        """T07: Total loss equals sum of 2 manually computed CE values within 1e-5."""
        torch.manual_seed(42)
        logits = [torch.randn(self.batch_size, 64) for _ in range(2)]
        labels = [torch.randint(0, 64, (self.batch_size,)) for _ in range(2)]

        preds = PredictionOutput(*logits)
        label_tensors = LabelTensors(*labels)

        expected_loss = sum(
            F.cross_entropy(logits[i], labels[i]) for i in range(2)
        )
        actual_loss = self.lc.compute(preds, label_tensors)

        self.assertAlmostEqual(
            actual_loss.item(), expected_loss.item(), places=4,
            msg="Loss does not match hand-computed cross-entropy sum",
        )

    def test_compute_returns_scalar_tensor(self) -> None:
        """T07: compute() returns a 0-dimensional tensor."""
        preds = make_prediction_output(self.batch_size)
        labels = make_label_tensors(self.batch_size)
        loss = self.lc.compute(preds, labels)
        self.assertEqual(loss.dim(), 0)

    def test_compute_supports_backward(self) -> None:
        """T07: The returned loss tensor supports .backward() without error."""
        preds = make_prediction_output(self.batch_size)
        labels = make_label_tensors(self.batch_size)
        loss = self.lc.compute(preds, labels)
        loss.backward()  # should not raise


if __name__ == "__main__":
    unittest.main()
