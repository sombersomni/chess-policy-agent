"""LossComputer: computes the combined cross-entropy loss across both heads.

Loss formula:
  loss = CE(src_sq) + CE(tgt_sq)

Both heads have equal weight (1.0). Each ply is always predicted from the
side-to-move's perspective, so no ignore_index handling is needed.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from chess_sim.types import LabelTensors, PredictionOutput


class LossComputer:
    """Computes the sum of two cross-entropy losses, one per prediction head.

    Equal loss weights (1.0 each) are used. No per-head weighting in v1.

    Example:
        >>> lc = LossComputer()
        >>> loss = lc.compute(predictions, labels)
        >>> loss.item()
        8.317...  # sum of 2 CE losses near log(64) ~ 4.16 each at random init
    """

    def __init__(self) -> None:
        """Initialize CrossEntropyLoss instance.

        Example:
            >>> lc = LossComputer()
        """
        self._ce = nn.CrossEntropyLoss()

    def compute(self, predictions: PredictionOutput, labels: LabelTensors) -> Tensor:
        """Compute and return the summed cross-entropy loss across both heads.

        Each head's CE loss is computed independently. The total is the unweighted
        sum: CE(src_sq) + CE(tgt_sq).

        Args:
            predictions: PredictionOutput with two [B, 64] logit tensors.
            labels: LabelTensors with two [B] integer label tensors.

        Returns:
            Scalar tensor representing the total loss. Supports .backward().

        Example:
            >>> loss = lc.compute(preds, labels)
            >>> loss.backward()
        """
        return (
            self._ce(predictions.src_sq_logits, labels.src_sq)
            + self._ce(predictions.tgt_sq_logits, labels.tgt_sq)
        )
