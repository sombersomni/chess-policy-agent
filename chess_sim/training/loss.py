"""LossComputer: computes the combined cross-entropy loss across all four heads.

Loss formula:
  loss = CE(src_sq) + CE(tgt_sq) + CE(opp_src_sq) + CE(opp_tgt_sq)

All four heads have equal weight (1.0). The opponent heads use ignore_index=-1
so that terminal-move examples (where no opponent response exists) are skipped.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.types import LabelTensors, PredictionOutput

# Sentinel value used in labels when no opponent move follows (last move in game).
IGNORE_INDEX: int = -1


class LossComputer:
    """Computes the sum of four cross-entropy losses, one per prediction head.

    The two opponent heads (opp_src_sq, opp_tgt_sq) use ignore_index=-1 so that
    terminal board states — where there is no subsequent opponent move — do not
    contribute to the gradient.

    Equal loss weights (1.0 each) are used. No per-head weighting in v1.

    Example:
        >>> lc = LossComputer()
        >>> loss = lc.compute(predictions, labels)
        >>> loss.item()
        8.317...  # sum of 4 CE losses near log(64) ≈ 4.16 each at random init
    """

    def __init__(self) -> None:
        """Initialize four CrossEntropyLoss instances.

        Player heads use the default ignore_index=-100 (no valid label is -100).
        Opponent heads use ignore_index=-1 to skip terminal-move examples.

        Example:
            >>> lc = LossComputer()
        """
        raise NotImplementedError("To be implemented")

    def compute(self, predictions: PredictionOutput, labels: LabelTensors) -> Tensor:
        """Compute and return the summed cross-entropy loss across all four heads.

        Each head's CE loss is computed independently. The total is the unweighted
        sum: CE(src_sq) + CE(tgt_sq) + CE(opp_src_sq) + CE(opp_tgt_sq).

        Args:
            predictions: PredictionOutput with four [B, 64] logit tensors.
            labels: LabelTensors with four [B] integer label tensors.
                    Opponent labels may contain -1 (ignore_index).

        Returns:
            Scalar tensor representing the total loss. Supports .backward().

        Example:
            >>> loss = lc.compute(preds, labels)
            >>> loss.backward()
        """
        raise NotImplementedError("To be implemented")
        return torch.tensor(0.0, requires_grad=True)
