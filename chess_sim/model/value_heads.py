"""Action-conditioned Q-value critic head: two-layer MLP on concat fusion.

Estimates Q(S_t, A_teacher_t) from detached board CLS embedding and
detached teacher action embedding. Replaces the prior ReturnValueHead.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ActionConditionedValueHead(nn.Module):
    """Two-layer MLP Q-critic: concat([cls, action]) -> 2*d_model -> d_model//2 -> 1.

    Caller must detach both cls_emb and action_emb before passing.
    This head performs no detach internally — keeping it testable
    with attached tensors when gradient flow tests require it.

    Internal layers:
        _fc1:  nn.Linear(2 * d_model, d_model // 2)
        _relu: nn.ReLU()
        _fc2:  nn.Linear(d_model // 2, 1)

    Example:
        >>> head = ActionConditionedValueHead(128)
        >>> q = head(torch.rand(4, 128), torch.rand(4, 128))
        >>> q.shape
        torch.Size([4, 1])
    """

    def __init__(self, d_model: int) -> None:
        """Initialize the two-layer MLP for Q-value estimation.

        Args:
            d_model: Embedding dimension (same as encoder/decoder).

        Example:
            >>> head = ActionConditionedValueHead(128)
        """
        raise NotImplementedError("To be implemented")

    def forward(
        self, cls_emb: Tensor, action_emb: Tensor
    ) -> Tensor:
        """Estimate Q-value from board state and action embeddings.

        Concatenates cls_emb and action_emb along the last dim,
        then projects through _fc1 -> _relu -> _fc2 to produce
        a scalar Q-value per batch element.

        Args:
            cls_emb: [B, d_model] CLS embedding, caller must detach.
            action_emb: [B, d_model] action embedding, caller must
                detach.

        Returns:
            [B, 1] scalar Q-value estimate per batch element.

        Example:
            >>> head = ActionConditionedValueHead(128)
            >>> q = head(torch.rand(4, 128), torch.rand(4, 128))
            >>> q.shape
            torch.Size([4, 1])
        """
        raise NotImplementedError("To be implemented")
