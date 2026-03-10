"""Value heads for chess model critic components.

ActionConditionedValueHead: Q-value critic via concat fusion MLP.
ValueHeads: legacy Phase 2 dual-head (win + surprise) critic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.types import ValueHeadOutput


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
        super().__init__()
        self._fc1 = nn.Linear(2 * d_model, d_model // 2)
        self._relu = nn.ReLU()
        self._fc2 = nn.Linear(d_model // 2, 1)

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
        fused = torch.cat([cls_emb, action_emb], dim=-1)
        return self._fc2(self._relu(self._fc1(fused)))


class ValueHeads(nn.Module):
    """Legacy Phase 2 dual-head critic: win + surprise on CLS embedding.

    V_win predicts probability of winning from the current state.
    V_surprise predicts the expected surprise score. Both heads
    receive a detached CLS embedding so value gradients do not
    distort the shared encoder representations.
    """

    def __init__(self, d_model: int) -> None:
        """Initialise win and surprise projection heads.

        Args:
            d_model: Dimensionality of the encoder CLS embedding.
        """
        super().__init__()
        self._win_head = nn.Linear(d_model, 1)
        self._surprise_head = nn.Linear(d_model, 1)

    def forward(self, cls_emb: Tensor) -> ValueHeadOutput:
        """Project detached CLS embedding through both heads.

        Args:
            cls_emb: [B, d_model] CLS embedding, must be detached.

        Returns:
            ValueHeadOutput with v_win and v_surprise each [B, 1].
        """
        v_win = self._win_head(cls_emb)
        v_surprise = self._surprise_head(cls_emb)
        return ValueHeadOutput(
            v_win=v_win, v_surprise=v_surprise
        )
