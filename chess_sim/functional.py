"""Reusable tensor operations for chess-sim, following torch.nn.functional.

Provides pure-function utilities that operate on raw tensors with no
module state. Each function is type-hinted, documented, and tested.
"""

from __future__ import annotations

from torch import Tensor
from torch.distributions import Categorical


def entropy_from_logits(logits: Tensor) -> Tensor:
    """Compute per-position Shannon entropy from raw logits.

    Uses torch.distributions.Categorical for numerically stable
    computation: H = -sum(p * log(p)) in nats.

    Args:
        logits: Raw logits of shape (..., V) where V is vocab size.

    Returns:
        Entropy tensor of shape (...), one scalar per position.

    Example:
        >>> h = entropy_from_logits(torch.randn(2, 10, 100))
        >>> h.shape
        torch.Size([2, 10])
    """
    return Categorical(logits=logits).entropy()


def mean_entropy(
    logits: Tensor, mask: Tensor
) -> float:
    """Compute mean prediction entropy over masked positions.

    Args:
        logits: Shape (B, T, V) raw logits.
        mask: Boolean mask (B, T); True = valid position.

    Returns:
        Mean entropy in nats as a Python float. Returns 0.0
        when mask has no True entries.

    Example:
        >>> logits = torch.randn(2, 10, 100)
        >>> mask = torch.ones(2, 10, dtype=torch.bool)
        >>> h = mean_entropy(logits, mask)
        >>> isinstance(h, float)
        True
    """
    if not mask.any():
        return 0.0
    h = entropy_from_logits(logits)  # (B, T)
    return h[mask].mean().item()
