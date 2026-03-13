"""Protocol interfaces for the chess encoder system.

Each Protocol defines one capability. Concrete classes implement these
implicitly via structural subtyping — no explicit `implements` declaration needed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class StructuralMaskable(Protocol):
    """Builds a per-batch structural move mask from color tokens."""

    def build(self, color_tokens: Tensor) -> Tensor:
        """Return a [B, V] bool mask from color_tokens [B, 65].

        True = token's from-square has a player piece.

        Args:
            color_tokens: LongTensor [B, 65] with values
                0=empty, 1=player, 2=opponent.

        Returns:
            BoolTensor [B, V].
        """
        ...
