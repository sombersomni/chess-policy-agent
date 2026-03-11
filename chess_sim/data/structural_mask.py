"""StructuralMaskBuilder: precomputed [65, V] bool LUT for move masking.

Maps each board slot (0=CLS, 1..64=squares) to the set of move vocab
tokens whose UCI from-square corresponds to that slot. During training,
`build(color_tokens)` OR-reduces the LUT rows for player-occupied
squares, producing a [B, V] mask that suppresses structurally impossible
moves (from-square has no player piece).

Implements the StructuralMaskable protocol.
"""

from __future__ import annotations

import torch
from torch import Tensor

from chess_sim.data.move_vocab import MoveVocab

# Files a-h mapped to integer indices 0-7.
_FILE_TO_IDX: dict[str, int] = {
    f: i for i, f in enumerate("abcdefgh")
}


def _uci_from_square_slot(uci: str) -> int:
    """Parse the from-square of a UCI string into a board slot index.

    Slot index formula: (rank - 1) * 8 + file_index + 1.
    Slot 0 is reserved for CLS; slots 1..64 map to a1..h8.

    Args:
        uci: UCI move string, e.g. "e2e4" or "e7e8q".

    Returns:
        Integer slot index in [1, 64].

    Example:
        >>> _uci_from_square_slot("e2e4")
        13
    """
    file_char = uci[0]
    rank_char = uci[1]
    file_idx = _FILE_TO_IDX[file_char]
    rank = int(rank_char)  # 1-indexed
    return (rank - 1) * 8 + file_idx + 1


class StructuralMaskBuilder:
    """Precomputed structural mask for decoder move logits.

    Holds a static [65, V] bool LUT built once from MoveVocab.
    `build(color_tokens)` returns a [B, V] bool mask where True
    means the token's from-square has a player piece.

    Implements the StructuralMaskable protocol.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Build the static slot-to-token LUT from MoveVocab.

        Args:
            device: Target device for the LUT tensor.

        Example:
            >>> builder = StructuralMaskBuilder()
            >>> builder._slot_mask.shape
            torch.Size([65, 1971])
        """
        vocab = MoveVocab()
        v = len(vocab)  # 1971
        # [65, V] bool: row i is True for tokens from slot i.
        slot_mask = torch.zeros(65, v, dtype=torch.bool)

        # Iterate non-special tokens (indices 3..V-1).
        for idx in range(3, v):
            uci = vocab.decode(idx)
            slot = _uci_from_square_slot(uci)
            slot_mask[slot, idx] = True

        self._slot_mask: Tensor = slot_mask.to(device)

    def build(self, color_tokens: Tensor) -> Tensor:
        """Build a per-batch structural mask from color tokens.

        For each batch item, identifies player-occupied squares
        (color_tokens == 1), looks up their LUT rows, and
        OR-reduces to produce a [B, V] mask.

        Args:
            color_tokens: LongTensor [B, 65] with values
                0=empty, 1=player, 2=opponent.

        Returns:
            BoolTensor [B, V] where True = token's from-square
            has a player piece (structurally valid move).

        Example:
            >>> builder = StructuralMaskBuilder()
            >>> ct = torch.zeros(1, 65, dtype=torch.long)
            >>> ct[0, 13] = 1  # e2
            >>> mask = builder.build(ct)
            >>> mask.shape
            torch.Size([1, 1971])
        """
        # player_slots: [B, 65] bool — True where player piece
        player_slots = (color_tokens == 1).to(
            self._slot_mask.device
        )
        # Matrix multiply: [B, 65] float @ [65, V] float -> [B, V]
        # Any nonzero entry means at least one player slot has
        # a True LUT row for that token.
        mask = torch.matmul(
            player_slots.float(),
            self._slot_mask.float(),
        ) > 0  # [B, V] bool
        return mask
