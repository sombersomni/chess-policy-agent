"""MoveVocab: enumerates all legal UCI move strings and assigns integer indices.

Special tokens:
  PAD_IDX = 0  — padding token for batched sequences
  SOS_IDX = 1  — start-of-sequence token
  EOS_IDX = 2  — end-of-sequence token

Non-special moves (~1968): all from-to square combinations (a1-h8 x a1-h8,
excluding same-square) plus promotion suffixes (q/r/b/n) for pawn moves
reaching rank 1 or rank 8. Total vocab ~ 1971.
"""

from __future__ import annotations

PAD_IDX: int = 0
SOS_IDX: int = 1
EOS_IDX: int = 2

_SQUARES: list[str] = [
    f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"
]

_PROMOTION_SUFFIXES: list[str] = ["q", "r", "b", "n"]


class MoveVocab:
    """Enumerates all UCI move strings and assigns integer indices.

    Builds the full vocabulary at construction time. All standard UCI
    moves (from-to square combos, excluding same-square) plus promotion
    suffixes for pawn moves to rank 1 or rank 8 are included.

    Attributes:
        _move_to_idx: Mapping from UCI string to integer index.
        _idx_to_move: Mapping from integer index to UCI string.

    Example:
        >>> vocab = MoveVocab()
        >>> vocab.encode("e2e4")
        42  # (actual index depends on enumeration order)
        >>> vocab.decode(vocab.encode("e2e4"))
        'e2e4'
        >>> len(vocab)
        1971
    """

    def __init__(self) -> None:
        """Build the full move vocabulary.

        Enumerates 3 special tokens (PAD, SOS, EOS) followed by all
        standard UCI moves and promotion moves. Populates bidirectional
        lookup dicts.

        Example:
            >>> vocab = MoveVocab()
            >>> len(vocab) > 1900
            True
        """
        raise NotImplementedError("To be implemented")

    def encode(self, uci: str) -> int:
        """Convert a UCI move string to its integer vocabulary index.

        Args:
            uci: UCI move string, e.g. "e2e4" or "e7e8q".

        Returns:
            Integer index in the vocabulary.

        Raises:
            KeyError: If the move string is not in the vocabulary.

        Example:
            >>> vocab = MoveVocab()
            >>> idx = vocab.encode("e2e4")
            >>> isinstance(idx, int)
            True
        """
        raise NotImplementedError("To be implemented")

    def decode(self, idx: int) -> str:
        """Convert an integer vocabulary index back to its UCI move string.

        Args:
            idx: Integer index in the vocabulary.

        Returns:
            UCI move string corresponding to the index.

        Raises:
            KeyError: If the index is not in the vocabulary.

        Example:
            >>> vocab = MoveVocab()
            >>> vocab.decode(3)
            'a1a2'  # (actual value depends on enumeration)
        """
        raise NotImplementedError("To be implemented")

    def __len__(self) -> int:
        """Return the total vocabulary size including special tokens.

        Returns:
            Integer count of all tokens (special + moves).

        Example:
            >>> len(MoveVocab())
            1971
        """
        raise NotImplementedError("To be implemented")

    def __contains__(self, uci: str) -> bool:
        """Check whether a UCI move string is in the vocabulary.

        Args:
            uci: UCI move string to check.

        Returns:
            True if the move exists in the vocabulary.

        Example:
            >>> "e2e4" in MoveVocab()
            True
        """
        raise NotImplementedError("To be implemented")
