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
        import chess

        self._encode: dict[str, int] = {}
        self._decode: dict[int, str] = {}
        # Reserve special tokens at indices 0, 1, 2.
        idx = 3
        moves: list[str] = []
        for sq in range(64):
            rank, file = sq // 8, sq % 8
            moves.extend(
                self._piece_moves(sq, rank, file)
            )
        # Deduplicate preserving order, assign indices.
        seen: set[str] = set()
        for uci in moves:
            if uci not in seen:
                seen.add(uci)
                self._encode[uci] = idx
                self._decode[idx] = uci
                idx += 1

    @staticmethod
    def _piece_moves(
        sq: int, rank: int, file: int
    ) -> list[str]:
        """Enumerate all piece-reachable UCI strings from sq."""
        import chess

        result: list[str] = []
        promos = [
            chess.QUEEN, chess.ROOK,
            chess.BISHOP, chess.KNIGHT,
        ]

        def _add(to: int, promo: bool = False) -> None:
            if promo:
                for p in promos:
                    result.append(
                        chess.Move(sq, to, promotion=p).uci()
                    )
            else:
                result.append(chess.Move(sq, to).uci())

        # Knight
        for dr, df in [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1),
        ]:
            r2, f2 = rank + dr, file + df
            if 0 <= r2 < 8 and 0 <= f2 < 8:
                _add(r2 * 8 + f2)

        # King (1-square moves)
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                r2, f2 = rank + dr, file + df
                if 0 <= r2 < 8 and 0 <= f2 < 8:
                    _add(r2 * 8 + f2)

        # Rook / Queen (straight lines)
        for d in range(1, 8):
            for dr, df in [(d, 0), (-d, 0), (0, d), (0, -d)]:
                r2, f2 = rank + dr, file + df
                if 0 <= r2 < 8 and 0 <= f2 < 8:
                    _add(r2 * 8 + f2)

        # Bishop / Queen (diagonals)
        for d in range(1, 8):
            for dr, df in [
                (d, d), (d, -d), (-d, d), (-d, -d),
            ]:
                r2, f2 = rank + dr, file + df
                if 0 <= r2 < 8 and 0 <= f2 < 8:
                    _add(r2 * 8 + f2)

        # White pawn (rank increases)
        if rank < 7:
            to_rank = rank + 1
            is_promo = to_rank == 7
            _add(to_rank * 8 + file, promo=is_promo)
            if rank == 1:
                _add((rank + 2) * 8 + file)
            for df in [-1, 1]:
                f2 = file + df
                if 0 <= f2 < 8:
                    _add(to_rank * 8 + f2, promo=is_promo)

        # Black pawn (rank decreases)
        if rank > 0:
            to_rank = rank - 1
            is_promo = to_rank == 0
            _add(to_rank * 8 + file, promo=is_promo)
            if rank == 6:
                _add((rank - 2) * 8 + file)
            for df in [-1, 1]:
                f2 = file + df
                if 0 <= f2 < 8:
                    _add(to_rank * 8 + f2, promo=is_promo)

        return result

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
        return self._encode[uci]

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
        return self._decode[idx]

    def __len__(self) -> int:
        """Return the total vocabulary size including special tokens.

        Returns:
            Integer count of all tokens (special + moves).

        Example:
            >>> len(MoveVocab())
            1971
        """
        return len(self._encode) + 3  # 3 special tokens

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
        return uci in self._encode
