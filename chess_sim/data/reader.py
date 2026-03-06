"""StreamingPGNReader: streams chess games one at a time from .zst compressed PGN files.

Uses the `zstandard` library for streaming decompression to avoid loading the
full file into memory. Yields chess.pgn.Game objects via an iterator interface.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import chess.pgn
import zstandard


class StreamingPGNReader:
    """Streams games one at a time from a .zst-compressed PGN file.

    Memory usage is O(1) with respect to file size — only one game is held in
    memory at a time. Compatible with ReservoirSampler for uniform random sampling.

    Example:
        >>> reader = StreamingPGNReader()
        >>> for game in reader.stream(Path("lichess_db.pgn.zst")):
        ...     process(game)
    """

    def stream(self, path: Path) -> Iterator[chess.pgn.Game]:
        """Yield chess.pgn.Game objects one at a time from a .zst PGN file.

        Decompresses the .zst stream on-the-fly without buffering the full file.
        Skips None returns from read_game() that indicate parse errors.

        Args:
            path: Filesystem path to a .zst-compressed PGN file.

        Returns:
            Iterator of chess.pgn.Game objects.

        Example:
            >>> games = list(reader.stream(Path("small_test.pgn.zst")))
            >>> len(games)
            3
        """
        raise NotImplementedError("To be implemented")
        return iter([])  # satisfies return type for type checkers
