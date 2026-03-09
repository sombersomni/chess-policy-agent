"""RLPGNReader: stream chess.pgn.Game objects for RL preprocessing.

Thin wrapper over StreamingPGNReader that also supports plain .pgn
files. Composes rather than duplicates .zst decompression logic.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chess.pgn


class RLPGNReader:
    """Stream chess.pgn.Game objects from .pgn or .pgn.zst files.

    Delegates .zst decompression to StreamingPGNReader; handles plain
    .pgn directly with chess.pgn.read_game.

    Example:
        >>> reader = RLPGNReader()
        >>> for game in reader.stream(Path("data/games.pgn"), max_games=10):
        ...     print(game.headers["Result"])
    """

    def stream(
        self,
        path: Path,
        max_games: int = 0,
    ) -> Iterator[chess.pgn.Game]:
        """Yield chess.pgn.Game objects from a PGN file.

        Args:
            path: Path to .pgn or .pgn.zst file.
            max_games: Maximum games to yield. 0 means all.

        Yields:
            chess.pgn.Game objects in file order.

        Raises:
            FileNotFoundError: If path does not exist.

        Example:
            >>> list(RLPGNReader().stream(Path("x.pgn"), max_games=1))
            [<chess.pgn.Game at ...>]
        """
        raise NotImplementedError("To be implemented")
