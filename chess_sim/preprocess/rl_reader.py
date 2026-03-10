"""RLPGNReader: stream chess.pgn.Game objects for RL preprocessing.

Thin wrapper over StreamingPGNReader that also supports plain .pgn
files. Composes rather than duplicates .zst decompression logic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import chess.pgn

from chess_sim.data.reader import StreamingPGNReader

logger = logging.getLogger(__name__)


class RLPGNReader:
    """Stream chess.pgn.Game objects from .pgn or .pgn.zst files.

    Delegates .zst decompression to StreamingPGNReader; handles
    plain .pgn directly with chess.pgn.read_game.

    Example:
        >>> reader = RLPGNReader()
        >>> for g in reader.stream(Path("data/games.pgn"), 10):
        ...     print(g.headers["Result"])
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
            >>> list(RLPGNReader().stream(Path("x.pgn"), 1))
            [<chess.pgn.Game at ...>]
        """
        if not path.exists():
            raise FileNotFoundError(f"PGN not found: {path}")

        count = 0
        if path.suffix == ".zst":
            for game in StreamingPGNReader().stream(path):
                if max_games > 0 and count >= max_games:
                    break
                yield game
                count += 1
        else:
            with open(
                path, encoding="utf-8", errors="replace"
            ) as fh:
                while True:
                    if max_games > 0 and count >= max_games:
                        break
                    game = chess.pgn.read_game(fh)
                    if game is None:
                        break
                    yield game
                    count += 1

        logger.info("Streamed %d games from %s", count, path)
