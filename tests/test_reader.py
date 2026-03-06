"""Tests for StreamingPGNReader: T12."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import chess.pgn
import zstandard

from chess_sim.data.reader import StreamingPGNReader
from tests.utils import make_small_pgn_fixture


def _write_zst_pgn(pgn_text: str, path: Path) -> None:
    """Compress a PGN string to a .zst file at the given path."""
    cctx = zstandard.ZstdCompressor()
    with open(path, "wb") as f:
        f.write(cctx.compress(pgn_text.encode("utf-8")))


class TestStreamingPGNReader(unittest.TestCase):
    """Tests for StreamingPGNReader.stream() (T12)."""

    def setUp(self) -> None:
        self.reader = StreamingPGNReader()
        self.pgn_text = make_small_pgn_fixture()

    def test_stream_yields_correct_number_of_games(self) -> None:
        """T12: Iterator yields exactly 3 chess.pgn.Game objects from a 3-game fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pgn_path = Path(tmpdir) / "test.pgn.zst"
            _write_zst_pgn(self.pgn_text, pgn_path)
            games = list(self.reader.stream(pgn_path))
        self.assertEqual(len(games), 3)

    def test_stream_yields_game_objects(self) -> None:
        """T12: Each yielded item is a chess.pgn.Game instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pgn_path = Path(tmpdir) / "test.pgn.zst"
            _write_zst_pgn(self.pgn_text, pgn_path)
            for game in self.reader.stream(pgn_path):
                self.assertIsInstance(game, chess.pgn.Game)

    def test_stream_is_lazy(self) -> None:
        """T12: stream() returns an iterator, not a list — memory-efficient."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pgn_path = Path(tmpdir) / "test.pgn.zst"
            _write_zst_pgn(self.pgn_text, pgn_path)
            result = self.reader.stream(pgn_path)
        import types
        self.assertIsInstance(result, types.GeneratorType)


if __name__ == "__main__":
    unittest.main()
