"""HDF5Preprocessor: orchestrates PGN-to-HDF5 conversion.

Streams games from StreamingPGNReader, fans out parsing to a
multiprocessing pool, collects results in the main process,
and writes to HDF5 via HDF5Writer.
"""

from __future__ import annotations

import io
import logging
import multiprocessing
from pathlib import Path
from typing import Iterator

import chess.pgn

from chess_sim.config import PreprocessV2Config
from chess_sim.data.reader import StreamingPGNReader
from chess_sim.preprocess.parse import GameParser
from chess_sim.preprocess.validate import HDF5Validator
from chess_sim.preprocess.writer import HDF5Writer
from chess_sim.types import RawTurnRecord

logger = logging.getLogger(__name__)


def _parse_worker(
    args: tuple[str, int, int, int, bool, int],
) -> list[RawTurnRecord]:
    """Module-level worker function for multiprocessing.

    Creates a fresh GameParser per call (stateless, cheap).
    Parses PGN text into RawTurnRecords.

    Args:
        args: Tuple of (pgn_text, game_id, min_moves,
              max_moves, winners_only, min_elo).

    Returns:
        List of RawTurnRecord for one game.
    """
    (pgn_text, game_id, min_moves,
     max_moves, winners_only, min_elo) = args
    parser = GameParser(
        min_moves, max_moves, winners_only, min_elo
    )
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return []
    return parser.parse_game(game, game_id)


def _game_to_pgn_text(game: chess.pgn.Game) -> str:
    """Serialize a game to PGN text for pickling."""
    exporter = chess.pgn.StringExporter(
        headers=True, variations=False, comments=False,
    )
    return game.accept(exporter)


class HDF5Preprocessor:
    """Orchestrates end-to-end PGN-to-HDF5 preprocessing.

    Implements the HDF5Preprocessable protocol. All dependencies
    are injected via constructor.

    Attributes:
        _reader: PGN streaming reader.
        _parser: Game parser (used for config extraction only).
        _writer: HDF5 file writer.
        _validator: Post-write validator.
    """

    def __init__(
        self,
        reader: StreamingPGNReader,
        parser: GameParser,
        writer: HDF5Writer,
        validator: HDF5Validator,
    ) -> None:
        """Initialize with injected dependencies.

        Args:
            reader: StreamingPGNReader for PGN iteration.
            parser: GameParser for filter config extraction.
            writer: HDF5Writer for output.
            validator: HDF5Validator for post-write checks.
        """
        self._reader = reader
        self._parser = parser
        self._writer = writer
        self._validator = validator

    def run(self, config: PreprocessV2Config) -> None:
        """Execute the full preprocessing pipeline.

        Streams games, parses in parallel, writes HDF5,
        then validates.

        Args:
            config: Full preprocessing configuration.
        """
        pgn_path = Path(config.input.pgn_path)
        hdf5_path = Path(config.output.hdf5_path)
        max_games = config.input.max_games
        workers = config.processing.workers
        train_pct = round(config.split.train * 100)

        # Filter config for workers
        min_moves = config.filter.min_moves
        max_moves = config.filter.max_moves
        winners_only = config.filter.winners_only
        min_elo = config.filter.min_elo

        self._writer.open(hdf5_path)

        game_count = 0
        total_records = 0

        if workers > 1:
            self._run_parallel(
                pgn_path, max_games, workers,
                min_moves, max_moves, winners_only,
                min_elo, train_pct,
            )
        else:
            self._run_serial(
                pgn_path, max_games,
                min_moves, max_moves, winners_only,
                min_elo, train_pct,
            )

        self._writer.close()
        logger.info("Preprocessing complete: %s", hdf5_path)

        # Validate output
        self._validator.validate(hdf5_path, config)

    def _stream_games(
        self, pgn_path: Path
    ) -> Iterator[chess.pgn.Game]:
        """Stream games handling both .zst and plain .pgn."""
        if pgn_path.suffix == ".zst":
            yield from self._reader.stream(pgn_path)
        else:
            # Plain PGN: read directly
            with open(
                pgn_path, encoding="utf-8",
                errors="replace",
            ) as fh:
                while True:
                    game = chess.pgn.read_game(fh)
                    if game is None:
                        break
                    yield game

    def _run_serial(
        self,
        pgn_path: Path,
        max_games: int,
        min_moves: int,
        max_moves: int,
        winners_only: bool,
        min_elo: int,
        train_pct: int,
    ) -> None:
        """Process games in the main process (no pool)."""
        game_count = 0
        for game in self._stream_games(pgn_path):
            if max_games > 0 and game_count >= max_games:
                break

            records = self._parser.parse_game(
                game, game_count
            )
            if records:
                split = (
                    "train"
                    if game_count % 100 < train_pct
                    else "val"
                )
                self._writer.write_batch(records, split)

            game_count += 1
            if game_count % 10_000 == 0:
                logger.info(
                    "Processed %d games", game_count
                )

        logger.info("Total games processed: %d", game_count)

    def _run_parallel(
        self,
        pgn_path: Path,
        max_games: int,
        workers: int,
        min_moves: int,
        max_moves: int,
        winners_only: bool,
        min_elo: int,
        train_pct: int,
    ) -> None:
        """Process games with a multiprocessing pool."""
        def task_iter() -> Iterator[
            tuple[str, int, int, int, bool, int]
        ]:
            gid = 0
            for game in self._stream_games(pgn_path):
                if max_games > 0 and gid >= max_games:
                    break
                pgn_text = _game_to_pgn_text(game)
                yield (
                    pgn_text, gid, min_moves,
                    max_moves, winners_only, min_elo,
                )
                gid += 1

        game_count = 0
        with multiprocessing.Pool(workers) as pool:
            for records in pool.imap_unordered(
                _parse_worker, task_iter(), chunksize=32
            ):
                if not records:
                    game_count += 1
                    if game_count % 10_000 == 0:
                        logger.info(
                            "Processed %d games",
                            game_count,
                        )
                    continue

                gid = records[0].game_id
                split = (
                    "train"
                    if gid % 100 < train_pct
                    else "val"
                )
                self._writer.write_batch(records, split)
                game_count += 1

                if game_count % 10_000 == 0:
                    logger.info(
                        "Processed %d games", game_count
                    )

        logger.info("Total games processed: %d", game_count)
