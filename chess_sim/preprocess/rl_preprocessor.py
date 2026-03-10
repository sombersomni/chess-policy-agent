"""RLHdf5Preprocessor: orchestrate the full RL HDF5 pipeline.

Stream PGN -> parse -> write -> validate. Supports serial and
multiprocessing execution. RL analogue of HDF5Preprocessor.
"""

from __future__ import annotations

import io
import logging
import multiprocessing
from collections.abc import Iterator
from pathlib import Path

import chess.pgn

from chess_sim.config import RLPreprocessConfig
from chess_sim.preprocess.rl_parser import RLPlyParser
from chess_sim.preprocess.rl_reader import RLPGNReader
from chess_sim.preprocess.rl_validator import RLHdf5Validator
from chess_sim.preprocess.rl_writer import RLHdf5Writer
from chess_sim.types import RLPlyRecord

logger = logging.getLogger(__name__)


def _rl_parse_worker(
    args: tuple[str, int, str, int, int],
) -> list[RLPlyRecord]:
    """Module-level worker function for multiprocessing.

    Creates a fresh RLPlyParser per call (stateless, cheap).

    Args:
        args: (pgn_text, game_id, train_color,
               min_moves, max_moves).

    Returns:
        List of RLPlyRecord for one game.
    """
    (
        pgn_text,
        game_id,
        train_color,
        min_moves,
        max_moves,
    ) = args
    parser = RLPlyParser(train_color, min_moves, max_moves)
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


class RLHdf5Preprocessor:
    """Orchestrate the RL PGN-to-HDF5 pipeline.

    Composes RLPGNReader, RLPlyParser, RLHdf5Writer, and
    RLHdf5Validator into a single run() call. Supports parallel
    game parsing via multiprocessing.

    Example:
        >>> preprocessor = RLHdf5Preprocessor(
        ...     reader, parser, writer, validator
        ... )
        >>> preprocessor.run(config)
    """

    def __init__(
        self,
        reader: RLPGNReader,
        parser: RLPlyParser,
        writer: RLHdf5Writer,
        validator: RLHdf5Validator,
    ) -> None:
        """Store pipeline components for later execution.

        Args:
            reader: Streams chess.pgn.Game from PGN files.
            parser: Converts games to RLPlyRecord lists.
            writer: Buffers and flushes records to HDF5.
            validator: Validates the written HDF5 file.

        Example:
            >>> p = RLHdf5Preprocessor(r, pa, w, v)
        """
        self._reader = reader
        self._parser = parser
        self._writer = writer
        self._validator = validator

    def run(self, config: RLPreprocessConfig) -> None:
        """Execute the full preprocessing pipeline.

        Steps: stream games -> parse to RLPlyRecord -> split
        train/val -> write to HDF5 -> validate output file.

        Args:
            config: Full RLPreprocessConfig.

        Raises:
            FileNotFoundError: If PGN file is missing.
            ValueError: If validation fails after writing.

        Example:
            >>> preprocessor.run(config)
        """
        pgn_path = Path(config.input.pgn_path)
        hdf5_path = Path(config.output.hdf5_path)
        max_games = config.input.max_games
        workers = config.processing.workers
        train_pct = round(config.split.train * 100)

        train_color = config.filter.train_color
        min_moves = config.filter.min_moves
        max_moves = config.filter.max_moves

        self._writer.open(hdf5_path)

        if workers > 1:
            self._run_parallel(
                pgn_path,
                max_games,
                workers,
                train_color,
                min_moves,
                max_moves,
                train_pct,
            )
        else:
            self._run_serial(
                pgn_path, max_games, train_pct,
            )

        self._writer.close()
        logger.info(
            "Preprocessing complete: %s", hdf5_path
        )

        self._validator.validate(hdf5_path, config)

    def _run_serial(
        self,
        pgn_path: Path,
        max_games: int,
        train_pct: int,
    ) -> None:
        """Process games in the main process."""
        game_count = 0
        for game in self._reader.stream(
            pgn_path, max_games
        ):
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

        logger.info(
            "Total games processed: %d", game_count
        )

    def _run_parallel(
        self,
        pgn_path: Path,
        max_games: int,
        workers: int,
        train_color: str,
        min_moves: int,
        max_moves: int,
        train_pct: int,
    ) -> None:
        """Process games with a multiprocessing pool."""

        def task_iter() -> Iterator[
            tuple[str, int, str, int, int]
        ]:
            gid = 0
            for game in self._reader.stream(
                pgn_path, max_games
            ):
                pgn_text = _game_to_pgn_text(game)
                yield (
                    pgn_text,
                    gid,
                    train_color,
                    min_moves,
                    max_moves,
                )
                gid += 1

        game_count = 0
        with multiprocessing.Pool(workers) as pool:
            for records in pool.imap_unordered(
                _rl_parse_worker,
                task_iter(),
                chunksize=32,
            ):
                if records:
                    gid = records[0].game_id
                    split = (
                        "train"
                        if gid % 100 < train_pct
                        else "val"
                    )
                    self._writer.write_batch(
                        records, split
                    )

                game_count += 1
                if game_count % 10_000 == 0:
                    logger.info(
                        "Processed %d games", game_count
                    )

        logger.info(
            "Total games processed: %d", game_count
        )
