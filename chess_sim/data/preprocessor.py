"""PGNPreprocessor: orchestrates the full PGN-to-shards pipeline.

Implements the Preprocessable protocol. Reads games from a PGN file,
processes them in chunks, writes numbered shard files, and produces a
manifest for cache validation and training-time dataset construction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import chess.pgn

from chess_sim.data.cache_manager import CacheManager
from chess_sim.data.chunk_processor import ChunkProcessor
from chess_sim.data.reader import StreamingPGNReader
from chess_sim.data.shard_writer import ShardWriter
from chess_sim.data.streaming_types import ManifestInfo, PreprocessConfig

logger = logging.getLogger(__name__)


class PGNPreprocessor:
    """Orchestrates PGN preprocessing: stream, chunk, tokenize, shards.

    Implements the Preprocessable protocol. This is the top-level entry
    point for Phase 1 of the streaming data pipeline.

    Algorithm:
        1. Compute source checksum. If cached, return ManifestInfo.
        2. Stream games via reader.stream(pgn_path).
        3. Accumulate config.chunk_size games into a buffer.
        4. Pass buffer to chunk_processor.process_chunk -> tensors.
        5. Pass tensors to shard_writer.flush -> shard path.
        6. After all games, write manifest.

    Example:
        >>> pp = PGNPreprocessor(reader, cp, sw, cm)
        >>> info = pp.preprocess(Path("games.pgn.zst"), Path("out"), cfg)
        >>> len(info.shard_paths)
        42
    """

    def __init__(
        self,
        reader: StreamingPGNReader,
        chunk_processor: ChunkProcessor,
        shard_writer: ShardWriter,
        cache_manager: CacheManager,
    ) -> None:
        """Initialize the preprocessor with its collaborators.

        Args:
            reader: StreamingPGNReader for streaming games from PGN.
            chunk_processor: ChunkProcessor for converting games to
                tensors.
            shard_writer: ShardWriter for serializing tensor dicts.
            cache_manager: CacheManager for checksum and manifest ops.

        Example:
            >>> pp = PGNPreprocessor(reader, cp, sw, cm)
        """
        self._reader = reader
        self._chunk_processor = chunk_processor
        self._shard_writer = shard_writer
        self._cache_manager = cache_manager

    def preprocess(
        self,
        pgn_path: Path,
        output_dir: Path,
        config: PreprocessConfig,
    ) -> ManifestInfo:
        """Run the full preprocessing pipeline on a PGN file.

        If a valid cache exists (matching checksum), returns the cached
        ManifestInfo without reprocessing.

        Args:
            pgn_path: Path to the source PGN file (plain or .zst).
            output_dir: Directory where shards and manifest go.
            config: PreprocessConfig with chunk_size, winners_only,
                max_games.

        Returns:
            ManifestInfo with shard paths, counts, and checksum.

        Example:
            >>> info = pp.preprocess(Path("g.pgn"), Path("out/"), cfg)
            >>> info.total_examples
            150000
        """
        manifest_path = output_dir / "manifest.json"
        source_checksum = self._cache_manager.checksum(pgn_path)

        if self._cache_manager.is_cached(
            manifest_path, source_checksum
        ):
            logger.info("Cache hit for %s, loading manifest", pgn_path)
            return self._cache_manager.load_manifest(manifest_path)

        logger.info("Cache miss for %s, preprocessing", pgn_path)
        shard_paths: list[Path] = []
        examples_per_shard: list[int] = []
        shard_idx = 0
        total_examples = 0

        game_stream: Iterator[chess.pgn.Game] = (
            self._reader.stream(pgn_path)
        )

        buffer: list[chess.pgn.Game] = []
        games_read = 0

        for game in game_stream:
            if config.max_games and games_read >= config.max_games:
                break
            buffer.append(game)
            games_read += 1

            if len(buffer) >= config.chunk_size:
                shard_idx, total_examples = self._flush_buffer(
                    buffer,
                    shard_idx,
                    total_examples,
                    shard_paths,
                    examples_per_shard,
                    output_dir,
                )
                buffer = []

        # Flush remaining games
        if buffer:
            shard_idx, total_examples = self._flush_buffer(
                buffer,
                shard_idx,
                total_examples,
                shard_paths,
                examples_per_shard,
                output_dir,
            )

        info = ManifestInfo(
            source_checksum=source_checksum,
            shard_paths=shard_paths,
            total_examples=total_examples,
            examples_per_shard=examples_per_shard,
            config=config,
        )
        self._cache_manager.write_manifest(info, output_dir)

        logger.info(
            "Preprocessing complete: %d shards, %d examples",
            len(shard_paths),
            total_examples,
        )
        return info

    def _flush_buffer(
        self,
        buffer: list[chess.pgn.Game],
        shard_idx: int,
        total_examples: int,
        shard_paths: list[Path],
        examples_per_shard: list[int],
        output_dir: Path,
    ) -> tuple[int, int]:
        """Process a buffer of games and write a shard.

        Args:
            buffer: Games to process.
            shard_idx: Current shard index.
            total_examples: Running total of examples.
            shard_paths: List to append new shard path to.
            examples_per_shard: List to append count to.
            output_dir: Directory for shard files.

        Returns:
            Tuple of (next shard_idx, updated total_examples).
        """
        tensors = self._chunk_processor.process_chunk(buffer)
        count = tensors["board_tokens"].shape[0]
        if count > 0:
            path = self._shard_writer.flush(
                tensors, shard_idx, output_dir
            )
            shard_paths.append(path)
            examples_per_shard.append(count)
            total_examples += count
            shard_idx += 1
        return shard_idx, total_examples
