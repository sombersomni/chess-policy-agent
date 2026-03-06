"""PGNPreprocessor: orchestrates the full PGN-to-shards preprocessing pipeline.

Implements the Preprocessable protocol. Reads games from a PGN file,
processes them in chunks, writes numbered shard files, and produces a
manifest for cache validation and training-time dataset construction.
"""

from __future__ import annotations

from pathlib import Path

from chess_sim.data.cache_manager import CacheManager
from chess_sim.data.chunk_processor import ChunkProcessor
from chess_sim.data.reader import StreamingPGNReader
from chess_sim.data.shard_writer import ShardWriter
from chess_sim.data.streaming_types import ManifestInfo, PreprocessConfig


class PGNPreprocessor:
    """Orchestrates PGN preprocessing: stream, chunk, tokenize, write shards.

    Implements the Preprocessable protocol. This is the top-level entry
    point for Phase 1 of the streaming data pipeline.

    Algorithm:
        1. Compute source checksum. If cached, return existing ManifestInfo.
        2. Stream games via reader.stream(pgn_path).
        3. Accumulate config.chunk_size games into a buffer.
        4. Pass buffer to chunk_processor.process_chunk -> tensor dict.
        5. Pass tensor dict to shard_writer.flush -> shard path.
        6. After all games, write manifest via cache_manager.write_manifest.

    Example:
        >>> pp = PGNPreprocessor(reader, chunk_proc, shard_writer, cache_mgr)
        >>> info = pp.preprocess(Path("games.pgn.zst"), Path("output/"), cfg)
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
            reader: StreamingPGNReader for streaming games from PGN files.
            chunk_processor: ChunkProcessor for converting games to tensors.
            shard_writer: ShardWriter for serializing tensor dicts to disk.
            cache_manager: CacheManager for checksum and manifest operations.

        Example:
            >>> pp = PGNPreprocessor(reader, cp, sw, cm)
        """
        raise NotImplementedError("To be implemented")

    def preprocess(
        self,
        pgn_path: Path,
        output_dir: Path,
        config: PreprocessConfig,
    ) -> ManifestInfo:
        """Run the full preprocessing pipeline on a PGN file.

        If a valid cache exists (matching checksum), returns the cached
        ManifestInfo without reprocessing. Otherwise, streams all games,
        processes them in chunks of config.chunk_size, writes shard files,
        and creates a manifest.

        Args:
            pgn_path: Path to the source PGN file (plain or .zst compressed).
            output_dir: Directory where shards and manifest will be written.
            config: PreprocessConfig with chunk_size, winners_only, max_games.

        Returns:
            ManifestInfo with shard paths, counts, and checksum metadata.

        Example:
            >>> info = pp.preprocess(Path("games.pgn"), Path("out/"), cfg)
            >>> info.total_examples
            150000
        """
        raise NotImplementedError("To be implemented")
