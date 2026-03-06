"""Dataclass types for the streaming data pipeline.

PreprocessConfig holds preprocessing hyperparameters. ManifestInfo
captures the output of a preprocessing run for cache validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PreprocessConfig:
    """Immutable configuration for the PGN preprocessing pipeline.

    Attributes:
        chunk_size: Number of games to accumulate before flushing a shard.
        winners_only: If True, only include positions where the winner moves.
        max_games: Stop after this many games (0 = no limit).
        num_workers: Number of DataLoader workers at training time.

    Example:
        >>> cfg = PreprocessConfig(chunk_size=512, winners_only=True)
        >>> cfg.chunk_size
        512
    """

    chunk_size: int = 1024
    winners_only: bool = False
    max_games: int = 0
    num_workers: int = 1


@dataclass(frozen=True)
class ManifestInfo:
    """Typed representation of the manifest.json written after preprocessing.

    Attributes:
        source_checksum: SHA-256 based checksum of the source PGN file.
        shard_paths: Ordered list of paths to .pt shard files on disk.
        total_examples: Sum of examples across all shards.
        examples_per_shard: Number of examples in each shard, same order as shard_paths.
        config: The PreprocessConfig used to generate these shards.

    Example:
        >>> info = ManifestInfo("abc123", [Path("s0.pt")], 100, [100], cfg)
        >>> info.total_examples
        100
    """

    source_checksum: str
    shard_paths: list[Path] = field(default_factory=list)
    total_examples: int = 0
    examples_per_shard: list[int] = field(default_factory=list)
    config: PreprocessConfig = field(default_factory=PreprocessConfig)
