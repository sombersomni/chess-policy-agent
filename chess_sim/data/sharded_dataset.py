"""ShardedChessDataset: memory-efficient Dataset backed by pre-tensorized shard files.

Maps a global index to a ChessBatch by resolving which shard contains that
index via binary search on cumulative counts, loading the shard with LRU
caching, and returning a zero-copy tensor slice.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from chess_sim.types import ChessBatch


class ShardedChessDataset(Dataset):
    """Dataset backed by numbered .pt shard files with LRU shard caching.

    Uses bisect on cumulative example counts for O(log S) shard lookup
    where S is the number of shards. Loaded shards are cached in an
    OrderedDict with LRU eviction bounded by max_cached_shards.

    Attributes:
        _shard_paths: Ordered list of paths to .pt shard files.
        _examples_per_shard: Number of examples in each shard.
        _cumulative_counts: Prefix sums for binary search shard resolution.
        _shard_cache: LRU cache of loaded shard tensor dicts.
        _max_cached_shards: Maximum number of shards to hold in memory.

    Example:
        >>> ds = ShardedChessDataset.from_manifest(Path("output/manifest.json"))
        >>> len(ds)
        350000
        >>> batch = ds[0]
    """

    def __init__(
        self,
        shard_paths: list[Path],
        examples_per_shard: list[int],
        max_cached_shards: int = 8,
    ) -> None:
        """Initialize the sharded dataset.

        Args:
            shard_paths: Ordered list of paths to .pt shard files.
            examples_per_shard: Number of examples in each shard (same order).
            max_cached_shards: Max shards to cache in memory. Default 8.

        Raises:
            ValueError: If shard_paths and examples_per_shard have different lengths.

        Example:
            >>> ds = ShardedChessDataset([Path("s0.pt")], [100])
            >>> len(ds)
            100
        """
        raise NotImplementedError("To be implemented")

    @staticmethod
    def from_manifest(manifest_path: Path) -> ShardedChessDataset:
        """Construct a ShardedChessDataset from a manifest.json file.

        Reads the manifest using CacheManager.load_manifest and extracts
        shard_paths and examples_per_shard.

        Args:
            manifest_path: Path to the manifest.json file.

        Returns:
            ShardedChessDataset instance ready for DataLoader use.

        Example:
            >>> ds = ShardedChessDataset.from_manifest(Path("out/manifest.json"))
        """
        raise NotImplementedError("To be implemented")

    def __len__(self) -> int:
        """Return total number of examples across all shards.

        Returns:
            Integer count of all training examples.

        Example:
            >>> len(ds)
            350000
        """
        raise NotImplementedError("To be implemented")

    def __getitem__(self, idx: int) -> ChessBatch:
        """Return one training example as a ChessBatch.

        Resolves the shard via binary search on cumulative counts, loads
        the shard into cache if needed (evicting LRU if full), and
        returns a zero-copy tensor slice.

        Args:
            idx: Global index into the dataset (0 <= idx < len(self)).

        Returns:
            ChessBatch with all fields as torch.long tensors.

        Raises:
            IndexError: If idx is out of range.

        Example:
            >>> batch = ds[42]
            >>> batch.board_tokens.shape
            torch.Size([65])
        """
        raise NotImplementedError("To be implemented")
