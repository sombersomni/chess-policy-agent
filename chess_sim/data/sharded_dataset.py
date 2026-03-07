"""ShardedChessDataset: memory-efficient Dataset backed by shard files.

Maps a global index to a ChessBatch by resolving which shard contains
that index via binary search on cumulative counts, loading the shard
with LRU caching, and returning a zero-copy tensor slice.

Also provides ShardAwareBatchSampler for memory-efficient multi-worker
DataLoader usage -- batches are drawn from one shard at a time.
"""

from __future__ import annotations

import bisect
import logging
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import BatchSampler, Dataset

from chess_sim.types import ChessBatch

logger = logging.getLogger(__name__)


class ShardedChessDataset(Dataset):
    """Dataset backed by numbered .pt shard files with LRU caching.

    Uses bisect on cumulative example counts for O(log S) shard lookup
    where S is the number of shards. Loaded shards are cached in an
    OrderedDict with LRU eviction bounded by max_cached_shards.

    Attributes:
        _shard_paths: Ordered list of paths to .pt shard files.
        _examples_per_shard: Number of examples in each shard.
        _cumulative_counts: Prefix sums for binary search resolution.
        _shard_cache: LRU cache of loaded shard tensor dicts.
        _max_cached_shards: Maximum shards to hold in memory.

    Example:
        >>> ds = ShardedChessDataset.from_manifest(Path("out/m.json"))
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
            examples_per_shard: Number of examples in each shard.
            max_cached_shards: Max shards to cache in memory.

        Raises:
            ValueError: If shard_paths and examples_per_shard differ
                in length.

        Example:
            >>> ds = ShardedChessDataset([Path("s0.pt")], [100])
            >>> len(ds)
            100
        """
        if len(shard_paths) != len(examples_per_shard):
            raise ValueError(
                f"shard_paths length ({len(shard_paths)}) != "
                f"examples_per_shard length "
                f"({len(examples_per_shard)})"
            )
        self._shard_paths = shard_paths
        self._examples_per_shard = examples_per_shard
        self._max_cached_shards = max_cached_shards

        # Build prefix sums for O(log S) shard lookup
        self._cumulative_counts: list[int] = []
        running = 0
        for count in examples_per_shard:
            running += count
            self._cumulative_counts.append(running)

        self._shard_cache: OrderedDict[
            int, dict[str, Tensor]
        ] = OrderedDict()

    @staticmethod
    def from_manifest(
        manifest_path: Path,
    ) -> ShardedChessDataset:
        """Construct from a manifest.json file.

        Reads the manifest using CacheManager.load_manifest and
        extracts shard_paths and examples_per_shard.

        Args:
            manifest_path: Path to the manifest.json file.

        Returns:
            ShardedChessDataset instance ready for DataLoader use.

        Example:
            >>> ds = ShardedChessDataset.from_manifest(Path("m.json"))
        """
        from chess_sim.data.cache_manager import CacheManager
        cm = CacheManager()
        info = cm.load_manifest(manifest_path)
        return ShardedChessDataset(
            shard_paths=info.shard_paths,
            examples_per_shard=info.examples_per_shard,
        )

    def __len__(self) -> int:
        """Return total number of examples across all shards.

        Returns:
            Integer count of all training examples.

        Example:
            >>> len(ds)
            350000
        """
        if not self._cumulative_counts:
            return 0
        return self._cumulative_counts[-1]

    def __getitem__(self, idx: int) -> ChessBatch:
        """Return one training example as a ChessBatch.

        Resolves the shard via binary search on cumulative counts,
        loads the shard into cache if needed (evicting LRU if full),
        and returns a zero-copy tensor slice.

        Args:
            idx: Global index into the dataset.

        Returns:
            ChessBatch with all fields as torch.long tensors.

        Raises:
            IndexError: If idx is out of range.

        Example:
            >>> batch = ds[42]
            >>> batch.board_tokens.shape
            torch.Size([65])
        """
        total = len(self)
        if idx < 0 or idx >= total:
            raise IndexError(
                f"Index {idx} out of range for dataset "
                f"of size {total}"
            )

        # Binary search for shard index
        shard_idx = bisect.bisect_right(
            self._cumulative_counts, idx
        )
        local_idx = idx
        if shard_idx > 0:
            local_idx = idx - self._cumulative_counts[shard_idx - 1]

        shard_data = self._load_shard(shard_idx)

        return ChessBatch(
            board_tokens=shard_data["board_tokens"][local_idx],
            color_tokens=shard_data["color_tokens"][local_idx],
            trajectory_tokens=shard_data[
                "trajectory_tokens"
            ][local_idx],
            src_sq=shard_data["src_sq"][local_idx],
            tgt_sq=shard_data["tgt_sq"][local_idx],
        )

    def _load_shard(
        self, shard_idx: int
    ) -> dict[str, Tensor]:
        """Load a shard into cache, evicting LRU if needed.

        Args:
            shard_idx: Index of the shard to load.

        Returns:
            Dict of tensors for the shard.
        """
        if shard_idx in self._shard_cache:
            # Move to end (most recently used)
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]

        path = self._shard_paths[shard_idx]
        logger.debug("Loading shard %d from %s", shard_idx, path)

        shard_data = torch.load(
            path, map_location="cpu", weights_only=True
        )

        # Evict LRU if at capacity
        if len(self._shard_cache) >= self._max_cached_shards:
            evicted_key, _ = self._shard_cache.popitem(last=False)
            logger.debug("Evicted shard %d from cache", evicted_key)

        self._shard_cache[shard_idx] = shard_data
        return shard_data


class ShardAwareBatchSampler(BatchSampler):
    """Batch sampler that yields batches from one shard at a time.

    Prevents cross-shard access within a single batch, ensuring each
    DataLoader worker only needs one shard in memory. Shuffles shard
    order and within-shard indices each epoch for stochasticity.

    Reason: Eliminates shard thrashing when using shuffle=True with
    ShardedChessDataset, enabling safe num_workers > 0.
    """

    def __init__(
        self,
        dataset: ShardedChessDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        """Initialize the shard-aware batch sampler.

        Args:
            dataset: The sharded dataset to sample from.
            batch_size: Number of examples per batch.
            drop_last: If True, drop the last incomplete batch
                within each shard.
            shuffle: If True, shuffle shard order and
                within-shard indices each epoch.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._shuffle = shuffle

        # Precompute global offset for each shard
        self._shard_offsets: list[int] = []
        for i in range(len(dataset._examples_per_shard)):
            if i == 0:
                self._shard_offsets.append(0)
            else:
                self._shard_offsets.append(
                    dataset._cumulative_counts[i - 1]
                )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of global indices, all from the same shard.

        Shuffles shard order each call if shuffle=True. Within each
        shard, shuffles local indices then yields consecutive
        batch_size chunks as global indices.

        Yields:
            List of global indices forming one batch.
        """
        shard_order = list(range(
            len(self._dataset._examples_per_shard)
        ))
        if self._shuffle:
            random.shuffle(shard_order)

        for shard_idx in shard_order:
            n = self._dataset._examples_per_shard[shard_idx]
            if n == 0:
                continue
            offset = self._shard_offsets[shard_idx]

            local_indices = list(range(n))
            if self._shuffle:
                random.shuffle(local_indices)

            # Yield full batches from this shard
            for start in range(0, n, self._batch_size):
                end = min(start + self._batch_size, n)
                if end - start < self._batch_size and self._drop_last:
                    continue
                batch = [
                    offset + local_indices[i]
                    for i in range(start, end)
                ]
                yield batch

    def __len__(self) -> int:
        """Return total number of batches across all shards.

        Returns:
            Integer count of batches that __iter__ will yield.
        """
        total = 0
        for n in self._dataset._examples_per_shard:
            if self._drop_last:
                total += n // self._batch_size
            else:
                total += math.ceil(n / self._batch_size)
        return total
