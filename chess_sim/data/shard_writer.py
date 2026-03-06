"""ShardWriter: serializes tensor dicts to numbered .pt shard files.

Implements the ShardWritable protocol. Each shard is a dict saved via
torch.save with a deterministic filename based on the shard index.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor


class ShardWriter:
    """Writes tensor dicts to numbered shard files on disk.

    Implements the ShardWritable protocol. File naming convention:
    shard_{shard_idx:06d}.pt (e.g., shard_000042.pt).

    Example:
        >>> sw = ShardWriter()
        >>> path = sw.flush(tensors, shard_idx=0, output_dir=Path("/tmp/shards"))
        >>> path.name
        'shard_000000.pt'
    """

    def flush(self, tensors: dict[str, Tensor], shard_idx: int, output_dir: Path) -> Path:
        """Serialize a tensor dict to a numbered .pt file on disk.

        Creates the output directory if it does not exist. The saved dict
        includes a "count" key with the number of examples (rows in first tensor).

        Args:
            tensors: Dict mapping field names to torch.long tensors.
                     All tensors must have the same first dimension.
            shard_idx: Zero-based shard index used in the filename.
            output_dir: Directory where the shard file will be written.

        Returns:
            Path to the written shard file.

        Example:
            >>> sw = ShardWriter()
            >>> p = sw.flush({"x": torch.zeros(10)}, 5, Path("/tmp"))
            >>> p.name
            'shard_000005.pt'
        """
        raise NotImplementedError("To be implemented")
