"""ShardWriter: serializes tensor dicts to numbered .pt shard files.

Implements the ShardWritable protocol. Each shard is a dict saved via
torch.save with a deterministic filename based on the shard index.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ShardWriter:
    """Writes tensor dicts to numbered shard files on disk.

    Implements the ShardWritable protocol. File naming convention:
    shard_{shard_idx:06d}.pt (e.g., shard_000042.pt).

    Example:
        >>> sw = ShardWriter()
        >>> path = sw.flush(tensors, shard_idx=0, output_dir=Path("/tmp"))
        >>> path.name
        'shard_000000.pt'
    """

    def flush(
        self,
        tensors: dict[str, Tensor],
        shard_idx: int,
        output_dir: Path,
    ) -> Path:
        """Serialize a tensor dict to a numbered .pt file on disk.

        Creates the output directory if it does not exist. The saved
        dict includes a "count" key with the number of examples.

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
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"shard_{shard_idx:06d}.pt"
        path = output_dir / filename

        # Determine count from first tensor's first dimension
        first_tensor = next(iter(tensors.values()))
        count = first_tensor.shape[0]

        save_dict: dict[str, Tensor | int] = {**tensors}
        save_dict["count"] = count

        torch.save(save_dict, path)
        logger.info(
            "Wrote shard %s with %d examples", path.name, count
        )
        return path
