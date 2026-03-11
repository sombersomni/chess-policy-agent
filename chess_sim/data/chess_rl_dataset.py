"""ChessRLDataset: PyTorch Dataset wrapping a preprocessed RSCE HDF5 file.

Returns (board [65,3], target_move, multiplier, color_tokens [65])
per sample. Supports train/val splitting by game_id: the last
val_split_fraction of unique game_ids form the val set.

The HDF5 file handle is opened lazily on first __getitem__ call
to support multi-worker DataLoader (h5py is not fork-safe if
opened before fork).
"""
from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset


class ChessRLDataset(Dataset):  # type: ignore[type-arg]
    """Dataset over a preprocessed RSCE HDF5 file.

    Lazily opens the HDF5 handle on first access to support
    num_workers > 0 in DataLoader.

    Example:
        >>> ds = ChessRLDataset(Path("data.h5"), split="train")
        >>> board, target, mult, ct = ds[0]
        >>> board.shape
        torch.Size([65, 3])
    """

    def __init__(
        self,
        hdf5_path: Path,
        val_split_fraction: float = 0.1,
        split: str = "train",
    ) -> None:
        """Initialize dataset with HDF5 path and split config.

        The val split consists of the last val_split_fraction
        of unique game_ids (ordered, not shuffled). The h5py
        file is NOT opened here -- deferred to first __getitem__.

        Args:
            hdf5_path: Path to preprocessed HDF5 file.
            val_split_fraction: Fraction of games for val split.
            split: "train" or "val".

        Raises:
            ValueError: If split is not "train" or "val".
            ValueError: If val_split_fraction not in (0, 1).

        Example:
            >>> ds = ChessRLDataset(Path("d.h5"), split="val")
        """
        raise NotImplementedError("To be implemented")

    def __len__(self) -> int:
        """Return number of samples in this split.

        Returns:
            Integer count of rows in the train or val subset.

        Example:
            >>> len(ChessRLDataset(Path("d.h5")))
            1800
        """
        raise NotImplementedError("To be implemented")

    def __getitem__(
        self, idx: int,
    ) -> tuple[Tensor, int, float, Tensor]:
        """Return one sample: (board, target_move, mult, color_tokens).

        Opens the HDF5 file lazily on first call. Maps the
        split-local index to the global HDF5 row index.

        Args:
            idx: Index into this split (0-based).

        Returns:
            Tuple of:
                board: float32 tensor [65, 3]
                target_move: int (vocab index)
                multiplier: float (pre-normalized m_hat)
                color_tokens: long tensor [65]

        Raises:
            IndexError: If idx is out of range.

        Example:
            >>> board, tgt, mult, ct = ds[0]
            >>> isinstance(mult, float)
            True
        """
        raise NotImplementedError("To be implemented")

    @property
    def n_games(self) -> int:
        """Number of unique game_ids in this split.

        Returns:
            Integer count of distinct games.

        Example:
            >>> ds.n_games
            90
        """
        raise NotImplementedError("To be implemented")

    def close(self) -> None:
        """Close the HDF5 file handle if open.

        Safe to call multiple times.

        Example:
            >>> ds.close()
        """
        raise NotImplementedError("To be implemented")
