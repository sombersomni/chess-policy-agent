"""ChessRLDataset: PyTorch Dataset wrapping a preprocessed RSCE HDF5 file.

Returns 6-tuple per sample:
    (board [65,3], target_move, multiplier, color_tokens [65],
     outcome, loss_mode)
Supports train/val splitting by game_id: the last
val_split_fraction of unique game_ids form the val set.

The HDF5 file handle is opened lazily on first __getitem__ call
to support multi-worker DataLoader (h5py is not fork-safe if
opened before fork).
"""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ChessRLDataset(Dataset):  # type: ignore[type-arg]
    """Dataset over a preprocessed RSCE HDF5 file.

    Lazily opens the HDF5 handle on first access to support
    num_workers > 0 in DataLoader.

    Example:
        >>> ds = ChessRLDataset(Path("data.h5"), split="train")
        >>> board, tgt, mult, ct, outcome, lm = ds[0]
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
        if split not in ("train", "val"):
            raise ValueError(
                f"split must be 'train' or 'val', "
                f"got '{split}'"
            )
        if not (0.0 < val_split_fraction < 1.0):
            raise ValueError(
                "val_split_fraction must be in (0, 1), "
                f"got {val_split_fraction}"
            )

        self._hdf5_path = hdf5_path
        self._split = split
        self._h5: h5py.File | None = None

        # Open briefly to compute indices, then close
        with h5py.File(hdf5_path, "r") as hf:
            game_ids = hf["game_id"][:]

        unique_ids = np.unique(game_ids)
        n_val = int(len(unique_ids) * val_split_fraction)

        if n_val == 0 and len(unique_ids) > 1:
            # Ensure at least some val games when possible
            n_val = 0

        val_game_ids = set(unique_ids[-n_val:]) if n_val > 0 else set()
        train_game_ids = set(unique_ids) - val_game_ids

        if split == "train":
            mask = np.isin(game_ids, list(train_game_ids))
            self._game_ids_set = train_game_ids
        else:
            mask = np.isin(game_ids, list(val_game_ids))
            self._game_ids_set = val_game_ids

        self._indices = np.where(mask)[0].astype(np.int64)

    def __len__(self) -> int:
        """Return number of samples in this split.

        Returns:
            Integer count of rows in the train or val subset.

        Example:
            >>> len(ChessRLDataset(Path("d.h5")))
            1800
        """
        return len(self._indices)

    def __getitem__(
        self, idx: int,
    ) -> tuple[Tensor, int, float, Tensor, int, int]:
        """Return one sample as a 6-tuple.

        Returns (board, target_move, multiplier, color_tokens,
        outcome, loss_mode). Opens the HDF5 file lazily on
        first call. Maps split-local index to global HDF5 row.

        Args:
            idx: Index into this split (0-based).

        Returns:
            Tuple of:
                board: float32 tensor [65, 3]
                target_move: int (vocab index)
                multiplier: float (pre-normalized m_hat)
                color_tokens: long tensor [65]
                outcome: int (+1 winner, 0 draw, -1 loser)
                loss_mode: int (+1 imitation, -1 repulsion)

        Raises:
            IndexError: If idx is out of range.

        Example:
            >>> board, tgt, mult, ct, out, lm = ds[0]
            >>> isinstance(mult, float)
            True
        """
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(
                f"index {idx} out of range for "
                f"dataset of size {len(self._indices)}"
            )

        if self._h5 is None:
            self._h5 = h5py.File(self._hdf5_path, "r")

        global_idx = int(self._indices[idx])

        board = torch.tensor(
            self._h5["board"][global_idx],
            dtype=torch.float32,
        )
        target_move = int(
            self._h5["target_move"][global_idx]
        )
        multiplier = float(
            self._h5["multiplier"][global_idx]
        )
        color_tokens = torch.tensor(
            self._h5["color_tokens"][global_idx],
            dtype=torch.long,
        )
        outcome = int(self._h5["outcome"][global_idx])
        loss_mode = int(
            self._h5["loss_mode"][global_idx]
        )

        return (
            board, target_move, multiplier,
            color_tokens, outcome, loss_mode,
        )

    @property
    def n_games(self) -> int:
        """Number of unique game_ids in this split.

        Returns:
            Integer count of distinct games.

        Example:
            >>> ds.n_games
            90
        """
        return len(self._game_ids_set)

    def close(self) -> None:
        """Close the HDF5 file handle if open.

        Safe to call multiple times.

        Example:
            >>> ds.close()
        """
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
