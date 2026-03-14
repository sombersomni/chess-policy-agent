"""ChessRLDataset: PyTorch Dataset wrapping a preprocessed HDF5 file.

Returns an 8-tuple per sample:
    (board [65,3], target_move, color_tokens [65],
     outcome, legal_mask [1971],
     capture_map [64], move_category, ply_idx)

Aux fields (capture_map, move_category) are auto-detected from
the HDF5 file. When absent, zeros are returned for backward
compatibility. ply_idx is always available (used for phase label).

All HDF5 data is loaded into memory at init time for zero-I/O
__getitem__ access. The HDF5 file is closed immediately after
loading.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Return type: 8-tuple
_SampleTuple = tuple[
    Tensor, int, Tensor, int, Tensor,
    Tensor, int, int,
]


class ChessRLDataset(Dataset):  # type: ignore[type-arg]
    """Dataset over a preprocessed HDF5 file.

    Loads all data into memory at init for fast __getitem__.

    Example:
        >>> ds = ChessRLDataset(Path("data.h5"), split="train")
        >>> sample = ds[0]
        >>> sample[0].shape  # board
        torch.Size([65, 3])
    """

    def __init__(
        self,
        hdf5_path: Path,
        val_split_fraction: float = 0.1,
        split: str = "train",
    ) -> None:
        """Load all split data into memory from HDF5.

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

        with h5py.File(hdf5_path, "r") as hf:
            game_ids = hf["game_id"][:]

            unique_ids = np.unique(game_ids)
            n_val = int(
                len(unique_ids) * val_split_fraction
            )

            if n_val == 0 and len(unique_ids) > 1:
                n_val = 0

            val_game_ids = (
                set(unique_ids[-n_val:])
                if n_val > 0 else set()
            )
            train_game_ids = (
                set(unique_ids) - val_game_ids
            )

            if split == "train":
                mask = np.isin(
                    game_ids, list(train_game_ids)
                )
                self._game_ids_set = train_game_ids
            else:
                mask = np.isin(
                    game_ids, list(val_game_ids)
                )
                self._game_ids_set = val_game_ids

            self._indices = np.where(mask)[0].astype(
                np.int64
            )

            idx = self._indices
            n = len(idx)
            logger.info(
                "Loading %d rows (%s split) into memory",
                n, split,
            )
            self._board = hf["board"][idx]
            self._target_move = hf["target_move"][idx]
            self._color_tokens = hf["color_tokens"][idx]
            self._outcome = hf["outcome"][idx]
            self._legal_mask = hf["legal_mask"][idx]

            # ply_idx: needed for phase label computation
            if "ply_idx" in hf:
                self._ply_idx: np.ndarray = (
                    hf["ply_idx"][idx]
                )
            else:
                self._ply_idx = np.zeros(
                    n, dtype=np.int16
                )

            # Aux fields: auto-detect from HDF5 presence
            self._has_aux = "capture_map" in hf
            if self._has_aux:
                self._capture_map: np.ndarray | None = (
                    hf["capture_map"][idx]
                )
                self._move_category: np.ndarray | None = (
                    hf["move_category"][idx]
                )
                logger.info(
                    "Aux fields loaded: capture_map, "
                    "move_category"
                )
            else:
                self._capture_map = None
                self._move_category = None

            logger.info(
                "Loaded %s split: %d rows, "
                "board=%.1f MB, legal_mask=%.1f MB",
                split, n,
                self._board.nbytes / 1e6,
                self._legal_mask.nbytes / 1e6,
            )

    @property
    def has_aux(self) -> bool:
        """Whether aux fields (capture_map, move_category) exist.

        Returns:
            True if HDF5 contained aux datasets.
        """
        return self._has_aux

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
    ) -> tuple[Any, ...]:
        """Return one sample as an 8-tuple.

        The first 5 elements match the legacy 5-tuple format.
        Elements 5-7 are aux fields (zeros when unavailable).

        Args:
            idx: Index into this split (0-based).

        Returns:
            Tuple of:
                board: float32 tensor [65, 3]
                target_move: int (vocab index)
                color_tokens: long tensor [65]
                outcome: int (+1 winner, 0 draw, -1 loser)
                legal_mask: bool tensor [1971]
                capture_map: float32 tensor [64]
                move_category: int (0-6)
                ply_idx: int

        Raises:
            IndexError: If idx is out of range.

        Example:
            >>> board, tgt, ct, out, mask, *_ = ds[0]
            >>> mask.shape
            torch.Size([1971])
        """
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(
                f"index {idx} out of range for "
                f"dataset of size {len(self._indices)}"
            )

        board = torch.tensor(
            self._board[idx], dtype=torch.float32,
        )
        target_move = int(self._target_move[idx])
        color_tokens = torch.tensor(
            self._color_tokens[idx], dtype=torch.long,
        )
        outcome = int(self._outcome[idx])
        legal_mask = torch.tensor(
            self._legal_mask[idx], dtype=torch.bool,
        )

        # Aux fields: zeros when HDF5 lacks them
        if self._has_aux and self._capture_map is not None:
            capture_map = torch.tensor(
                self._capture_map[idx],
                dtype=torch.float32,
            )
            move_category = int(
                self._move_category[idx]  # type: ignore
            )
        else:
            capture_map = torch.zeros(
                64, dtype=torch.float32
            )
            move_category = 0

        ply_idx = int(self._ply_idx[idx])

        return (
            board, target_move, color_tokens,
            outcome, legal_mask,
            capture_map, move_category, ply_idx,
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
        """No-op. HDF5 is closed at init after in-memory load.

        Kept for API compatibility.

        Example:
            >>> ds.close()
        """
