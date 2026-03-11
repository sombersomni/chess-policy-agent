"""RLPlyHDF5Dataset: reads OfflinePlyTuple records from RL HDF5.

Provides random access to preprocessed RL data without PGN parsing.
Compatible with standard DataLoader and rl_hdf5_worker_init for
multi-worker loading.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from chess_sim.types import OfflinePlyTuple

logger = logging.getLogger(__name__)


class RLPlyHDF5Dataset(Dataset[OfflinePlyTuple]):
    """Dataset reading OfflinePlyTuple records from RL HDF5.

    Each __getitem__ call reads integer/byte arrays from disk
    and reconstructs an OfflinePlyTuple with proper tensor types.
    move_prefix is sliced to prefix_lengths[idx] to strip padding.

    Example:
        >>> ds = RLPlyHDF5Dataset(Path("data/chess_rl.h5"))
        >>> len(ds)
        5000
        >>> ply = ds[0]
        >>> ply.move_uci
        'e2e4'
    """

    def __init__(
        self,
        hdf5_path: Path,
        split: str = "train",
    ) -> None:
        """Open HDF5 file and cache dataset length.

        Args:
            hdf5_path: Path to the RL HDF5 dataset file.
            split: 'train' or 'val'.

        Raises:
            FileNotFoundError: If hdf5_path does not exist.
            KeyError: If split group is missing in the file.

        Example:
            >>> ds = RLPlyHDF5Dataset(Path("data/chess_rl.h5"))
        """
        super().__init__()
        self._path = Path(hdf5_path)
        self._split = split
        self._file: h5py.File | None = None
        self._open()
        grp = self._file[self._split]  # type: ignore[index]
        self._length: int = int(
            grp["board_tokens"].shape[0]
        )
        logger.info(
            "RLPlyHDF5Dataset(%s/%s): %d samples",
            self._path.name,
            split,
            self._length,
        )

    def _open(self) -> None:
        """Open or re-open the HDF5 file handle.

        Called during __init__ and by rl_hdf5_worker_init
        in each DataLoader worker process.

        Example:
            >>> ds._open()
        """
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._file = h5py.File(str(self._path), "r")

    def __len__(self) -> int:
        """Return number of samples in this split.

        Returns:
            Integer count of ply records.

        Example:
            >>> len(ds)
            5000
        """
        return self._length

    def __getitem__(self, idx: int) -> OfflinePlyTuple:
        """Read one OfflinePlyTuple by index.

        Slices move_prefix to actual length (no padding).
        Decodes move_uci from ASCII bytes.

        Args:
            idx: Sample index (non-negative).

        Returns:
            OfflinePlyTuple with tensors and decoded strings.

        Raises:
            IndexError: If idx < 0 or idx >= len(self).

        Example:
            >>> ply = ds[0]
            >>> ply.board_tokens.shape
            torch.Size([65])
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(
                f"Index {idx} out of range "
                f"[0, {self._length})"
            )

        grp = self._file[self._split]  # type: ignore[index]
        pl = int(grp["prefix_lengths"][idx])

        board_tokens = torch.tensor(
            grp["board_tokens"][idx], dtype=torch.long
        )
        color_tokens = torch.tensor(
            grp["color_tokens"][idx], dtype=torch.long
        )
        traj_tokens = torch.tensor(
            grp["traj_tokens"][idx], dtype=torch.long
        )
        move_prefix = torch.tensor(
            grp["move_prefix"][idx, :pl], dtype=torch.long
        )
        move_uci: str = (
            grp["move_uci"][idx]
            .decode("ascii")
            .rstrip("\x00")
        )
        is_winner_ply = bool(grp["is_winner_ply"][idx])
        is_white_ply = bool(grp["is_white_ply"][idx])
        is_draw_ply = bool(grp["is_draw_ply"][idx])

        material_delta: float = float(
            grp["material_delta"][idx]
            if "material_delta" in grp
            else 0.0
        )

        return OfflinePlyTuple(
            board_tokens=board_tokens,
            color_tokens=color_tokens,
            traj_tokens=traj_tokens,
            move_prefix=move_prefix,
            move_uci=move_uci,
            is_winner_ply=is_winner_ply,
            is_white_ply=is_white_ply,
            is_draw_ply=is_draw_ply,
            material_delta=material_delta,
        )


def rl_hdf5_worker_init(worker_id: int) -> None:
    """Re-open HDF5 handle in each DataLoader worker.

    h5py file handles are not safe to share across forked
    processes. This function re-opens the file in each worker.

    Args:
        worker_id: DataLoader worker index (unused).

    Example:
        >>> DataLoader(ds, num_workers=2,
        ...            worker_init_fn=rl_hdf5_worker_init)
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    ds = worker_info.dataset
    if isinstance(ds, RLPlyHDF5Dataset):
        ds._open()
