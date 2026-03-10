"""RLHdf5Writer: buffer RLPlyRecord lists and flush to HDF5.

Manages resizable chunked datasets, compression, and split routing.
Stores train_color as a root-level HDF5 attribute. RL analogue of
HDF5Writer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from chess_sim.types import RLPlyRecord

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val")


class RLHdf5Writer:
    """Buffer and write RLPlyRecord batches to split-aware HDF5.

    Creates two groups ('train' and 'val') with identical dataset
    schemas. move_uci is stored as fixed-length 5-byte ASCII.
    move_prefix is zero-padded to max_prefix_len.

    Example:
        >>> writer = RLHdf5Writer(512, 1000, "gzip", 4, "white")
        >>> writer.open(Path("data/chess_rl.h5"))
        >>> writer.write_batch(records, split="train")
        >>> writer.flush("train")
        >>> writer.close()
    """

    def __init__(
        self,
        max_prefix_len: int,
        chunk_size: int,
        compression: str,
        compression_opts: int,
        train_color: str,
    ) -> None:
        """Initialize writer with HDF5 layout parameters.

        Args:
            max_prefix_len: Padded width of move_prefix.
            chunk_size: HDF5 chunk / auto-flush threshold.
            compression: Compression filter name.
            compression_opts: Compression level (0-9).
            train_color: "white" or "black"; stored as attr.

        Example:
            >>> w = RLHdf5Writer(512, 1000, "gzip", 4, "white")
        """
        self._max_prefix_len = max_prefix_len
        self._chunk_size = chunk_size
        self._compression = compression
        self._compression_opts = compression_opts
        self._train_color = train_color
        self._file: Optional[h5py.File] = None
        self._buffers: dict[str, list[RLPlyRecord]] = {
            "train": [],
            "val": [],
        }

    def open(self, path: Path, mode: str = "w") -> None:
        """Open or create the HDF5 file.

        Args:
            path: Path to the HDF5 file.
            mode: File open mode ("w" or "a").

        Raises:
            OSError: If the file cannot be opened.

        Example:
            >>> writer.open(Path("data/chess_rl.h5"))
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(str(path), mode)
        for split in _SPLITS:
            if split not in self._file:
                grp = self._file.create_group(split)
                self._init_datasets(grp)
        logger.info(
            "Opened RL HDF5: %s (mode=%s)", path, mode
        )

    def _init_datasets(self, grp: h5py.Group) -> None:
        """Create resizable datasets in a group."""
        cs = min(self._chunk_size, 1024)
        mpl = self._max_prefix_len

        # (name, dtype, trailing_shape)
        specs: list[
            tuple[str, str | type, tuple[int, ...]]
        ] = [
            ("board_tokens", "uint8", (65,)),
            ("color_tokens", "uint8", (65,)),
            ("traj_tokens", "uint8", (65,)),
            ("move_prefix", "uint16", (mpl,)),
            ("prefix_lengths", "uint16", ()),
            ("move_uci", "S5", ()),
            ("is_winner_ply", "uint8", ()),
            ("is_white_ply", "uint8", ()),
            ("is_draw_ply", "uint8", ()),
            ("game_id", "uint32", ()),
            ("ply_index", "uint32", ()),
        ]

        for name, dtype, trailing in specs:
            if trailing:
                shape = (0,) + trailing
                maxshape = (None,) + trailing
                chunks = (cs,) + trailing
            else:
                shape = (0,)
                maxshape = (None,)
                chunks = (cs,)

            grp.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=chunks,
                compression=self._compression,
                compression_opts=self._compression_opts,
            )

    def write_batch(
        self,
        records: list[RLPlyRecord],
        split: str,
    ) -> None:
        """Buffer a batch of records for the given split.

        Args:
            records: List of RLPlyRecord to write.
            split: "train" or "val".

        Raises:
            RuntimeError: If the file is not open.

        Example:
            >>> writer.write_batch(records, "train")
        """
        if self._file is None:
            raise RuntimeError("HDF5 file not open")
        self._buffers[split].extend(records)
        if len(self._buffers[split]) >= self._chunk_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        """Flush buffered records to disk for the given split.

        Args:
            split: "train" or "val".

        Example:
            >>> writer.flush("train")
        """
        buf = self._buffers[split]
        if not buf or self._file is None:
            return

        grp = self._file[split]
        n = len(buf)
        mpl = self._max_prefix_len

        # Build numpy arrays
        board = np.array(
            [r.board_tokens for r in buf], dtype=np.uint8
        )
        color = np.array(
            [r.color_tokens for r in buf], dtype=np.uint8
        )
        traj = np.array(
            [r.traj_tokens for r in buf], dtype=np.uint8
        )

        # Pad move_prefix to max_prefix_len
        prefix_arr = np.zeros(
            (n, mpl), dtype=np.uint16
        )
        lengths = np.zeros(n, dtype=np.uint16)
        for i, r in enumerate(buf):
            plen = len(r.move_prefix)
            if plen > mpl:
                logger.warning(
                    "Prefix len %d > max %d; truncating "
                    "(game_id=%d, ply=%d)",
                    plen, mpl, r.game_id, r.ply_index,
                )
                plen = mpl
            prefix_arr[i, :plen] = r.move_prefix[:plen]
            lengths[i] = plen

        uci_arr = np.array(
            [
                r.move_uci.encode("ascii").ljust(
                    5, b"\x00"
                )
                for r in buf
            ],
            dtype="S5",
        )

        winner = np.array(
            [int(r.is_winner_ply) for r in buf],
            dtype=np.uint8,
        )
        white = np.array(
            [int(r.is_white_ply) for r in buf],
            dtype=np.uint8,
        )
        draw = np.array(
            [int(r.is_draw_ply) for r in buf],
            dtype=np.uint8,
        )
        gid = np.array(
            [r.game_id for r in buf], dtype=np.uint32
        )
        ply_idx = np.array(
            [r.ply_index for r in buf], dtype=np.uint32
        )

        # Resize and append
        old = grp["board_tokens"].shape[0]
        new_size = old + n

        arrays: dict[str, np.ndarray] = {
            "board_tokens": board,
            "color_tokens": color,
            "traj_tokens": traj,
            "move_prefix": prefix_arr,
            "prefix_lengths": lengths,
            "move_uci": uci_arr,
            "is_winner_ply": winner,
            "is_white_ply": white,
            "is_draw_ply": draw,
            "game_id": gid,
            "ply_index": ply_idx,
        }

        for name, arr in arrays.items():
            ds: h5py.Dataset = grp[name]
            ds.resize(new_size, axis=0)
            ds[old:new_size] = arr

        self._buffers[split] = []
        logger.debug(
            "Flushed %d records to %s/ (total=%d)",
            n, split, new_size,
        )

    def close(self) -> None:
        """Write root-level attributes and close HDF5 file.

        Example:
            >>> writer.close()
        """
        if self._file is None:
            return
        for split in _SPLITS:
            self.flush(split)

        self._file.attrs["version"] = "rl_1.0"
        self._file.attrs["train_color"] = self._train_color
        self._file.attrs["max_prefix_len"] = (
            self._max_prefix_len
        )
        self._file.attrs["created_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

        self._file.close()
        self._file = None
        logger.info("RL HDF5 file closed")
