"""HDF5Writer: buffers RawTurnRecords and flushes to HDF5.

Manages resizable chunked datasets with gzip compression.
One writer instance owns a single h5py.File handle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from chess_sim.types import RawTurnRecord

logger = logging.getLogger(__name__)

# Dataset names and their dtypes/shapes (per-sample)
_FIXED_65 = ("board_tokens", "color_tokens", "trajectory_tokens")
_SCALAR = ("outcome", "turn", "game_id")
_SEQ = ("move_tokens", "target_tokens")

_DATASETS: dict[str, tuple[str, tuple[int, ...]]] = {
    "board_tokens":      ("uint8",  (65,)),
    "color_tokens":      ("uint8",  (65,)),
    "trajectory_tokens": ("uint8",  (65,)),
    "move_tokens":       ("uint16", (-1,)),  # placeholder
    "target_tokens":     ("uint16", (-1,)),  # placeholder
    "move_lengths":      ("uint16", ()),
    "outcome":           ("int8",   ()),
    "turn":              ("uint16", ()),
    "game_id":           ("uint32", ()),
}

_SPLITS = ("train", "val")


class HDF5Writer:
    """Buffers RawTurnRecord batches and flushes to HDF5.

    Implements the HDF5Writable protocol. All datasets are created
    with resizable first dimension and chunked storage.

    Attributes:
        _max_seq_len: Padded move sequence dimension.
        _chunk_size: Buffer flush threshold.
        _compression: Compression algorithm name.
        _compression_opts: Compression level.
    """

    def __init__(
        self,
        max_seq_len: int,
        chunk_size: int,
        compression: str,
        compression_opts: int,
    ) -> None:
        """Initialize writer settings.

        Args:
            max_seq_len: Fixed dimension for move sequences.
            chunk_size: Flush buffer when this many records.
            compression: HDF5 compression algorithm.
            compression_opts: Compression level parameter.
        """
        self._max_seq_len = max_seq_len
        self._chunk_size = chunk_size
        self._compression = compression
        self._compression_opts = compression_opts
        self._file: Optional[h5py.File] = None
        self._buffers: dict[str, list[RawTurnRecord]] = {
            "train": [], "val": [],
        }

    def open(self, path: Path, mode: str = "w") -> None:
        """Create or open HDF5 file and initialize datasets.

        Args:
            path: Filesystem path for the HDF5 file.
            mode: File open mode ('w' for create, 'a' for append).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(str(path), mode)
        for split in _SPLITS:
            if split not in self._file:
                grp = self._file.create_group(split)
                self._init_datasets(grp)
        logger.info("Opened HDF5 file: %s (mode=%s)", path, mode)

    def _init_datasets(self, grp: h5py.Group) -> None:
        """Create resizable datasets in a group."""
        seq = self._max_seq_len
        cs = min(self._chunk_size, 1024)

        specs: list[tuple[str, str, tuple[int, ...]]] = [
            ("board_tokens", "uint8", (65,)),
            ("color_tokens", "uint8", (65,)),
            ("trajectory_tokens", "uint8", (65,)),
            ("move_tokens", "uint16", (seq,)),
            ("target_tokens", "uint16", (seq,)),
            ("move_lengths", "uint16", ()),
            ("outcome", "int8", ()),
            ("turn", "uint16", ()),
            ("game_id", "uint32", ()),
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
        records: list[RawTurnRecord],
        split: str,
    ) -> None:
        """Buffer records; auto-flush when >= chunk_size.

        Args:
            records: List of RawTurnRecord to buffer.
            split: Target split ('train' or 'val').
        """
        self._buffers[split].extend(records)
        if len(self._buffers[split]) >= self._chunk_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        """Write buffered records to HDF5 and reset buffer.

        Args:
            split: The split to flush ('train' or 'val').
        """
        buf = self._buffers[split]
        if not buf or self._file is None:
            return

        grp = self._file[split]
        n = len(buf)
        seq = self._max_seq_len

        # Build numpy arrays from buffer
        board = np.array(
            [r.board_tokens for r in buf], dtype=np.uint8
        )
        color = np.array(
            [r.color_tokens for r in buf], dtype=np.uint8
        )
        traj = np.array(
            [r.trajectory_tokens for r in buf],
            dtype=np.uint8,
        )

        # Pad move sequences
        mt = np.zeros((n, seq), dtype=np.uint16)
        tt = np.zeros((n, seq), dtype=np.uint16)
        lengths = np.zeros(n, dtype=np.uint16)
        for i, r in enumerate(buf):
            ml = min(len(r.move_tokens), seq)
            mt[i, :ml] = r.move_tokens[:ml]
            tl = min(len(r.target_tokens), seq)
            tt[i, :tl] = r.target_tokens[:tl]
            lengths[i] = ml

        outcome = np.array(
            [r.outcome for r in buf], dtype=np.int8
        )
        turn = np.array(
            [r.turn for r in buf], dtype=np.uint16
        )
        game_id = np.array(
            [r.game_id for r in buf], dtype=np.uint32
        )

        # Resize and append
        old = grp["board_tokens"].shape[0]
        new = old + n

        arrays = {
            "board_tokens": board,
            "color_tokens": color,
            "trajectory_tokens": traj,
            "move_tokens": mt,
            "target_tokens": tt,
            "move_lengths": lengths,
            "outcome": outcome,
            "turn": turn,
            "game_id": game_id,
        }

        for name, arr in arrays.items():
            ds: h5py.Dataset = grp[name]
            if ds.ndim == 1:
                ds.resize(new, axis=0)
                ds[old:new] = arr
            else:
                ds.resize(new, axis=0)
                ds[old:new] = arr

        self._buffers[split] = []
        logger.debug(
            "Flushed %d records to %s/ (total=%d)",
            n, split, new,
        )

    def close(self) -> None:
        """Flush all remaining buffers and close file."""
        if self._file is None:
            return
        for split in _SPLITS:
            self.flush(split)

        # Write file attributes
        now = datetime.now(timezone.utc).isoformat()
        self._file.attrs["version"] = "2.0"
        self._file.attrs["created_at"] = now
        self._file.attrs["source_checksum"] = ""

        self._file.close()
        self._file = None
        logger.info("HDF5 file closed")
