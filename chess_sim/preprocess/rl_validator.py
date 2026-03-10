"""RLHdf5Validator: post-write sanity checks on RL HDF5 files.

Validates schema presence, row-count consistency, value ranges,
and train_color attribute. RL analogue of HDF5Validator.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from chess_sim.config import RLPreprocessConfig

logger = logging.getLogger(__name__)

_REQUIRED_DATASETS = (
    "board_tokens",
    "color_tokens",
    "traj_tokens",
    "move_prefix",
    "prefix_lengths",
    "move_uci",
    "is_winner_ply",
    "is_white_ply",
    "is_draw_ply",
    "game_id",
    "ply_index",
)


class RLHdf5Validator:
    """Validate an RL HDF5 file against its preprocessing config.

    Checks required datasets, row-count consistency, flag ranges,
    prefix_lengths bounds, and train_color attribute match.

    Example:
        >>> validator = RLHdf5Validator()
        >>> validator.validate(Path("data/chess_rl.h5"), cfg)
    """

    def validate(
        self,
        path: Path,
        config: RLPreprocessConfig,
    ) -> None:
        """Run all validation checks on the HDF5 file.

        Args:
            path: Path to the HDF5 file to validate.
            config: The RLPreprocessConfig used to write it.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If any check fails.

        Example:
            >>> RLHdf5Validator().validate(
            ...     Path("data/chess_rl.h5"), config
            ... )
        """
        if not path.exists():
            raise FileNotFoundError(
                f"HDF5 not found: {path}"
            )

        max_pl = config.output.max_prefix_len

        with h5py.File(str(path), "r") as f:
            # Version check
            version = f.attrs.get("version", "")
            if version != "rl_1.0":
                raise ValueError(
                    "Expected version='rl_1.0', "
                    f"got '{version}'"
                )

            # train_color check
            stored_color = f.attrs.get("train_color", "")
            expected = config.filter.train_color
            if stored_color != expected:
                raise ValueError(
                    f"train_color mismatch: file has "
                    f"'{stored_color}', config has "
                    f"'{expected}'"
                )

            # Both splits present
            for split in ("train", "val"):
                if split not in f:
                    raise ValueError(
                        f"Missing group: {split}"
                    )
                self._validate_group(
                    f[split], split, max_pl
                )

        logger.info("RL HDF5 validation passed: %s", path)

    def _validate_group(
        self,
        grp: h5py.Group,
        split: str,
        max_prefix_len: int,
    ) -> None:
        """Validate a single split group."""
        # All datasets present
        for name in _REQUIRED_DATASETS:
            if name not in grp:
                raise ValueError(
                    f"{split}/ missing dataset: {name}"
                )

        # Consistent row counts
        n = grp["board_tokens"].shape[0]
        for name in _REQUIRED_DATASETS:
            ds_n = grp[name].shape[0]
            if ds_n != n:
                raise ValueError(
                    f"{split}/{name} has {ds_n} rows, "
                    f"expected {n}"
                )

        if n == 0:
            return

        # Value range checks
        bt = grp["board_tokens"][:]
        if np.any(bt > 7):
            raise ValueError(
                f"{split}/board_tokens has values > 7"
            )

        ct = grp["color_tokens"][:]
        if np.any(ct > 2):
            raise ValueError(
                f"{split}/color_tokens has values > 2"
            )

        tt = grp["traj_tokens"][:]
        if np.any(tt > 4):
            raise ValueError(
                f"{split}/traj_tokens has values > 4"
            )

        # Flag fields must be 0 or 1
        for flag_name in (
            "is_winner_ply",
            "is_white_ply",
            "is_draw_ply",
        ):
            vals = grp[flag_name][:]
            if not np.all(np.isin(vals, [0, 1])):
                raise ValueError(
                    f"{split}/{flag_name} has values "
                    "outside {{0, 1}}"
                )

        # prefix_lengths bounds
        pl = grp["prefix_lengths"][:]
        if np.any(pl > max_prefix_len):
            raise ValueError(
                f"{split}/prefix_lengths has values "
                f"> {max_prefix_len}"
            )
