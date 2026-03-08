"""HDF5Validator: post-preprocessing schema and value-range checks.

Raises ValueError on any violation. Designed to run immediately
after HDF5Writer.close() to catch corruption early.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from chess_sim.config import PreprocessV2Config

logger = logging.getLogger(__name__)

_REQUIRED_DATASETS = (
    "board_tokens", "color_tokens", "trajectory_tokens",
    "move_tokens", "target_tokens", "move_lengths",
    "outcome", "turn", "game_id",
)


class HDF5Validator:
    """Validates HDF5 file schema, shapes, and value ranges.

    Implements the HDF5Validatable protocol.
    """

    def validate(
        self, path: Path, config: PreprocessV2Config
    ) -> None:
        """Check schema, shapes, and value ranges.

        Args:
            path: Path to the HDF5 file.
            config: Preprocessing config for max_seq_len etc.

        Raises:
            ValueError: On any schema or data violation.
        """
        max_seq = config.output.max_seq_len

        with h5py.File(str(path), "r") as f:
            # Check version attribute
            version = f.attrs.get("version", "")
            if version != "2.0":
                raise ValueError(
                    f"Expected version='2.0', got '{version}'"
                )

            # Check both splits present
            for split in ("train", "val"):
                if split not in f:
                    raise ValueError(
                        f"Missing group: {split}"
                    )
                grp = f[split]
                self._validate_group(grp, split, max_seq)

        logger.info("Validation passed: %s", path)

    def _validate_group(
        self,
        grp: h5py.Group,
        split: str,
        max_seq_len: int,
    ) -> None:
        """Validate a single split group."""
        # Check all datasets present
        for name in _REQUIRED_DATASETS:
            if name not in grp:
                raise ValueError(
                    f"{split}/ missing dataset: {name}"
                )

        # Check consistent N across datasets
        n = grp["board_tokens"].shape[0]
        for name in _REQUIRED_DATASETS:
            ds_n = grp[name].shape[0]
            if ds_n != n:
                raise ValueError(
                    f"{split}/{name} has {ds_n} rows, "
                    f"expected {n}"
                )

        # Skip value range checks if empty
        if n == 0:
            return

        # Value range checks (sample all data)
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

        tt = grp["trajectory_tokens"][:]
        if np.any(tt > 4):
            raise ValueError(
                f"{split}/trajectory_tokens values > 4"
            )

        oc = grp["outcome"][:]
        valid_outcomes = np.isin(oc, [-1, 0, 1])
        if not np.all(valid_outcomes):
            raise ValueError(
                f"{split}/outcome has invalid values"
            )

        ml = grp["move_lengths"][:]
        if np.any(ml < 1) or np.any(ml > max_seq_len):
            raise ValueError(
                f"{split}/move_lengths out of "
                f"[1, {max_seq_len}]"
            )
