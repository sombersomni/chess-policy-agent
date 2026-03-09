"""RLHdf5Validator: post-write sanity checks on RL HDF5 files.

Validates schema presence, row-count consistency, value ranges,
and train_color attribute. RL analogue of HDF5Validator.
"""

from __future__ import annotations

from pathlib import Path

from chess_sim.config import RLPreprocessConfig


class RLHdf5Validator:
    """Validate an RL HDF5 file against its preprocessing config.

    Checks that all required datasets exist in each split, row
    counts are consistent, flag fields are 0/1, prefix_lengths
    are within bounds, and the train_color attribute matches.

    Example:
        >>> validator = RLHdf5Validator()
        >>> validator.validate(Path("data/chess_rl.h5"), config)
    """

    def validate(
        self,
        path: Path,
        config: RLPreprocessConfig,
    ) -> None:
        """Run all validation checks on the HDF5 file.

        Args:
            path: Path to the HDF5 file to validate.
            config: The RLPreprocessConfig used to write the file.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If any schema, value, or attribute check fails.

        Example:
            >>> RLHdf5Validator().validate(
            ...     Path("data/chess_rl.h5"), config
            ... )
        """
        raise NotImplementedError("To be implemented")
