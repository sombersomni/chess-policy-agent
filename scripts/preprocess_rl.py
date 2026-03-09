"""Entry point: run once to preprocess PGN into RL HDF5.

Usage:
    python -m scripts.preprocess_rl --config configs/preprocess_rl.yaml
    python -m scripts.preprocess_rl --config configs/preprocess_rl.yaml \
        --pgn data/small.pgn --max-games 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from chess_sim.config import (
    RLPreprocessConfig,
    load_rl_preprocess_config,
)
from chess_sim.preprocess import (
    RLHdf5Preprocessor,
    RLHdf5Validator,
    RLHdf5Writer,
    RLPGNReader,
    RLPlyParser,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser with None defaults."""
    p = argparse.ArgumentParser(
        description="Preprocess PGN to RL HDF5"
    )
    p.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML preprocess config.",
    )
    p.add_argument(
        "--pgn", type=str, default=None,
        help="Override PGN input path.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Override HDF5 output path.",
    )
    p.add_argument(
        "--max-games", type=int, default=None,
        help="Override max games to process.",
    )
    return p


def main() -> None:
    """Entry point for PGN-to-RL-HDF5 preprocessing.

    Loads config, builds pipeline components, runs preprocessing.

    Example:
        python -m scripts.preprocess_rl \
            --config configs/preprocess_rl.yaml
    """
    raise NotImplementedError("To be implemented")


if __name__ == "__main__":
    main()
