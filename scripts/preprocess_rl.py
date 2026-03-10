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

from chess_sim.config import load_rl_preprocess_config
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
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(name)s %(levelname)s "
            "%(message)s"
        ),
    )

    args = _build_parser().parse_args()
    cfg = load_rl_preprocess_config(Path(args.config))

    if args.pgn:
        cfg.input.pgn_path = args.pgn
    if args.output:
        cfg.output.hdf5_path = args.output
    if args.max_games is not None:
        cfg.input.max_games = args.max_games

    reader = RLPGNReader()
    parser = RLPlyParser(
        cfg.filter.train_color,
        cfg.filter.min_moves,
        cfg.filter.max_moves,
    )
    writer = RLHdf5Writer(
        cfg.output.max_prefix_len,
        cfg.output.chunk_size,
        cfg.output.compression,
        cfg.output.compression_opts,
        cfg.filter.train_color,
    )
    validator = RLHdf5Validator()
    preprocessor = RLHdf5Preprocessor(
        reader, parser, writer, validator
    )
    preprocessor.run(cfg)


if __name__ == "__main__":
    main()
