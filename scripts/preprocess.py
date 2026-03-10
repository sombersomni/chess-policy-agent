"""Entry point: run once to preprocess PGN into HDF5.

Usage:
    python -m scripts.preprocess --config configs/preprocess_v2.yaml
    python -m scripts.preprocess --config configs/preprocess_v2.yaml --pgn data/small.pgn --max-games 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from chess_sim.config import (
    load_preprocess_v2_config,
)
from chess_sim.data.reader import StreamingPGNReader
from chess_sim.preprocess import (
    GameParser,
    HDF5Preprocessor,
    HDF5Validator,
    HDF5Writer,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser with None defaults."""
    p = argparse.ArgumentParser(
        description="Preprocess PGN to HDF5"
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
    """Entry point for PGN-to-HDF5 preprocessing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_preprocess_v2_config(Path(args.config))
    if args.pgn:
        cfg.input.pgn_path = args.pgn
    if args.output:
        cfg.output.hdf5_path = args.output
    if args.max_games is not None:
        cfg.input.max_games = args.max_games

    reader = StreamingPGNReader()
    parser_obj = GameParser(
        cfg.filter.min_moves,
        cfg.filter.max_moves,
        cfg.filter.winners_only,
        cfg.filter.min_elo,
    )
    writer = HDF5Writer(
        cfg.output.max_seq_len,
        cfg.output.chunk_size,
        cfg.output.compression,
        cfg.output.compression_opts,
    )
    validator = HDF5Validator()
    preprocessor = HDF5Preprocessor(
        reader, parser_obj, writer, validator
    )
    preprocessor.run(cfg)


if __name__ == "__main__":
    main()
