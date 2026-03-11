"""Offline RL v4 training script (batched RSCE via DataLoader).

Preprocesses PGN to HDF5 with pre-normalized RSCE multipliers,
then trains using minibatch DataLoader instead of sequential
per-game loop.

Usage:
    python -m scripts.train_rl_v4 --config configs/train_rl_v4.yaml
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from chess_sim.config import PGNRLConfig
from chess_sim.data.chess_rl_dataset import ChessRLDataset
from chess_sim.tracking.protocol import MetricTracker
from chess_sim.training.pgn_rl_trainer_v4 import PGNRLTrainerV4

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for v4 training."""
    p = argparse.ArgumentParser(
        description=(
            "Offline RL v4: batched RSCE training via "
            "DataLoader on preprocessed HDF5."
        )
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return p


def _setup_reproducibility(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed: %d", seed)


def _run_training_loop(
    trainer: PGNRLTrainerV4,
    cfg: PGNRLConfig,
    train_dataset: ChessRLDataset,
    val_dataset: ChessRLDataset,
    ckpt_path: Path | None,
    tracker: MetricTracker,
) -> None:
    """Run the epoch training loop with eval and checkpointing.

    Args:
        trainer: Initialized PGNRLTrainerV4.
        cfg: Full PGN RL config.
        train_dataset: Training split dataset.
        val_dataset: Validation split dataset.
        ckpt_path: Checkpoint save path, or None to skip.
        tracker: Metric tracker for logging.
    """
    raise NotImplementedError("To be implemented")


def main(config_path: Path | None = None) -> None:
    """Entry point for offline RL v4 batched training.

    Args:
        config_path: Optional path override; if None, reads
            from --config CLI argument.

    Example:
        >>> main(Path("configs/train_rl_v4.yaml"))
    """
    raise NotImplementedError("To be implemented")


if __name__ == "__main__":
    main()
