"""Self-play RL fine-tuning entry point.

Loads a CE-trained checkpoint and fine-tunes via self-play
REINFORCE with EMA shadow opponent and KL anchor.

Usage:
    python -m scripts.finetune_rl --config configs/finetune_rl.yaml
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for fine-tune training.

    Returns:
        Configured ArgumentParser with --config argument.

    Example:
        >>> parser = _build_parser()
        >>> args = parser.parse_args(["--config", "cfg.yaml"])
        >>> args.config
        'cfg.yaml'
    """
    p = argparse.ArgumentParser(
        description=(
            "Self-play RL fine-tuning with EMA shadow "
            "opponent and KL anchor."
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
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed for all RNGs.

    Example:
        >>> _setup_reproducibility(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed: %d", seed)


def main(config_path: Path | None = None) -> None:
    """Entry point for self-play RL fine-tuning.

    Loads config, constructs RLFinetuneTrainer, and runs the
    training loop. Catches KeyboardInterrupt and saves an
    emergency checkpoint with _interrupted suffix.

    Args:
        config_path: Optional path override; if None, reads
            from --config CLI argument.

    Example:
        >>> main(Path("configs/finetune_rl.yaml"))
    """
    raise NotImplementedError("To be implemented")


if __name__ == "__main__":
    main()
