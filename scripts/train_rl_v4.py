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

from chess_sim.config import PGNRLConfig, load_pgn_rl_config
from chess_sim.data.chess_rl_dataset import ChessRLDataset
from chess_sim.data.pgn_reward_preprocessor import (
    PGNRewardPreprocessor,
)
from chess_sim.tracking import AimLogHandler, make_tracker
from chess_sim.tracking.protocol import MetricTracker
from chess_sim.training.pgn_rl_trainer_v4 import PGNRLTrainerV4

logger = logging.getLogger(__name__)

_AVG_PLIES_PER_GAME: int = 40


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
    try:
        for epoch in range(1, cfg.rl.epochs + 1):
            metrics = trainer.train_epoch(train_dataset)
            val_metrics = trainer.evaluate(val_dataset)
            lr_main = trainer.current_lr

            all_metrics: dict[str, float] = {
                **metrics,
                **val_metrics,
            }
            tracker.track_epoch(
                all_metrics, epoch, lr_main
            )

            logger.info(
                "Epoch %02d: loss=%.4f train_acc=%.4f "
                "| val_loss=%.4f val_acc=%.4f "
                "| n_samples=%d n_games=%d "
                "lr=%.2e",
                epoch,
                metrics["total_loss"],
                metrics["train_accuracy"],
                val_metrics["val_loss"],
                val_metrics["val_accuracy"],
                metrics["n_samples"],
                metrics["n_games"],
                lr_main,
            )

            if ckpt_path is not None:
                trainer.save_checkpoint(ckpt_path)
                logger.info(
                    "Checkpoint saved: %s", ckpt_path
                )

    except KeyboardInterrupt:
        logger.warning(
            "Training interrupted — saving emergency "
            "checkpoint"
        )
        if ckpt_path is not None:
            emergency = ckpt_path.with_stem(
                ckpt_path.stem + "_interrupted"
            )
            trainer.save_checkpoint(emergency)
            logger.info(
                "Emergency checkpoint saved: %s",
                emergency,
            )
        raise


def main(config_path: Path | None = None) -> None:
    """Entry point for offline RL v4 batched training.

    Args:
        config_path: Optional path override; if None, reads
            from --config CLI argument.

    Example:
        >>> main(Path("configs/train_rl_v4.yaml"))
    """
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(name)s %(levelname)s "
            "%(message)s"
        ),
    )

    if config_path is None:
        args = _build_parser().parse_args()
        config_path = Path(args.config)

    cfg = load_pgn_rl_config(config_path)
    logger.info("Loaded config from %s", config_path)

    _setup_reproducibility(cfg.rl.seed)

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    pgn_path = Path(cfg.data.pgn)
    if not pgn_path.exists():
        raise FileNotFoundError(
            f"PGN file not found: {pgn_path}"
        )

    ckpt_path: Path | None = (
        Path(cfg.rl.checkpoint) if cfg.rl.checkpoint
        else None
    )
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(
            parents=True, exist_ok=True
        )

    # Estimate total optimizer steps for LR schedule
    max_g = max(cfg.data.max_games, 1)
    est_samples = max_g * _AVG_PLIES_PER_GAME
    est_steps_per_epoch = max(
        est_samples // cfg.rl.batch_size, 1
    )
    total_steps = cfg.rl.epochs * est_steps_per_epoch
    logger.info(
        "Estimated total_steps: %d", total_steps
    )

    tracker = make_tracker(cfg.aim)
    aim_handler = AimLogHandler(tracker)
    logging.getLogger().addHandler(aim_handler)

    train_ds: ChessRLDataset | None = None
    val_ds: ChessRLDataset | None = None

    try:
        # Preprocess PGN to HDF5
        preprocessor = PGNRewardPreprocessor(
            cfg, device="cpu"
        )
        hdf5_path = Path(
            cfg.rl.hdf5_path
            or pgn_path.with_suffix(".h5")
        )
        preprocessor.generate(
            pgn_path, hdf5_path, cfg.data.max_games
        )

        # Build datasets
        train_ds = ChessRLDataset(
            hdf5_path,
            val_split_fraction=cfg.rl.val_split_fraction,
            split="train",
        )
        val_ds = ChessRLDataset(
            hdf5_path,
            val_split_fraction=cfg.rl.val_split_fraction,
            split="val",
        )

        logger.info(
            "Train: %d samples (%d games), "
            "Val: %d samples (%d games)",
            len(train_ds),
            train_ds.n_games,
            len(val_ds),
            val_ds.n_games,
        )

        trainer = PGNRLTrainerV4(
            cfg=cfg,
            device=device,
            total_steps=total_steps,
            tracker=tracker,
        )

        if cfg.rl.resume:
            resume_path = Path(cfg.rl.resume)
            if not resume_path.exists():
                raise FileNotFoundError(
                    "Resume checkpoint not found: "
                    f"{resume_path}"
                )
            trainer.load_checkpoint(resume_path)
            logger.info(
                "Resumed from %s", resume_path
            )

        n_params = sum(
            p.numel()
            for p in trainer.model.parameters()
        )
        logger.info(
            "Model parameters: %s", f"{n_params:,}"
        )

        _run_training_loop(
            trainer,
            cfg,
            train_ds,
            val_ds,
            ckpt_path,
            tracker,
        )
    finally:
        if train_ds is not None:
            train_ds.close()
        if val_ds is not None:
            val_ds.close()
        logging.getLogger().removeHandler(aim_handler)
        tracker.close()


if __name__ == "__main__":
    main()
