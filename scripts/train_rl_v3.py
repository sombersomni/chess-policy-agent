"""Offline RL v3 training script (RSCE: reward-scaled CE loss).

Usage:
    python -m scripts.train_rl_v3 --config configs/train_rl_v3.yaml
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from chess_sim.config import PGNRLConfig, load_pgn_rl_config
from chess_sim.tracking import AimLogHandler, make_tracker
from chess_sim.tracking.protocol import MetricTracker
from chess_sim.training.pgn_rl_trainer_v3 import (
    PGNRLTrainerV3,
)

try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    _MPL_AVAILABLE = False

_VISUAL_N_PLIES: int = 4

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Offline RL v3 training on PGN master games "
            "(RSCE: reward-scaled cross-entropy loss)."
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
    trainer: PGNRLTrainerV3,
    cfg: PGNRLConfig,
    pgn_path: Path,
    ckpt_path: Path | None,
    tracker: MetricTracker,
) -> None:
    """Run the epoch training loop with eval and checkpointing.

    Args:
        trainer: Initialized PGNRLTrainerV3.
        cfg: Full PGN RL config.
        pgn_path: Path to the PGN file for training/eval.
        ckpt_path: Checkpoint save path, or None to skip.
        tracker: Metric tracker for logging.
    """
    try:
        for epoch in range(1, cfg.rl.epochs + 1):
            metrics = trainer.train_epoch(
                pgn_path,
                max_games=cfg.data.max_games,
            )
            lr_main = trainer.current_lr

            val_metrics = trainer.evaluate(
                pgn_path,
                max_games=cfg.data.max_games,
            )

            overlap = metrics.keys() & val_metrics.keys()
            if overlap:
                raise ValueError(
                    "Metric key collision between "
                    f"train and val: {overlap}"
                )
            all_metrics: dict[str, float] = {
                **metrics,
                **val_metrics,
            }
            tracker.track_epoch(
                all_metrics, epoch, lr_main
            )

            logger.info(
                "Epoch %02d: total=%.4f rsce=%.4f "
                "| val_loss=%.4f val_acc=%.4f "
                "| reward=%.4f "
                "games=%d "
                "lr=%.2e",
                epoch,
                metrics["total_loss"],
                metrics["rsce_loss"],
                val_metrics["val_loss"],
                val_metrics["val_accuracy"],
                metrics["mean_reward"],
                metrics["n_games"],
                lr_main,
            )

            if _MPL_AVAILABLE:
                try:
                    figs = trainer.sample_visuals(
                        pgn_path,
                        n_plies=_VISUAL_N_PLIES,
                        train_accuracy=val_metrics.get(
                            "val_accuracy"
                        ),
                    )
                    for i, fig in enumerate(figs):
                        tracker.track_image(
                            fig,
                            name=f"ply_sample_{i}",
                            step=epoch,
                        )
                        plt.close(fig)
                except (
                    RuntimeError,
                    ValueError,
                    OSError,
                ):
                    logger.warning(
                        "sample_visuals failed at "
                        "epoch %d",
                        epoch,
                        exc_info=True,
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
            emergency_ckpt_path = ckpt_path.with_stem(
                ckpt_path.stem + "_interrupted"
            )
            trainer.save_checkpoint(emergency_ckpt_path)
            logger.info(
                "Emergency checkpoint saved: %s",
                emergency_ckpt_path,
            )
        raise


def main() -> None:
    """Entry point for offline RL v3 PGN training."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(name)s %(levelname)s "
            "%(message)s"
        ),
    )

    args = _build_parser().parse_args()
    cfg = load_pgn_rl_config(Path(args.config))
    logger.info("Loaded config from %s", args.config)

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
        Path(cfg.rl.checkpoint)
        if cfg.rl.checkpoint
        else None
    )
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = cfg.rl.epochs * max(
        cfg.data.max_games, 1
    )
    if cfg.data.max_games == 0:
        logger.warning(
            "max_games=0 (all games) — total_steps "
            "estimate is epochs=%d only. LR schedule "
            "may be inaccurate. Set data.max_games "
            "explicitly for a correct schedule.",
            cfg.rl.epochs,
        )
    logger.info("Estimated total_steps: %d", total_steps)

    tracker = make_tracker(cfg.aim)
    aim_handler = AimLogHandler(tracker)
    logging.getLogger().addHandler(aim_handler)

    try:
        trainer = PGNRLTrainerV3(
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
            trainer, cfg, pgn_path, ckpt_path, tracker
        )
    finally:
        logging.getLogger().removeHandler(aim_handler)
        tracker.close()


if __name__ == "__main__":
    main()
