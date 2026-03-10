"""Offline RL training script for ChessModel on PGN master games.

Usage:
    python -m scripts.train_rl --config configs/train_rl.yaml
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from chess_sim.config import load_pgn_rl_config
from chess_sim.tracking import AimLogHandler, make_tracker
from chess_sim.training.pgn_rl_trainer import PGNRLTrainer

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
        description="Offline RL training on PGN master games."
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return p


def main() -> None:
    """Entry point for offline RL PGN training."""
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

    # Reproducibility
    random.seed(cfg.rl.seed)
    np.random.seed(cfg.rl.seed)
    torch.manual_seed(cfg.rl.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.rl.seed)
    logger.info("Random seed: %d", cfg.rl.seed)

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
        ckpt_path.parent.mkdir(
            parents=True, exist_ok=True
        )

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

    trainer = PGNRLTrainer(
        cfg=cfg,
        device=device,
        total_steps=total_steps,
        tracker=tracker,
    )

    # Load Phase1 checkpoint for warm-start if specified
    if cfg.rl.resume:
        resume_path = Path(cfg.rl.resume)
        trainer.load_checkpoint(resume_path)
        logger.info("Resumed from %s", resume_path)

    n_params = sum(
        p.numel() for p in trainer.model.parameters()
    )
    logger.info("Model parameters: %s", f"{n_params:,}")

    try:
        for epoch in range(1, cfg.rl.epochs + 1):
            metrics = trainer.train_epoch(
                pgn_path,
                max_games=cfg.data.max_games,
            )
            lr_main, lr_value = trainer.current_lrs

            # Evaluation pass (same games, no grad)
            val_metrics = trainer.evaluate(
                pgn_path,
                max_games=cfg.data.max_games,
            )

            # Merge all metrics and ship to Aim
            all_metrics = {
                **metrics,
                **val_metrics,
                "lr_value": lr_value,
            }
            tracker.track_epoch(
                all_metrics, epoch, lr_main
            )

            logger.info(
                "Epoch %02d: total=%.4f pg=%.4f "
                "ce=%.4f "
                "| val_loss=%.4f val_acc=%.4f "
                "| reward=%.4f "
                "adv=%.4f\u00b1%.4f "
                "vl=%.4f games=%d "
                "lr=%.2e lr_v=%.2e",
                epoch,
                metrics["total_loss"],
                metrics["pg_loss"],
                metrics["ce_loss"],
                val_metrics["val_loss"],
                val_metrics["val_accuracy"],
                metrics["mean_reward"],
                metrics["mean_advantage"],
                metrics["std_advantage"],
                metrics["value_loss"],
                metrics["n_games"],
                lr_main,
                lr_value,
            )

            # Board + prediction visuals → Aim
            if _MPL_AVAILABLE:
                try:
                    figs = trainer.sample_visuals(
                        pgn_path,
                        n_plies=_VISUAL_N_PLIES,
                    )
                    for i, fig in enumerate(figs):
                        tracker.track_image(
                            fig,
                            name=f"ply_sample_{i}",
                            step=epoch,
                        )
                        plt.close(fig)
                except Exception:
                    logger.warning(
                        "sample_visuals failed at "
                        "epoch %d",
                        epoch,
                        exc_info=True,
                    )

            # Save checkpoint after every epoch
            if ckpt_path is not None:
                trainer.save_checkpoint(ckpt_path)
                logger.info(
                    "Checkpoint saved: %s", ckpt_path
                )
    except KeyboardInterrupt:
        logger.warning(
            "Training interrupted — saving emergency checkpoint"
        )
        if ckpt_path is not None:
            emerg = ckpt_path.with_stem(
                ckpt_path.stem + "_interrupted"
            )
            trainer.save_checkpoint(emerg)
            logger.info(
                "Emergency checkpoint saved: %s", emerg
            )
        raise
    finally:
        logging.getLogger().removeHandler(aim_handler)
        tracker.close()


if __name__ == "__main__":
    main()
