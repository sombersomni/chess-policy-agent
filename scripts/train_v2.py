"""Phase 1 training script for ChessModel v2 (encoder-decoder).

Generates synthetic games via random legal moves, builds a
PGNSequenceDataset, and trains using Phase1Trainer with AdamW +
CosineAnnealingLR + CrossEntropyLoss.

Usage:
    # From YAML config (recommended):
    python -m scripts.train_v2 --config configs/train_v2.yaml

    # With CLI overrides:
    python -m scripts.train_v2 --config configs/train_v2.yaml --epochs 10

    # Pure CLI (no YAML):
    python -m scripts.train_v2 --num-games 50 --epochs 5
"""

from __future__ import annotations

import argparse
import logging
import random
import tempfile
from pathlib import Path

import chess.pgn
import torch
from torch.utils.data import DataLoader, random_split

from chess_sim.config import ChessModelV2Config, load_v2_config
from chess_sim.data.hdf5_dataset import (
    ChessHDF5Dataset,
    hdf5_worker_init,
)
from chess_sim.data.pgn_sequence_dataset import (
    PGNSequenceCollator,
    PGNSequenceDataset,
)
from chess_sim.training.phase1_trainer import Phase1Trainer
from scripts.train_real import generate_random_game

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic PGN generation
# ---------------------------------------------------------------------------

def write_synthetic_pgn(num_games: int, seed: int = 42) -> Path:
    """Generate random-legal-move games and write to a temp PGN file.

    Args:
        num_games: Number of games to generate.
        seed: Random seed for reproducibility.

    Returns:
        Path to the temporary PGN file.
    """
    random.seed(seed)
    tmp = Path(tempfile.mktemp(suffix=".pgn"))
    with open(tmp, "w") as fh:
        for i in range(num_games):
            game = generate_random_game()
            print(game, file=fh, end="\n\n")
            if (i + 1) % 10 == 0:
                logger.info("Generated %d/%d games", i + 1, num_games)
    logger.info("Wrote %d games to %s", num_games, tmp)
    return tmp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser with None defaults for merge logic."""
    p = argparse.ArgumentParser(
        description="Phase 1 training for ChessModel v2."
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML v2 config file.",
    )
    p.add_argument(
        "--pgn", type=str, default=None,
        help="Path to a real PGN file (.pgn or .pgn.zst).",
    )
    p.add_argument(
        "--num-games", type=int, default=None,
        help="Number of synthetic games to generate.",
    )
    p.add_argument(
        "--max-games", type=int, default=None,
        help="Max games to load from PGN file.",
    )
    p.add_argument(
        "--winners-only", action="store_true", default=False,
        help="Only include games with decisive result.",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size.",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to save final checkpoint (.pt).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--hdf5", type=str, default=None,
        help="Path to preprocessed HDF5 dataset file.",
    )
    return p


def _merge_v2_config(
    args: argparse.Namespace,
    cfg: ChessModelV2Config,
) -> ChessModelV2Config:
    """Apply non-None CLI args on top of cfg (mutates and returns cfg).

    Args:
        args: Parsed argparse namespace (None = not provided).
        cfg: ChessModelV2Config loaded from YAML or default.

    Returns:
        Updated cfg with CLI overrides applied.
    """
    if args.pgn is not None:
        cfg.data.pgn = args.pgn
    if args.num_games is not None:
        cfg.data.num_games = args.num_games
    if args.max_games is not None:
        cfg.data.max_games = args.max_games
    if args.winners_only:
        cfg.data.winners_only = True
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.trainer.epochs = args.epochs
    if args.checkpoint is not None:
        cfg.trainer.checkpoint = args.checkpoint
    if args.hdf5 is not None:
        cfg.data.hdf5_path = args.hdf5
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for Phase 1 v2 training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = load_v2_config(Path(args.config))
        logger.info("Loaded v2 config from %s", args.config)
    else:
        cfg = ChessModelV2Config()

    cfg = _merge_v2_config(args, cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)

    # Build dataset: HDF5 path takes precedence over PGN
    collator = PGNSequenceCollator()
    is_synthetic = False

    if cfg.data.hdf5_path:
        hdf5_p = Path(cfg.data.hdf5_path)
        logger.info("Using HDF5 dataset: %s", hdf5_p)
        train_ds = ChessHDF5Dataset(hdf5_p, "train")
        val_ds = ChessHDF5Dataset(hdf5_p, "val")
        logger.info(
            "Train: %d  Val: %d",
            len(train_ds), len(val_ds),
        )
        wk_init = hdf5_worker_init
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=cfg.data.num_workers,
            worker_init_fn=wk_init,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.data.num_workers,
            worker_init_fn=wk_init,
        )
    else:
        # Determine PGN source: real file or synthetic
        if cfg.data.pgn:
            pgn_path = Path(cfg.data.pgn)
            logger.info("Using real PGN: %s", pgn_path)
            max_games = cfg.data.max_games
        else:
            pgn_path = write_synthetic_pgn(
                cfg.data.num_games, seed=args.seed
            )
            max_games = cfg.data.num_games
            is_synthetic = True

        logger.info(
            "Building PGNSequenceDataset from %s ...",
            pgn_path,
        )
        full_ds = PGNSequenceDataset(
            pgn_path=str(pgn_path),
            max_games=max_games,
            winners_only=cfg.data.winners_only,
        )
        logger.info("Total samples: %d", len(full_ds))

        n_total = len(full_ds)
        n_train = int(n_total * cfg.data.train_frac)
        n_val = n_total - n_train
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(
                args.seed
            ),
        )
        logger.info(
            "Train: %d  Val: %d",
            len(train_ds), len(val_ds),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

    # Build trainer
    total_steps = cfg.trainer.epochs * len(train_loader)
    trainer = Phase1Trainer(
        device=device,
        total_steps=max(total_steps, 1),
        v2_cfg=cfg,
    )

    # Count parameters
    n_params = sum(
        p.numel() for p in trainer.model.parameters()
    )
    logger.info("Model parameters: %s", f"{n_params:,}")

    # Training loop
    logger.info(
        "Training %d epochs | batches/epoch=%d | total_steps=%d",
        cfg.trainer.epochs, len(train_loader), total_steps,
    )

    epoch_train_losses: list[float] = []
    epoch_val_metrics: list[dict[str, float]] = []

    for epoch in range(1, cfg.trainer.epochs + 1):
        avg_loss = trainer.train_epoch(train_loader)
        epoch_train_losses.append(avg_loss)

        val_metrics = trainer.evaluate(val_loader)
        epoch_val_metrics.append(val_metrics)

        logger.info(
            "Epoch %02d: train_loss=%.4f | "
            "val_loss=%.4f | val_acc=%.4f",
            epoch,
            avg_loss,
            val_metrics["val_loss"],
            val_metrics["val_accuracy"],
        )

    # Summary
    first, last = epoch_train_losses[0], epoch_train_losses[-1]
    logger.info(
        "Train loss: %.4f -> %.4f (delta=%+.4f)",
        first, last, last - first,
    )
    if epoch_val_metrics:
        best_val = min(m["val_loss"] for m in epoch_val_metrics)
        best_acc = max(m["val_accuracy"] for m in epoch_val_metrics)
        logger.info(
            "Best val_loss=%.4f | Best val_acc=%.4f",
            best_val, best_acc,
        )

    # Save checkpoint
    if cfg.trainer.checkpoint:
        ckpt_path = Path(cfg.trainer.checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(ckpt_path)
        logger.info("Checkpoint saved to %s", ckpt_path)

    # Cleanup temp PGN (only if synthetic)
    if is_synthetic:
        pgn_path.unlink(missing_ok=True)
        logger.info("Cleaned up temp PGN: %s", pgn_path)


if __name__ == "__main__":
    main()
