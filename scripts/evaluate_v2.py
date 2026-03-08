"""Evaluate a ChessModel v2 (encoder-decoder) checkpoint.

Loads a trained v2 checkpoint, runs evaluation on a PGN dataset,
and reports val_loss, val_accuracy, and mean_entropy.

Usage:
    python -m scripts.evaluate_v2 --config configs/evaluate_v2.yaml
    python -m scripts.evaluate_v2 \
        --checkpoint checkpoints/chess_v2_50games.pt \
        --pgn data/games.pgn --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from chess_sim.config import ChessModelV2Config, load_v2_config
from chess_sim.data.pgn_sequence_dataset import (
    PGNSequenceCollator,
    PGNSequenceDataset,
)
from chess_sim.training.phase1_trainer import Phase1Trainer

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with None defaults for merge logic."""
    p = argparse.ArgumentParser(
        description="Evaluate a ChessModel v2 checkpoint on PGN data.",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (e.g. configs/evaluate_v2.yaml).",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt checkpoint file.",
    )
    p.add_argument(
        "--pgn", type=str, default=None,
        help="Path to .pgn or .pgn.zst file.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Evaluation batch size.",
    )
    return p


def _load_eval_extras(path: Path) -> dict[str, Any]:
    """Extract the eval section from a YAML config (not part of ChessModelV2Config)."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return raw.get("eval", {})


def main() -> None:
    """CLI entry point for v2 model evaluation."""
    parser = _build_parser()
    args = parser.parse_args()

    # Load config
    eval_extras: dict[str, Any] = {}
    if args.config:
        cfg = load_v2_config(Path(args.config))
        eval_extras = _load_eval_extras(Path(args.config))
        logger.info("Loaded config from %s", args.config)
    else:
        cfg = ChessModelV2Config()

    # CLI overrides
    if args.pgn is not None:
        cfg.data.pgn = args.pgn
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size

    # Resolve checkpoint: CLI > YAML eval section > empty
    checkpoint_path = (
        args.checkpoint
        or eval_extras.get("checkpoint", "")
        or ""
    )

    if not checkpoint_path:
        parser.error(
            "--checkpoint is required (or set eval.checkpoint in YAML)."
        )
    if not cfg.data.pgn:
        parser.error("--pgn is required (or set data.pgn in YAML).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build dataset and loader
    logger.info("Loading dataset from %s ...", cfg.data.pgn)
    dataset = PGNSequenceDataset(
        pgn_path=cfg.data.pgn,
        max_games=0,
        winners_only=cfg.data.winners_only,
    )
    collator = PGNSequenceCollator()
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    # Build trainer and load checkpoint
    trainer = Phase1Trainer(
        device=device,
        total_steps=1,
        v2_cfg=cfg,
    )
    trainer.load_checkpoint(Path(checkpoint_path))

    # Run evaluation
    logger.info("Evaluating on %d samples (device=%s) ...", len(dataset), device)
    metrics = trainer.evaluate(loader)

    # Print results
    val_acc_pct = metrics["val_accuracy"] * 100.0
    print()
    print("=== ChessModel v2 Evaluation ===")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Dataset    : {cfg.data.pgn} ({len(dataset)} samples)")
    print(f"Device     : {device}")
    print()
    print(f"val_loss     : {metrics['val_loss']:.4f}")
    print(f"val_accuracy : {val_acc_pct:.2f}%")
    print(f"mean_entropy : {metrics['mean_entropy']:.4f} nats")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
