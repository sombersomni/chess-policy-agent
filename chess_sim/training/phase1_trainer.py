"""Phase1Trainer: supervised cross-entropy training on game move sequences.

Trains the ChessModel (encoder-decoder) end-to-end using teacher forcing.
The decoder receives ground-truth move tokens as input and predicts the
next move at each step. Loss is standard cross-entropy over the move
vocabulary, ignoring PAD positions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from chess_sim.config import ChessModelV2Config, DecoderConfig, ModelConfig, TrainerConfig
from chess_sim.model.chess_model import ChessModel
from chess_sim.training.trainer import device_aware, log_metrics
from chess_sim.types import GameTurnBatch


class Phase1Trainer:
    """Supervised CE trainer for the encoder-decoder chess model.

    Implements the Trainable protocol (train_step, train_epoch).

    Owns:
      model:     ChessModel (encoder + decoder)
      optimizer: AdamW
      scheduler: CosineAnnealingLR
      criterion: CrossEntropyLoss (ignore_index=PAD_IDX)

    Example:
        >>> trainer = Phase1Trainer(device="cpu")
        >>> loss = trainer.train_step(batch)
    """

    def __init__(
        self,
        device: str = "cpu",
        total_steps: int = 10_000,
        v2_cfg: ChessModelV2Config | None = None,
    ) -> None:
        """Initialize model, optimizer, scheduler, and loss function.

        Args:
            device: Torch device string. Use 'cpu' for tests.
            total_steps: Total optimizer steps for scheduler. Default 10000.
            v2_cfg: Optional ChessModelV2Config. When None, uses defaults.

        Example:
            >>> trainer = Phase1Trainer(device="cpu")
        """
        raise NotImplementedError("To be implemented")

    @log_metrics
    @device_aware
    def train_step(self, batch: GameTurnBatch) -> float:
        """Execute one gradient update on a GameTurnBatch.

        Forward pass with teacher forcing, compute CE loss on move logits
        vs target_tokens (ignoring PAD), backward, clip gradients, step.

        Args:
            batch: GameTurnBatch with board tokens, move tokens, and targets.

        Returns:
            Scalar loss as a Python float.

        Example:
            >>> loss = trainer.train_step(batch)
            >>> isinstance(loss, float)
            True
        """
        raise NotImplementedError("To be implemented")

    @log_metrics
    def train_epoch(self, loader: DataLoader) -> float:
        """Run one full epoch and return the mean loss.

        Args:
            loader: DataLoader yielding GameTurnBatch instances.

        Returns:
            Mean loss across all batches in the epoch.

        Example:
            >>> avg_loss = trainer.train_epoch(loader)
        """
        raise NotImplementedError("To be implemented")

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a validation DataLoader.

        Computes loss and top-1 move accuracy without gradient computation.

        Args:
            loader: DataLoader yielding GameTurnBatch instances.

        Returns:
            Dict with keys 'val_loss' and 'val_accuracy'.

        Example:
            >>> metrics = trainer.evaluate(val_loader)
            >>> metrics['val_loss']
            2.5
        """
        raise NotImplementedError("To be implemented")

    def save_checkpoint(self, path: Path) -> None:
        """Save model state_dict to a checkpoint file.

        Args:
            path: Destination path for the .pt checkpoint file.

        Example:
            >>> trainer.save_checkpoint(Path("checkpoints/phase1_best.pt"))
        """
        raise NotImplementedError("To be implemented")

    def load_checkpoint(self, path: Path) -> None:
        """Load model state_dict from a checkpoint file.

        Args:
            path: Path to the .pt checkpoint file.

        Example:
            >>> trainer.load_checkpoint(Path("checkpoints/phase1_best.pt"))
        """
        raise NotImplementedError("To be implemented")
