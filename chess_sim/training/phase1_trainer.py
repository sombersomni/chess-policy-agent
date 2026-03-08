"""Phase1Trainer: supervised cross-entropy training on game move sequences.

Trains the ChessModel (encoder-decoder) end-to-end using teacher forcing.
The decoder receives ground-truth move tokens as input and predicts the
next move at each step. Loss is standard cross-entropy over the move
vocabulary, ignoring PAD positions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from chess_sim.config import (
    ChessModelV2Config,
    DecoderConfig,
    ModelConfig,
    TrainerConfig,
)
from chess_sim.data.move_vocab import PAD_IDX
from chess_sim.functional import entropy_from_logits, mean_entropy
from chess_sim.model.chess_model import ChessModel
from chess_sim.tracking.noop_tracker import NoOpTracker
from chess_sim.tracking.protocol import MetricTracker
from chess_sim.training.trainer import log_metrics
from chess_sim.types import GameTurnBatch

logger = logging.getLogger(__name__)


def _to_device(
    batch: GameTurnBatch, device: torch.device
) -> GameTurnBatch:
    """Move all tensor fields in a GameTurnBatch to device."""
    return GameTurnBatch(
        *(
            t.to(device) if isinstance(t, Tensor) else t
            for t in batch
        )
    )


class Phase1Trainer:
    """Supervised CE trainer for the encoder-decoder chess model.

    Implements the Trainable protocol (train_step, train_epoch).

    Owns:
      model:     ChessModel (encoder + decoder)
      optimizer: AdamW
      scheduler: LinearLR warmup + CosineAnnealingLR
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
        tracker: MetricTracker | None = None,
    ) -> None:
        """Initialize model, optimizer, scheduler, and loss function.

        Args:
            device: Torch device string. Use 'cpu' for tests.
            total_steps: Total optimizer steps for scheduler.
            v2_cfg: Optional ChessModelV2Config. When None, uses defaults.
            tracker: Optional MetricTracker for experiment logging.
                     When None, uses NoOpTracker (silent fallback).

        Example:
            >>> trainer = Phase1Trainer(device="cpu")
        """
        cfg = v2_cfg or ChessModelV2Config()
        self.device = torch.device(device)
        self.model = ChessModel(
            cfg.model, cfg.decoder
        ).to(self.device)
        # Expose encoder for compatibility with device_aware
        self.encoder = self.model.encoder
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.trainer.learning_rate,
            weight_decay=cfg.trainer.weight_decay,
        )
        warmup_steps = cfg.trainer.warmup_steps
        cosine_steps = max(total_steps - warmup_steps, 1)
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cosine_steps,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_IDX
        )
        self._gradient_clip = cfg.trainer.gradient_clip
        self._tracker: MetricTracker = tracker or NoOpTracker()
        self._global_step: int = 0

    @log_metrics
    def train_step(self, batch: GameTurnBatch) -> float:
        """Execute one gradient update on a GameTurnBatch.

        Forward pass with teacher forcing, compute CE loss on move logits
        vs target_tokens (ignoring PAD), backward, clip gradients, step.

        Args:
            batch: GameTurnBatch with board tokens, move tokens, targets.

        Returns:
            Scalar loss as a Python float.

        Example:
            >>> loss = trainer.train_step(batch)
            >>> isinstance(loss, float)
            True
        """
        batch = _to_device(batch, self.device)
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(
            batch.board_tokens,
            batch.color_tokens,
            batch.trajectory_tokens,
            batch.move_tokens,
            batch.move_pad_mask,
        )
        B, T, V = logits.shape
        loss = self.criterion(
            logits.view(B * T, V),
            batch.target_tokens.view(B * T),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self._gradient_clip,
        )
        self.optimizer.step()
        self.scheduler.step()
        loss_val = loss.detach().item()
        self._global_step += 1
        self._tracker.track_step(loss_val, self._global_step)
        return loss_val

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
        total_loss = 0.0
        steps = 0
        for batch in loader:
            total_loss += self.train_step(batch)
            steps += 1
        return total_loss / max(steps, 1)

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
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        entropy_sum = 0.0
        entropy_count = 0
        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, self.device)
                logits = self.model(
                    batch.board_tokens,
                    batch.color_tokens,
                    batch.trajectory_tokens,
                    batch.move_tokens,
                    batch.move_pad_mask,
                )
                B, T, V = logits.shape
                loss = self.criterion(
                    logits.view(B * T, V),
                    batch.target_tokens.view(B * T),
                )
                total_loss += loss.item()
                preds = logits.argmax(-1)
                mask = batch.target_tokens != PAD_IDX
                correct += (
                    preds[mask] == batch.target_tokens[mask]
                ).sum().item()
                total += mask.sum().item()
                # Accumulate entropy per batch (weighted)
                n_valid = mask.sum().item()
                if n_valid > 0:
                    h = entropy_from_logits(logits)
                    entropy_sum += h[mask].sum().item()
                    entropy_count += n_valid
        n_batches = max(len(loader), 1)
        mean_h = entropy_sum / max(entropy_count, 1)
        metrics = {
            "val_loss": total_loss / n_batches,
            "val_accuracy": correct / max(total, 1),
            "mean_entropy": mean_h,
        }
        return metrics

    def save_checkpoint(self, path: Path) -> None:
        """Save model state_dict to a checkpoint file.

        Args:
            path: Destination path for the .pt checkpoint file.

        Example:
            >>> trainer.save_checkpoint(Path("checkpoints/p1.pt"))
        """
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model state_dict from a checkpoint file.

        Args:
            path: Path to the .pt checkpoint file.

        Example:
            >>> trainer.load_checkpoint(Path("checkpoints/p1.pt"))
        """
        ckpt = torch.load(
            path, map_location=self.device
        )
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("Checkpoint loaded from %s", path)


def _mean_entropy(
    logits: torch.Tensor, mask: torch.Tensor
) -> float:
    """Compute mean prediction entropy over non-PAD positions.

    Thin wrapper around chess_sim.functional.mean_entropy for
    backward compatibility with existing imports.

    Args:
        logits: Shape (B, T, V) raw logits from the decoder.
        mask: Boolean mask (B, T); True = valid (non-PAD) position.

    Returns:
        Mean entropy in nats as a Python float.

    Example:
        >>> logits = torch.randn(2, 10, 1971)
        >>> mask = torch.ones(2, 10, dtype=torch.bool)
        >>> h = _mean_entropy(logits, mask)
        >>> isinstance(h, float)
        True
    """
    return mean_entropy(logits, mask)
