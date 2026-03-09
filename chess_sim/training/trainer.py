"""Trainer: orchestrates the training loop for the chess encoder.

Owns ChessEncoder, PredictionHeads, LossComputer, AdamW optimizer, and
cosine LR scheduler with linear warmup. Handles gradient clipping,
checkpointing, early stopping, and metric logging via decorators.

Decorators (cross-cutting concerns):
  @log_metrics   -- emits structured loss/accuracy logs to stdout.
  @device_aware  -- handles tensor device placement (GPU training, CPU testing).
  @timed         -- optional profiling decorator (no-op stub in this scaffold).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from chess_sim.config import ModelConfig, TrainerConfig
from chess_sim.model.encoder import ChessEncoder
from chess_sim.model.heads import PredictionHeads
from chess_sim.protocols import Trainable
from chess_sim.training.loss import LossComputer
from chess_sim.types import ChessBatch

F = TypeVar("F", bound=Callable[..., Any])

# Optimizer hyperparameters (from design doc).
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 0.01
WARMUP_STEPS: int = 1000
GRADIENT_CLIP: float = 1.0
PATIENCE: int = 3  # early stopping patience in epochs


# ---------------------------------------------------------------------------
# Cross-cutting concern decorators
# ---------------------------------------------------------------------------

def log_metrics(fn: F) -> F:
    """Decorator: logs loss and per-head accuracy to stdout after each call.

    Wraps train_step or train_epoch. Emits structured log lines without
    modifying the return value. Implementation logs at minimum: step loss,
    learning rate, and per-head top-1 accuracy.

    Args:
        fn: The function to wrap (train_step or train_epoch).

    Returns:
        Wrapped function with identical signature and return value.

    Example:
        >>> @log_metrics
        ... def train_step(self, batch): ...
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        self_arg = args[0]
        lr = (
            self_arg.scheduler.get_last_lr()[0]
            if hasattr(self_arg, 'scheduler')
            else 0.0
        )
        print(
            f"[log_metrics] loss={result:.4f} lr={lr:.2e}"
        )
        return result
    return wrapper  # type: ignore[return-value]


def device_aware(fn: F) -> F:
    """Decorator: ensures all tensors in the batch are moved to the correct device.

    Reads the device from the Trainer's encoder parameters. Moves all tensors
    in the ChessBatch namedtuple to the target device before calling fn.
    Defaults to GPU when available; CPU for unit tests.

    Args:
        fn: The function to wrap (train_step or train_epoch).

    Returns:
        Wrapped function with identical signature and return value.

    Example:
        >>> @device_aware
        ... def train_step(self, batch): ...
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        self_arg = args[0]
        device = next(self_arg.encoder.parameters()).device
        if len(args) > 1 and hasattr(args[1], '_fields'):
            batch = args[1]
            moved = type(batch)(
                *[
                    t.to(device)
                    if isinstance(t, torch.Tensor)
                    else t
                    for t in batch
                ]
            )
            args = (self_arg, moved) + args[2:]
        return fn(*args, **kwargs)
    return wrapper  # type: ignore[return-value]


def timed(fn: F) -> F:
    """Decorator: records wall-clock time of the wrapped call for profiling.

    No-op in the scaffold — records nothing. Implementation should log elapsed
    ms to a profiling dict or stdout. Useful for profiling data pipeline stages.

    Args:
        fn: The function to profile.

    Returns:
        Wrapped function with identical signature and return value.

    Example:
        >>> @timed
        ... def heavy_preprocessing(): ...
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import time
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(
            f"[timed] {fn.__name__} took {elapsed_ms:.1f}ms"
        )
        return result
    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the chess encoder pretraining loop.

    Implements the Trainable protocol.

    Owns:
      encoder:   ChessEncoder
      heads:     PredictionHeads
      loss_fn:   LossComputer
      optimizer: AdamW(lr=3e-4, weight_decay=0.01)
      scheduler: CosineAnnealingLR with 1000-step linear warmup

    Gradient clipping is applied at max_norm=1.0 before each optimizer step.
    Checkpointing saves the best validation loss. Early stopping uses patience=3.

    Example:
        >>> trainer = Trainer(device='cpu')
        >>> loss = trainer.train_step(batch)
    """

    def __init__(
        self,
        device: str = "cpu",
        total_steps: int = 10_000,
        trainer_cfg: TrainerConfig | None = None,
        model_cfg: ModelConfig | None = None,
    ) -> None:
        """Initialize encoder, heads, optimizer, and scheduler.

        Falls back to module-level constants when trainer_cfg / model_cfg
        are None — preserving full backward compatibility.

        Args:
            device: torch device string. Use 'cpu' for tests, 'cuda' for training.
            total_steps: Optimizer steps across all epochs (epochs * batches_per_epoch).
                Defaults to 10_000.
            trainer_cfg: Optional TrainerConfig for lr, weight_decay, warmup_steps,
                and gradient_clip. When None, module constants are used.
            model_cfg: Optional ModelConfig for encoder / heads architecture.
                When None, the default architecture (d_model=256, 6 layers) is used.

        Example:
            >>> trainer = Trainer(device='cpu', total_steps=5 * 9)
            >>> from chess_sim.config import TrainerConfig
            >>> trainer = Trainer(trainer_cfg=TrainerConfig(learning_rate=1e-3))
        """
        self.device = device
        lr = trainer_cfg.learning_rate if trainer_cfg else LEARNING_RATE
        wd = trainer_cfg.weight_decay if trainer_cfg else WEIGHT_DECAY
        self._gradient_clip = (
            trainer_cfg.gradient_clip if trainer_cfg else GRADIENT_CLIP
        )
        self.encoder = ChessEncoder(model_cfg).to(device)
        self.heads = PredictionHeads(model_cfg).to(device)
        self.loss_fn = LossComputer()
        params = (
            list(self.encoder.parameters())
            + list(self.heads.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=wd
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )

    @log_metrics
    @device_aware
    def train_step(self, batch: ChessBatch) -> float:
        """Execute one gradient update and return the scalar loss.

        Steps: forward pass -> loss computation -> zero_grad -> backward ->
        gradient clip -> optimizer step -> scheduler step.

        Args:
            batch: A ChessBatch with board/color tokens and four label tensors.

        Returns:
            Combined cross-entropy loss as a Python float.

        Example:
            >>> loss = trainer.train_step(batch)
            >>> isinstance(loss, float)
            True
        """
        self.encoder.train()
        self.heads.train()
        output = self.encoder(
            batch.board_tokens,
            batch.color_tokens,
            batch.trajectory_tokens,
        )
        preds = self.heads(output.cls_embedding)
        from chess_sim.types import LabelTensors
        labels = LabelTensors(
            batch.src_sq, batch.tgt_sq,
        )
        loss = self.loss_fn.compute(preds, labels)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.heads.parameters()),
            self._gradient_clip,
        )
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach().item()

    @log_metrics
    def train_epoch(self, loader: DataLoader) -> float:
        """Run all batches in loader through train_step; return mean epoch loss.

        Args:
            loader: DataLoader yielding ChessBatch instances.

        Returns:
            Mean loss across all batches in the epoch.

        Example:
            >>> avg = trainer.train_epoch(loader)
        """
        total = 0.0
        count = 0
        for batch in loader:
            total += self.train_step(batch)
            count += 1
        return total / count if count > 0 else 0.0

    def save_checkpoint(self, path: Path) -> None:
        """Save encoder and heads state_dicts to a .pt checkpoint file.

        Args:
            path: Destination path for the checkpoint file.

        Example:
            >>> trainer.save_checkpoint(Path("checkpoints/best.pt"))
        """
        torch.save({
            'encoder': self.encoder.state_dict(),
            'heads': self.heads.state_dict(),
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load encoder and heads state_dicts from a .pt checkpoint file.

        Args:
            path: Path to the checkpoint file to load.

        Example:
            >>> trainer.load_checkpoint(Path("checkpoints/best.pt"))
        """
        ckpt = torch.load(
            path, map_location=self.device, weights_only=True
        )
        self.encoder.load_state_dict(
            ckpt['encoder'], strict=False
        )
        self.heads.load_state_dict(ckpt['heads'], strict=False)
