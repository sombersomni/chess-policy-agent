"""PGNRLTrainerV4: batched minibatch CE trainer using DataLoader.

Replaces the sequential per-game loop of V3 with a standard
DataLoader minibatch loop over a preprocessed HDF5 dataset.
Does NOT inherit from V2/V3 -- standalone trainer with its own
model, optimizer, scheduler, and structural mask.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from chess_sim.config import PGNRLConfig
from chess_sim.data.move_vocab import SOS_IDX
from chess_sim.data.structural_mask import (
    StructuralMaskBuilder,
)
from chess_sim.model.chess_model import ChessModel
from chess_sim.protocols import StructuralMaskable
from chess_sim.tracking.noop_tracker import NoOpTracker
from chess_sim.tracking.protocol import MetricTracker

if TYPE_CHECKING:
    from chess_sim.data.chess_rl_dataset import (
        ChessRLDataset,
    )

logger = logging.getLogger(__name__)


class PGNRLTrainerV4:
    """Batched CE trainer over preprocessed HDF5 dataset.

    Owns model, optimizer, scheduler, and structural mask.
    Accepts ChessRLDataset and wraps it in a DataLoader internally.

    Example:
        >>> trainer = PGNRLTrainerV4(cfg, device="cpu",
        ...                          total_steps=10_000)
        >>> metrics = trainer.train_epoch(train_dataset)
        >>> assert "total_loss" in metrics
    """

    def __init__(
        self,
        cfg: PGNRLConfig,
        device: str = "cpu",
        total_steps: int = 10_000,
        tracker: MetricTracker | None = None,
    ) -> None:
        """Initialize model, optimizer, scheduler, structural mask.

        LR schedule: linear warmup -> constant -> cosine decay,
        parameterized by warmup_fraction and decay_start_fraction.

        Args:
            cfg: PGNRLConfig with all training parameters.
            device: Torch device string ("cpu" or "cuda").
            total_steps: Total optimizer steps for LR schedule.
            tracker: Optional MetricTracker for logging.

        Example:
            >>> trainer = PGNRLTrainerV4(PGNRLConfig())
        """
        self._cfg = cfg
        self._device = torch.device(device)
        self._total_steps = total_steps
        self._tracker: MetricTracker = (
            tracker or NoOpTracker()
        )

        self._model = ChessModel(
            cfg.model, cfg.decoder
        ).to(self._device)

        self._opt = torch.optim.AdamW(
            self._model.parameters(),
            lr=cfg.rl.learning_rate,
            weight_decay=cfg.rl.weight_decay,
        )

        # LR schedule: warmup -> constant -> cosine decay
        warmup_steps = max(
            int(cfg.rl.warmup_fraction * total_steps), 1
        )
        decay_start = max(
            int(
                cfg.rl.decay_start_fraction * total_steps
            ),
            warmup_steps + 1,
        )
        constant_steps = decay_start - warmup_steps
        cosine_steps = max(total_steps - decay_start, 1)

        warmup = torch.optim.lr_scheduler.LinearLR(
            self._opt,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        constant = torch.optim.lr_scheduler.ConstantLR(
            self._opt,
            factor=1.0,
            total_iters=constant_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._opt,
            T_max=cosine_steps,
            eta_min=cfg.rl.min_lr,
        )
        self._sched = (
            torch.optim.lr_scheduler.SequentialLR(
                self._opt,
                schedulers=[warmup, constant, cosine],
                milestones=[warmup_steps, decay_start],
            )
        )

        # Structural mask: suppress logits for tokens whose
        # from-square has no player piece.
        self._struct_mask: StructuralMaskable | None = (
            StructuralMaskBuilder(self._device)
            if cfg.rl.use_structural_mask
            else None
        )

        self._global_step: int = 0

    @property
    def model(self) -> ChessModel:
        """Expose the underlying ChessModel.

        Returns:
            The ChessModel instance owned by this trainer.

        Example:
            >>> trainer.model.encoder
        """
        return self._model

    @property
    def current_lr(self) -> float:
        """Return the current learning rate (single param group).

        Returns:
            Float learning rate.

        Example:
            >>> trainer.current_lr
            0.0001
        """
        return self._opt.param_groups[0]["lr"]

    def train_epoch(
        self,
        dataset: ChessRLDataset,
    ) -> dict[str, float]:
        """Train one epoch over the dataset using DataLoader.

        Creates a DataLoader with cfg batch_size, num_workers,
        shuffle=True, pin_memory=(device != "cpu"). For each
        batch: forward pass, structural mask, per-sample CE with
        reduction="none", element-wise multiply by multiplier,
        scalar mean, backward, grad clip, optimizer + scheduler
        step.

        Args:
            dataset: ChessRLDataset (train split).

        Returns:
            Dict with keys: total_loss, n_samples,
            mean_multiplier, n_games.

        Example:
            >>> m = trainer.train_epoch(train_ds)
            >>> m["rsce_loss"]
            3.14
        """
        self._model.train()
        dl = self._build_dataloader(dataset, shuffle=True)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for (
            board, targets, ct, _outcomes, legal_mask,
        ) in dl:
            board = board.to(self._device)
            targets = targets.to(
                self._device, dtype=torch.long
            )
            ct = ct.to(self._device, dtype=torch.long)
            legal_mask = legal_mask.to(self._device)

            step_result = self._train_step(
                board, targets, ct, legal_mask,
            )
            batch_n = step_result["n_total"]
            total_loss += step_result["loss"] * batch_n
            total_correct += step_result["n_correct"]
            total_samples += batch_n

            self._tracker.track_scalars(
                {
                    "train_loss": step_result["loss"],
                    "train_accuracy": (
                        step_result["n_correct"]
                        / max(batch_n, 1)
                    ),
                    "mean_entropy": step_result[
                        "mean_entropy"
                    ],
                    "grad_norm": step_result["grad_norm"],
                    "lr": self.current_lr,
                },
                step=self._global_step,
            )

        denom = max(total_samples, 1)
        return {
            "total_loss": total_loss / denom,
            "train_accuracy": total_correct / denom,
            "n_samples": total_samples,
            "n_games": dataset.n_games,
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataset: ChessRLDataset,
    ) -> dict[str, float]:
        """Evaluate unweighted CE and top-1 accuracy on val set.

        Uses @torch.no_grad(). DataLoader with shuffle=False.

        Args:
            dataset: ChessRLDataset (val split).

        Returns:
            Dict with keys: val_loss, val_accuracy, val_n_samples.

        Example:
            >>> m = trainer.evaluate(val_ds)
            >>> m["val_accuracy"]
            0.42
        """
        if len(dataset) == 0:
            return {
                "val_loss": 0.0,
                "val_accuracy": 0.0,
                "val_n_samples": 0,
            }

        self._model.eval()
        dl = self._build_dataloader(
            dataset, shuffle=False
        )

        total_ce = 0.0
        correct = 0
        total = 0
        # Stratified: {outcome_val: [n_correct, n_total]}
        strat: dict[int, list[int]] = {
            1: [0, 0], 0: [0, 0], -1: [0, 0]
        }

        for (
            board, targets, ct, outcomes, legal_mask,
        ) in dl:
            board = board.to(self._device)
            targets = targets.to(
                self._device, dtype=torch.long
            )
            ct = ct.to(self._device, dtype=torch.long)
            legal_mask = legal_mask.to(self._device)

            # Split board channels
            bt = board[:, :, 0].long()
            ct_tok = board[:, :, 1].long()
            tt = board[:, :, 2].long()

            # SOS-only prefix: batched [B, 1]
            b_size = bt.size(0)
            prefix = torch.full(
                (b_size, 1),
                SOS_IDX,
                dtype=torch.long,
                device=self._device,
            )
            move_colors = torch.zeros(
                b_size, 1, dtype=torch.long,
                device=self._device,
            )

            logits = self._model(
                bt, ct_tok, tt, prefix, None, move_colors
            )
            # Take last token logits: [B, V]
            last_logits = logits[:, -1, :]

            # Apply legal move mask (true legality per position)
            last_logits = last_logits.masked_fill(
                ~legal_mask, float("-inf")
            )

            # Unweighted CE
            ce = F.cross_entropy(
                last_logits,
                targets,
                reduction="sum",
            )
            total_ce += ce.item()

            preds = last_logits.argmax(dim=-1)
            correct_mask = preds == targets
            correct += int(correct_mask.sum().item())
            total += b_size

            # Accumulate per-outcome accuracy
            outcomes_cpu = outcomes.cpu()
            for o in (1, 0, -1):
                omask = outcomes_cpu == o
                strat[o][0] += int(
                    correct_mask.cpu()[omask].sum().item()
                )
                strat[o][1] += int(omask.sum().item())

        denom = max(total, 1)
        return {
            "val_loss": total_ce / denom,
            "val_accuracy": correct / denom,
            "val_n_samples": total,
            "acc_winners": (
                strat[1][0] / strat[1][1]
                if strat[1][1] > 0 else 0.0
            ),
            "acc_draws": (
                strat[0][0] / strat[0][1]
                if strat[0][1] > 0 else 0.0
            ),
            "acc_losers": (
                strat[-1][0] / strat[-1][1]
                if strat[-1][1] > 0 else 0.0
            ),
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save model, optimizer, and scheduler state to .pt file.

        Args:
            path: Destination .pt file path.

        Example:
            >>> trainer.save_checkpoint(Path("ckpt.pt"))
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self._model.state_dict(),
                "optimizer": self._opt.state_dict(),
                "scheduler": self._sched.state_dict(),
                "global_step": self._global_step,
            },
            path,
        )
        logger.info(
            "RL v4 checkpoint saved to %s", path
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint with weights_only=True (security).

        Supports loading Phase1/V2/V3 checkpoints that only
        have a 'model' key. Skips optimizer/scheduler if absent.

        Args:
            path: Source .pt file path.

        Example:
            >>> trainer.load_checkpoint(Path("ckpt.pt"))
        """
        ckpt = torch.load(
            path,
            map_location=self._device,
            weights_only=True,
        )
        self._model.load_state_dict(ckpt["model"])
        self._global_step = ckpt.get("global_step", 0)
        opt_state = ckpt.get("optimizer")
        if opt_state is not None:
            try:
                self._opt.load_state_dict(opt_state)
            except ValueError:
                logger.warning(
                    "Optimizer state mismatch — "
                    "skipping optimizer restore"
                )
        sched_state = ckpt.get("scheduler")
        if sched_state is not None:
            self._sched.load_state_dict(sched_state)
        logger.info(
            "RL v4 checkpoint loaded from %s", path
        )

    def _build_dataloader(
        self,
        dataset: ChessRLDataset,
        shuffle: bool,
    ) -> DataLoader:  # type: ignore[type-arg]
        """Wrap dataset in a DataLoader with configured params.

        Args:
            dataset: ChessRLDataset instance.
            shuffle: Whether to shuffle the data.

        Returns:
            Configured DataLoader.

        Example:
            >>> dl = trainer._build_dataloader(ds, shuffle=True)
        """
        nw = self._cfg.rl.num_workers
        extra: dict[str, object] = {}
        if nw > 0:
            extra["persistent_workers"] = True
            extra["prefetch_factor"] = 4
        return DataLoader(
            dataset,
            batch_size=self._cfg.rl.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=(str(self._device) != "cpu"),
            **extra,
        )

    def _train_step(
        self,
        board: Tensor,
        targets: Tensor,
        color_tokens: Tensor,
        legal_mask: Tensor,
    ) -> dict[str, float]:
        """Single minibatch forward + backward with plain CE.

        Args:
            board: Float tensor [B, 65, 3].
            targets: Long tensor [B] of vocab indices.
            color_tokens: Long tensor [B, 65] for masking.
            legal_mask: Bool tensor [B, 1971].

        Returns:
            Dict with keys: loss, n_correct, n_total,
            mean_entropy, grad_norm.

        Example:
            >>> out = trainer._train_step(b, t, ct, lm)
            >>> out["loss"]
            2.5
        """
        # Split board channels
        bt = board[:, :, 0].long()
        ct_tok = board[:, :, 1].long()
        tt = board[:, :, 2].long()

        # SOS-only prefix for decoder
        b_size = bt.size(0)
        prefix = torch.full(
            (b_size, 1),
            SOS_IDX,
            dtype=torch.long,
            device=self._device,
        )
        move_colors = torch.zeros(
            b_size, 1, dtype=torch.long,
            device=self._device,
        )

        logits = self._model(
            bt, ct_tok, tt, prefix, None, move_colors
        )
        last_logits = logits[:, -1, :]

        # Apply legal move mask
        last_logits = last_logits.masked_fill(
            ~legal_mask, float("-inf")
        )

        loss = F.cross_entropy(
            last_logits,
            targets,
            label_smoothing=(
                self._cfg.rl.label_smoothing
            ),
        )

        self._opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self._model.parameters(),
            self._cfg.rl.gradient_clip,
        ).item()
        self._opt.step()
        self._sched.step()
        self._global_step += 1

        with torch.no_grad():
            probs = torch.softmax(
                last_logits.detach(), dim=-1
            )
            entropy = -(
                probs * probs.clamp(min=1e-9).log()
            ).sum(dim=-1).mean()

        preds = last_logits.detach().argmax(dim=-1)
        n_correct = int(
            (preds == targets).sum().item()
        )

        return {
            "loss": loss.item(),
            "n_correct": n_correct,
            "n_total": b_size,
            "mean_entropy": entropy.item(),
            "grad_norm": grad_norm,
        }
