"""PGNRLTrainerV4: batched minibatch CE trainer using DataLoader.

Replaces the sequential per-game loop of V3 with a standard
DataLoader minibatch loop over a preprocessed HDF5 dataset.
Does NOT inherit from V2/V3 -- standalone trainer with its own
model, optimizer, scheduler, and structural mask.

Optionally attaches AuxiliaryHeads for denser encoder supervision
(capture target, move category, game phase) when
cfg.rl.use_aux_heads is True.
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
from chess_sim.model.auxiliary_heads import AuxiliaryHeads
from chess_sim.model.chess_model import ChessModel
from chess_sim.protocols import StructuralMaskable
from chess_sim.tracking.noop_tracker import NoOpTracker
from chess_sim.tracking.protocol import MetricTracker

if TYPE_CHECKING:
    from chess_sim.data.chess_rl_dataset import (
        ChessRLDataset,
    )

logger = logging.getLogger(__name__)

# Phase classification thresholds
_OPENING_PLY_THRESHOLD: int = 20
_ENDGAME_MATERIAL_THRESHOLD: int = 15


def compute_phase_labels(
    ply_idx: Tensor,
    board: Tensor,
) -> Tensor:
    """Compute game phase labels from ply index and board.

    Opening(0): ply < 20.
    Endgame(1): total non-pawn material <= 15 cp equivalent.
    Midgame(2): otherwise.

    Non-pawn material is estimated from the board channel 0
    (piece type indices): knight(3)=3, bishop(4)=3, rook(5)=5,
    queen(6)=9. Summed for both sides (squares 1-64).

    Args:
        ply_idx: [B] long tensor of ply indices.
        board: [B, 65, 3] float tensor (channel 0 = piece type).

    Returns:
        [B] long tensor with values 0, 1, or 2.
    """
    b_size = ply_idx.size(0)
    labels = torch.full(
        (b_size,), 2,
        dtype=torch.long, device=ply_idx.device,
    )

    # Opening: ply < 20
    opening_mask = ply_idx < _OPENING_PLY_THRESHOLD
    labels[opening_mask] = 0

    # Endgame: total non-pawn material <= 15
    # board[:, 1:, 0] = piece types for squares 1-64
    pieces = board[:, 1:, 0].long()  # [B, 64]

    # Piece values: idx 3=knight(3), 4=bishop(3), 5=rook(5), 6=queen(9)
    val_map = torch.zeros(
        8, dtype=torch.float32, device=board.device
    )
    val_map[3] = 3.0   # knight
    val_map[4] = 3.0   # bishop
    val_map[5] = 5.0   # rook
    val_map[6] = 9.0   # queen

    material = val_map[pieces].sum(dim=1)  # [B]
    endgame_mask = (
        material <= _ENDGAME_MATERIAL_THRESHOLD
    )
    # Endgame overrides opening if both conditions met
    labels[endgame_mask] = 1

    return labels


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

        # Aux heads: instantiated only when enabled
        self._aux_heads: AuxiliaryHeads | None = None
        if cfg.rl.use_aux_heads:
            self._aux_heads = AuxiliaryHeads(
                cfg.model.d_model
            ).to(self._device)

        # Collect all parameters for optimizer
        params = list(self._model.parameters())
        if self._aux_heads is not None:
            params += list(self._aux_heads.parameters())

        self._opt = torch.optim.AdamW(
            params,
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
    def aux_heads(self) -> AuxiliaryHeads | None:
        """Expose the auxiliary heads (None when disabled).

        Returns:
            AuxiliaryHeads or None.
        """
        return self._aux_heads

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
            mean_multiplier, n_games. When use_aux_heads is
            True, also includes capture_loss, category_loss,
            phase_loss.

        Example:
            >>> m = trainer.train_epoch(train_ds)
            >>> m["total_loss"]
            3.14
        """
        self._model.train()
        if self._aux_heads is not None:
            self._aux_heads.train()
        dl = self._build_dataloader(dataset, shuffle=True)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        aux_accum: dict[str, float] = {
            "capture_loss": 0.0,
            "category_loss": 0.0,
            "phase_loss": 0.0,
        }

        for batch in dl:
            (
                board, targets, ct, _outcomes,
                legal_mask, capture_map,
                move_category, ply_idx,
            ) = batch
            board = board.to(self._device)
            targets = targets.to(
                self._device, dtype=torch.long
            )
            ct = ct.to(self._device, dtype=torch.long)
            legal_mask = legal_mask.to(self._device)
            capture_map = capture_map.to(self._device)
            move_category = move_category.to(
                self._device, dtype=torch.long
            )
            ply_idx = ply_idx.to(
                self._device, dtype=torch.long
            )

            step_result = self._train_step(
                board, targets, ct, legal_mask,
                capture_map, move_category, ply_idx,
            )
            batch_n = step_result["n_total"]
            total_loss += step_result["loss"] * batch_n
            total_correct += step_result["n_correct"]
            total_samples += batch_n

            for k in aux_accum:
                if k in step_result:
                    aux_accum[k] += (
                        step_result[k] * batch_n
                    )

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
        result: dict[str, float] = {
            "total_loss": total_loss / denom,
            "train_accuracy": total_correct / denom,
            "n_samples": total_samples,
            "n_games": dataset.n_games,
        }
        if self._aux_heads is not None:
            for k, v in aux_accum.items():
                result[k] = v / denom
        return result

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
            Dict with keys: val_loss, val_accuracy,
            val_n_samples. When use_aux_heads is True, also
            includes capture_loss, category_loss, phase_loss.

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
        if self._aux_heads is not None:
            self._aux_heads.eval()
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
        aux_accum: dict[str, float] = {
            "capture_loss": 0.0,
            "category_loss": 0.0,
            "phase_loss": 0.0,
        }

        for batch in dl:
            (
                board, targets, ct, outcomes,
                legal_mask, capture_map,
                move_category, ply_idx,
            ) = batch
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

            # Get encoder output for aux heads
            enc_out = self._model.encoder.encode(
                bt, ct_tok, tt,
            )
            memory = torch.cat(
                [
                    enc_out.cls_embedding.unsqueeze(1),
                    enc_out.square_embeddings,
                ],
                dim=1,
            )
            logits = self._model.decoder.decode(
                prefix, memory, None, move_colors,
            ).logits

            # Take last token logits: [B, V]
            last_logits = logits[:, -1, :]

            # Apply legal move mask
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

            # Aux losses during eval
            if self._aux_heads is not None:
                capture_map = capture_map.to(
                    self._device
                )
                move_category = move_category.to(
                    self._device, dtype=torch.long
                )
                ply_idx = ply_idx.to(
                    self._device, dtype=torch.long
                )
                phase_gt = compute_phase_labels(
                    ply_idx, board.to(self._device)
                )
                aux_out = self._aux_heads(
                    enc_out.square_embeddings,
                    enc_out.cls_embedding,
                    capture_map,
                    move_category,
                    phase_gt,
                )
                aux_accum["capture_loss"] += (
                    aux_out.capture_loss.item() * b_size
                )
                aux_accum["category_loss"] += (
                    aux_out.category_loss.item() * b_size
                )
                aux_accum["phase_loss"] += (
                    aux_out.phase_loss.item() * b_size
                )

            # Accumulate per-outcome accuracy
            outcomes_cpu = outcomes.cpu()
            for o in (1, 0, -1):
                omask = outcomes_cpu == o
                strat[o][0] += int(
                    correct_mask.cpu()[omask].sum().item()
                )
                strat[o][1] += int(omask.sum().item())

        denom = max(total, 1)
        result: dict[str, float] = {
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
        if self._aux_heads is not None:
            for k, v in aux_accum.items():
                result[k] = v / denom
        return result

    def save_checkpoint(self, path: Path) -> None:
        """Save model, optimizer, and scheduler state to .pt file.

        Args:
            path: Destination .pt file path.

        Example:
            >>> trainer.save_checkpoint(Path("ckpt.pt"))
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, object] = {
            "model": self._model.state_dict(),
            "optimizer": self._opt.state_dict(),
            "scheduler": self._sched.state_dict(),
            "global_step": self._global_step,
        }
        if self._aux_heads is not None:
            state["aux_heads"] = (
                self._aux_heads.state_dict()
            )
        torch.save(state, path)
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
        # Load aux heads if present
        aux_state = ckpt.get("aux_heads")
        if aux_state is not None and self._aux_heads:
            self._aux_heads.load_state_dict(aux_state)
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
        capture_map: Tensor | None = None,
        move_category: Tensor | None = None,
        ply_idx: Tensor | None = None,
    ) -> dict[str, float]:
        """Single minibatch forward + backward with plain CE.

        When aux heads are enabled, computes and adds auxiliary
        losses to the total loss before backward.

        Args:
            board: Float tensor [B, 65, 3].
            targets: Long tensor [B] of vocab indices.
            color_tokens: Long tensor [B, 65] for masking.
            legal_mask: Bool tensor [B, 1971].
            capture_map: Optional float tensor [B, 64].
            move_category: Optional long tensor [B].
            ply_idx: Optional long tensor [B].

        Returns:
            Dict with keys: loss, n_correct, n_total,
            mean_entropy, grad_norm. When aux heads enabled,
            also: capture_loss, category_loss, phase_loss.

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

        # Encode and decode (split for aux head access)
        enc_out = self._model.encoder.encode(
            bt, ct_tok, tt,
        )
        memory = torch.cat(
            [
                enc_out.cls_embedding.unsqueeze(1),
                enc_out.square_embeddings,
            ],
            dim=1,
        )
        logits = self._model.decoder.decode(
            prefix, memory, None, move_colors,
        ).logits

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

        # Aux losses
        result_extra: dict[str, float] = {}
        if self._aux_heads is not None:
            assert capture_map is not None
            assert move_category is not None
            assert ply_idx is not None
            phase_gt = compute_phase_labels(
                ply_idx, board
            )
            aux_out = self._aux_heads(
                enc_out.square_embeddings,
                enc_out.cls_embedding,
                capture_map,
                move_category,
                phase_gt,
            )
            rl = self._cfg.rl
            aux_loss = (
                rl.lambda_capture
                * aux_out.capture_loss
                + rl.lambda_category
                * aux_out.category_loss
                + rl.lambda_phase
                * aux_out.phase_loss
            )
            loss = loss + aux_loss
            result_extra = {
                "capture_loss": (
                    aux_out.capture_loss.item()
                ),
                "category_loss": (
                    aux_out.category_loss.item()
                ),
                "phase_loss": (
                    aux_out.phase_loss.item()
                ),
            }

        self._opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            list(self._model.parameters())
            + (
                list(self._aux_heads.parameters())
                if self._aux_heads else []
            ),
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

        result: dict[str, float] = {
            "loss": loss.item(),
            "n_correct": n_correct,
            "n_total": b_size,
            "mean_entropy": entropy.item(),
            "grad_norm": grad_norm,
        }
        result.update(result_extra)
        return result
