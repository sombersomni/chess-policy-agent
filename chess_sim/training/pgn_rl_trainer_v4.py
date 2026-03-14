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
from chess_sim.data.capture_map_builder import (
    compute_from_board_tensor as compute_capture_map,
)
from chess_sim.data.move_category_builder import (
    batch_build as batch_compute_category,
)
from chess_sim.data.move_vocab import SOS_IDX
from chess_sim.data.src_move_lut import SrcMoveLUT
from chess_sim.data.structural_mask import (
    StructuralMaskBuilder,
)
from chess_sim.model.auxiliary_heads import AuxiliaryHeads
from chess_sim.model.chess_model import ChessModel
from chess_sim.protocols import StructuralMaskable
from chess_sim.tracking.noop_tracker import NoOpTracker
from chess_sim.tracking.protocol import MetricTracker
from chess_sim.types import AuxLossOutput, EncoderOutput

if TYPE_CHECKING:
    from chess_sim.data.chess_rl_dataset import (
        ChessRLDataset,
    )

logger = logging.getLogger(__name__)

# Phase classification thresholds
_OPENING_PLY_THRESHOLD: int = 20
_ENDGAME_MATERIAL_THRESHOLD: int = 15

# Piece values indexed by piece-type idx:
# 0=unused, 1=EMPTY, 2=PAWN, 3=KNIGHT, 4=BISHOP, 5=ROOK, 6=QUEEN, 7=KING
_PIECE_VALUES: tuple[float, ...] = (
    0.0, 0.0, 0.0, 3.0, 3.0, 5.0, 9.0, 0.0
)

# LR warmup starting multiplier and DataLoader read-ahead depth
_LR_WARMUP_START_FACTOR: float = 1e-4
_DATALOADER_PREFETCH_FACTOR: int = 4

# Numerical floor for entropy computation to avoid log(0)
_ENTROPY_CLAMP_MIN: float = 1e-9

# Shared key set for aux-head losses — single source of truth across
# _train_step / train_epoch / evaluate to prevent silent key mismatches.
_AUX_LOSS_KEYS: tuple[str, ...] = (
    "capture_loss", "category_loss", "phase_loss"
)


def compute_phase_labels(
    ply_idx: Tensor,
    board: Tensor,
) -> Tensor:
    """Compute game phase labels from ply index and board.

    Opening(0): ply < 20.
    Endgame(1): total non-pawn material <= 15 cp equivalent.
    Midgame(2): otherwise. Endgame overrides opening when both
    conditions hold simultaneously.

    Non-pawn material is estimated from board channel 0
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
    # board[:, 1:, 0] = piece types for squares 1-64.
    # Clamp guards against any out-of-range values in the board tensor.
    pieces = board[:, 1:, 0].long().clamp(
        0, len(_PIECE_VALUES) - 1
    )  # [B, 64]
    val_map = torch.tensor(
        _PIECE_VALUES, dtype=torch.float32, device=board.device
    )
    material = val_map[pieces].sum(dim=1)  # [B]
    endgame_mask = material <= _ENDGAME_MATERIAL_THRESHOLD
    labels[endgame_mask] = 1

    return labels


def _piece_type_from_board(
    board: Tensor,
    src_square: Tensor,
) -> Tensor:
    """Extract piece type at each sample's src square from board tensor.

    Args:
        board: Float tensor [B, 65, 3]. Channel 0 = piece type
            (1=EMPTY, 2=PAWN, 3=KNIGHT, 4=BISHOP, 5=ROOK, 6=QUEEN, 7=KING).
        src_square: Long tensor [B]. 0-based source square (0-63).

    Returns:
        Long tensor [B]. Piece type values 0-7; EMPTY (1) is mapped
        to 0 (no conditioning) since no piece is being selected.
    """
    b_idx = torch.arange(
        board.size(0), device=board.device
    )
    raw = board[b_idx, src_square + 1, 0].long()  # [B]
    # EMPTY (1) → 0 (no conditioning); board 2-7 → chess type 1-6
    return torch.where(
        raw == 1, torch.zeros_like(raw), raw - 1
    )


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

        # Cache all parameters once for optimizer and grad-clip
        # to avoid rebuilding the list on every step.
        params = list(self._model.parameters())
        if self._aux_heads is not None:
            params += list(self._aux_heads.parameters())
        self._all_params: list[nn.Parameter] = params

        self._opt = torch.optim.AdamW(
            self._all_params,
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
            start_factor=_LR_WARMUP_START_FACTOR,
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

        # Src-square LUT: narrow legal mask to moves from the
        # selected piece when use_src_conditioning is enabled.
        self._src_lut: SrcMoveLUT | None = (
            SrcMoveLUT(device=self._device)
            if cfg.model.use_src_conditioning
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
        batch: forward pass, structural mask, CE loss, backward,
        grad clip, optimizer + scheduler step.

        Args:
            dataset: ChessRLDataset (train split).

        Returns:
            Dict with keys: total_loss, train_accuracy, n_samples,
            n_games. When use_aux_heads is True, also includes
            capture_loss, category_loss, phase_loss.

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
            k: 0.0 for k in _AUX_LOSS_KEYS
        }

        for batch in dl:
            (
                board, targets, _ct, _outcomes,
                legal_mask, src_square, ply_idx,
            ) = batch
            board, targets, legal_mask, src_square, ply_idx = (
                self._move_batch_to_device(
                    board, targets, legal_mask,
                    src_square, ply_idx,
                )
            )

            step_result = self._train_step(
                board, targets, legal_mask,
                src_square, ply_idx,
            )
            batch_n = int(step_result["n_total"])
            total_loss += float(step_result["loss"]) * batch_n
            total_correct += int(step_result["n_correct"])
            total_samples += batch_n

            for k in _AUX_LOSS_KEYS:
                if k in step_result:
                    aux_accum[k] += (
                        float(step_result[k]) * batch_n
                    )

            scalars: dict[str, float] = {
                "train_loss": float(step_result["loss"]),
                "train_accuracy": (
                    int(step_result["n_correct"])
                    / max(batch_n, 1)
                ),
                "mean_entropy": float(
                    step_result["mean_entropy"]
                ),
                "grad_norm": float(step_result["grad_norm"]),
                "lr": self.current_lr,
            }
            for k in _AUX_LOSS_KEYS:
                if k in step_result:
                    scalars[k] = float(step_result[k])
            self._tracker.track_scalars(
                scalars, step=self._global_step,
            )

        denom = max(total_samples, 1)
        result: dict[str, float] = {
            "total_loss": total_loss / denom,
            "train_accuracy": total_correct / denom,
            "n_samples": float(total_samples),
            "n_games": float(dataset.n_games),
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
            Dict with keys: val_loss, val_accuracy, val_n_samples,
            acc_winners, acc_draws, acc_losers. When use_aux_heads
            is True, also includes capture_loss, category_loss,
            phase_loss.

        Example:
            >>> m = trainer.evaluate(val_ds)
            >>> m["val_accuracy"]
            0.42
        """
        if len(dataset) == 0:
            return {
                "val_loss": 0.0,
                "val_accuracy": 0.0,
                "val_n_samples": 0.0,
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
        # [n_correct, n_total] per outcome: +1=win, 0=draw, -1=loss
        outcome_acc: dict[int, list[int]] = {
            1: [0, 0], 0: [0, 0], -1: [0, 0]
        }
        aux_accum: dict[str, float] = {
            k: 0.0 for k in _AUX_LOSS_KEYS
        }

        for batch in dl:
            (
                board, targets, _ct, outcomes,
                legal_mask, src_square, ply_idx,
            ) = batch
            board, targets, legal_mask, src_square, ply_idx = (
                self._move_batch_to_device(
                    board, targets, legal_mask,
                    src_square, ply_idx,
                )
            )

            last_logits, enc_out = self._forward(
                board, legal_mask, src_square
            )

            b_size = targets.size(0)
            ce = F.cross_entropy(
                last_logits, targets, reduction="sum",
            )
            total_ce += ce.item()

            preds = last_logits.argmax(dim=-1)
            correct_mask = preds == targets
            correct += int(correct_mask.sum().item())
            total += b_size

            if self._aux_heads is not None:
                aux_out = self._compute_aux_losses(
                    enc_out, targets, board, ply_idx
                )
                for k in _AUX_LOSS_KEYS:
                    aux_accum[k] += (
                        getattr(aux_out, k).item() * b_size
                    )

            # Accumulate per-outcome accuracy (outcomes stays on CPU)
            outcomes_cpu = outcomes.cpu()
            for o in (1, 0, -1):
                omask = outcomes_cpu == o
                outcome_acc[o][0] += int(
                    correct_mask.cpu()[omask].sum().item()
                )
                outcome_acc[o][1] += int(omask.sum().item())

        denom = max(total, 1)
        result: dict[str, float] = {
            "val_loss": total_ce / denom,
            "val_accuracy": correct / denom,
            "val_n_samples": float(total),
            "acc_winners": (
                outcome_acc[1][0] / outcome_acc[1][1]
                if outcome_acc[1][1] > 0 else 0.0
            ),
            "acc_draws": (
                outcome_acc[0][0] / outcome_acc[0][1]
                if outcome_acc[0][1] > 0 else 0.0
            ),
            "acc_losers": (
                outcome_acc[-1][0] / outcome_acc[-1][1]
                if outcome_acc[-1][1] > 0 else 0.0
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
            extra["prefetch_factor"] = _DATALOADER_PREFETCH_FACTOR
        return DataLoader(
            dataset,
            batch_size=self._cfg.rl.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=(str(self._device) != "cpu"),
            **extra,
        )

    def _move_batch_to_device(
        self,
        board: Tensor,
        targets: Tensor,
        legal_mask: Tensor,
        src_square: Tensor,
        ply_idx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Move primary training tensors to the configured device.

        Outcomes tensor is intentionally excluded so callers keep
        it on CPU for stratified accuracy computation.

        Args:
            board: Float tensor [B, 65, 3].
            targets: Long tensor [B].
            legal_mask: Bool tensor [B, 1971].
            src_square: Long tensor [B].
            ply_idx: Long tensor [B].

        Returns:
            Tuple (board, targets, legal_mask, src_square, ply_idx)
            on self._device.
        """
        return (
            board.to(self._device),
            targets.to(self._device, dtype=torch.long),
            legal_mask.to(self._device),
            src_square.to(self._device, dtype=torch.long),
            ply_idx.to(self._device, dtype=torch.long),
        )

    def _forward(
        self,
        board: Tensor,
        legal_mask: Tensor,
        src_square: Tensor | None,
    ) -> tuple[Tensor, EncoderOutput]:
        """Shared encoder-decoder forward pass with legal masking.

        Splits board channels, optionally conditions on src square,
        applies structural mask (if enabled), then applies the legal
        move mask. Called by both _train_step and evaluate.

        Args:
            board: Float tensor [B, 65, 3] on device.
            legal_mask: Bool tensor [B, 1971] on device.
            src_square: Optional long tensor [B] (0-based 0-63).

        Returns:
            Tuple of (masked_last_logits [B, V], EncoderOutput).
        """
        piece_types = board[:, :, 0].long()
        color_toks = board[:, :, 1].long()
        traj_toks = board[:, :, 2].long()

        batch_size = piece_types.size(0)
        prefix = torch.full(
            (batch_size, 1), SOS_IDX,
            dtype=torch.long, device=self._device,
        )
        move_colors = torch.zeros(
            batch_size, 1, dtype=torch.long,
            device=self._device,
        )

        src_tok: Tensor | None = None
        pt_tok: Tensor | None = None
        if (
            self._cfg.model.use_src_conditioning
            and src_square is not None
        ):
            src_tok = src_square + 1  # [B]
            pt_tok = _piece_type_from_board(board, src_square)

        enc_out = self._model.encoder.encode(
            piece_types, color_toks, traj_toks,
            src_tok, pt_tok,
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

        # Suppress logits for moves from squares with no player piece.
        if self._struct_mask is not None:
            struct_mask = self._struct_mask.build(color_toks)
            last_logits = last_logits.masked_fill(
                ~struct_mask, float("-inf")
            )

        # Narrow legal mask to moves from the selected source square.
        if self._src_lut is not None and src_square is not None:
            legal_mask = self._src_lut.filter_legal_mask(
                legal_mask, src_square
            )
        last_logits = last_logits.masked_fill(
            ~legal_mask, float("-inf")
        )

        return last_logits, enc_out

    def _compute_aux_losses(
        self,
        enc_out: EncoderOutput,
        targets: Tensor,
        board: Tensor,
        ply_idx: Tensor,
    ) -> AuxLossOutput:
        """Compute auxiliary head losses from encoder outputs.

        Caller must ensure self._aux_heads is not None and all
        tensors are on device before calling.

        Args:
            enc_out: EncoderOutput from encoder.encode().
            targets: Long tensor [B] of vocab indices.
            board: Float tensor [B, 65, 3] on device.
            ply_idx: Long tensor [B] of ply indices.

        Returns:
            AuxLossOutput with scalar losses for each head.
        """
        assert self._aux_heads is not None  # guarded by callers
        cap_map = compute_capture_map(board)
        move_cat = batch_compute_category(targets, board)
        phase_gt = compute_phase_labels(ply_idx, board)
        return self._aux_heads(
            enc_out.square_embeddings,
            enc_out.cls_embedding,
            cap_map,
            move_cat,
            phase_gt,
        )

    def _train_step(
        self,
        board: Tensor,
        targets: Tensor,
        legal_mask: Tensor,
        src_square: Tensor | None = None,
        ply_idx: Tensor | None = None,
    ) -> dict[str, float | int]:
        """Single minibatch forward + backward with plain CE.

        When aux heads are enabled, auxiliary targets are computed
        per-batch from the board tensor. When src conditioning is on,
        the encoder is conditioned on the from-square and the legal
        mask is narrowed to moves from that square.

        Args:
            board: Float tensor [B, 65, 3].
            targets: Long tensor [B] of vocab indices.
            legal_mask: Bool tensor [B, 1971].
            src_square: Optional long tensor [B] (0-based 0-63).
            ply_idx: Optional long tensor [B]. Required when
                use_aux_heads is True.

        Returns:
            Dict with keys: loss, n_correct, n_total,
            mean_entropy, grad_norm. When aux heads enabled,
            also: capture_loss, category_loss, phase_loss.

        Example:
            >>> out = trainer._train_step(b, t, lm)
            >>> out["loss"]
            2.5
        """
        last_logits, enc_out = self._forward(
            board, legal_mask, src_square
        )

        loss = F.cross_entropy(
            last_logits,
            targets,
            label_smoothing=self._cfg.rl.label_smoothing,
        )

        # Aux losses — computed per-batch from board tensor
        result_extra: dict[str, float | int] = {}
        if self._aux_heads is not None:
            if ply_idx is None:
                raise ValueError(
                    "ply_idx required when use_aux_heads=True"
                )
            aux_out = self._compute_aux_losses(
                enc_out, targets, board, ply_idx
            )
            rl = self._cfg.rl
            aux_loss = (
                rl.lambda_capture * aux_out.capture_loss
                + rl.lambda_category * aux_out.category_loss
                + rl.lambda_phase * aux_out.phase_loss
            )
            loss = loss + aux_loss
            result_extra = {
                k: getattr(aux_out, k).item()
                for k in _AUX_LOSS_KEYS
            }

        self._opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self._all_params,
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
                probs * probs.clamp(min=_ENTROPY_CLAMP_MIN).log()
            ).sum(dim=-1).mean()

        preds = last_logits.detach().argmax(dim=-1)
        n_correct = int((preds == targets).sum().item())

        result: dict[str, float | int] = {
            "loss": loss.item(),
            "n_correct": n_correct,
            "n_total": targets.size(0),
            "mean_entropy": entropy.item(),
            "grad_norm": grad_norm,
        }
        result.update(result_extra)
        return result
