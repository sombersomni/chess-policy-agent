"""Per-move game evaluation with loss, accuracy, and entropy.

Evaluates a trained chess encoder checkpoint on a held-out game,
producing per-ply CE loss, top-1 accuracy, and Shannon entropy
for each of the two prediction heads (src, tgt).

Each ply is evaluated from the side-to-move's perspective.

Usage:
    # From a YAML config file (recommended):
    python -m scripts.evaluate \
        --config configs/evaluate.yaml \
        --checkpoint checkpoints/lichess_50k_v1.pt

    # Pure CLI (backward compatible):
    python -m scripts.evaluate \
        --checkpoint checkpoints/real_run_01.pt \
        --pgn data/games.pgn \
        [--game-index 0] \
        [--top-n 3]
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
from pathlib import Path
from typing import NamedTuple, Protocol

import chess
import chess.pgn
import torch
import torch.nn as nn
from torch import Tensor

from chess_sim.config import EvaluateConfig, load_eval_config
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.model.encoder import ChessEncoder
from chess_sim.model.heads import PredictionHeads
from chess_sim.training.trainer import Trainer, timed
from chess_sim.types import TrainingExample
from chess_sim.utils import winner_color

from scripts.train_real import (
    _make_trajectory_tokens,
    game_to_examples,
    stream_pgn,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data contract
# ------------------------------------------------------------------

class StepResult(NamedTuple):
    """Per-ply evaluation metrics for both heads."""

    ply: int
    move_uci: str
    loss_src: float
    loss_tgt: float
    total_loss: float
    acc_src: int
    acc_tgt: int
    entropy_src: float
    entropy_tgt: float
    mean_entropy: float


# ------------------------------------------------------------------
# Protocol
# ------------------------------------------------------------------

class Evaluatable(Protocol):
    """Evaluates a full game and returns per-ply metrics."""

    def evaluate_game(
        self,
        game: chess.pgn.Game,
        winners_only: bool = False,
    ) -> list[StepResult]:
        """Evaluate every ply of a game.

        Args:
            game: A parsed PGN game.
            winners_only: If True, only evaluate positions
                where the winning side is to move.

        Returns:
            One StepResult per ply.
        """
        ...


# ------------------------------------------------------------------
# Pure functions
# ------------------------------------------------------------------

def shannon_entropy(logits: Tensor) -> float:
    """Compute Shannon entropy in nats from raw logits.

    H = -sum(p * log(p + eps)) where p = softmax(logits).
    Max for 64 classes: log(64) ~ 4.158.

    Args:
        logits: 1-D tensor of shape [64].

    Returns:
        Entropy as a Python float (nats).
    """
    p = torch.softmax(logits, dim=0)
    h = -(p * torch.log(p + 1e-9)).sum().item()
    return h


def top1_accuracy(logits: Tensor, label: int) -> int:
    """Return 1 if argmax matches label, 0 otherwise.

    Args:
        logits: 1-D tensor of shape [64].
        label: Ground-truth square index.

    Returns:
        1 (correct) or 0 (wrong).
    """
    return int(torch.argmax(logits).item() == label)


def per_head_ce(
    logits: Tensor,
    label: int,
) -> float:
    """Compute CE loss for a single example and head.

    Args:
        logits: 1-D tensor of shape [64].
        label: Ground-truth square index.

    Returns:
        CE loss as float.
    """
    ce = nn.CrossEntropyLoss()
    loss = ce(
        logits.unsqueeze(0),
        torch.tensor([label], dtype=torch.long,
                      device=logits.device),
    )
    return loss.item()


# ------------------------------------------------------------------
# evaluate_step (no_grad)
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_step(
    example: TrainingExample,
    move_uci: str,
    ply: int,
    encoder: ChessEncoder,
    heads: PredictionHeads,
    device: str,
) -> StepResult:
    """Run a single forward pass and compute all per-head metrics.

    Args:
        example: A TrainingExample with tokens and labels.
        move_uci: UCI string of the move (e.g. "e2e4").
        ply: Zero-based ply index.
        encoder: The chess encoder model (eval mode).
        heads: The prediction heads (eval mode).
        device: Torch device string.

    Returns:
        Fully populated StepResult.
    """
    bt = torch.tensor(
        [example.board_tokens],
        dtype=torch.long, device=device,
    )
    ct = torch.tensor(
        [example.color_tokens],
        dtype=torch.long, device=device,
    )
    tt = torch.tensor(
        [example.trajectory_tokens],
        dtype=torch.long, device=device,
    )
    enc_out = encoder(bt, ct, tt)
    pred = heads(enc_out.cls_embedding)

    # Squeeze batch dim: [1, 64] -> [64]
    src_logits = pred.src_sq_logits.squeeze(0)
    tgt_logits = pred.tgt_sq_logits.squeeze(0)

    # Per-head CE losses
    l_src = per_head_ce(src_logits, example.src_sq)
    l_tgt = per_head_ce(tgt_logits, example.tgt_sq)
    total = l_src + l_tgt

    # Top-1 accuracy
    a_src = top1_accuracy(src_logits, example.src_sq)
    a_tgt = top1_accuracy(tgt_logits, example.tgt_sq)

    # Shannon entropy
    h_src = shannon_entropy(src_logits)
    h_tgt = shannon_entropy(tgt_logits)
    mean_h = (h_src + h_tgt) / 2.0

    return StepResult(
        ply=ply,
        move_uci=move_uci,
        loss_src=l_src,
        loss_tgt=l_tgt,
        total_loss=total,
        acc_src=a_src,
        acc_tgt=a_tgt,
        entropy_src=h_src,
        entropy_tgt=h_tgt,
        mean_entropy=mean_h,
    )


# ------------------------------------------------------------------
# GameEvaluator
# ------------------------------------------------------------------

class GameEvaluator:
    """Evaluates a full chess game move-by-move.

    Implements the Evaluatable protocol.

    Loads a trained checkpoint via Trainer.load_checkpoint,
    sets both encoder and heads to eval mode, then runs
    evaluate_step for every ply of a game.
    """

    def __init__(
        self, checkpoint_path: Path, device: str = "cpu"
    ) -> None:
        """Load checkpoint and prepare models for evaluation.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            device: Torch device string.
        """
        self.device = device
        self.tokenizer = BoardTokenizer()
        self._trainer = Trainer(device=device)
        self._trainer.load_checkpoint(checkpoint_path)
        self._trainer.encoder.eval()
        self._trainer.heads.eval()

    @property
    def encoder(self) -> ChessEncoder:
        """Access the loaded encoder."""
        return self._trainer.encoder

    @property
    def heads(self) -> PredictionHeads:
        """Access the loaded prediction heads."""
        return self._trainer.heads

    @timed
    def evaluate_game(
        self,
        game: chess.pgn.Game,
        winners_only: bool = False,
    ) -> list[StepResult]:
        """Evaluate every ply of a game.

        Args:
            game: A parsed PGN game object.
            winners_only: If True, only evaluate positions
                where the winning side is to move. Draws
                return [].

        Returns:
            List of StepResult, one per evaluated ply.
        """
        winner = winner_color(game) if winners_only else None
        if winners_only and winner is None:
            return []

        board = game.board()
        moves = list(game.mainline_moves())
        tokenizer = self.tokenizer
        results: list[StepResult] = []
        move_history: list[chess.Move] = []

        for i, move in enumerate(moves):
            if winners_only and board.turn != winner:
                move_history.append(move)
                board.push(move)
                continue

            tokenized = tokenizer.tokenize(
                board, board.turn
            )
            trajectory_tokens = _make_trajectory_tokens(
                move_history
            )

            ex = TrainingExample(
                board_tokens=tokenized.board_tokens,
                color_tokens=tokenized.color_tokens,
                trajectory_tokens=trajectory_tokens,
                src_sq=move.from_square,
                tgt_sq=move.to_square,
            )

            result = evaluate_step(
                example=ex,
                move_uci=move.uci(),
                ply=i,
                encoder=self.encoder,
                heads=self.heads,
                device=self.device,
            )
            results.append(result)
            move_history.append(move)
            board.push(move)

        return results


# ------------------------------------------------------------------
# Output formatting
# ------------------------------------------------------------------

def print_table(results: list[StepResult]) -> None:
    """Print a fixed-width terminal table of per-ply results.

    Args:
        results: List of StepResult from evaluate_game.
    """
    header = (
        f"{'Ply':>3}  {'Move':<8} "
        f"{'L_src':>7} {'L_tgt':>7} "
        f"{'Total':>8} "
        f"{'A_s':>3} {'A_t':>3} "
        f"{'H_src':>7} {'H_tgt':>7}"
    )
    print(header)
    for r in results:
        line = (
            f"{r.ply:>3}  {r.move_uci:<8} "
            f"{r.loss_src:>7.4f} {r.loss_tgt:>7.4f} "
            f"{r.total_loss:>8.4f} "
            f"{r.acc_src:>3} "
            f"{r.acc_tgt:>3} "
            f"{r.entropy_src:>7.4f} "
            f"{r.entropy_tgt:>7.4f}"
        )
        print(line)


def print_summary(
    results: list[StepResult], top_n: int = 3
) -> None:
    """Print aggregate summary statistics and top-N by entropy.

    Args:
        results: List of StepResult from evaluate_game.
        top_n: Number of highest-entropy plies to show.
    """
    if not results:
        print("No results to summarize.")
        return

    def _mean_std(
        vals: list[float],
    ) -> tuple[float, float]:
        if len(vals) < 2:
            m = vals[0] if vals else 0.0
            return m, 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    # Collect numeric fields
    l_src = [r.loss_src for r in results]
    l_tgt = [r.loss_tgt for r in results]
    total = [r.total_loss for r in results]
    h_src = [r.entropy_src for r in results]
    h_tgt = [r.entropy_tgt for r in results]
    m_ent = [r.mean_entropy for r in results]

    # Accuracies
    a_src = [float(r.acc_src) for r in results]
    a_tgt = [float(r.acc_tgt) for r in results]

    print("\n=== Summary ===")
    for name, vals in [
        ("loss_src", l_src),
        ("loss_tgt", l_tgt),
        ("total_loss", total),
        ("acc_src", a_src),
        ("acc_tgt", a_tgt),
        ("entropy_src", h_src),
        ("entropy_tgt", h_tgt),
        ("mean_entropy", m_ent),
    ]:
        if vals:
            m, s = _mean_std(vals)
            print(f"  {name:<16} mean={m:.4f}  std={s:.4f}")
        else:
            print(f"  {name:<16} (no data)")

    # Top-N by mean_entropy descending
    ranked = sorted(
        results, key=lambda r: r.mean_entropy, reverse=True
    )
    print(f"\nTop-{top_n} plies by mean_entropy:")
    for r in ranked[:top_n]:
        print(
            f"  ply={r.ply:>3}  move={r.move_uci:<8} "
            f"mean_H={r.mean_entropy:.4f}"
        )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_eval_parser() -> argparse.ArgumentParser:
    """Build the evaluation argument parser.

    All overrideable args default to None so the merge step can
    distinguish 'explicitly set' from 'not provided'.
    """
    p = argparse.ArgumentParser(
        description="Evaluate a chess encoder on a PGN game."
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML evaluation config file.",
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
        "--game-index", type=int, default=None,
        help="Zero-based game index in the PGN file.",
    )
    p.add_argument(
        "--top-n", type=int, default=None,
        help="Number of top-entropy plies to show.",
    )
    p.add_argument(
        "--winners-only",
        action="store_true",
        default=None,
        help="Evaluate only the winning player's positions.",
    )
    return p


def _merge_eval_config(
    args: argparse.Namespace,
    cfg: EvaluateConfig,
) -> EvaluateConfig:
    """Apply non-None CLI args on top of cfg (mutates and returns cfg).

    Args:
        args: Parsed argparse namespace (None means not provided).
        cfg: EvaluateConfig loaded from YAML or default.

    Returns:
        Updated cfg with CLI overrides applied.
    """
    if args.checkpoint is not None:
        cfg.eval.checkpoint = args.checkpoint
    if args.pgn is not None:
        cfg.data.pgn = args.pgn
    if args.game_index is not None:
        cfg.eval.game_index = args.game_index
    if args.top_n is not None:
        cfg.eval.top_n = args.top_n
    if args.winners_only:
        cfg.data.winners_only = True
    return cfg


def main() -> None:
    """CLI entry point for per-move game evaluation."""
    parser = _build_eval_parser()
    args = parser.parse_args()

    if args.config:
        cfg = load_eval_config(Path(args.config))
        logger.info("Loaded config from %s", args.config)
    else:
        cfg = EvaluateConfig()

    cfg = _merge_eval_config(args, cfg)

    if not cfg.eval.checkpoint:
        parser.error("--checkpoint is required (or set eval.checkpoint in YAML).")
    if not cfg.data.pgn:
        parser.error("--pgn is required (or set data.pgn in YAML).")

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    evaluator = GameEvaluator(
        checkpoint_path=Path(cfg.eval.checkpoint),
        device=device,
    )

    games = list(stream_pgn(Path(cfg.data.pgn)))
    if cfg.eval.game_index >= len(games):
        logger.error(
            "Game index %d out of range (%d games).",
            cfg.eval.game_index, len(games),
        )
        return

    game = games[cfg.eval.game_index]
    results = evaluator.evaluate_game(
        game, winners_only=cfg.data.winners_only
    )

    print_table(results)
    print_summary(results, top_n=cfg.eval.top_n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
