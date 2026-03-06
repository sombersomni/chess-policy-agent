"""End-to-end training script on real chess positions.

Accepts either a PGN file (plain or .zst compressed) via --pgn, or
generates synthetic games with random legal moves when --pgn is omitted.

Usage:
    # From a downloaded PGN file (plain or .zst):
    python -m scripts.train_real --pgn games.pgn --epochs 10
    python -m scripts.train_real --pgn games.pgn.zst --epochs 10

    # From synthetic random-legal-move games:
    python -m scripts.train_real --num-games 20 --epochs 10
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import torch
from torch.utils.data import DataLoader

from chess_sim.data.dataset import ChessDataset
from chess_sim.data.scorer import ActivityScorer
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.training.trainer import Trainer
from chess_sim.types import TrainingExample
from chess_sim.utils import winner_color


# ---------------------------------------------------------------------------
# Game sources
# ---------------------------------------------------------------------------

def stream_pgn(path: Path) -> Iterator[chess.pgn.Game]:
    """Yield games from a plain .pgn or .zst-compressed PGN file."""
    if path.suffix == ".zst":
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        with open(path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_io = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                yield from _read_games(text_io)
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            yield from _read_games(fh)


def _read_games(fh: io.TextIOWrapper) -> Iterator[chess.pgn.Game]:
    while True:
        game = chess.pgn.read_game(fh)
        if game is None:
            break
        yield game


def generate_random_game(max_moves: int = 80) -> chess.pgn.Game:
    """Play a game with uniformly random legal moves up to max_moves plies."""
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        move = random.choice(list(board.legal_moves))
        node = node.add_variation(move)
        board.push(move)
    game.headers["Result"] = board.result()
    return game


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------

def game_to_examples(
    game: chess.pgn.Game,
    tokenizer: BoardTokenizer,
    winners_only: bool = False,
) -> list[TrainingExample]:
    """Walk every ply of a game and emit one TrainingExample per position.

    Args:
        game: A parsed PGN game object.
        tokenizer: Board tokenizer instance.
        winners_only: If True, only include positions where
            the winning side is to move. Draws return [].

    Returns:
        List of TrainingExample namedtuples.
    """
    winner = winner_color(game) if winners_only else None
    if winners_only and winner is None:
        return []

    scorer = ActivityScorer()
    examples: list[TrainingExample] = []
    board = game.board()
    moves = list(game.mainline_moves())
    move_history: list[chess.Move] = []

    for i, move in enumerate(moves):
        if winners_only and board.turn != winner:
            move_history.append(move)
            board.push(move)
            continue

        tokenized = tokenizer.tokenize(board, board.turn)
        activity_tokens = scorer.score(
            move_history, board, n=4
        )
        if i + 1 < len(moves):
            opp = moves[i + 1]
            opp_src_sq = opp.from_square
            opp_tgt_sq = opp.to_square
        else:
            opp_src_sq, opp_tgt_sq = -1, -1

        examples.append(TrainingExample(
            board_tokens=tokenized.board_tokens,
            color_tokens=tokenized.color_tokens,
            activity_tokens=activity_tokens,
            src_sq=move.from_square,
            tgt_sq=move.to_square,
            opp_src_sq=opp_src_sq,
            opp_tgt_sq=opp_tgt_sq,
        ))
        move_history.append(move)
        board.push(move)

    return examples


def build_examples_from_file(
    pgn_path: Path,
    winners_only: bool = False,
) -> list[TrainingExample]:
    """Build training examples from a PGN file.

    Args:
        pgn_path: Path to .pgn or .pgn.zst file.
        winners_only: If True, only include winning side positions.

    Returns:
        List of TrainingExample namedtuples.
    """
    tokenizer = BoardTokenizer()
    all_examples: list[TrainingExample] = []
    for i, game in enumerate(stream_pgn(pgn_path)):
        all_examples.extend(
            game_to_examples(
                game, tokenizer, winners_only=winners_only
            )
        )
        if (i + 1) % 50 == 0:
            print(
                f"  Processed {i + 1} games "
                f"({len(all_examples)} examples)"
            )
    return all_examples


def build_examples_synthetic(num_games: int, seed: int = 42) -> list[TrainingExample]:
    random.seed(seed)
    tokenizer = BoardTokenizer()
    all_examples: list[TrainingExample] = []
    for i in range(num_games):
        game = generate_random_game()
        all_examples.extend(game_to_examples(game, tokenizer))
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_games} games "
                  f"({len(all_examples)} examples so far)")
    return all_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=str, default="",
                        help="Path to a .pgn or .pgn.zst file (optional)")
    parser.add_argument("--num-games", type=int, default=20,
                        help="Synthetic games to generate when --pgn is not given")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to save final checkpoint (.pt)")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from (.pt)")
    parser.add_argument(
        "--winners-only",
        action="store_true",
        default=False,
        help=(
            "Include only the winning player's positions."
            " Skips draws."
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.pgn:
        pgn_path = Path(args.pgn)
        print(f"\nLoading examples from {pgn_path} ...")
        examples = build_examples_from_file(
            pgn_path, winners_only=args.winners_only
        )
    else:
        print(f"\nGenerating examples from {args.num_games} synthetic games...")
        examples = build_examples_synthetic(args.num_games)

    print(f"Total examples: {len(examples)}")

    train_ds, val_ds = ChessDataset.split(examples, train_frac=0.9)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    total_steps = args.epochs * len(loader)
    trainer = Trainer(device=device, total_steps=max(total_steps, 1))

    if args.resume:
        ckpt = torch.load(
            args.resume, map_location=device
        )
        # strict=False allows loading old 3-stream checkpoints
        # that lack activity_emb weights.
        trainer.encoder.load_state_dict(
            ckpt['encoder'], strict=False
        )
        trainer.heads.load_state_dict(
            ckpt['heads'], strict=False
        )
        print(f"Resumed from {args.resume}")

    print(f"\nTraining {args.epochs} epochs | "
          f"batches/epoch={len(loader)} | total_steps={total_steps}\n")

    epoch_losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        avg = trainer.train_epoch(loader)
        epoch_losses.append(avg)
        print(f"Epoch {epoch:02d}: avg_loss={avg:.4f}")

    first, last = epoch_losses[0], epoch_losses[-1]
    print(f"\nLoss: {first:.4f} → {last:.4f}  (Δ={last - first:+.4f})")

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
