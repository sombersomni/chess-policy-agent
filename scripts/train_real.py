"""End-to-end training script on real chess positions.

Accepts either a PGN file (plain or .zst compressed) via --pgn, or
generates synthetic games with random legal moves when --pgn is omitted.

When --pgn is given, uses the streaming data pipeline: preprocesses
the PGN into shard files on first run, then trains from cached shards.

Usage:
    # From a YAML config file (recommended):
    python -m scripts.train_real --config configs/train_50k.yaml

    # YAML config with CLI overrides:
    python -m scripts.train_real --config configs/train_50k.yaml --epochs 5

    # Pure CLI (backward compatible, no YAML):
    python -m scripts.train_real --pgn games.pgn --epochs 10
    python -m scripts.train_real --pgn games.pgn.zst --epochs 10

    # From synthetic random-legal-move games:
    python -m scripts.train_real --num-games 20 --epochs 10
"""

from __future__ import annotations

import argparse
import io
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from chess_sim.data.sharded_dataset import (
        ShardAwareBatchSampler,
        ShardedChessDataset,
    )

import chess
import chess.pgn
import torch
from torch.utils.data import DataLoader

from chess_sim.config import (
    TrainConfig,
    load_train_config,
)
from chess_sim.data.dataset import ChessDataset
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.training.trainer import Trainer
from chess_sim.types import TrainingExample
from chess_sim.utils import winner_color

logger = logging.getLogger(__name__)


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
                text_io = io.TextIOWrapper(
                    reader, encoding="utf-8",
                    errors="replace",
                )
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
# Trajectory tokens
# ---------------------------------------------------------------------------

def _make_trajectory_tokens(move_history: list[chess.Move]) -> list[int]:
    """Return trajectory_tokens of length 65 from the last two half-moves.

    Index 0 is CLS (always 0). Indices 1-64 map to squares a1-h8.
    Values: 0=none, 1=player prev loc, 2=player curr loc,
            3=opp prev loc, 4=opp curr loc.

    The most recent half-move (move_history[-1]) is the opponent's last move.
    The second most recent (move_history[-2]) is the player's last move.
    Opp marks overwrite player marks on collision (semantically correct when
    the opponent captured the piece the player just moved).

    Args:
        move_history: All moves played up to (not including) the current ply.

    Returns:
        List of 65 ints, values 0-4.
    """
    tokens: list[int] = [0] * 65
    if len(move_history) >= 2:
        pl = move_history[-2]
        tokens[pl.from_square + 1] = 1
        tokens[pl.to_square + 1] = 2
    if len(move_history) >= 1:
        opp = move_history[-1]
        tokens[opp.from_square + 1] = 3
        tokens[opp.to_square + 1] = 4
    return tokens


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
        trajectory_tokens = _make_trajectory_tokens(move_history)

        examples.append(TrainingExample(
            board_tokens=tokenized.board_tokens,
            color_tokens=tokenized.color_tokens,
            trajectory_tokens=trajectory_tokens,
            src_sq=move.from_square,
            tgt_sq=move.to_square,
        ))
        move_history.append(move)
        board.push(move)

    return examples


def build_examples_from_file(
    pgn_path: Path,
    winners_only: bool = False,
    max_games: int = 0,
) -> list[TrainingExample]:
    """Build training examples from a PGN file.

    Args:
        pgn_path: Path to .pgn or .pgn.zst file.
        winners_only: If True, only include winning side positions.
        max_games: Stop after this many games (0 = no limit).

    Returns:
        List of TrainingExample namedtuples.
    """
    tokenizer = BoardTokenizer()
    all_examples: list[TrainingExample] = []
    for i, game in enumerate(stream_pgn(pgn_path)):
        if max_games and i >= max_games:
            break
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


def build_examples_synthetic(
    num_games: int, seed: int = 42,
) -> list[TrainingExample]:
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

class _PlainPGNReader:
    """Adapter that streams games from plain .pgn or .zst files.

    Wraps the existing stream_pgn function to match the
    StreamingPGNReader.stream interface expected by PGNPreprocessor.
    """

    def stream(self, path: Path) -> Iterator[chess.pgn.Game]:
        """Yield games from a .pgn or .zst file."""
        return stream_pgn(path)


def _build_streaming_datasets(
    pgn_path: Path,
    winners_only: bool,
    max_games: int,
    batch_size: int = 128,
    train_frac: float = 0.9,
    chunk_size: int = 1024,
) -> tuple[
    ShardedChessDataset,
    ShardedChessDataset,
    int,
    ShardAwareBatchSampler,
]:
    """Preprocess a PGN file into shards and build datasets.

    Args:
        pgn_path: Path to the PGN file.
        winners_only: Only include winning-side positions.
        max_games: Stop after N games (0 = no limit).
        batch_size: Batch size for the train sampler.
        train_frac: Fraction of shards for training.
        chunk_size: Games per shard chunk.

    Returns:
        Tuple of (train_ds, val_ds, total_examples, sampler).
    """
    from chess_sim.data.cache_manager import CacheManager
    from chess_sim.data.chunk_processor import ChunkProcessor
    from chess_sim.data.preprocessor import PGNPreprocessor
    from chess_sim.data.shard_writer import ShardWriter
    from chess_sim.data.sharded_dataset import (
        ShardAwareBatchSampler,
        ShardedChessDataset,
    )
    from chess_sim.data.streaming_types import PreprocessConfig

    cache_dir = pgn_path.parent / ".shard_cache"

    reader = _PlainPGNReader()
    tokenizer = BoardTokenizer()
    cp = ChunkProcessor(tokenizer, winners_only=winners_only)
    sw = ShardWriter()
    cm = CacheManager()
    pp = PGNPreprocessor(reader, cp, sw, cm)

    config = PreprocessConfig(
        chunk_size=chunk_size,
        winners_only=winners_only,
        max_games=max_games,
    )
    info = pp.preprocess(pgn_path, cache_dir, config)
    logger.info(
        "Preprocessed %d examples in %d shards",
        info.total_examples,
        len(info.shard_paths),
    )

    n_shards = len(info.shard_paths)
    split_idx = max(int(n_shards * train_frac), 1)

    if n_shards <= 1:
        train_ds = ShardedChessDataset(
            info.shard_paths,
            info.examples_per_shard,
        )
        val_ds = ShardedChessDataset([], [])
    else:
        train_ds = ShardedChessDataset(
            info.shard_paths[:split_idx],
            info.examples_per_shard[:split_idx],
        )
        val_ds = ShardedChessDataset(
            info.shard_paths[split_idx:],
            info.examples_per_shard[split_idx:],
        )

    train_sampler = ShardAwareBatchSampler(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_ds, val_ds, info.total_examples, train_sampler


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    All overrideable args default to None so that the merge step
    can distinguish 'explicitly set' from 'not provided'.
    """
    p = argparse.ArgumentParser(
        description="Train chess encoder (with optional YAML config)."
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML training config file.",
    )
    p.add_argument(
        "--pgn", type=str, default=None,
        help="Path to a .pgn or .pgn.zst file (optional).",
    )
    p.add_argument(
        "--num-games", type=int, default=None,
        help="Synthetic games when --pgn is not given.",
    )
    p.add_argument(
        "--max-games", type=int, default=None,
        help="Max games to read from PGN (0 = no limit).",
    )
    p.add_argument(
        "--winners-only",
        action="store_true",
        default=None,
        help="Include only the winning player's positions. Skips draws.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size.",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to save final checkpoint (.pt).",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (.pt).",
    )
    return p


def _merge_train_config(
    args: argparse.Namespace,
    cfg: TrainConfig,
) -> TrainConfig:
    """Apply non-None CLI args on top of cfg (mutates and returns cfg).

    Args:
        args: Parsed argparse namespace (None means not provided).
        cfg: TrainConfig loaded from YAML or default.

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
    if args.resume is not None:
        cfg.trainer.resume = args.resume
    return cfg


def main() -> None:
    """Entry point for training on real or synthetic chess data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    if args.config:
        cfg = load_train_config(Path(args.config))
        logger.info("Loaded config from %s", args.config)
    else:
        cfg = TrainConfig()

    cfg = _merge_train_config(args, cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    if cfg.data.pgn:
        pgn_path = Path(cfg.data.pgn)
        logger.info("Streaming pipeline for %s ...", pgn_path)
        train_ds, val_ds, total, sampler = (
            _build_streaming_datasets(
                pgn_path,
                winners_only=cfg.data.winners_only,
                max_games=cfg.data.max_games,
                batch_size=cfg.data.batch_size,
                train_frac=cfg.data.train_frac,
                chunk_size=cfg.data.chunk_size,
            )
        )
        logger.info(
            "Total: %d  Train: %d  Val: %d",
            total, len(train_ds), len(val_ds),
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=cfg.data.num_workers,
        )
    else:
        logger.info(
            "Generating %d synthetic games...", cfg.data.num_games
        )
        examples = build_examples_synthetic(cfg.data.num_games)
        logger.info("Total examples: %d", len(examples))
        train_ds, val_ds = ChessDataset.split(
            examples, train_frac=cfg.data.train_frac
        )
        logger.info(
            "Train: %d  Val: %d", len(train_ds), len(val_ds)
        )
        loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=0,
        )

    total_steps = cfg.trainer.epochs * len(loader)
    trainer = Trainer(
        device=device,
        total_steps=max(total_steps, 1),
        trainer_cfg=cfg.trainer,
        model_cfg=cfg.model,
    )

    if cfg.trainer.resume:
        ckpt = torch.load(
            cfg.trainer.resume, map_location=device
        )
        trainer.encoder.load_state_dict(
            ckpt['encoder'], strict=False
        )
        trainer.heads.load_state_dict(
            ckpt['heads'], strict=False
        )
        logger.info("Resumed from %s", cfg.trainer.resume)

    logger.info(
        "Training %d epochs | batches/epoch=%d | steps=%d",
        cfg.trainer.epochs, len(loader), total_steps,
    )

    epoch_losses: list[float] = []
    for epoch in range(1, cfg.trainer.epochs + 1):
        avg = trainer.train_epoch(loader)
        epoch_losses.append(avg)
        logger.info("Epoch %02d: avg_loss=%.4f", epoch, avg)

    first, last = epoch_losses[0], epoch_losses[-1]
    logger.info(
        "Loss: %.4f -> %.4f (delta=%+.4f)",
        first, last, last - first,
    )

    if cfg.trainer.checkpoint:
        ckpt_path = Path(cfg.trainer.checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(ckpt_path)
        logger.info("Checkpoint saved to %s", ckpt_path)


if __name__ == "__main__":
    main()
