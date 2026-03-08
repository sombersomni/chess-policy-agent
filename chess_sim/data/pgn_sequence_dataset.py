"""PGNSequenceDataset: produces one GameTurnSample per game turn.

Each sample contains the board state at a given turn plus the decoder
input/target move-token sequences up to that point. PGNSequenceCollator
pads variable-length move sequences within a batch.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.move_vocab import EOS_IDX, PAD_IDX, SOS_IDX
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.data.tokenizer_utils import (
    make_trajectory_tokens as _make_trajectory_tokens,
)
from chess_sim.types import GameTurnBatch, GameTurnSample

logger = logging.getLogger(__name__)


def _stream_pgn(path: Path) -> Iterator[chess.pgn.Game]:
    """Yield games from a plain .pgn or .zst file."""
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
        with open(
            path, encoding="utf-8", errors="replace"
        ) as fh:
            yield from _read_games(fh)


def _read_games(
    fh: io.TextIOWrapper,
) -> Iterator[chess.pgn.Game]:
    """Read games from a text stream until exhausted."""
    while True:
        game = chess.pgn.read_game(fh)
        if game is None:
            break
        yield game


class PGNSequenceDataset(Dataset[GameTurnSample]):
    """Dataset producing one GameTurnSample per game turn.

    Each sample contains the board state tokens, color tokens, trajectory
    tokens, and the move-token sequences (decoder input and shifted target).

    Attributes:
        samples: Pre-computed list of GameTurnSample namedtuples.

    Example:
        >>> ds = PGNSequenceDataset(pgn_path="games.pgn")
        >>> sample = ds[0]
        >>> sample.board_tokens.shape
        torch.Size([65])
    """

    def __init__(
        self,
        pgn_path: str,
        max_games: int = 0,
        winners_only: bool = False,
    ) -> None:
        """Load and preprocess games from a PGN file.

        Parses all games (or up to max_games), iterates over each ply,
        and builds a GameTurnSample for every turn.

        Args:
            pgn_path: Path to a PGN file (plain text or .zst).
            max_games: Maximum number of games to load. 0 = all.
            winners_only: If True, only include games with decisive result.

        Example:
            >>> ds = PGNSequenceDataset("games.pgn", max_games=100)
        """
        super().__init__()
        self._samples: list[GameTurnSample] = []
        self._build(
            Path(pgn_path), max_games, winners_only
        )

    def _build(
        self,
        pgn_path: Path,
        max_games: int,
        winners_only: bool,
    ) -> None:
        """Parse PGN and build per-ply samples."""
        board_tok = BoardTokenizer()
        move_tok = MoveTokenizer()
        game_count = 0

        for game in _stream_pgn(pgn_path):
            if max_games > 0 and game_count >= max_games:
                break
            result = game.headers.get("Result", "*")
            if winners_only and result not in (
                "1-0", "0-1",
            ):
                continue

            board = game.board()
            moves = list(game.mainline_moves())
            if not moves:
                continue

            move_history: list[chess.Move] = []
            uci_history: list[str] = []

            for t, move in enumerate(moves):
                # Board state BEFORE this move
                tb = board_tok.tokenize(board, board.turn)
                traj = _make_trajectory_tokens(move_history)

                board_tokens = torch.tensor(
                    tb.board_tokens, dtype=torch.long
                )
                color_tokens = torch.tensor(
                    tb.color_tokens, dtype=torch.long
                )
                trajectory_tokens = torch.tensor(
                    traj, dtype=torch.long
                )

                # Decoder input: SOS + moves[:t]
                input_ids = [SOS_IDX] + [
                    move_tok.tokenize_move(u)
                    for u in uci_history
                ]
                move_tokens = torch.tensor(
                    input_ids, dtype=torch.long
                )

                # Target: moves[:t] + current move
                target_ids = [
                    move_tok.tokenize_move(u)
                    for u in uci_history
                ] + [move_tok.tokenize_move(move.uci())]
                target_tokens = torch.tensor(
                    target_ids, dtype=torch.long
                )

                move_pad_mask = torch.zeros(
                    len(input_ids), dtype=torch.bool
                )

                self._samples.append(GameTurnSample(
                    board_tokens=board_tokens,
                    color_tokens=color_tokens,
                    trajectory_tokens=trajectory_tokens,
                    move_tokens=move_tokens,
                    target_tokens=target_tokens,
                    move_pad_mask=move_pad_mask,
                ))

                uci_history.append(move.uci())
                move_history.append(move)
                board.push(move)

            game_count += 1

        logger.info(
            "Built %d samples from %d games",
            len(self._samples), game_count,
        )

    def __len__(self) -> int:
        """Return the total number of game-turn samples.

        Returns:
            Integer count of samples.

        Example:
            >>> len(ds)
            5000
        """
        return len(self._samples)

    def __getitem__(self, idx: int) -> GameTurnSample:
        """Return the GameTurnSample at the given index.

        Args:
            idx: Integer index into the sample list.

        Returns:
            GameTurnSample namedtuple for the requested turn.

        Raises:
            IndexError: If idx is out of range.

        Example:
            >>> sample = ds[42]
            >>> sample.move_tokens.dtype
            torch.int64
        """
        return self._samples[idx]


class PGNSequenceCollator:
    """Collate function that pads move sequences in a batch.

    Used as the collate_fn argument to DataLoader. Pads move_tokens,
    target_tokens, and move_pad_mask to the longest sequence in batch.
    Board/color/trajectory tokens are stacked directly (fixed len 65).

    Attributes:
        pad_idx: Integer padding index for move tokens. Default PAD_IDX.

    Example:
        >>> collator = PGNSequenceCollator()
        >>> loader = DataLoader(ds, batch_size=32, collate_fn=collator)
    """

    def __init__(self, pad_idx: int = PAD_IDX) -> None:
        """Initialize with a padding index.

        Args:
            pad_idx: Token index used for padding. Default PAD_IDX (0).

        Example:
            >>> collator = PGNSequenceCollator()
        """
        self.pad_idx = pad_idx

    def __call__(
        self, samples: list[GameTurnSample]
    ) -> GameTurnBatch:
        """Collate GameTurnSamples into a padded GameTurnBatch.

        Pads move_tokens, target_tokens with pad_idx and move_pad_mask
        with True at padding positions. Stacks fixed-size board tokens.

        Args:
            samples: List of GameTurnSample namedtuples.

        Returns:
            GameTurnBatch with all tensors batched and padded.

        Example:
            >>> batch = collator([ds[0], ds[1], ds[2]])
            >>> batch.move_tokens.shape[0]
            3
        """
        board_tokens = torch.stack(
            [s.board_tokens for s in samples]
        )
        color_tokens = torch.stack(
            [s.color_tokens for s in samples]
        )
        trajectory_tokens = torch.stack(
            [s.trajectory_tokens for s in samples]
        )

        # Pad variable-length move sequences
        move_tokens = pad_sequence(
            [s.move_tokens for s in samples],
            batch_first=True,
            padding_value=self.pad_idx,
        )
        target_tokens = pad_sequence(
            [s.target_tokens for s in samples],
            batch_first=True,
            padding_value=self.pad_idx,
        )
        # Build pad mask: True where padded
        max_len = move_tokens.size(1)
        move_pad_mask = torch.stack([
            torch.cat([
                s.move_pad_mask,
                torch.ones(
                    max_len - len(s.move_pad_mask),
                    dtype=torch.bool,
                ),
            ])
            for s in samples
        ])

        return GameTurnBatch(
            board_tokens=board_tokens,
            color_tokens=color_tokens,
            trajectory_tokens=trajectory_tokens,
            move_tokens=move_tokens,
            target_tokens=target_tokens,
            move_pad_mask=move_pad_mask,
        )
