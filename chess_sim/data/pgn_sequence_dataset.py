"""PGNSequenceDataset: produces one GameTurnSample per game turn.

Each sample contains the board state at a given turn plus the decoder
input/target move-token sequences up to that point. PGNSequenceCollator
pads variable-length move sequences within a batch.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from chess_sim.data.move_vocab import PAD_IDX
from chess_sim.types import GameTurnBatch, GameTurnSample


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
            winners_only: If True, only include games with a decisive result.

        Example:
            >>> ds = PGNSequenceDataset("games.pgn", max_games=100)
        """
        raise NotImplementedError("To be implemented")

    def __len__(self) -> int:
        """Return the total number of game-turn samples.

        Returns:
            Integer count of samples.

        Example:
            >>> len(ds)
            5000
        """
        raise NotImplementedError("To be implemented")

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
        raise NotImplementedError("To be implemented")


class PGNSequenceCollator:
    """Collate function that pads move sequences to the max length in a batch.

    Used as the collate_fn argument to DataLoader. Pads move_tokens,
    target_tokens, and move_pad_mask to the longest sequence in the batch.
    Board/color/trajectory tokens are stacked directly (fixed length 65).

    Attributes:
        pad_idx: Integer padding index for move tokens. Default PAD_IDX.

    Example:
        >>> collator = PGNSequenceCollator()
        >>> loader = DataLoader(ds, batch_size=32, collate_fn=collator)
    """

    def __init__(self, pad_idx: int = PAD_IDX) -> None:
        """Initialize with a padding index.

        Args:
            pad_idx: Token index used for padding. Defaults to PAD_IDX (0).

        Example:
            >>> collator = PGNSequenceCollator()
        """
        raise NotImplementedError("To be implemented")

    def __call__(self, samples: list[GameTurnSample]) -> GameTurnBatch:
        """Collate a list of GameTurnSamples into a padded GameTurnBatch.

        Pads move_tokens, target_tokens with pad_idx and move_pad_mask
        with True at padding positions. Stacks fixed-size board tokens.

        Args:
            samples: List of GameTurnSample namedtuples from the dataset.

        Returns:
            GameTurnBatch with all tensors batched and padded.

        Example:
            >>> batch = collator([ds[0], ds[1], ds[2]])
            >>> batch.move_tokens.shape[0]
            3
        """
        raise NotImplementedError("To be implemented")
