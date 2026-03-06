"""ChessDataset: torch.utils.data.Dataset for preprocessed training examples.

Preprocessed examples are saved to disk once and memory-mapped at training time.
The train/val split is performed at the game level (not example level) to prevent
data leakage from consecutive board states of the same game.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from chess_sim.types import ChessBatch, TrainingExample

# Fraction of games used for training (remaining go to validation).
TRAIN_SPLIT: float = 0.95


class ChessDataset(Dataset):
    """Stores preprocessed training examples and serves them as typed tensors.

    Each example corresponds to one board state and its associated move labels.
    Implements torch.utils.data.Dataset so it can be wrapped in a DataLoader.

    Attributes:
        examples: List of TrainingExample namedtuples loaded from disk.

    Example:
        >>> ds = ChessDataset(examples)
        >>> batch = ds[0]
        >>> batch.board_tokens.shape
        torch.Size([65])
    """

    def __init__(self, examples: list[TrainingExample]) -> None:
        """Initialize the dataset with a list of preprocessed training examples.

        Args:
            examples: Preprocessed examples produced by the data pipeline.
                      Each contains board_tokens, color_tokens, and four square labels.

        Example:
            >>> ds = ChessDataset(examples)
            >>> len(ds)
            40000000
        """
        self.examples = examples

    def __len__(self) -> int:
        """Return the number of examples in this dataset split.

        Returns:
            Integer count of training examples.

        Example:
            >>> len(ChessDataset(examples))
            40000000
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> ChessBatch:
        """Return one training example as a ChessBatch of scalar/1D tensors.

        All returned tensors have dtype=torch.long. Labels use -1 as ignore_index
        for the opponent heads when the current move is the last in the game.

        Args:
            idx: Integer index into the dataset.

        Returns:
            ChessBatch with board_tokens [65], color_tokens [65], and four label scalars.

        Example:
            >>> item = ds[0]
            >>> item.board_tokens.dtype
            torch.int64
        """
        ex = self.examples[idx]
        return ChessBatch(
            board_tokens=torch.tensor(
                ex.board_tokens, dtype=torch.long
            ),
            color_tokens=torch.tensor(
                ex.color_tokens, dtype=torch.long
            ),
            trajectory_tokens=torch.tensor(
                ex.trajectory_tokens, dtype=torch.long
            ),
            src_sq=torch.tensor(
                ex.src_sq, dtype=torch.long
            ),
            tgt_sq=torch.tensor(
                ex.tgt_sq, dtype=torch.long
            ),
            opp_src_sq=torch.tensor(
                ex.opp_src_sq, dtype=torch.long
            ),
            opp_tgt_sq=torch.tensor(
                ex.opp_tgt_sq, dtype=torch.long
            ),
        )

    @staticmethod
    def from_disk(path: Path) -> "ChessDataset":
        """Load a preprocessed dataset from a .pt or Arrow file on disk.

        Args:
            path: Path to the serialized dataset file.

        Returns:
            ChessDataset instance with examples loaded from disk.

        Example:
            >>> ds = ChessDataset.from_disk(Path("data/train.pt"))
        """
        examples = torch.load(path)
        return ChessDataset(examples)

    @staticmethod
    def split(
        examples: list[TrainingExample], train_frac: float = TRAIN_SPLIT
    ) -> tuple["ChessDataset", "ChessDataset"]:
        """Split examples into train and validation datasets at the game level.

        The split is performed on whole games to prevent leakage from consecutive
        board states. Examples within a game stay together in one split.

        Args:
            examples: Full list of TrainingExample namedtuples.
            train_frac: Fraction of games to use for training. Default 0.95.

        Returns:
            Tuple (train_dataset, val_dataset).

        Example:
            >>> train_ds, val_ds = ChessDataset.split(examples)
        """
        split_idx = int(len(examples) * train_frac)
        return (
            ChessDataset(examples[:split_idx]),
            ChessDataset(examples[split_idx:]),
        )
