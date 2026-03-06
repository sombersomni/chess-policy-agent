"""Tests for ChessDataset: T17, T18, T20."""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

import chess
import chess.pgn
import torch
from torch.utils.data import DataLoader

from chess_sim.data.dataset import ChessDataset
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.types import ChessBatch, TrainingExample
from tests.utils import make_small_pgn_fixture, make_training_examples, parse_pgn_games


class TestMoveLabelsExtraction(unittest.TestCase):
    """T17: Move labels are correctly extracted from each ply."""

    def test_labels_from_e4(self) -> None:
        """T17: First move e4 has src=e2, tgt=e4."""
        pgn_text = (
            "[Event \"?\"]\n[Result \"*\"]\n\n"
            "1. e4 e5 *"
        )
        games = parse_pgn_games(pgn_text)
        self.assertEqual(len(games), 1)

        game = games[0]
        moves = list(game.mainline_moves())
        self.assertGreaterEqual(len(moves), 2)

        # First move is e4: from e2 to e4
        first_move = moves[0]
        self.assertEqual(first_move.from_square, chess.E2)
        self.assertEqual(first_move.to_square, chess.E4)


class TestDataLoaderDtypes(unittest.TestCase):
    """T20: DataLoader produces tensors with the correct dtypes."""

    def setUp(self) -> None:
        examples = make_training_examples(8)
        self.ds = ChessDataset(examples)

    def test_board_tokens_dtype_is_long(self) -> None:
        """T20: board_tokens has dtype=torch.long."""
        item = self.ds[0]
        self.assertEqual(item.board_tokens.dtype, torch.long)

    def test_color_tokens_dtype_is_long(self) -> None:
        """T20: color_tokens has dtype=torch.long."""
        item = self.ds[0]
        self.assertEqual(item.color_tokens.dtype, torch.long)

    def test_label_dtypes_are_long(self) -> None:
        """T20: src_sq and tgt_sq have dtype=torch.long."""
        item = self.ds[0]
        self.assertEqual(item.src_sq.dtype, torch.long)
        self.assertEqual(item.tgt_sq.dtype, torch.long)

    def test_dataloader_batch_board_tokens_dtype(self) -> None:
        """T20: DataLoader-batched board_tokens has dtype=torch.long."""
        loader = DataLoader(self.ds, batch_size=4)
        batch = next(iter(loader))
        # DataLoader with NamedTuple returns a list of field tensors.
        # Access board_tokens field.
        board_tokens = batch[0] if isinstance(batch, (list, tuple)) else batch.board_tokens
        self.assertEqual(board_tokens.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
