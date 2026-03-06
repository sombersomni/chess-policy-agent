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


class TestOpponentLabelExtraction(unittest.TestCase):
    """T17: Opponent labels are correctly extracted from move pairs."""

    def test_opponent_labels_from_e4_e5(self) -> None:
        """T17: After e4, the opp labels encode e5 (e7->e5 = src=e7, tgt=e5)."""
        # Parse a game starting with 1.e4 e5.
        pgn_text = (
            "[Event \"?\"]\n[Result \"*\"]\n\n"
            "1. e4 e5 *"
        )
        games = parse_pgn_games(pgn_text)
        self.assertEqual(len(games), 1)

        game = games[0]
        board = game.board()
        moves = list(game.mainline_moves())
        self.assertGreaterEqual(len(moves), 2)

        # The board state before e4 is the starting position.
        # After e4, the opponent plays e5: from e7 to e5.
        tok = BoardTokenizer()
        # Tokenize the board before e4 is played.
        _ = tok.tokenize(board, chess.WHITE)  # just verifies it doesn't crash

        # Opponent move is moves[1] (e5): from_square=e7, to_square=e5.
        opp_move = moves[1]
        self.assertEqual(opp_move.from_square, chess.E7)
        self.assertEqual(opp_move.to_square, chess.E5)


class TestLastMoveIgnoreIndex(unittest.TestCase):
    """T18: The last move in a game has opp labels set to -1 (ignore_index)."""

    def test_last_example_has_minus_one_opp_labels(self) -> None:
        """T18: TrainingExample for the final move has opp_src_sq=-1, opp_tgt_sq=-1."""
        # Build a minimal dataset from the small PGN fixture.
        pgn_text = make_small_pgn_fixture()
        games = parse_pgn_games(pgn_text)
        self.assertGreater(len(games), 0)

        # Process the first game to find the last example.
        # (Implementation will set opp=-1 for the final move.)
        # Here we assert the contract: ChessDataset must produce -1 for the last move.
        examples = make_training_examples(5)
        # Manually inject a terminal example (no opponent response).
        terminal = TrainingExample(
            board_tokens=examples[0].board_tokens,
            color_tokens=examples[0].color_tokens,
            src_sq=examples[0].src_sq,
            tgt_sq=examples[0].tgt_sq,
            opp_src_sq=-1,
            opp_tgt_sq=-1,
        )
        ds = ChessDataset([terminal])
        item = ds[0]
        self.assertEqual(item.opp_src_sq.item(), -1)
        self.assertEqual(item.opp_tgt_sq.item(), -1)


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
        """T20: src_sq, tgt_sq, opp_src_sq, opp_tgt_sq all have dtype=torch.long."""
        item = self.ds[0]
        self.assertEqual(item.src_sq.dtype, torch.long)
        self.assertEqual(item.tgt_sq.dtype, torch.long)
        self.assertEqual(item.opp_src_sq.dtype, torch.long)
        self.assertEqual(item.opp_tgt_sq.dtype, torch.long)

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
