"""Tests for per-batch auxiliary target computation.

Covers:
  - compute_from_board_tensor() in capture_map_builder
  - batch_build() in move_category_builder
"""

from __future__ import annotations

import unittest

import chess
import torch

from chess_sim.data.capture_map_builder import compute_from_board_tensor
from chess_sim.data.move_category_builder import (
    CAPTURE_MINOR,
    CAPTURE_PAWN,
    CAPTURE_QUEEN,
    CAPTURE_ROOK,
    CASTLE,
    PROMOTION,
    QUIET,
    batch_build,
)
from chess_sim.data.move_category_builder import (
    build as category_build,
)
from chess_sim.data.move_vocab import MoveVocab

_VOCAB = MoveVocab()


def _enc(uci: str) -> int:
    return _VOCAB.encode(uci)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _board_from_pieces(
    pieces: list[tuple[int, int, int]],
) -> torch.Tensor:
    """Build a [1, 65, 3] board tensor.

    Each piece is (slot, piece_type, color) where slot is 1-64 (square+1),
    piece_type is 0-7, color is 0=empty/1=player/2=opponent.
    Unfilled slots are (piece_type=1, color=0) = EMPTY.
    """
    board = torch.zeros(1, 65, 3)
    # Default all squares to EMPTY (piece_type=1, color=0)
    board[0, 1:, 0] = 1.0
    for slot, ptype, color in pieces:
        board[0, slot, 0] = ptype
        board[0, slot, 1] = color
    return board


# ---------------------------------------------------------------------------
# compute_from_board_tensor
# ---------------------------------------------------------------------------

class TestComputeCaptureMap(unittest.TestCase):
    """Tests for compute_from_board_tensor()."""

    def test_output_shape(self) -> None:
        """Output is [B, 64] float tensor."""
        board = torch.zeros(3, 65, 3)
        out = compute_from_board_tensor(board)
        self.assertEqual(out.shape, (3, 64))
        self.assertEqual(out.dtype, torch.float32)

    def test_empty_board_all_zeros(self) -> None:
        """Board with no pieces → no capture targets."""
        board = torch.zeros(1, 65, 3)
        out = compute_from_board_tensor(board)
        self.assertTrue((out == 0).all().item())

    def test_opponent_piece_marked(self) -> None:
        """Opponent piece at slot 5 → index 4 in output is 1."""
        board = _board_from_pieces([(5, 2, 2)])  # PAWN, opponent
        out = compute_from_board_tensor(board)
        self.assertEqual(out[0, 4].item(), 1.0)

    def test_player_piece_not_marked(self) -> None:
        """Player piece at slot 5 → index 4 in output is 0."""
        board = _board_from_pieces([(5, 2, 1)])  # PAWN, player
        out = compute_from_board_tensor(board)
        self.assertEqual(out[0, 4].item(), 0.0)

    def test_empty_slot_not_marked(self) -> None:
        """EMPTY piece type (1) at opponent color → not a capture target."""
        board = _board_from_pieces([(5, 1, 2)])  # EMPTY type, opponent color
        out = compute_from_board_tensor(board)
        self.assertEqual(out[0, 4].item(), 0.0)

    def test_cls_token_excluded(self) -> None:
        """CLS token at slot 0 is excluded; output covers slots 1-64 only."""
        board = torch.zeros(1, 65, 3)
        board[0, 0, 0] = 6.0  # queen at CLS — must not appear in output
        board[0, 0, 1] = 2.0
        out = compute_from_board_tensor(board)
        self.assertEqual(out.shape, (1, 64))
        # Output index 0 corresponds to slot 1, not slot 0
        self.assertEqual(out[0, 0].item(), 0.0)

    def test_multiple_opponent_pieces(self) -> None:
        """Multiple opponent pieces all appear in the capture map."""
        board = _board_from_pieces([
            (1, 5, 2),   # ROOK opponent at slot 1
            (10, 6, 2),  # QUEEN opponent at slot 10
            (20, 3, 2),  # KNIGHT opponent at slot 20
        ])
        out = compute_from_board_tensor(board)
        self.assertEqual(out[0, 0].item(), 1.0)   # slot 1 → index 0
        self.assertEqual(out[0, 9].item(), 1.0)   # slot 10 → index 9
        self.assertEqual(out[0, 19].item(), 1.0)  # slot 20 → index 19

    def test_batch_independence(self) -> None:
        """Each batch item produces its own capture map."""
        board = torch.zeros(2, 65, 3)
        board[0, 1:, 0] = 1.0
        board[1, 1:, 0] = 1.0
        # Only batch item 0 has an opponent piece
        board[0, 3, 0] = 2.0
        board[0, 3, 1] = 2.0
        out = compute_from_board_tensor(board)
        self.assertEqual(out[0, 2].item(), 1.0)
        self.assertEqual(out[1, 2].item(), 0.0)


# ---------------------------------------------------------------------------
# batch_build
# ---------------------------------------------------------------------------

class TestBatchBuild(unittest.TestCase):
    """Tests for batch_build() against the per-move build() function."""

    def _single(
        self, uci: str, board: torch.Tensor,
    ) -> int:
        """Run batch_build on a single move and return the scalar category."""
        idx = torch.tensor([_enc(uci)], dtype=torch.long)
        return batch_build(idx, board).item()  # type: ignore[return-value]

    def test_output_shape(self) -> None:
        """Output is [B] LongTensor."""
        B = 4
        moves = torch.zeros(B, dtype=torch.long)
        board = torch.zeros(B, 65, 3)
        out = batch_build(moves, board)
        self.assertEqual(out.shape, (B,))
        self.assertEqual(out.dtype, torch.long)

    def test_quiet_move(self) -> None:
        """e2e4 on an otherwise empty board → QUIET."""
        board = torch.zeros(1, 65, 3)
        # Set all squares to EMPTY
        board[0, 1:, 0] = 1.0
        # Put a pawn (player) at e2 = square 12, slot 13
        board[0, 13, 0] = 2.0
        board[0, 13, 1] = 1.0
        cat = self._single("e2e4", board)
        self.assertEqual(cat, QUIET)

    def test_promotion(self) -> None:
        """Any promotion UCI → PROMOTION regardless of board state."""
        board = torch.zeros(1, 65, 3)
        board[0, 1:, 0] = 1.0
        cat = self._single("a7a8q", board)
        self.assertEqual(cat, PROMOTION)

    def test_castle_kingside(self) -> None:
        """e1g1 → CASTLE regardless of board state."""
        board = torch.zeros(1, 65, 3)
        board[0, 1:, 0] = 1.0
        cat = self._single("e1g1", board)
        self.assertEqual(cat, CASTLE)

    def test_castle_queenside(self) -> None:
        """e1c1 → CASTLE."""
        board = torch.zeros(1, 65, 3)
        board[0, 1:, 0] = 1.0
        cat = self._single("e1c1", board)
        self.assertEqual(cat, CASTLE)

    def test_capture_pawn(self) -> None:
        """Move landing on opponent PAWN → CAPTURE_PAWN."""
        # d4xe5: d4=square 27 (slot 28), e5=square 36 (slot 37)
        board = _board_from_pieces([
            (28, 2, 1),  # player PAWN at d4
            (37, 2, 2),  # opponent PAWN at e5
        ])
        cat = self._single("d4e5", board)
        self.assertEqual(cat, CAPTURE_PAWN)

    def test_capture_knight(self) -> None:
        """Move landing on opponent KNIGHT → CAPTURE_MINOR."""
        # b1xc3: b1=slot 2, c3=slot 19
        board = _board_from_pieces([
            (2, 3, 1),   # player KNIGHT at b1
            (19, 3, 2),  # opponent KNIGHT at c3
        ])
        cat = self._single("b1c3", board)
        self.assertEqual(cat, CAPTURE_MINOR)

    def test_capture_bishop(self) -> None:
        """Move landing on opponent BISHOP → CAPTURE_MINOR."""
        # a1xb2: a1=slot 1, b2=slot 10
        board = _board_from_pieces([
            (1, 5, 1),   # player ROOK at a1
            (10, 4, 2),  # opponent BISHOP at b2
        ])
        cat = self._single("a1b2", board)
        self.assertEqual(cat, CAPTURE_MINOR)

    def test_capture_rook(self) -> None:
        """Move landing on opponent ROOK → CAPTURE_ROOK."""
        # a1xa2 (queen takes rook)
        board_fixed = _board_from_pieces([
            (1, 6, 1),
            (9, 5, 2),
        ])
        cat = self._single("a1a2", board_fixed)
        self.assertEqual(cat, CAPTURE_ROOK)

    def test_capture_queen(self) -> None:
        """Move landing on opponent QUEEN → CAPTURE_QUEEN."""
        # a1xa8: a1=slot 1, a8=slot 57
        board = _board_from_pieces([
            (1, 5, 1),   # player ROOK at a1
            (57, 6, 2),  # opponent QUEEN at a8
        ])
        cat = self._single("a1a8", board)
        self.assertEqual(cat, CAPTURE_QUEEN)

    def test_batch_multiple_categories(self) -> None:
        """Batch of moves with mixed categories returns correct per-sample result."""
        quiet_idx = _enc("e2e4")
        promo_idx = _enc("a7a8q")
        castle_idx = _enc("e1g1")
        moves = torch.tensor(
            [quiet_idx, promo_idx, castle_idx], dtype=torch.long
        )
        board = torch.zeros(3, 65, 3)
        board[:, 1:, 0] = 1.0  # all squares EMPTY
        out = batch_build(moves, board)
        self.assertEqual(out[0].item(), QUIET)
        self.assertEqual(out[1].item(), PROMOTION)
        self.assertEqual(out[2].item(), CASTLE)

    def test_matches_per_move_build_quiet(self) -> None:
        """batch_build QUIET agrees with build() on starting position."""
        board_obj = chess.Board()
        uci = "e2e4"
        expected = category_build(uci, board_obj)

        board_t = torch.zeros(1, 65, 3)
        board_t[0, 1:, 0] = 1.0
        cat = self._single(uci, board_t)
        self.assertEqual(cat, expected)


if __name__ == "__main__":
    unittest.main()
