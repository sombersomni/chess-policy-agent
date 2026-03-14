"""Tests for MoveCategoryBuilder.

T1: Quiet move (e2e4).
T2: Pawn capture.
T3: Minor piece capture (knight takes bishop).
T4: Rook capture.
T5: Queen capture.
T6: Castling.
T7: Promotion (highest priority even if also a capture).
T8: En passant is classified as CAPTURE_PAWN.
"""
from __future__ import annotations

import unittest

import chess

from chess_sim.data.move_category_builder import (
    CAPTURE_MINOR,
    CAPTURE_PAWN,
    CAPTURE_QUEEN,
    CAPTURE_ROOK,
    CASTLE,
    PROMOTION,
    QUIET,
    build,
)


class TestMoveCategoryBuilder(unittest.TestCase):
    """Tests for move_category_builder.build()."""

    def test_t1_quiet_move(self) -> None:
        """e2e4 on starting board is QUIET."""
        board = chess.Board()
        self.assertEqual(build("e2e4", board), QUIET)

    def test_t2_pawn_capture(self) -> None:
        """Pawn captures pawn -> CAPTURE_PAWN."""
        board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/"
            "PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
        )
        self.assertEqual(
            build("e4d5", board), CAPTURE_PAWN
        )

    def test_t3_minor_piece_capture(self) -> None:
        """Knight takes bishop -> CAPTURE_MINOR."""
        board = chess.Board(
            "r1bqkbnr/pppppppp/2n5/8/8/2N2b2/"
            "PPPPPPPP/R1BQKBNR w KQkq - 2 3"
        )
        # Nc3 takes bishop on f3... wait that's not right
        # Let's use a cleaner position
        board = chess.Board(
            "r1bqk1nr/pppppppp/2n5/8/1b6/2N5/"
            "PPPPPPPP/R1BQKBNR w KQkq - 2 3"
        )
        # Nc3 captures Bb4
        self.assertEqual(
            build("c3b4", board), CAPTURE_MINOR  # bishop
        )

    def test_t4_rook_capture(self) -> None:
        """Piece captures rook -> CAPTURE_ROOK."""
        board = chess.Board(
            "4k3/8/8/8/8/8/8/r3K3 w - - 0 1"
        )
        # King captures rook on a1
        self.assertEqual(
            build("e1d1", board), QUIET  # no rook on d1
        )
        # Better: bishop captures rook
        board = chess.Board(
            "4k3/8/8/8/8/2B5/8/r3K3 w - - 0 1"
        )
        self.assertEqual(
            build("c3a1", board), CAPTURE_ROOK
        )

    def test_t5_queen_capture(self) -> None:
        """Piece captures queen -> CAPTURE_QUEEN."""
        board = chess.Board(
            "4k3/8/8/3q4/4B3/8/8/4K3 w - - 0 1"
        )
        self.assertEqual(
            build("e4d5", board), CAPTURE_QUEEN
        )

    def test_t6_castling(self) -> None:
        """Kingside castling -> CASTLE."""
        board = chess.Board(
            "r1bqkbnr/pppppppp/2n5/8/8/5N2/"
            "PPPPPPPP/RNBQKB1R w KQkq - 2 2"
        )
        # Can't castle yet — need to clear f1/g1.
        board = chess.Board(
            "r1bqk2r/pppppppp/2n2n2/2b5/2B5/5N2/"
            "PPPPPPPP/RNBQK2R w KQkq - 4 4"
        )
        self.assertEqual(
            build("e1g1", board), CASTLE
        )

    def test_t7_promotion_highest_priority(self) -> None:
        """Promotion beats all other categories."""
        # Pawn on e7 promotes: e7e8q
        board = chess.Board(
            "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"
        )
        self.assertEqual(
            build("e7e8q", board), PROMOTION
        )

    def test_t8_promotion_capture_still_promotion(
        self,
    ) -> None:
        """Promotion with capture -> PROMOTION (not CAPTURE)."""
        board = chess.Board(
            "3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1"
        )
        self.assertEqual(
            build("e7d8q", board), PROMOTION
        )

    def test_t9_en_passant_is_capture_pawn(self) -> None:
        """En passant -> CAPTURE_PAWN."""
        board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/"
            "PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        )
        self.assertEqual(
            build("e5d6", board), CAPTURE_PAWN
        )


if __name__ == "__main__":
    unittest.main()
