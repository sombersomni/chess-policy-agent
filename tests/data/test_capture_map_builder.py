"""Tests for CaptureMapBuilder.

T1: Starting position — no capturable squares.
T2: After 1. e4 e5 2. Nf3 — e5 pawn is capturable by Nf3.
T3: Multiple captures available.
T4: Black's perspective.
T5: Empty board (only kings).
T6: Output is always length 64.
T7: En passant square is NOT counted (we check board.attackers
    which uses piece occupation, not en passant flag).
"""
from __future__ import annotations

import unittest

import chess

from chess_sim.data.capture_map_builder import build


class TestCaptureMapBuilder(unittest.TestCase):
    """Tests for capture_map_builder.build()."""

    def test_t1_starting_position_no_captures(
        self,
    ) -> None:
        """Starting position: no opponent piece is attacked."""
        board = chess.Board()
        result = build(board, chess.WHITE)
        self.assertEqual(len(result), 64)
        self.assertEqual(sum(result), 0)

    def test_t2_e5_pawn_capturable_after_nf3(
        self,
    ) -> None:
        """After 1.e4 e5 2.Nf3, e5 pawn is attacked by Nf3."""
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "g1f3"):
            board.push(chess.Move.from_uci(uci))
        # White's turn is done; now it's Black's turn.
        # From White's perspective, e5 (square 36) has
        # a black pawn attacked by Nf3.
        result = build(board, chess.WHITE)
        self.assertEqual(result[chess.E5], 1)

    def test_t3_multiple_captures_available(
        self,
    ) -> None:
        """Multiple opponent pieces attacked at once."""
        # Italian Game position where white attacks d5, f7
        board = chess.Board(
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/"
            "PPPP1PPP/RNBQK2R w KQkq - 2 3"
        )
        result = build(board, chess.WHITE)
        # Bc4 attacks f7 pawn, Nf3 attacks e5 pawn
        self.assertEqual(result[chess.F7], 1)
        self.assertEqual(result[chess.E5], 1)

    def test_t4_black_perspective(self) -> None:
        """Capture map from Black's side."""
        board = chess.Board()
        for uci in ("e2e4",):
            board.push(chess.Move.from_uci(uci))
        # Black to move. Does Black attack any white pieces?
        result = build(board, chess.BLACK)
        # e4 pawn is NOT attacked by any black piece yet
        self.assertEqual(result[chess.E4], 0)
        self.assertEqual(sum(result), 0)

    def test_t5_only_kings(self) -> None:
        """Board with only kings — no captures possible."""
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        result = build(board, chess.WHITE)
        self.assertEqual(sum(result), 0)
        result_b = build(board, chess.BLACK)
        self.assertEqual(sum(result_b), 0)

    def test_t6_output_length_always_64(self) -> None:
        """Output is always exactly 64 elements."""
        for fen in (
            chess.STARTING_FEN,
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/"
            "PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        ):
            board = chess.Board(fen)
            result = build(board, board.turn)
            self.assertEqual(
                len(result), 64,
                f"Expected len 64 for fen={fen}",
            )

    def test_t7_values_are_binary(self) -> None:
        """All values are 0 or 1."""
        board = chess.Board(
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/"
            "PPPP1PPP/RNBQK2R w KQkq - 2 3"
        )
        result = build(board, chess.WHITE)
        self.assertTrue(
            all(v in (0, 1) for v in result)
        )


if __name__ == "__main__":
    unittest.main()
