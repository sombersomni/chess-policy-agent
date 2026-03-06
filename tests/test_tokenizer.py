"""Tests for BoardTokenizer: T01, T02, T03, T13, T14, T15, T16."""

from __future__ import annotations

import unittest

import chess

from chess_sim.data.tokenizer import (
    COLOR_EMPTY,
    COLOR_OPPONENT,
    COLOR_PLAYER,
    CLS_TOKEN,
    EMPTY_TOKEN,
    BoardTokenizer,
)
from tests.utils import make_initial_board_tokens


class TestBoardTokenizerInitialPosition(unittest.TestCase):
    """Tests for tokenizing the standard starting position (T01, T02, T13)."""

    def setUp(self) -> None:
        self.tok = BoardTokenizer()
        self.board = chess.Board()

    def test_tokenize_white_cls_token(self) -> None:
        """T01: CLS token at index 0 has board_token=0 and color_token=0."""
        result = self.tok.tokenize(self.board, chess.WHITE)
        self.assertEqual(result.board_tokens[0], CLS_TOKEN)
        self.assertEqual(result.color_tokens[0], COLOR_EMPTY)

    def test_tokenize_white_a1_is_rook(self) -> None:
        """T01: a1 (index 1) is ROOK (5) with PLAYER color when White to move."""
        result = self.tok.tokenize(self.board, chess.WHITE)
        self.assertEqual(result.board_tokens[1], 5)  # ROOK
        self.assertEqual(result.color_tokens[1], COLOR_PLAYER)

    def test_tokenize_white_sequence_length(self) -> None:
        """T01: Output lists are exactly length 65."""
        result = self.tok.tokenize(self.board, chess.WHITE)
        self.assertEqual(len(result.board_tokens), 65)
        self.assertEqual(len(result.color_tokens), 65)

    def test_tokenize_black_board_tokens_identical(self) -> None:
        """T02/T13: board_tokens are the same regardless of turn (no board flip)."""
        white_result = self.tok.tokenize(self.board, chess.WHITE)
        black_result = self.tok.tokenize(self.board, chess.BLACK)
        self.assertEqual(white_result.board_tokens, black_result.board_tokens)

    def test_tokenize_black_a1_color_is_opponent(self) -> None:
        """T02: When Black is to move, White pieces at a1 get color=2 (OPPONENT)."""
        result = self.tok.tokenize(self.board, chess.BLACK)
        self.assertEqual(result.color_tokens[1], COLOR_OPPONENT)

    def test_tokenize_color_indices_swap_between_turns(self) -> None:
        """T13: For occupied squares, color indices swap (1<->2) between White/Black."""
        white_result = self.tok.tokenize(self.board, chess.WHITE)
        black_result = self.tok.tokenize(self.board, chess.BLACK)
        # Check all 64 squares (skip CLS at index 0).
        for i in range(1, 65):
            w_color = white_result.color_tokens[i]
            b_color = black_result.color_tokens[i]
            if w_color == COLOR_EMPTY:
                self.assertEqual(b_color, COLOR_EMPTY)
            elif w_color == COLOR_PLAYER:
                self.assertEqual(b_color, COLOR_OPPONENT)
            else:
                self.assertEqual(b_color, COLOR_PLAYER)


class TestBoardTokenizerEmptySquare(unittest.TestCase):
    """Tests for tokenizing empty squares (T03)."""

    def setUp(self) -> None:
        self.tok = BoardTokenizer()
        # e4 is empty in the starting position.
        self.board = chess.Board()
        self.e4_idx = chess.E4 + 1  # +1 because index 0 is CLS

    def test_tokenize_empty_square_board_token(self) -> None:
        """T03: Empty square has board_token=1 (EMPTY)."""
        result = self.tok.tokenize(self.board, chess.WHITE)
        self.assertEqual(result.board_tokens[self.e4_idx], EMPTY_TOKEN)

    def test_tokenize_empty_square_color_token(self) -> None:
        """T03: Empty square has color_token=0 (COLOR_EMPTY)."""
        result = self.tok.tokenize(self.board, chess.WHITE)
        self.assertEqual(result.color_tokens[self.e4_idx], COLOR_EMPTY)


class TestBoardTokenizerEdgeCases(unittest.TestCase):
    """Tests for castling (T16), en passant (T15), and promotion (T14) tokenization."""

    def setUp(self) -> None:
        self.tok = BoardTokenizer()

    def test_promotion_src_sq_is_pawn(self) -> None:
        """T14: Before promotion, the source square holds a PAWN (board_token=2)."""
        # Set up a board with a pawn about to promote on h7.
        board = chess.Board("8/7P/8/8/8/8/8/4K1k1 w - - 0 1")
        result = self.tok.tokenize(board, chess.WHITE)
        h7_idx = chess.H7 + 1  # +1 for CLS offset
        self.assertEqual(result.board_tokens[h7_idx], 2)  # PAWN

    def test_castling_src_sq_is_king(self) -> None:
        """T16: King's source square contains KING token (board_token=7)."""
        # Position where White can castle kingside.
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        result = self.tok.tokenize(board, chess.WHITE)
        e1_idx = chess.E1 + 1
        self.assertEqual(result.board_tokens[e1_idx], 7)  # KING

    def test_en_passant_square_is_empty_before_capture(self) -> None:
        """T15: The en passant target square is empty on the board (the capture is virtual)."""
        # After 1.e4 e5 2.e5 d5 — en passant on d6 is available; d6 is empty.
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
        result = self.tok.tokenize(board, chess.WHITE)
        d6_idx = chess.D6 + 1
        self.assertEqual(result.board_tokens[d6_idx], EMPTY_TOKEN)


if __name__ == "__main__":
    unittest.main()
