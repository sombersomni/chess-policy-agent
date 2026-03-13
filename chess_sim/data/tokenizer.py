"""BoardTokenizer: converts chess.Board into 65-token integer sequences.

Token layout:
  Index 0: CLS token (board_token=0, color_token=0).
  Indices 1-64: squares a1..h8 (python-chess square indices 0..63, shifted by +1).

Piece-type vocabulary (board_tokens):
  0=CLS, 1=EMPTY, 2=PAWN, 3=KNIGHT, 4=BISHOP, 5=ROOK, 6=QUEEN, 7=KING

Color vocabulary (color_tokens):
  0=EMPTY/CLS, 1=PLAYER (piece belongs to the player to move), 2=OPPONENT
"""

from __future__ import annotations

import chess

from chess_sim.types import TokenizedBoard

# Piece-type index constants.
CLS_TOKEN: int = 0
EMPTY_TOKEN: int = 1
_PIECE_TYPE_MAP: dict[chess.PieceType, int] = {
    chess.PAWN:   2,
    chess.KNIGHT: 3,
    chess.BISHOP: 4,
    chess.ROOK:   5,
    chess.QUEEN:  6,
    chess.KING:   7,
}

# Color index constants.
COLOR_EMPTY: int = 0
COLOR_PLAYER: int = 1
COLOR_OPPONENT: int = 2

SEQ_LEN: int = 65  # CLS + 64 squares


class BoardTokenizer:
    """Converts a chess.Board into the 65-token encoding used by EmbeddingLayer.

    Implements the Tokenizable protocol.

    The board is always encoded from the fixed geometric perspective (a1=index 1,
    h8=index 64). No board flipping is applied. The color_tokens stream encodes
    which pieces belong to the player to move (PLAYER=1) vs the opponent (OPPONENT=2).
    """

    def tokenize(self, board: chess.Board, turn: chess.Color) -> TokenizedBoard:
        """Convert board to (board_tokens, color_tokens) integer lists of length 65.

        CLS is prepended at index 0. Squares are ordered a1=1, b1=2, ..., h8=64.
        The player's pieces receive color_token=1; the opponent's receive color_token=2.
        No board flip — square semantics are stable regardless of who is to move.

        Args:
            board: Current chess board state.
            turn: The color of the player whose perspective is encoded.

        Returns:
            TokenizedBoard(board_tokens, color_tokens), each of length 65.

        Example:
            >>> tok = BoardTokenizer()
            >>> result = tok.tokenize(chess.Board(), chess.WHITE)
            >>> result.board_tokens[0]  # CLS
            0
        """
        board_tokens: list[int] = [CLS_TOKEN]
        color_tokens: list[int] = [COLOR_EMPTY]
        for sq in range(64):
            piece = board.piece_at(sq)
            if piece is None:
                board_tokens.append(EMPTY_TOKEN)
                color_tokens.append(COLOR_EMPTY)
            else:
                board_tokens.append(_PIECE_TYPE_MAP[piece.piece_type])
                if piece.color == turn:
                    color_tokens.append(COLOR_PLAYER)
                else:
                    color_tokens.append(COLOR_OPPONENT)
        return TokenizedBoard(board_tokens, color_tokens)
