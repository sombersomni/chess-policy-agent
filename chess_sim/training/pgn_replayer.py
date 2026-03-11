"""PGNReplayer: replays a chess.pgn.Game into OfflinePlyTuple records.

Implements the Replayable protocol. Iterates all mainline moves,
captures board state BEFORE each move, and determines winner/loser
and draw ply assignment from the game result header.
"""
from __future__ import annotations

import logging

import chess
import chess.pgn
import torch

from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.training.phase2_trainer import (
    _make_trajectory_tokens,
)
from chess_sim.types import OfflinePlyTuple

logger = logging.getLogger(__name__)

# Standard piece values for material evaluation.
_PIECE_VALUES: dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def _material_of(
    board: chess.Board, color: chess.Color,
) -> float:
    """Sum of piece values for color on board.

    Args:
        board: Current board state.
        color: chess.WHITE or chess.BLACK.

    Returns:
        Total material value for the given color.

    Example:
        >>> _material_of(chess.Board(), chess.WHITE)
        39.0
    """
    return sum(
        _PIECE_VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.color == color
    )


# Result header -> (white_wins, black_wins) mapping
_RESULT_MAP: dict[str, tuple[bool, bool]] = {
    "1-0": (True, False),
    "0-1": (False, True),
    "1/2-1/2": (True, True),  # draw: both sides "win"
}


class PGNReplayer:
    """Replays a chess.pgn.Game into a list of OfflinePlyTuple.

    Implements the Replayable protocol.
    """

    def __init__(self) -> None:
        """Initialize tokenizers for board and move encoding."""
        self._board_tok = BoardTokenizer()
        self._move_tok = MoveTokenizer()

    def replay(
        self,
        game: chess.pgn.Game,
    ) -> list[OfflinePlyTuple]:
        """Replay all moves, returning one OfflinePlyTuple per ply.

        Skips games with unknown result (e.g. '*').

        Args:
            game: A parsed PGN game object.

        Returns:
            List of OfflinePlyTuple in game order.
            Empty if result is unknown or no moves.
        """
        result = game.headers.get("Result", "*")
        if result not in _RESULT_MAP:
            return []

        white_wins, black_wins = _RESULT_MAP[result]
        is_draw = white_wins and black_wins
        board = game.board()
        move_history: list[chess.Move] = []
        plies: list[OfflinePlyTuple] = []
        last_balance: dict[chess.Color, float | None] = {
            chess.WHITE: None,
            chess.BLACK: None,
        }

        for move in game.mainline_moves():
            is_white_ply = board.turn == chess.WHITE
            is_winner = (
                white_wins if is_white_ply else black_wins
            )
            side = board.turn

            # Material balance delta: (own - opp) change since last same-color ply.
            # Positive = net gain (we captured), negative = net loss (opponent captured).
            opp = not side
            cur_balance = (
                _material_of(board, side) - _material_of(board, opp)
            )
            prev_balance = last_balance[side]
            delta = (
                cur_balance - prev_balance
                if prev_balance is not None
                else 0.0
            )
            last_balance[side] = cur_balance

            # Tokenize board BEFORE push
            tb = self._board_tok.tokenize(board, board.turn)
            bt = torch.tensor(
                tb.board_tokens, dtype=torch.long
            )
            ct = torch.tensor(
                tb.color_tokens, dtype=torch.long
            )
            traj = _make_trajectory_tokens(move_history)
            tt = torch.tensor(traj, dtype=torch.long)

            # Build move prefix: SOS + prior moves (no EOS)
            prior_ucis = [m.uci() for m in move_history]
            prefix = self._move_tok.tokenize_game(prior_ucis)
            prefix = prefix[:-1]  # drop EOS

            board.push(move)

            plies.append(OfflinePlyTuple(
                board_tokens=bt,
                color_tokens=ct,
                traj_tokens=tt,
                move_prefix=prefix,
                move_uci=move.uci(),
                is_winner_ply=is_winner,
                is_white_ply=is_white_ply,
                is_draw_ply=is_draw,
                material_delta=delta,
            ))
            move_history.append(move)

        return plies
