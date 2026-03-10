"""RLPlyParser: convert a chess.pgn.Game into RLPlyRecord list.

Applies train_color filter, encodes board/color/trajectory/prefix
tokens using existing BoardTokenizer and MoveTokenizer. RL analogue
of GameParser.
"""

from __future__ import annotations

import logging

import chess
import chess.pgn

from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.training.phase2_trainer import (
    _make_trajectory_tokens,
)
from chess_sim.types import RLPlyRecord

logger = logging.getLogger(__name__)

# Result header -> (white_wins, black_wins)
_RESULT_MAP: dict[str, tuple[bool, bool]] = {
    "1-0": (True, False),
    "0-1": (False, True),
    "1/2-1/2": (True, True),
}


class RLPlyParser:
    """Parse a PGN game into filtered RLPlyRecord instances.

    Only plies for the configured train_color are returned.
    Games with unknown result ("*") or fewer than min_moves
    return an empty list.

    Example:
        >>> parser = RLPlyParser(train_color="white")
        >>> records = parser.parse_game(game, game_id=0)
        >>> len(records)  # half the game's plies (white only)
        30
    """

    def __init__(
        self,
        train_color: str,
        min_moves: int = 5,
        max_moves: int = 512,
    ) -> None:
        """Initialize parser with filtering parameters.

        Args:
            train_color: "white" or "black" -- side to keep.
            min_moves: Minimum full moves to accept a game.
            max_moves: Maximum full moves before truncation.

        Raises:
            ValueError: If train_color is not valid.

        Example:
            >>> parser = RLPlyParser("white", min_moves=10)
        """
        if train_color not in ("white", "black"):
            raise ValueError(
                "train_color must be 'white' or 'black'"
                f", got '{train_color}'"
            )
        self._train_color = train_color
        self._min_moves = min_moves
        self._max_moves = max_moves
        self._board_tok = BoardTokenizer()
        self._move_tok = MoveTokenizer()

    def parse_game(
        self,
        game: chess.pgn.Game,
        game_id: int,
    ) -> list[RLPlyRecord]:
        """Convert one PGN game into a list of RLPlyRecord.

        Iterates mainline moves, tokenizes board state before
        each push, builds move_prefix from prior UCI strings.
        Filters to train_color side only.

        Args:
            game: A parsed chess.pgn.Game object.
            game_id: Unique integer index for this game.

        Returns:
            List of RLPlyRecord for the configured train_color.
            Empty list for games with result "*" or < min_moves.

        Example:
            >>> records = parser.parse_game(game, game_id=42)
            >>> records[0].move_uci
            'e2e4'
        """
        result = game.headers.get("Result", "*")
        if result not in _RESULT_MAP:
            return []

        # Count full moves to check min_moves
        moves = list(game.mainline_moves())
        full_moves = (len(moves) + 1) // 2
        if full_moves < self._min_moves:
            return []

        white_wins, black_wins = _RESULT_MAP[result]
        is_draw = white_wins and black_wins
        board = game.board()
        move_history: list[chess.Move] = []
        records: list[RLPlyRecord] = []
        max_half = self._max_moves * 2

        for ply_idx, move in enumerate(moves):
            if ply_idx >= max_half:
                break

            is_white_ply = board.turn == chess.WHITE
            keep = is_white_ply == (
                self._train_color == "white"
            )

            if keep:
                is_winner = (
                    white_wins
                    if is_white_ply
                    else black_wins
                )

                # Tokenize board BEFORE push
                tb = self._board_tok.tokenize(
                    board, board.turn
                )

                # Trajectory tokens
                traj = _make_trajectory_tokens(move_history)

                # Move prefix: SOS + prior moves (no EOS)
                prior_ucis = [
                    m.uci() for m in move_history
                ]
                prefix = self._move_tok.tokenize_game(
                    prior_ucis
                )
                prefix = prefix[:-1]  # drop EOS

                records.append(
                    RLPlyRecord(
                        board_tokens=list(tb.board_tokens),
                        color_tokens=list(tb.color_tokens),
                        traj_tokens=traj,
                        move_prefix=prefix.tolist(),
                        move_uci=move.uci(),
                        is_winner_ply=is_winner,
                        is_white_ply=is_white_ply,
                        is_draw_ply=is_draw,
                        game_id=game_id,
                        ply_index=ply_idx,
                    )
                )

            board.push(move)
            move_history.append(move)

        return records
