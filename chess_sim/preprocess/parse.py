"""GameParser: converts a PGN game into RawTurnRecord list.

Replays each ply, tokenizes board/color/trajectory/move sequences,
and produces one RawTurnRecord per turn. Applies configurable filters.
"""

from __future__ import annotations

import logging

import chess
import chess.pgn

from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.move_vocab import SOS_IDX
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.data.tokenizer_utils import make_trajectory_tokens
from chess_sim.types import RawTurnRecord

logger = logging.getLogger(__name__)

# Outcome map: result string -> raw outcome from White's perspective
_OUTCOME_MAP: dict[str, int] = {
    "1-0": 1,
    "0-1": -1,
    "1/2-1/2": 0,
}


class GameParser:
    """Parses a chess.pgn.Game into per-ply RawTurnRecords.

    Implements the HDF5Parseable protocol. Applies min/max move
    filters, winners-only filter, and ELO filter.

    Attributes:
        _min_moves: Minimum number of plies to include a game.
        _max_moves: Maximum plies to process per game.
        _winners_only: If True, skip draws.
        _min_elo: Minimum ELO for both players.
        _board_tok: Board tokenizer instance.
        _move_tok: Move tokenizer instance.
    """

    def __init__(
        self,
        min_moves: int,
        max_moves: int,
        winners_only: bool,
        min_elo: int,
    ) -> None:
        """Initialize parser with filter settings.

        Args:
            min_moves: Skip games with fewer plies.
            max_moves: Truncate games beyond this ply count.
            winners_only: Skip draws if True.
            min_elo: Skip if both ELOs below this threshold.
        """
        self._min_moves = min_moves
        self._max_moves = max_moves
        self._winners_only = winners_only
        self._min_elo = min_elo
        self._board_tok = BoardTokenizer()
        self._move_tok = MoveTokenizer()

    def parse_game(
        self, game: chess.pgn.Game, game_id: int
    ) -> list[RawTurnRecord]:
        """Parse one PGN game into per-ply records.

        Returns empty list if game is filtered out or
        contains unknown moves.

        Args:
            game: A parsed PGN game object.
            game_id: Sequential index of this game.

        Returns:
            List of RawTurnRecord, one per ply.
        """
        result = game.headers.get("Result", "*")

        # Winners-only filter
        if self._winners_only and result not in (
            "1-0", "0-1",
        ):
            return []

        # ELO filter
        if self._min_elo > 0:
            w_elo = self._parse_elo(
                game.headers.get("WhiteElo", "0")
            )
            b_elo = self._parse_elo(
                game.headers.get("BlackElo", "0")
            )
            if w_elo < self._min_elo and b_elo < self._min_elo:
                return []

        moves = list(game.mainline_moves())

        # Min moves filter
        if len(moves) < self._min_moves:
            return []

        # Raw outcome from White's perspective
        raw_outcome = _OUTCOME_MAP.get(result, 0)

        board = game.board()
        move_history: list[chess.Move] = []
        uci_history: list[str] = []
        records: list[RawTurnRecord] = []

        plies = min(len(moves), self._max_moves)
        for t in range(plies):
            move = moves[t]
            uci = move.uci()

            # Check move is in vocab
            try:
                self._move_tok.tokenize_move(uci)
            except KeyError:
                logger.warning(
                    "Unknown move %s in game %d, skipping",
                    uci, game_id,
                )
                return []

            # Tokenize board state BEFORE this move
            tb = self._board_tok.tokenize(
                board, board.turn
            )
            traj = make_trajectory_tokens(move_history)

            # Outcome from player-to-move perspective
            if board.turn == chess.WHITE:
                outcome = raw_outcome
            else:
                outcome = -raw_outcome

            # Decoder input: SOS + prior moves
            move_tokens = [SOS_IDX] + [
                self._move_tok.tokenize_move(u)
                for u in uci_history
            ]

            # Target: prior moves + current move
            target_tokens = [
                self._move_tok.tokenize_move(u)
                for u in uci_history
            ] + [self._move_tok.tokenize_move(uci)]

            records.append(RawTurnRecord(
                board_tokens=tb.board_tokens,
                color_tokens=tb.color_tokens,
                trajectory_tokens=traj,
                move_tokens=move_tokens,
                target_tokens=target_tokens,
                outcome=outcome,
                turn=t,
                game_id=game_id,
            ))

            uci_history.append(uci)
            move_history.append(move)
            board.push(move)

        return records

    @staticmethod
    def _parse_elo(elo_str: str) -> int:
        """Parse ELO string to int, defaulting to 0."""
        try:
            return int(elo_str)
        except (ValueError, TypeError):
            return 0
