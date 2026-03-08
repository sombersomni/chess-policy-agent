"""TerminalRenderer: renders a chess board and RenderContext to stdout.

Implements the TerminalRenderable protocol from chess_sim.env.

Features
--------
- Unicode piece symbols (♔♕♖♗♘♙ / ♚♛♜♝♞♟) with ASCII fallback.
- ANSI escape codes highlight the last-move source (yellow) and target (cyan).
- Sidebar: last N moves, game phase, material balance.
- Prediction panel: model top-N predictions with probabilities and hit marker.
"""

from __future__ import annotations

import sys

import chess

from chess_sim.env import MovePrediction, RenderContext

# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_ANSI_YELLOW_BG = "\033[43m"   # last-move source square
_ANSI_CYAN_BG = "\033[46m"     # last-move target square
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"

# ---------------------------------------------------------------------------
# Piece symbol tables
# ---------------------------------------------------------------------------

_UNICODE_PIECES: dict[tuple[chess.PieceType, chess.Color], str] = {
    (chess.KING,   chess.WHITE): "♔",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KING,   chess.BLACK): "♚",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN,   chess.BLACK): "♟",
}

_ASCII_PIECES: dict[tuple[chess.PieceType, chess.Color], str] = {
    (chess.KING,   chess.WHITE): "K",
    (chess.QUEEN,  chess.WHITE): "Q",
    (chess.ROOK,   chess.WHITE): "R",
    (chess.BISHOP, chess.WHITE): "B",
    (chess.KNIGHT, chess.WHITE): "N",
    (chess.PAWN,   chess.WHITE): "P",
    (chess.KING,   chess.BLACK): "k",
    (chess.QUEEN,  chess.BLACK): "q",
    (chess.ROOK,   chess.BLACK): "r",
    (chess.BISHOP, chess.BLACK): "b",
    (chess.KNIGHT, chess.BLACK): "n",
    (chess.PAWN,   chess.BLACK): "p",
}

_FILE_LABELS = "  a b c d e f g h"


def _parse_last_move(
    last_move_uci: str,
) -> tuple[int | None, int | None]:
    """Parse the source and target squares from a UCI move string.

    Args:
        last_move_uci: UCI string like "e2e4"; empty string means no move.

    Returns:
        Tuple of (src_sq, tgt_sq) as python-chess square indices [0, 63],
        or (None, None) if the string is empty or malformed.
    """
    if len(last_move_uci) < 4:
        return None, None
    try:
        src = chess.parse_square(last_move_uci[:2])
        tgt = chess.parse_square(last_move_uci[2:4])
        return src, tgt
    except ValueError:
        return None, None


def _piece_symbol(
    piece: chess.Piece | None,
    use_unicode: bool,
) -> str:
    """Return the display character for a piece (or a dot for empty).

    Args:
        piece: chess.Piece or None for an empty square.
        use_unicode: Use Unicode symbols when True, ASCII letters when False.

    Returns:
        Single-character string representing the piece.
    """
    if piece is None:
        return "."
    table = _UNICODE_PIECES if use_unicode else _ASCII_PIECES
    return table[(piece.piece_type, piece.color)]


# ---------------------------------------------------------------------------
# TerminalRenderer
# ---------------------------------------------------------------------------


class TerminalRenderer:
    """Renders a chess board and RenderContext to stdout as ANSI terminal art.

    Implements the TerminalRenderable protocol.

    Attributes:
        _use_unicode: Emit Unicode piece symbols when True.
        _highlight_last: Highlight last-move squares with ANSI colours.
        _sidebar_moves: Number of recent moves shown in the history sidebar.
    """

    def __init__(
        self,
        use_unicode: bool = True,
        highlight_last: bool = True,
        sidebar_moves: int = 6,
    ) -> None:
        """Initialise the renderer.

        Args:
            use_unicode: Use Unicode piece symbols (default True).
            highlight_last: Highlight source/target of last move (default True).
            sidebar_moves: Number of recent moves to show in the sidebar.
        """
        self._use_unicode = use_unicode
        self._highlight_last = highlight_last
        self._sidebar_moves = sidebar_moves

    # ------------------------------------------------------------------
    # Public API (TerminalRenderable protocol)
    # ------------------------------------------------------------------

    def render(
        self,
        board: chess.Board,
        ctx: RenderContext,
    ) -> None:
        """Write a full terminal frame to stdout.

        Clears the previous frame with a carriage-return block, then prints
        the board rows side-by-side with the sidebar and prediction panel.

        Args:
            board: Current board state to display.
            ctx: Contextual game data for the sidebar and prediction panel.
        """
        src_sq, tgt_sq = _parse_last_move(ctx.last_move_uci)
        board_lines = self._board_rows(board, src_sq, tgt_sq)
        sidebar_lines = self._sidebar_lines(ctx)
        pred_lines = self._prediction_panel(ctx.predictions, ctx.last_move_uci)

        # Pad shorter column to match board height.
        height = max(len(board_lines), len(sidebar_lines) + len(pred_lines))
        board_lines += [""] * (height - len(board_lines))

        right_lines = sidebar_lines + pred_lines
        right_lines += [""] * (height - len(right_lines))

        frame_lines = [
            f"{bl:<22}  {rl}"
            for bl, rl in zip(board_lines, right_lines)
        ]

        output = "\n" + "\n".join(frame_lines) + "\n"
        sys.stdout.write(output)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _board_rows(
        self,
        board: chess.Board,
        src_sq: int | None,
        tgt_sq: int | None,
    ) -> list[str]:
        """Build the 8 board rows plus the file-label footer.

        Ranks are displayed from 8 (top) to 1 (bottom). Highlight colours
        are applied to src and tgt squares when _highlight_last is True.

        Args:
            board: Current board state.
            src_sq: Source square index [0, 63] to highlight, or None.
            tgt_sq: Target square index [0, 63] to highlight, or None.

        Returns:
            List of strings, one per rank plus the file label row.
        """
        rows: list[str] = []
        for rank in range(7, -1, -1):
            row_parts = [f"{rank + 1} "]
            for file in range(8):
                sq = chess.square(file, rank)
                piece = board.piece_at(sq)
                symbol = _piece_symbol(piece, self._use_unicode)

                if self._highlight_last and sq == src_sq:
                    cell = f"{_ANSI_YELLOW_BG}{symbol}{_ANSI_RESET}"
                elif self._highlight_last and sq == tgt_sq:
                    cell = f"{_ANSI_CYAN_BG}{symbol}{_ANSI_RESET}"
                else:
                    cell = symbol
                row_parts.append(f"{cell} ")
            rows.append("".join(row_parts))
        rows.append(_FILE_LABELS)
        return rows

    def _sidebar_lines(self, ctx: RenderContext) -> list[str]:
        """Build game-info lines for the right-hand sidebar.

        Shows ply, last move, game phase, material balance, and recent history.

        Args:
            ctx: RenderContext snapshot.

        Returns:
            List of formatted strings.
        """
        lines: list[str] = []
        sep = "-"
        move_label = ctx.last_move_uci if ctx.last_move_uci else sep
        lines.append(
            f"{_ANSI_BOLD}Ply {ctx.ply}  {sep}  {move_label}{_ANSI_RESET}"
        )
        lines.append(f"Phase: {ctx.game_phase}")
        bal = ctx.material_balance
        bal_str = f"+{bal}" if bal > 0 else str(bal)
        lines.append(f"Material: {bal_str}")
        lines.append("")
        lines.append(f"{_ANSI_DIM}Move history:{_ANSI_RESET}")
        history = ctx.move_history[-self._sidebar_moves:]
        for i, uci in enumerate(history, start=1):
            lines.append(f"  {i}. {uci}")
        return lines

    def _prediction_panel(
        self,
        predictions: list[MovePrediction],
        actual_move: str,
    ) -> list[str]:
        """Build the agent prediction panel lines.

        Shows top-N predictions with probabilities.  Marks the prediction
        that matches the actual move with a green checkmark.

        Args:
            predictions: Ordered list of MovePrediction (rank 1 first).
            actual_move: The UCI move that was actually played.

        Returns:
            List of formatted strings, empty when predictions is empty.
        """
        if not predictions:
            return []

        lines: list[str] = ["", f"{_ANSI_BOLD}-- Agent Predictions --{_ANSI_RESET}"]
        for pred in predictions:
            pct = f"{pred.probability * 100:.1f}%"
            if pred.move_uci == actual_move:
                marker = f"  {_ANSI_GREEN}✓ correct{_ANSI_RESET}"
            else:
                marker = ""
            lines.append(
                f"  {pred.rank}. {pred.move_uci:<8} {pct:>6}{marker}"
            )
        return lines
