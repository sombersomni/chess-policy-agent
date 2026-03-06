"""BoardPanel: left-side chess board canvas with ply navigation."""
from __future__ import annotations

import tkinter as tk
from typing import Callable

import chess

from scripts.evaluate import StepResult
from scripts.gui import GameSource


class BoardPanel(tk.Frame):
    """Left panel: 8x8 canvas board and ply navigation controls.

    Implements Renderable and Navigable.

    Layout (top-to-bottom):
        tk.Canvas (400x400 px) -- the chess board
        Navigation bar: [Prev] [tk.Scale] [Next] [ply label]

    Attributes:
        _controller: GameSource providing board and metrics per ply.
        _canvas: tk.Canvas for board rendering (400x400).
        _slider: tk.Scale for ply selection.
        _ply_label: tk.Label displaying current ply info.
        _current_ply: int tracking the displayed ply index.
        _on_ply_change: Callable[[int], None] wired by ChessViewer.
    """

    SQUARE_PX: int = 50
    LIGHT_SQ: str = "#F0D9B5"
    DARK_SQ: str = "#B58863"
    ACTUAL_SRC_BDR: str = "#FFD700"   # yellow border: actual move source
    ACTUAL_TGT_BDR: str = "#FF8C00"   # orange border: actual move target

    PIECE_SYMBOLS: dict[int, str] = {
        chess.PAWN:   "\u265F",
        chess.KNIGHT: "\u265E",
        chess.BISHOP: "\u265D",
        chess.ROOK:   "\u265C",
        chess.QUEEN:  "\u265B",
        chess.KING:   "\u265A",
    }

    def __init__(
        self,
        parent: tk.Widget,
        controller: GameSource,
    ) -> None:
        """Build canvas and navigation bar.

        Args:
            parent: Parent Tk widget.
            controller: GameSource providing board snapshots.
        """
        super().__init__(parent)
        self._controller = controller
        self._current_ply: int = 0
        self._on_ply_change: Callable[[int], None] = lambda _: None

        # Canvas
        self._canvas = tk.Canvas(
            self, width=self.SQUARE_PX * 8, height=self.SQUARE_PX * 8
        )
        self._canvas.pack()

        # Navigation bar
        nav = tk.Frame(self)
        nav.pack(fill=tk.X)
        tk.Button(nav, text="\u25C0 Prev", command=self.prev_ply).pack(side=tk.LEFT)
        self._slider = tk.Scale(
            nav, orient=tk.HORIZONTAL, from_=0, to=0,
            command=self._on_slider_change,
        )
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(nav, text="Next \u25B6", command=self.next_ply).pack(side=tk.LEFT)
        self._ply_label = tk.Label(nav, text="Ply 0 / 0")
        self._ply_label.pack(side=tk.LEFT)

    def render(self) -> None:
        """Clear canvas and redraw board, pieces, and move highlights."""
        self._canvas.delete("all")
        self._draw_board()
        board = self._controller.board_at(self._current_ply)
        self._draw_pieces(board)
        result = self._controller.step_result_at(
            self._current_ply
        )
        self._draw_predictions(result)
        total = self._controller.total_plies()
        self._ply_label.config(
            text=f"Ply {self._current_ply + 1} / {total}"
        )
        self._slider.config(to=total - 1)
        self._slider.set(self._current_ply)

    def go_to_ply(self, ply: int) -> None:
        """Jump to ply; clamp to valid range; render; fire _on_ply_change.

        Args:
            ply: Target ply index.
        """
        total = self._controller.total_plies()
        self._current_ply = max(0, min(ply, total - 1))
        self._controller.current_ply = self._current_ply
        self.render()
        self._on_ply_change(self._current_ply)

    def next_ply(self) -> None:
        """Advance one ply if not at the last ply."""
        self.go_to_ply(self._current_ply + 1)

    def prev_ply(self) -> None:
        """Step back one ply if not at ply 0."""
        self.go_to_ply(self._current_ply - 1)

    def _draw_board(self) -> None:
        """Draw 64 alternating-colour square rectangles onto _canvas."""
        for rank in range(8):
            for file in range(8):
                x0 = file * self.SQUARE_PX
                y0 = (7 - rank) * self.SQUARE_PX
                x1 = x0 + self.SQUARE_PX
                y1 = y0 + self.SQUARE_PX
                color = (
                    self.LIGHT_SQ
                    if (rank + file) % 2 == 0
                    else self.DARK_SQ
                )
                self._canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=color, outline=""
                )

    def _draw_pieces(self, board: chess.Board) -> None:
        """Overlay Unicode piece symbols at each occupied square.

        Args:
            board: Board snapshot for the current ply.
        """
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            cx = file * self.SQUARE_PX + self.SQUARE_PX // 2
            cy = (
                (7 - rank) * self.SQUARE_PX
                + self.SQUARE_PX // 2
            )
            symbol = self.PIECE_SYMBOLS[piece.piece_type]
            color = (
                "white"
                if piece.color == chess.WHITE
                else "black"
            )
            self._canvas.create_text(
                cx, cy, text=symbol,
                fill=color, font=("Arial", 28)
            )

    def _draw_predictions(self, result: StepResult) -> None:
        """Highlight actual move source/target squares from move_uci.

        Args:
            result: StepResult for the current ply.
        """
        uci = result.move_uci
        if not uci or len(uci) < 4:
            return
        src_sq = chess.parse_square(uci[:2])
        tgt_sq = chess.parse_square(uci[2:4])
        for sq, color in [
            (src_sq, self.ACTUAL_SRC_BDR),
            (tgt_sq, self.ACTUAL_TGT_BDR),
        ]:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            x0 = file * self.SQUARE_PX + 2
            y0 = (7 - rank) * self.SQUARE_PX + 2
            x1 = x0 + self.SQUARE_PX - 4
            y1 = y0 + self.SQUARE_PX - 4
            self._canvas.create_rectangle(
                x0, y0, x1, y1,
                outline=color, width=3
            )

    def _on_slider_change(self, value: str) -> None:
        """Handle tk.Scale callback; value is a string integer.

        Args:
            value: String representation of the selected ply index.
        """
        self.go_to_ply(int(value))
