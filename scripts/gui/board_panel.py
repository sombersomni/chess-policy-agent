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
        raise NotImplementedError("To be implemented")

    def go_to_ply(self, ply: int) -> None:
        """Jump to ply; clamp to valid range; render; fire _on_ply_change.

        Args:
            ply: Target ply index.
        """
        raise NotImplementedError("To be implemented")

    def next_ply(self) -> None:
        """Advance one ply if not at the last ply."""
        raise NotImplementedError("To be implemented")

    def prev_ply(self) -> None:
        """Step back one ply if not at ply 0."""
        raise NotImplementedError("To be implemented")

    def _draw_board(self) -> None:
        """Draw 64 alternating-colour square rectangles onto _canvas."""
        raise NotImplementedError("To be implemented")

    def _draw_pieces(self, board: chess.Board) -> None:
        """Overlay Unicode piece symbols at each occupied square.

        Args:
            board: Board snapshot for the current ply.

        Note:
            White pieces use the filled Unicode symbols.
            Black pieces use the hollow Unicode symbols.
            Both are displayed as centred text within each square.
        """
        raise NotImplementedError("To be implemented")

    def _draw_predictions(self, result: StepResult) -> None:
        """Highlight actual move source/target squares from move_uci.

        Args:
            result: StepResult for the current ply.

        Note:
            StepResult does not carry logits, so predicted-square
            highlights are limited to actual move squares derived from
            result.move_uci. Coloured borders are drawn:
            ACTUAL_SRC_BDR (yellow) on source, ACTUAL_TGT_BDR (orange)
            on target.
        """
        raise NotImplementedError("To be implemented")

    def _on_slider_change(self, value: str) -> None:
        """Handle tk.Scale callback; value is a string integer.

        Args:
            value: String representation of the selected ply index.
        """
        raise NotImplementedError("To be implemented")
