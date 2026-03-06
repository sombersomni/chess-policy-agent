"""ChessViewer: root window wiring BoardPanel and StatsPanel."""
from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path

from scripts.gui.board_panel import BoardPanel
from scripts.gui.game_controller import GameController
from scripts.gui.stats_panel import StatsPanel


class ChessViewer:
    """Root window: 1000x600, title "Chess Encoder Viewer".

    Left column (400px): BoardPanel.
    Right column (600px): StatsPanel.

    Attributes:
        _root: tk.Tk root window.
        _controller: GameController providing game data.
        _board_panel: BoardPanel left-side widget.
        _stats_panel: StatsPanel right-side widget.
    """

    def __init__(
        self,
        pgn_path: Path,
        checkpoint_path: Path,
        game_index: int = 0,
    ) -> None:
        """Build window, load data, instantiate panels, wire callbacks.

        Args:
            pgn_path: Path to .pgn or .pgn.zst file.
            checkpoint_path: Path to .pt checkpoint.
            game_index: Zero-based game index within the PGN.
        """
        self._root = tk.Tk()
        self._root.title("Chess Encoder Viewer")
        self._root.geometry("1000x600")
        self._root.resizable(False, False)

        self._controller = GameController()
        self._controller.load(
            pgn_path, checkpoint_path, game_index
        )

        # Left column (400px)
        left = tk.Frame(self._root, width=400)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        self._board_panel = BoardPanel(
            left, self._controller
        )
        self._board_panel.pack(fill=tk.BOTH, expand=True)

        # Right column (600px)
        right = tk.Frame(self._root, width=600)
        right.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True
        )
        right.pack_propagate(False)
        self._stats_panel = StatsPanel(
            right, self._controller
        )
        self._stats_panel.pack(fill=tk.BOTH, expand=True)

        # Wire callback: board nav refreshes stats
        self._board_panel._on_ply_change = (
            lambda ply: self._stats_panel.render()
        )

        # Initial render
        self._board_panel.go_to_ply(0)

    def run(self) -> None:
        """Enter tk.mainloop(). Blocks until window is closed."""
        self._root.mainloop()


def main() -> None:
    """Parse CLI args and launch ChessViewer."""
    parser = argparse.ArgumentParser(
        description="Chess Encoder Viewer"
    )
    parser.add_argument(
        "--pgn",
        type=str,
        default="data/games.pgn",
        help="Path to PGN file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/winner_run_01.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--game-index",
        type=int,
        default=0,
        help="Zero-based game index",
    )
    args = parser.parse_args()
    viewer = ChessViewer(
        pgn_path=Path(args.pgn),
        checkpoint_path=Path(args.checkpoint),
        game_index=args.game_index,
    )
    viewer.run()


if __name__ == "__main__":
    main()
