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
        raise NotImplementedError("To be implemented")

    def run(self) -> None:
        """Enter tk.mainloop(). Blocks until window is closed."""
        raise NotImplementedError("To be implemented")


def main() -> None:
    """Parse CLI args and launch ChessViewer.

    CLI flags:
        --pgn: Path to PGN file (default: data/games.pgn)
        --checkpoint: Path to .pt checkpoint (default: checkpoints/winner_run_01.pt)
        --game-index: Zero-based game index (default: 0)

    Example:
        python -m scripts.gui.viewer \\
            --pgn data/games.pgn \\
            --checkpoint checkpoints/winner_run_01.pt \\
            --game-index 0
    """
    raise NotImplementedError("To be implemented")


if __name__ == "__main__":
    main()
