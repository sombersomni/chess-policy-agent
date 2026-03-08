"""Entry-point for Phase 2 self-play RL training.

Usage:
    python -m scripts.train_phase2 --config configs/phase2.yaml
"""

from __future__ import annotations

import argparse  # noqa: F401
from pathlib import Path  # noqa: F401

from chess_sim.config import load_v2_config  # noqa: F401
from chess_sim.model.chess_model import ChessModel  # noqa: F401
from chess_sim.training.self_play_loop import SelfPlayLoop  # noqa: F401


def main() -> None:
    """Parse args, load config, and run the self-play training loop."""
    raise NotImplementedError("To be implemented")


if __name__ == "__main__":
    main()
