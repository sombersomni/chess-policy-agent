"""SelfPlayLoop: orchestrates Phase 2 self-play RL training.

Implements the Trainable protocol where train_step corresponds to one
complete self-play episode. Drives ChessSimEnv, coordinates
EpisodeRecorder, calls RewardComputer, trains ValueHeads, computes
REINFORCE policy loss, and triggers EMAUpdater after each episode.

Checkpoint saves both player and EMA opponent state dicts for exact
resume.
"""

from __future__ import annotations

from pathlib import Path

from chess_sim.config import Phase2Config
from chess_sim.model.chess_model import ChessModel
from chess_sim.protocols import Trainable  # noqa: F401


class SelfPlayLoop:
    """Orchestrates the full Phase 2 self-play RL loop.

    Implements the Trainable protocol where train_step corresponds
    to one complete self-play episode. Drives ChessSimEnv, coordinates
    EpisodeRecorder, calls RewardComputer, trains ValueHeads, computes
    REINFORCE policy loss, and triggers EMAUpdater after each episode.

    Checkpoint saves both player and EMA opponent state dicts for
    exact resume.
    """

    def __init__(
        self,
        player: ChessModel,
        cfg: Phase2Config,
        device: str = "cpu",
    ) -> None:
        """Initialise the self-play loop.

        Args:
            player: The learning ChessModel (weights updated).
            cfg: Phase2Config with all RL hyperparameters.
            device: Torch device string, e.g. "cpu" or "cuda".
        """
        raise NotImplementedError("To be implemented")

    def run(self, episodes: int) -> None:
        """Run self-play training for the given number of episodes.

        Each episode: play full game -> record plies -> compute
        rewards -> train value heads -> compute REINFORCE loss ->
        step optimizer -> update EMA opponent weights.

        Args:
            episodes: Total number of self-play games to run.
        """
        raise NotImplementedError("To be implemented")

    def save_checkpoint(self, path: Path) -> None:
        """Save player and EMA opponent state dicts to path.

        Args:
            path: Destination .pt file path.
        """
        raise NotImplementedError("To be implemented")

    def load_checkpoint(self, path: Path) -> None:
        """Load player and EMA opponent state dicts from path.

        Args:
            path: Source .pt file path.
        """
        raise NotImplementedError("To be implemented")
