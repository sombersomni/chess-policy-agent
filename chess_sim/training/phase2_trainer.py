"""Phase2Trainer: REINFORCE self-play trainer for the chess model.

Uses policy gradient (REINFORCE) to fine-tune the ChessModel through
self-play games. The model plays against itself, and the game outcome
(win/loss/draw) provides the reward signal for policy updates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from chess_sim.config import ChessModelV2Config, Phase2Config
from chess_sim.model.chess_model import ChessModel
from chess_sim.training.trainer import log_metrics
from chess_sim.types import SelfPlayGame


class Phase2Trainer:
    """REINFORCE self-play trainer for the encoder-decoder chess model.

    Plays games against itself, computes REINFORCE policy gradients
    using game outcomes as rewards, and updates the model parameters.

    Owns:
      model:     ChessModel (encoder + decoder)
      optimizer: AdamW
      phase2_cfg: Phase2Config with reward values and hyperparams

    Example:
        >>> trainer = Phase2Trainer(device="cpu")
        >>> game = trainer.self_play_game(trainer.model)
        >>> loss = trainer.train_step(game)
    """

    def __init__(
        self,
        device: str = "cpu",
        v2_cfg: ChessModelV2Config | None = None,
    ) -> None:
        """Initialize model, optimizer, and phase2 config.

        Args:
            device: Torch device string. Use 'cpu' for tests.
            v2_cfg: Optional ChessModelV2Config. When None, uses defaults.

        Example:
            >>> trainer = Phase2Trainer(device="cpu")
        """
        raise NotImplementedError("To be implemented")

    def self_play_game(self, model: ChessModel) -> SelfPlayGame:
        """Play a complete game of chess via self-play.

        The model alternates as white and black, sampling moves from
        its own policy distribution with legal-move masking. The game
        ends by checkmate, stalemate, draw rules, or max-move limit.

        Args:
            model: ChessModel instance to use for move generation.

        Returns:
            SelfPlayGame with move list, board states, and outcome.

        Example:
            >>> game = trainer.self_play_game(model)
            >>> isinstance(game.outcome, float)
            True
        """
        raise NotImplementedError("To be implemented")

    def train_step(self, game: SelfPlayGame) -> float:
        """Execute one REINFORCE gradient update from a self-play game.

        Computes log-probs of the moves taken, multiplies by discounted
        rewards, and performs a gradient step.

        Args:
            game: SelfPlayGame record from self_play_game().

        Returns:
            Scalar policy loss as a Python float.

        Example:
            >>> loss = trainer.train_step(game)
            >>> isinstance(loss, float)
            True
        """
        raise NotImplementedError("To be implemented")

    def compute_rewards(self, game: SelfPlayGame) -> Tensor:
        """Compute per-move reward signal from the game outcome.

        Assigns the game outcome (+1/-1/0) to each move, potentially
        with discounting or shaping based on Phase2Config parameters.

        Args:
            game: SelfPlayGame record.

        Returns:
            FloatTensor [num_moves] of per-move rewards.

        Example:
            >>> rewards = trainer.compute_rewards(game)
            >>> rewards.shape[0] == len(game.moves)
            True
        """
        raise NotImplementedError("To be implemented")

    def compute_log_probs(self, game: SelfPlayGame) -> Tensor:
        """Compute log-probabilities of the moves actually taken.

        Feeds each board state and move history prefix through the model,
        then extracts the log-probability of the actual move taken.

        Args:
            game: SelfPlayGame record.

        Returns:
            FloatTensor [num_moves] of log-probabilities.

        Example:
            >>> log_probs = trainer.compute_log_probs(game)
            >>> log_probs.shape[0] == len(game.moves)
            True
        """
        raise NotImplementedError("To be implemented")

    def save_checkpoint(self, path: Path) -> None:
        """Save model state_dict to a checkpoint file.

        Args:
            path: Destination path for the .pt checkpoint file.

        Example:
            >>> trainer.save_checkpoint(Path("checkpoints/phase2_best.pt"))
        """
        raise NotImplementedError("To be implemented")

    def load_checkpoint(self, path: Path) -> None:
        """Load model state_dict from a checkpoint file.

        Args:
            path: Path to the .pt checkpoint file.

        Example:
            >>> trainer.load_checkpoint(Path("checkpoints/phase2_best.pt"))
        """
        raise NotImplementedError("To be implemented")
