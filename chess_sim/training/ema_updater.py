"""EMAUpdater: exponential moving average sync from player to opponent.

Implements the Updatable protocol. After each episode, step() updates
the opponent weights:
    theta_opp <- alpha * theta_opp + (1 - alpha) * theta_player
The opponent model always has requires_grad=False on all parameters.
"""

from __future__ import annotations

import torch.nn as nn

from chess_sim.protocols import Updatable  # noqa: F401


class EMAUpdater:
    """EMA parameter sync from player to opponent model.

    Implements the Updatable protocol. After each episode, step()
    updates opponent weights via exponential moving average from
    the player model parameters.
    """

    def __init__(self, alpha: float) -> None:
        """Initialise with EMA decay rate.

        Args:
            alpha: Decay rate in (0, 1). Typical value 0.995.
        """
        self._alpha = alpha

    def step(
        self,
        player: nn.Module,
        opponent: nn.Module,
    ) -> None:
        """Update opponent parameters via EMA from player.

        For each parameter pair (p_player, p_opp):
            p_opp <- alpha * p_opp + (1 - alpha) * p_player
        Player parameters are not modified.

        Args:
            player: The learning agent model (gradients enabled).
            opponent: The EMA-frozen opponent (requires_grad=False).
        """
        raise NotImplementedError("To be implemented")
