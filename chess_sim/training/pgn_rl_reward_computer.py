"""PGNRLRewardComputer: converts offline ply tuples into reward tensors.

Implements the OfflineComputable protocol. Pure temporal discount:
last ply gets gamma^0=1.0, first ply gets the largest discount.
Outcome: win_reward / draw_reward / loss_reward per ply type.
"""
from __future__ import annotations

import torch
from torch import Tensor

from chess_sim.config import RLConfig
from chess_sim.types import OfflinePlyTuple


class PGNRLRewardComputer:
    """Converts list[OfflinePlyTuple] into per-ply reward tensor.

    Implements the OfflineComputable protocol.

    R(t) = base_outcome * gamma^(T-1-t)
    where base_outcome is draw_reward, win_reward, or loss_reward.
    """

    def compute(
        self,
        plies: list[OfflinePlyTuple],
        cfg: RLConfig,
    ) -> Tensor:
        """Return reward tensor of shape [T], one value per ply.

        Args:
            plies: List of OfflinePlyTuple from PGNReplayer.
            cfg: RLConfig with gamma and reward values.

        Returns:
            FloatTensor of shape [T] with temporally discounted rewards.
        """
        T = len(plies)
        if T == 0:
            return torch.zeros(0)

        return torch.tensor(
            [
                (
                    cfg.draw_reward
                    if p.is_draw_ply
                    else cfg.win_reward
                    if p.is_winner_ply
                    else cfg.loss_reward
                )
                * (cfg.gamma ** (T - 1 - t))
                for t, p in enumerate(plies)
            ],
            dtype=torch.float32,
        )
