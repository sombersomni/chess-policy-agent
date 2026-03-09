"""PGNRLRewardComputer: converts offline ply tuples into reward tensors.

Implements the OfflineComputable protocol. Combines temporal discount
(correct direction: last ply gets gamma^0=1.0), material delta, and
check bonus into a single per-ply reward tensor.
"""
from __future__ import annotations

import torch
from torch import Tensor

from chess_sim.config import RLConfig
from chess_sim.training.training_utils import l1_normalize
from chess_sim.types import OfflinePlyTuple


class PGNRLRewardComputer:
    """Converts list[OfflinePlyTuple] into per-ply reward tensor.

    Implements the OfflineComputable protocol.

    Temporal discount: gamma^(T-1-t) so the last ply (closest to
    outcome) gets gamma^0 = 1.0, and the first ply gets the
    largest discount.
    """

    def compute(
        self,
        plies: list[OfflinePlyTuple],
        cfg: RLConfig,
    ) -> Tensor:
        """Return reward tensor of shape [T], one value per ply.

        Args:
            plies: List of OfflinePlyTuple from PGNReplayer.
            cfg: RLConfig with gamma, reward values, lambdas.

        Returns:
            FloatTensor of shape [T] with combined rewards.
        """
        T = len(plies)
        if T == 0:
            return torch.zeros(0)

        temporal = torch.tensor(
            [
                (
                    cfg.win_reward
                    if p.is_winner_ply
                    else cfg.loss_reward
                )
                * (cfg.gamma ** (T - 1 - t))
                for t, p in enumerate(plies)
            ],
            dtype=torch.float32,
        )
        material = torch.tensor(
            [p.material_delta for p in plies],
            dtype=torch.float32,
        )
        check = torch.tensor(
            [p.gave_check for p in plies],
            dtype=torch.float32,
        )
        return (
            temporal
            + cfg.lambda_material * l1_normalize(material)
            + cfg.lambda_check * l1_normalize(check)
        )
