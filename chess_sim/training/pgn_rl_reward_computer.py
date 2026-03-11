"""PGNRLRewardComputer: composite reward from outcome sign + material delta.

R(t) = lambda_outcome * sign_outcome(t) + lambda_material * material_delta(t)
where sign_outcome = +1.0 (winner), -1.0 (loser), draw_reward_norm (draw).
"""
from __future__ import annotations

from torch import Tensor

from chess_sim.config import RLConfig
from chess_sim.types import OfflinePlyTuple


class PGNRLRewardComputer:
    """Converts list[OfflinePlyTuple] into per-ply reward tensor.

    Implements the OfflineComputable protocol.

    R(t) = lambda_outcome * sign_outcome(t)
         + lambda_material * material_delta(t)

    where sign_outcome is +1.0 (winner), -1.0 (loser),
    or draw_reward_norm (draw).
    """

    def compute(
        self,
        plies: list[OfflinePlyTuple],
        cfg: RLConfig,
    ) -> Tensor:
        """Return reward tensor of shape [T], one value per ply.

        Composite formula:
        R(t) = lambda_outcome * sign_outcome(t)
             + lambda_material * material_delta(t)
        where sign_outcome = +1.0 (winner), -1.0 (loser),
        draw_reward_norm (draw).

        Args:
            plies: List of OfflinePlyTuple from PGNReplayer.
            cfg: RLConfig with lambda_outcome, lambda_material,
                and draw_reward_norm.

        Returns:
            FloatTensor of shape [T] with composite rewards.

        Example:
            >>> r = PGNRLRewardComputer().compute(plies, cfg)
            >>> r.shape
            torch.Size([10])
        """
        raise NotImplementedError("To be implemented")
