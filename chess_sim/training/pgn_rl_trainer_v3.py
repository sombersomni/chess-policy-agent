"""PGNRLTrainerV3: RSCE (Reward-Scaled Cross-Entropy) trainer.

v3 variant replaces the fixed three-bucket outcome weights of v2
with a continuous exponential multiplier m(t) = exp(-(R(t) - r_ref))
derived from the composite per-ply reward. Higher-reward plies get
lower CE weight; lower-reward plies get amplified gradient pressure.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from chess_sim.training.pgn_rl_trainer_v2 import (
    PGNRLTrainerV2,
)

if TYPE_CHECKING:
    import chess.pgn


class PGNRLTrainerV3(PGNRLTrainerV2):
    """RSCE loss: reward-scaled CE via exp multiplier.

    Inherits all infrastructure (model, optimizer, scheduler,
    replayer, reward computer, structural mask) from v2. Overrides
    only the loss computation path: train_game builds continuous
    multipliers from rewards instead of discrete outcome buckets.

    Example:
        >>> trainer = PGNRLTrainerV3(cfg, device="cpu")
        >>> metrics = trainer.train_game(game)
        >>> assert "rsce_loss" in metrics
    """

    def train_game(
        self,
        game: chess.pgn.Game,
        game_idx: int = 0,
    ) -> dict[str, float]:
        """Train on one PGN game using RSCE loss.

        Replaces v2's fixed ply_weights with continuous
        multipliers from _build_multipliers, then delegates
        to _compute_rsce_loss for the weighted CE.

        Args:
            game: A parsed PGN game object.
            game_idx: Index of this game in the epoch.

        Returns:
            Dict with keys {total_loss, rsce_loss,
            n_plies, mean_reward, n_games}. Empty dict
            if game is skipped (draw/empty/too long).

        Example:
            >>> m = trainer.train_game(game, game_idx=0)
            >>> m["rsce_loss"]  # scalar float
        """
        raise NotImplementedError("To be implemented")

    def _build_multipliers(
        self, rewards: Tensor,
    ) -> Tensor:
        """Map per-ply rewards to exponential multipliers.

        Computes m(t) = exp(-(R(t) - r_ref)) for each ply.
        When rsbc_normalize_per_game is True, rescales so
        sum(m) == N (mean multiplier = 1.0).

        Args:
            rewards: Per-ply composite rewards, shape [N].

        Returns:
            Non-negative multiplier tensor, shape [N].
            All values > 0 for any finite input.

        Example:
            >>> r = torch.tensor([1.0, 0.0, -1.0])
            >>> m = trainer._build_multipliers(r)
            >>> assert (m > 0).all()
            >>> assert m[0] < m[1] < m[2]
        """
        raise NotImplementedError("To be implemented")

    def _compute_rsce_loss(
        self,
        all_logits: list[Tensor],
        all_targets: list[int],
        rewards: Tensor,
        all_color_tokens: list[Tensor] | None = None,
    ) -> Tensor:
        """Reward-scaled CE: mean(m_hat(t) * CE(logits_t, y_t)).

        Drop-in replacement for v2's _compute_rsbc_loss. The
        structural mask is applied to logits before CE (same as
        v2). Teacher targets from PGN replay are always legal,
        so the masked-target path is unreachable in practice.

        Args:
            all_logits: Per-ply decoder logits, each [vocab].
            all_targets: Per-ply teacher move indices (int).
            rewards: Composite per-ply rewards [N].
            all_color_tokens: Per-ply color tokens [65] each,
                or None when structural masking is disabled.

        Returns:
            Scalar RSCE loss tensor (always >= 0).

        Example:
            >>> loss = trainer._compute_rsce_loss(
            ...     logits, targets, rewards, color_tokens
            ... )
            >>> assert loss.dim() == 0 and loss >= 0
        """
        raise NotImplementedError("To be implemented")
