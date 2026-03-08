"""RewardComputer: converts EpisodeRecord into per-player-ply rewards.

Implements the Computable protocol. Reward formula per player ply t:
    R(t) = temporal_advantage(t) + lambda_surprise * surprise(t)

Surprise formula (using softmax-normalized entropy H_norm):
    surprise(t) = H_norm(t) * correct(t) * reward_sign
    correct(t)  = +1 if argmax(probs_t) == vocab_idx(move_uci_t)
                  else -1
    reward_sign  = +1 win / -1 loss / 0 draw
"""

from __future__ import annotations

from torch import Tensor

from chess_sim.config import Phase2Config
from chess_sim.protocols import Computable  # noqa: F401
from chess_sim.types import EpisodeRecord


class RewardComputer:
    """Converts an EpisodeRecord into a per-player-ply reward tensor.

    Implements the Computable protocol. Filters plies to player-only,
    computes temporal advantage and surprise for each, and combines
    them according to the Phase2Config hyperparameters.
    """

    def compute(
        self,
        record: EpisodeRecord,
        cfg: Phase2Config,
    ) -> Tensor:
        """Return reward tensor of shape [T_player].

        Filters EpisodeRecord.plies to player plies only
        (is_player_ply=True), computes temporal advantage and
        surprise for each, and combines them.

        Args:
            record: Sealed episode with softmax-normalized player
                    entropies.
            cfg: Phase2Config with gamma and lambda_surprise.

        Returns:
            Tensor of shape [T_player] -- one scalar reward per
            player ply.
        """
        raise NotImplementedError("To be implemented")

    def _temporal_advantage(
        self,
        outcome: float,
        T: int,
        gamma: float,
    ) -> Tensor:
        """Compute discounted return for T player plies.

        Returns Tensor of shape [T] where index 0 is the earliest
        ply. R[t] = outcome * gamma^(T-1-t) so earlier plies have
        smaller discount.

        Args:
            outcome: Scalar game outcome (+1, -1, or draw_reward).
            T: Number of player plies.
            gamma: Temporal discount factor in (0, 1].

        Returns:
            Discounted return tensor of shape [T].
        """
        raise NotImplementedError("To be implemented")

    def _surprise(
        self,
        entropy: Tensor,
        correct: Tensor,
        reward_sign: float,
    ) -> Tensor:
        """Compute per-ply surprise scores.

        Formula: surprise(t) = entropy(t) * correct(t) * reward_sign
        entropy values must already be softmax-normalized (from
        EpisodeRecord). correct values are in {-1, +1}. When
        reward_sign=0 (draw), output is 0.

        Args:
            entropy: Normalized entropy tensor of shape [T].
            correct: Correctness tensor of shape [T] in {-1, +1}.
            reward_sign: +1.0 win / -1.0 loss / 0.0 draw.

        Returns:
            Surprise score tensor of shape [T].
        """
        raise NotImplementedError("To be implemented")
