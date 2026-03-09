"""RewardComputer: converts EpisodeRecord into per-player-ply rewards.

Implements the Computable protocol. Each reward component is L1-normalized
across player plies before combining, so every component's absolute values
sum to 1.0 and contribute equally at baseline. Sign is preserved so wins
produce positive rewards and losses produce negative rewards. Lambda weights
provide explicit relative control over each component.

Reward formula per player ply t:
    R(t) = l1_norm(temporal)[t]
           + lambda_surprise  * l1_norm(surprise)[t]
           + lambda_material  * l1_norm(material)[t]
           + lambda_illegal   * l1_norm(illegal)[t]
           + lambda_check     * l1_norm(check)[t]

All-zero components are left as zeros (no spurious uniform signal).
"""

from __future__ import annotations

import torch
from torch import Tensor

from chess_sim.config import Phase2Config
from chess_sim.protocols import Computable  # noqa: F401
from chess_sim.types import EpisodeRecord


class RewardComputer:
    """Converts an EpisodeRecord into a per-player-ply reward tensor.

    Implements the Computable protocol. Filters plies to player-only,
    computes each reward component, softmax-normalizes each so it sums
    to 1.0, then combines them with lambda weights.
    """

    @staticmethod
    def _l1_normalize(t: Tensor) -> Tensor:
        """L1-normalize t across plies, preserving sign.

        Divides by the sum of absolute values so the normalized tensor's
        absolute values sum to 1.0. Returns zeros if t is all-zero to
        avoid injecting spurious signal.

        Args:
            t: Tensor of shape [T].

        Returns:
            t / |t|.sum() if non-zero, else zeros_like(t).
        """
        s = t.abs().sum()
        if s < 1e-8:
            return torch.zeros_like(t)
        return t / s

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
        from chess_sim.data.move_tokenizer import MoveTokenizer

        tok = MoveTokenizer()
        player_plies = [
            p for p in record.plies if p.is_player_ply
        ]
        T = len(player_plies)
        if T == 0:
            return torch.zeros(0)
        temporal = self._temporal_advantage(
            record.outcome, T, cfg.gamma
        )
        entropies = torch.tensor(
            [p.entropy for p in player_plies],
            dtype=torch.float32,
        )
        correct_list: list[float] = []
        for p in player_plies:
            try:
                vocab_idx = tok.tokenize_move(p.move_uci)
            except (KeyError, ValueError):
                correct_list.append(-1.0)
                continue
            predicted_idx = int(torch.argmax(p.probs).item())
            correct_list.append(
                1.0 if predicted_idx == vocab_idx else -1.0
            )
        correct = torch.tensor(
            correct_list, dtype=torch.float32
        )
        if record.outcome > 0.5:
            reward_sign = 1.0
        elif record.outcome < 0.0:
            reward_sign = -1.0
            # Flip: correct move on loss gets partial credit
            correct = -correct
        else:
            reward_sign = 0.0
        surprise = self._surprise(
            entropies, correct, reward_sign
        )
        material = torch.tensor(
            [p.material_delta for p in player_plies],
            dtype=torch.float32,
        )
        illegal = torch.tensor(
            [p.illegal_mass for p in player_plies],
            dtype=torch.float32,
        )
        check = torch.tensor(
            [p.gave_check for p in player_plies],
            dtype=torch.float32,
        )
        norm = self._l1_normalize
        return (
            norm(temporal)
            + cfg.lambda_surprise * norm(surprise)
            + cfg.lambda_material * norm(material)
            + cfg.lambda_illegal * norm(illegal)
            + cfg.lambda_check * norm(check)
        )

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
        return torch.tensor(
            [outcome * (gamma ** t) for t in range(T)],
            dtype=torch.float32,
        )

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
        return entropy * correct * reward_sign
