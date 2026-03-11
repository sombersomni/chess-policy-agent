"""PGNRLTrainerV3: RSCE (Reward-Scaled Cross-Entropy) trainer.

v3 variant replaces the fixed three-bucket outcome weights of v2
with a continuous exponential multiplier m(t) = exp(-(R(t) - r_ref))
derived from the composite per-ply reward. Higher-reward plies get
lower CE weight; lower-reward plies get amplified gradient pressure.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chess_sim.training.pgn_rl_trainer_v2 import (
    PGNRLTrainerV2,
)

if TYPE_CHECKING:
    import chess.pgn

logger = logging.getLogger(__name__)


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
        if (
            self._cfg.rl.skip_draws
            and game.headers.get("Result") == "1/2-1/2"
        ):
            return {}
        plies = self._replayer.replay(game)
        train_white = self._cfg.rl.train_color == "white"
        plies = [
            p for p in plies if p.is_white_ply == train_white
        ]
        if not plies:
            return {}

        # VRAM guard: skip unusually long games
        if len(plies) > self._cfg.rl.max_plies_per_game:
            return {}

        board_snaps = self._build_board_snapshots(
            game, train_white
        )

        rewards = self._reward_fn.compute(
            plies, self._cfg.rl
        )
        self._model.train()

        # Single batched forward pass for all plies
        last_logits_all, cls_all, move_idxs = (
            self._encode_and_decode_batch(
                plies, [p.move_uci for p in plies]
            )
        )

        # Bulk increment ply counter before snapshot logging
        self._ply_step += len(plies)

        # Board snapshot logging with offset-based cadence.
        saved_ply_step = self._ply_step
        base_step = self._ply_step - len(plies)
        for j, snap in enumerate(board_snaps):
            self._ply_step = base_step + j + 1
            self._log_board_snapshot(
                snap,
                plies[j].move_uci,
                game_idx,
                is_winner_ply=plies[j].is_winner_ply,
                reward=float(rewards[j]),
                last_logits=last_logits_all[j],
                move_idx=move_idxs[j],
            )
        self._ply_step = saved_ply_step

        # Filter to valid (non-OOV) plies
        valid_mask: list[int] = [
            i
            for i, idx in enumerate(move_idxs)
            if idx is not None
        ]

        if not valid_mask:
            return {}

        valid_rewards = rewards[valid_mask].to(
            self._device
        )

        # Collect logits/targets for valid plies only
        all_logits: list[Tensor] = [
            last_logits_all[i] for i in valid_mask
        ]
        all_targets: list[int] = [
            move_idxs[i]  # type: ignore[misc]
            for i in valid_mask
        ]
        all_color_tokens: list[Tensor] = [
            plies[i].color_tokens.to(self._device)
            for i in valid_mask
        ]

        rsce_loss = self._compute_rsce_loss(
            all_logits=all_logits,
            all_targets=all_targets,
            rewards=valid_rewards,
            all_color_tokens=(
                all_color_tokens
                if self._struct_mask is not None
                else None
            ),
        )

        total_loss = self._cfg.rl.lambda_rsbc * rsce_loss

        self._opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self._model.parameters(),
            self._cfg.rl.gradient_clip,
        )
        self._opt.step()
        self._sched.step()
        self._global_step += 1

        mean_reward = float(valid_rewards.cpu().mean())

        return {
            "total_loss": total_loss.item(),
            "rsce_loss": rsce_loss.item(),
            "n_plies": len(plies),
            "mean_reward": mean_reward,
            "n_games": 1,
        }

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
        m = torch.exp(
            -(rewards - self._cfg.rl.rsce_r_ref)
        )
        if self._cfg.rl.rsbc_normalize_per_game:
            n = m.size(0)
            m = m * n / m.sum().clamp(min=1e-8)
        return m

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
        multipliers = self._build_multipliers(rewards)

        targets_t = torch.tensor(
            all_targets,
            dtype=torch.long,
            device=self._device,
        )
        logits_t = torch.stack(all_logits)  # [N, V]

        # Apply structural mask (same block as v2)
        if (
            self._struct_mask is not None
            and all_color_tokens is not None
        ):
            ct_stacked = torch.stack(all_color_tokens)
            smask = self._struct_mask.build(ct_stacked)
            logits_t = logits_t.masked_fill(~smask, -1e9)

        per_ply_ce: Tensor = F.cross_entropy(
            logits_t,
            targets_t,
            label_smoothing=self._cfg.rl.label_smoothing,
            reduction="none",
        )
        return (multipliers * per_ply_ce).mean()
