"""RL self-play fine-tune trainer with EMA shadow opponent.

Three model copies: live policy (trained), frozen CE reference
(KL anchor), and EMA shadow (lagging opponent). REINFORCE with
running-mean baseline on terminal-only rewards.

Example:
    >>> trainer = RLFinetuneTrainer(cfg, device="cpu")
    >>> metrics = trainer.train(n_updates=10)
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from chess_sim.config import FinetuneConfig, FinetuneRLConfig
from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.model.chess_model import ChessModel
from chess_sim.tracking.protocol import MetricTracker
from chess_sim.types import GameRecord


class RunningMeanBaseline:
    """Online running mean of discounted returns.

    Uses Welford-style accumulation with Python floats only.
    Provides a variance-reduction baseline for REINFORCE.

    Example:
        >>> b = RunningMeanBaseline()
        >>> b.update(torch.tensor([1.0, -1.0]))
        >>> abs(b.value()) < 0.1
        True
    """

    def __init__(self) -> None:
        """Initialize with zero mean and zero count."""
        self._mean: float = 0.0
        self._count: int = 0

    def update(self, returns: Tensor) -> None:
        """Incorporate a batch of returns into the running mean.

        Uses Welford's online algorithm for numerical stability.
        Each element of returns is treated as an independent sample.

        Args:
            returns: 1-D tensor of discounted return values.

        Example:
            >>> b = RunningMeanBaseline()
            >>> b.update(torch.tensor([2.0, 4.0]))
            >>> b.value()
            3.0
        """
        for val in returns.tolist():
            self._count += 1
            delta = val - self._mean
            self._mean += delta / self._count

    def value(self) -> float:
        """Return the current running mean.

        Returns 0.0 if no updates have been made yet.

        Returns:
            Current mean of all observed returns.

        Example:
            >>> RunningMeanBaseline().value()
            0.0
        """
        return self._mean


def play_game(
    policy: ChessModel,
    shadow: ChessModel,
    board_tok: BoardTokenizer,
    move_tok: MoveTokenizer,
    cfg: FinetuneConfig,
    device: torch.device,
) -> GameRecord:
    """Play one self-play game between policy (White) and shadow (Black).

    Policy moves are sampled at T=cfg.t_policy with log_prob retained
    for gradient computation. Shadow moves use T=cfg.t_opponent with
    no_grad. Game terminates on checkmate, stalemate, 50-move rule,
    threefold repetition, or max_ply.

    Args:
        policy: Live policy model (White).
        shadow: EMA shadow model (Black), eval mode.
        board_tok: Board tokenizer for encoding positions.
        move_tok: Move tokenizer for UCI-to-vocab conversion.
        cfg: Fine-tune hyperparameters.
        device: Torch device for tensors.

    Returns:
        GameRecord with trajectory of policy plies and outcome.

    Example:
        >>> record = play_game(pol, shd, bt, mt, cfg, dev)
        >>> record.termination in ("checkmate", "maxply")
        True
    """
    raise NotImplementedError("To be implemented")


def _log_prob_of_move(
    policy: ChessModel,
    board_tokens: Tensor,
    color_tokens: Tensor,
    traj_tokens: Tensor,
    move_token: int,
    legal_moves: list[str],
    move_tok: MoveTokenizer,
    device: torch.device,
) -> Tensor:
    """Compute log pi(a|s) with gradients for REINFORCE.

    Runs policy.forward in training mode (no torch.no_grad),
    applies legal-move mask, temperature, log_softmax, and
    indexes at the given move_token.

    Args:
        policy: Live policy model in training mode.
        board_tokens: [1, 65] long tensor of piece types.
        color_tokens: [1, 65] long tensor of player/opponent.
        traj_tokens: [1, 65] long tensor of trajectory roles.
        move_token: Vocab index of the sampled move.
        legal_moves: List of legal UCI move strings.
        move_tok: Move tokenizer for vocab index lookup.
        device: Torch device for tensors.

    Returns:
        Scalar tensor with requires_grad=True.

    Example:
        >>> lp = _log_prob_of_move(pol, bt, ct, tt, 42, [...], mt, dev)
        >>> lp.requires_grad
        True
    """
    raise NotImplementedError("To be implemented")


def compute_returns(
    records: list[GameRecord],
    gamma: float,
) -> list[Tensor]:
    """Compute per-ply discounted returns for each game.

    Reward is terminal-only: outcome at last ply, 0 elsewhere.
    Returns G_t = gamma^(T-1-t) * outcome for each live ply.

    Args:
        records: List of completed GameRecord objects.
        gamma: Discount factor in (0, 1].

    Returns:
        Flat list of scalar tensors, one per PlyRecord across
        all games, ordered game-by-game then ply-by-ply.

    Example:
        >>> rets = compute_returns([game], gamma=0.99)
        >>> len(rets) == len(game.plies)
        True
    """
    raise NotImplementedError("To be implemented")


class RLFinetuneTrainer:
    """Self-play fine-tune trainer with EMA shadow + KL anchor.

    Manages three model copies:
    - _policy: live parameters, trained via AdamW
    - _ref: frozen deep-copy of CE checkpoint (KL anchor)
    - _shadow: EMA-updated opponent in eval mode

    The training loop collects n_games_per_update trajectories
    via play_game(), computes REINFORCE + KL loss, and takes
    one optimizer step per update.

    Example:
        >>> trainer = RLFinetuneTrainer(cfg, device="cpu")
        >>> metrics = trainer.train(n_updates=5)
        >>> "pg_loss" in metrics
        True
    """

    def __init__(
        self,
        cfg: FinetuneRLConfig,
        device: str = "cpu",
        tracker: MetricTracker | None = None,
    ) -> None:
        """Initialize trainer with three model copies and optimizer.

        Loads CE checkpoint into _policy, deep-copies to _ref
        (frozen, requires_grad=False) and _shadow (eval mode,
        requires_grad=False). Creates AdamW optimizer on _policy.

        Args:
            cfg: Full fine-tune RL config with model/decoder/finetune.
            device: "cpu" or "cuda" device string.
            tracker: Optional metric tracker; defaults to NoOpTracker.

        Example:
            >>> trainer = RLFinetuneTrainer(cfg, device="cpu")
        """
        raise NotImplementedError("To be implemented")

    @property
    def policy(self) -> ChessModel:
        """Return the live policy model.

        Returns:
            The trainable ChessModel instance.

        Example:
            >>> model = trainer.policy
            >>> model.training
            True
        """
        raise NotImplementedError("To be implemented")

    def train(
        self, n_updates: int
    ) -> dict[str, float]:
        """Run the full self-play fine-tuning loop.

        For each update step: plays n_games_per_update games,
        computes REINFORCE + KL loss, takes one optimizer step,
        and conditionally updates the shadow via EMA.

        Args:
            n_updates: Number of gradient update steps.

        Returns:
            Dict with final metrics: pg_loss, kl_loss,
            total_loss, win_rate, mean_ply, baseline_value.

        Example:
            >>> metrics = trainer.train(n_updates=10)
            >>> metrics["win_rate"]
            0.5
        """
        raise NotImplementedError("To be implemented")

    def save_checkpoint(self, path: Path) -> None:
        """Save trainer state to disk.

        Stores model state_dict, optimizer state_dict,
        baseline state (_mean, _count), global_step, and
        game_count. Uses torch.save.

        Args:
            path: Destination file path for the checkpoint.

        Example:
            >>> trainer.save_checkpoint(Path("ckpt.pt"))
        """
        raise NotImplementedError("To be implemented")

    def load_checkpoint(self, path: Path) -> None:
        """Restore trainer state from a checkpoint.

        Loads model weights, optimizer state, baseline state,
        global_step, and game_count. Uses torch.load with
        weights_only=True for security.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If path does not exist.

        Example:
            >>> trainer.load_checkpoint(Path("ckpt.pt"))
        """
        raise NotImplementedError("To be implemented")

    def _play_games(self) -> list[GameRecord]:
        """Play n_games_per_update self-play games.

        Delegates to play_game() for each game. Policy plays
        White, shadow plays Black.

        Returns:
            List of GameRecord, one per completed game.

        Example:
            >>> records = trainer._play_games()
            >>> len(records) == cfg.finetune.n_games_per_update
            True
        """
        raise NotImplementedError("To be implemented")

    def _gradient_step(
        self, records: list[GameRecord]
    ) -> dict[str, float]:
        """Compute REINFORCE + KL loss and take one optimizer step.

        Computes discounted returns, subtracts baseline, collects
        log-probs from PlyRecords, and computes KL divergence
        against the frozen reference model.

        Args:
            records: List of GameRecord from _play_games().

        Returns:
            Dict with pg_loss, kl_loss, total_loss, grad_norm.

        Example:
            >>> metrics = trainer._gradient_step(records)
            >>> "total_loss" in metrics
            True
        """
        raise NotImplementedError("To be implemented")

    def _update_shadow(self) -> None:
        """Apply EMA update to shadow model weights.

        theta_shadow = alpha * theta_shadow + (1-alpha) * theta_policy
        Runs under torch.no_grad().

        Example:
            >>> trainer._update_shadow()
        """
        raise NotImplementedError("To be implemented")
