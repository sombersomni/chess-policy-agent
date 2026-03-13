"""RL self-play fine-tune trainer with EMA shadow opponent.

Three model copies: live policy (trained), frozen CE reference
(KL anchor), and EMA shadow (lagging opponent). REINFORCE with
running-mean baseline on terminal-only rewards.

Example:
    >>> trainer = RLFinetuneTrainer(cfg, device="cpu")
    >>> metrics = trainer.train(n_updates=10)
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chess_sim.config import FinetuneConfig, FinetuneRLConfig
from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.move_vocab import SOS_IDX
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.env.agent_adapter import _compute_trajectory_tokens
from chess_sim.model.chess_model import ChessModel
from chess_sim.tracking.noop_tracker import NoOpTracker
from chess_sim.tracking.protocol import MetricTracker
from chess_sim.types import GameRecord, PlyRecord

logger = logging.getLogger(__name__)


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
    prefix = torch.tensor(
        [[SOS_IDX]], dtype=torch.long, device=device
    )
    move_colors = torch.zeros(
        1, 1, dtype=torch.long, device=device
    )
    logits = policy(
        board_tokens, color_tokens, traj_tokens,
        prefix, None, move_colors,
    )
    last = logits[0, -1, :]  # [V]

    legal_mask = move_tok.build_legal_mask(
        legal_moves
    ).to(device)
    last = last.masked_fill(~legal_mask, -1e9)

    log_probs = torch.log_softmax(last, dim=-1)
    return log_probs[move_token]


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
    flat: list[Tensor] = []
    for game in records:
        n = len(game.plies)
        for t in range(n):
            g_t = (gamma ** (n - 1 - t)) * game.outcome
            flat.append(torch.tensor(g_t))
    return flat


def play_game(
    policy: ChessModel,
    shadow: ChessModel,
    board_tok: BoardTokenizer,
    move_tok: MoveTokenizer,
    cfg: FinetuneConfig,
    device: torch.device,
) -> GameRecord:
    """Play one self-play game: policy (White) vs shadow (Black).

    Policy moves are sampled at T=cfg.t_policy with log_prob
    retained for gradient computation. Shadow moves use
    T=cfg.t_opponent with no_grad. Game terminates on checkmate,
    stalemate, 50-move rule, threefold repetition, or max_ply.

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
    board = chess.Board()
    move_history: list[str] = []
    plies: list[PlyRecord] = []

    for _ in range(cfg.max_ply):
        tb = board_tok.tokenize(board, board.turn)
        traj = _compute_trajectory_tokens(move_history)

        bt = torch.tensor(
            tb.board_tokens
        ).unsqueeze(0).to(device)
        ct = torch.tensor(
            tb.color_tokens
        ).unsqueeze(0).to(device)
        tt = torch.tensor(traj).unsqueeze(0).to(device)

        if board.turn == chess.WHITE:
            legal = [m.uci() for m in board.legal_moves]
            move_uci = policy.predict_next_move(
                bt, ct, tt, move_history, legal,
                is_white_turn=True,
                temperature=cfg.t_policy,
                tokenizer=move_tok,
            )
            move_token = move_tok.tokenize_move(move_uci)
            log_prob = _log_prob_of_move(
                policy, bt, ct, tt,
                move_token, legal, move_tok, device,
            )
            plies.append(PlyRecord(
                bt, ct, tt, move_token,
                log_prob, is_white_ply=True,
            ))
        else:
            legal = [m.uci() for m in board.legal_moves]
            with torch.no_grad():
                move_uci = shadow.predict_next_move(
                    bt, ct, tt, move_history, legal,
                    is_white_turn=False,
                    temperature=cfg.t_opponent,
                    tokenizer=move_tok,
                )

        board.push(chess.Move.from_uci(move_uci))
        move_history.append(move_uci)

        if board.is_game_over():
            break

    # Determine termination and outcome via match/case
    match True:
        case _ if board.is_checkmate():
            termination = "checkmate"
            # White's turn => Black just mated White => loss
            outcome = (
                -1 if board.turn == chess.WHITE else 1
            )
        case _ if board.is_stalemate():
            termination = "stalemate"
            outcome = 0
        case _ if board.is_fifty_moves():
            termination = "50move"
            outcome = 0
        case _ if board.is_repetition(3):
            termination = "threefold"
            outcome = 0
        case _:
            termination = "maxply"
            outcome = 0

    return GameRecord(
        plies=plies,
        outcome=outcome,
        n_ply=len(move_history),
        termination=termination,
    )


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
        self._cfg = cfg
        self._device = torch.device(device)
        self._tracker = tracker or NoOpTracker()

        # Build policy from config
        self._policy = ChessModel(
            cfg.model, cfg.decoder
        ).to(self._device)

        if cfg.finetune.checkpoint_in:
            ckpt = torch.load(
                cfg.finetune.checkpoint_in,
                map_location=self._device,
                weights_only=True,
            )
            self._policy.load_state_dict(ckpt["model"])

        # Frozen reference (KL anchor) — never updated
        self._ref = copy.deepcopy(self._policy)
        self._ref.requires_grad_(False)
        self._ref.eval()

        # Lagging shadow opponent — EMA updated
        self._shadow = copy.deepcopy(self._policy)
        self._shadow.requires_grad_(False)
        self._shadow.eval()

        self._opt = torch.optim.AdamW(
            self._policy.parameters(),
            lr=cfg.finetune.learning_rate,
        )
        self._baseline = RunningMeanBaseline()
        self._board_tok = BoardTokenizer()
        self._move_tok = MoveTokenizer()
        self._game_count = 0
        self._global_step = 0

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
        return self._policy

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
        records: list[GameRecord] = []
        for _ in range(
            self._cfg.finetune.n_games_per_update
        ):
            rec = play_game(
                self._policy, self._shadow,
                self._board_tok, self._move_tok,
                self._cfg.finetune, self._device,
            )
            records.append(rec)
        return records

    def _update_shadow(self) -> None:
        """Apply EMA update to shadow model weights.

        theta_shadow = alpha * theta_shadow + (1-alpha) * theta_policy
        Runs under torch.no_grad().

        Example:
            >>> trainer._update_shadow()
        """
        alpha = self._cfg.finetune.ema_alpha
        with torch.no_grad():
            for p_s, p_p in zip(
                self._shadow.parameters(),
                self._policy.parameters(),
            ):
                p_s.copy_(
                    alpha * p_s + (1.0 - alpha) * p_p
                )

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
        all_plies = [
            p for g in records for p in g.plies
        ]
        if not all_plies:
            return {
                "pg_loss": 0.0,
                "kl_loss": 0.0,
                "total_loss": 0.0,
                "grad_norm": 0.0,
            }

        flat_returns = compute_returns(
            records, self._cfg.finetune.gamma
        )
        self._baseline.update(torch.stack(flat_returns))
        advantages = (
            torch.stack(flat_returns)
            - self._baseline.value()
        ).to(self._device)

        log_probs = torch.stack(
            [p.log_prob for p in all_plies]
        )
        pg_loss = -(advantages.detach() * log_probs).mean()

        # KL term: policy distribution vs frozen reference
        prefix = torch.tensor(
            [[SOS_IDX]], dtype=torch.long,
            device=self._device,
        )
        move_colors = torch.zeros(
            1, 1, dtype=torch.long,
            device=self._device,
        )
        kl_terms: list[Tensor] = []
        for ply in all_plies:
            p_logits = self._policy(
                ply.board_tokens, ply.color_tokens,
                ply.traj_tokens, prefix, None, move_colors,
            )[0, -1, :]
            with torch.no_grad():
                r_logits = self._ref(
                    ply.board_tokens, ply.color_tokens,
                    ply.traj_tokens, prefix, None,
                    move_colors,
                )[0, -1, :]
            p_dist = torch.softmax(p_logits, dim=-1)
            q_dist = torch.softmax(r_logits, dim=-1)
            kl = F.kl_div(
                p_dist.log(), q_dist, reduction="sum"
            )
            kl_terms.append(kl)

        kl_loss = (
            torch.stack(kl_terms).mean()
            if kl_terms
            else torch.tensor(0.0, device=self._device)
        )

        total = (
            pg_loss
            + self._cfg.finetune.lambda_kl * kl_loss
        )
        self._opt.zero_grad()
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self._policy.parameters(),
            self._cfg.finetune.gradient_clip,
        ).item()
        self._opt.step()
        self._global_step += 1

        return {
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total.item(),
            "grad_norm": grad_norm,
        }

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
        self._policy.train()
        metrics: dict[str, float] = {}
        for update_idx in range(n_updates):
            records = self._play_games()
            self._game_count += len(records)
            metrics = self._gradient_step(records)

            # Shadow update on crossing a games-multiple
            ngpu = self._cfg.finetune.n_games_per_update
            every = (
                self._cfg.finetune
                .shadow_update_every_n_games
            )
            if (self._game_count % every) < ngpu:
                self._update_shadow()

            win_rate = sum(
                1 for g in records if g.outcome == 1
            ) / max(len(records), 1)
            mean_ply = sum(
                g.n_ply for g in records
            ) / max(len(records), 1)
            metrics.update({
                "win_rate": win_rate,
                "mean_ply": mean_ply,
                "baseline": self._baseline.value(),
            })

            self._tracker.track_scalars(
                metrics, step=self._global_step,
            )
            logger.info(
                "Update %d | pg=%.4f kl=%.4f "
                "total=%.4f win=%.1f%% ply=%.1f",
                update_idx + 1,
                metrics["pg_loss"],
                metrics["kl_loss"],
                metrics["total_loss"],
                win_rate * 100,
                mean_ply,
            )
        return metrics

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
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self._policy.state_dict(),
            "optimizer": self._opt.state_dict(),
            "baseline_mean": self._baseline.value(),
            "baseline_count": self._baseline._count,
            "global_step": self._global_step,
            "game_count": self._game_count,
        }, path)

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
        ckpt = torch.load(
            path, map_location=self._device,
            weights_only=True,
        )
        self._policy.load_state_dict(ckpt["model"])
        self._global_step = ckpt.get("global_step", 0)
        self._game_count = ckpt.get("game_count", 0)
        opt_state = ckpt.get("optimizer")
        if opt_state:
            try:
                self._opt.load_state_dict(opt_state)
            except ValueError:
                logger.warning(
                    "Optimizer state mismatch — skipping"
                )
        if "baseline_mean" in ckpt:
            self._baseline._mean = ckpt["baseline_mean"]
            self._baseline._count = ckpt.get(
                "baseline_count", 1
            )
