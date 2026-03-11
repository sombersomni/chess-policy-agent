"""PGNRLTrainer: unified offline RL trainer on PGN master games.

Trains one side of each game (configured via train_color) using
pure REINFORCE policy gradient loss on recorded moves.

Does NOT implement the Trainable protocol (different train_epoch
signature -- takes a PGN path instead of a DataLoader).
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chess_sim.config import PGNRLConfig
from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.move_vocab import MoveVocab
from chess_sim.model.chess_model import ChessModel
from chess_sim.tracking.noop_tracker import NoOpTracker
from chess_sim.tracking.protocol import MetricTracker
from chess_sim.training.pgn_replayer import PGNReplayer
from chess_sim.training.pgn_rl_reward_computer import (
    PGNRLRewardComputer,
)

logger = logging.getLogger(__name__)


def _stream_pgn(
    pgn_path: Path,
    max_games: int = 0,
) -> list[chess.pgn.Game]:
    """Read games from a PGN file (plain or .zst).

    Args:
        pgn_path: Path to .pgn or .pgn.zst file.
        max_games: Maximum games to read (0 = all).

    Returns:
        List of parsed chess.pgn.Game objects.
    """
    games: list[chess.pgn.Game] = []
    if str(pgn_path).endswith(".zst"):
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        with open(pgn_path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_io = io.TextIOWrapper(
                    reader,
                    encoding="utf-8",
                    errors="replace",
                )
                while True:
                    game = chess.pgn.read_game(text_io)
                    if game is None:
                        break
                    games.append(game)
                    if 0 < max_games <= len(games):
                        break
    else:
        with open(pgn_path, "r") as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                games.append(game)
                if 0 < max_games <= len(games):
                    break
    return games


class PGNRLTrainer:
    """Unified offline RL trainer on PGN master games.

    Trains one side of each game (configured via train_color)
    using pure REINFORCE policy gradient loss on recorded moves.
    """

    def __init__(
        self,
        cfg: PGNRLConfig,
        device: str = "cpu",
        total_steps: int = 10_000,
        tracker: MetricTracker | None = None,
    ) -> None:
        """Initialize model, optimizer, scheduler, and components.

        Args:
            cfg: PGNRLConfig with all training parameters.
            device: Torch device string.
            total_steps: Total optimizer steps for LR schedule.
            tracker: Optional MetricTracker for logging.
        """
        self._cfg = cfg
        self._device = torch.device(device)
        self._model = ChessModel(
            cfg.model, cfg.decoder
        ).to(self._device)
        self._move_tok = MoveTokenizer()
        self._replayer = PGNReplayer()
        self._reward_fn = PGNRLRewardComputer()
        self._tracker: MetricTracker = (
            tracker or NoOpTracker()
        )

        # Split params: value head gets higher LR
        value_params = set(
            id(p)
            for p in self._model.value_head.parameters()
        )
        main_params = [
            p
            for p in self._model.parameters()
            if id(p) not in value_params
        ]
        self._opt = torch.optim.AdamW(
            [
                {
                    "params": main_params,
                    "lr": cfg.rl.learning_rate,
                },
                {
                    "params": list(
                        self._model.value_head.parameters()
                    ),
                    "lr": (
                        cfg.rl.learning_rate
                        * cfg.rl.value_lr_multiplier
                    ),
                },
            ],
            weight_decay=cfg.rl.weight_decay,
        )

        # LR schedule: warmup -> constant -> cosine decay
        warmup_steps = max(
            int(cfg.rl.warmup_fraction * total_steps), 1
        )
        decay_start = max(
            int(
                cfg.rl.decay_start_fraction * total_steps
            ),
            warmup_steps + 1,
        )
        constant_steps = decay_start - warmup_steps
        cosine_steps = max(total_steps - decay_start, 1)

        warmup = torch.optim.lr_scheduler.LinearLR(
            self._opt,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        constant = torch.optim.lr_scheduler.ConstantLR(
            self._opt,
            factor=1.0,
            total_iters=constant_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._opt,
            T_max=cosine_steps,
            eta_min=cfg.rl.min_lr,
        )
        self._sched = (
            torch.optim.lr_scheduler.SequentialLR(
                self._opt,
                schedulers=[warmup, constant, cosine],
                milestones=[warmup_steps, decay_start],
            )
        )
        self._global_step: int = 0
        self._ply_step: int = 0

    def _encode_and_decode(
        self,
        bt: Tensor,
        ct: Tensor,
        tt: Tensor,
        prefix: Tensor,
        move_uci: str,
    ) -> tuple[Tensor, Tensor, int | None]:
        """Shared encode->decode path; also resolves move_idx.

        Returns (last_logits [1, vocab], cls [1, d_model],
        move_idx | None). move_idx is None when tokenize_move
        raises KeyError for the given UCI string.

        Args:
            bt: [1, 65] board tokens.
            ct: [1, 65] color tokens.
            tt: [1, 65] trajectory tokens.
            prefix: [1, T] decoder prefix tokens.
            move_uci: UCI string of the teacher's move.

        Returns:
            Tuple of (last_logits, cls_embedding, move_idx).

        Example:
            >>> logits, cls, idx = trainer._encode_and_decode(
            ...     bt, ct, tt, prefix, "e2e4")
        """
        enc_out = self._model.encoder.encode(bt, ct, tt)
        memory = torch.cat(
            [
                enc_out.cls_embedding.unsqueeze(1),
                enc_out.square_embeddings,
            ],
            dim=1,
        )
        dec_out = self._model.decoder.decode(
            prefix, memory, None
        )
        last_logits = dec_out.logits[0, -1]
        try:
            move_idx: int | None = (
                self._move_tok.tokenize_move(move_uci)
            )
        except KeyError:
            move_idx = None
        return (last_logits, enc_out.cls_embedding, move_idx)

    @property
    def model(self) -> ChessModel:
        """Expose the underlying ChessModel."""
        return self._model

    @property
    def current_lrs(self) -> tuple[float, float]:
        """Return (main_lr, value_lr) for the current step."""
        return (
            self._opt.param_groups[0]["lr"],
            self._opt.param_groups[1]["lr"],
        )

    def _build_board_snapshots(
        self,
        game: chess.pgn.Game,
        train_white: bool,
    ) -> list[chess.Board]:
        """Replay game and collect pre-move boards for trained side.

        Args:
            game: A parsed PGN game object.
            train_white: True if training the white side.

        Returns:
            Board snapshots (before each move) for the trained
            side only, in game order.
        """
        board = game.board()
        snapshots: list[chess.Board] = []
        for move in game.mainline_moves():
            is_white = board.turn == chess.WHITE
            if is_white == train_white:
                snapshots.append(board.copy())
            board.push(move)
        return snapshots

    def _log_board_snapshot(
        self,
        board: chess.Board,
        move_uci: str,
        game_idx: int,
        is_winner_ply: bool,
        reward: float,
        last_logits: Tensor,
        move_idx: int | None,
    ) -> None:
        """Emit a board text block to tracker every 100 ply steps.

        Includes side label, reward, and top-1 prediction hit/miss.

        Args:
            board: Board state before the move.
            move_uci: UCI string of the teacher's move.
            game_idx: Index of the current game in the epoch.
            is_winner_ply: True if this ply belongs to the winner.
            reward: Composite reward at this ply.
            last_logits: Raw logits [vocab_size] from the decoder.
            move_idx: Vocab index of teacher move (None if OOV).
        """
        if self._ply_step % 100 != 0:
            return
        try:
            san = board.san(chess.Move.from_uci(move_uci))
        except (ValueError, chess.InvalidMoveError):
            san = "?"
        side = "winner" if is_winner_ply else "loser"
        probs = torch.softmax(last_logits.detach(), dim=-1)
        top1_idx = int(probs.argmax().item())
        top1_prob = float(probs[top1_idx].item()) * 100.0
        try:
            pred_uci = self._move_tok.decode(top1_idx)
            try:
                pred_san = board.san(
                    chess.Move.from_uci(pred_uci)
                )
            except (ValueError, chess.InvalidMoveError, AssertionError):
                pred_san = pred_uci
        except (KeyError, IndexError):
            pred_uci, pred_san = "?", "?"
        hit = (
            "[HIT]"
            if move_idx is not None and top1_idx == move_idx
            else "[MISS]"
        )
        header = (
            f"Step {self._ply_step} | "
            f"Game {game_idx} | "
            f"side={side} | "
            f"reward={reward:.2f} | "
            f"Move: {move_uci} ({san}) | "
            f"pred={pred_uci}({pred_san}){hit} "
            f"top1={top1_prob:.1f}%"
        )
        logger.info(
            "Board @ ply_step=%d game=%d side=%s reward=%.2f"
            " move=%s(%s) pred=%s(%s)%s top1=%.1f%%",
            self._ply_step, game_idx, side, reward,
            move_uci, san, pred_uci, pred_san, hit, top1_prob,
        )
        self._tracker.log_text(
            f"{header}\n{board.unicode()}", step=self._ply_step
        )

    def train_game(
        self,
        game: chess.pgn.Game,
        game_idx: int = 0,
    ) -> dict[str, float]:
        """Train on one complete PGN game.

        Args:
            game: A parsed PGN game object.
            game_idx: Index of this game in the epoch
                (used for board snapshot logging).

        Returns:
            Dict with loss metrics. Empty dict if game is
            skipped (unknown result, draw, or no valid plies).
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
        board_snaps = self._build_board_snapshots(
            game, train_white
        )

        rewards = self._reward_fn.compute(
            plies, self._cfg.rl
        )
        self._model.train()

        q_preds: list[Tensor] = []
        all_logits: list[Tensor] = []
        all_targets: list[int] = []
        valid_reward_idxs: list[int] = []

        for i, ply in enumerate(plies):
            self._ply_step += 1

            bt = ply.board_tokens.unsqueeze(0).to(
                self._device
            )
            ct = ply.color_tokens.unsqueeze(0).to(
                self._device
            )
            tt = ply.traj_tokens.unsqueeze(0).to(
                self._device
            )
            prefix = ply.move_prefix.unsqueeze(0).to(
                self._device
            )

            last_logits, cls, move_idx = (
                self._encode_and_decode(
                    bt, ct, tt, prefix, ply.move_uci
                )
            )
            if i < len(board_snaps):
                self._log_board_snapshot(
                    board_snaps[i],
                    ply.move_uci,
                    game_idx,
                    is_winner_ply=ply.is_winner_ply,
                    reward=float(rewards[i]),
                    last_logits=last_logits,
                    move_idx=move_idx,
                )
            if move_idx is None:
                continue

            # Look up action embedding from shared token table
            midx_t = torch.tensor(
                [move_idx],
                dtype=torch.long,
                device=self._device,
            )
            action_emb: Tensor = (
                self._model.move_token_emb(midx_t)
            )  # [1, d_model]
            # detach: value MSE must not train token_emb
            # via Q-head path; CE trains token_emb separately
            action_emb = action_emb.detach()

            # detach cls: value MSE must not reshape encoder
            q_t: Tensor = self._model.value_head(
                cls.detach(), action_emb
            )  # [1, 1]
            q_preds.append(q_t.squeeze())  # scalar

            all_logits.append(last_logits)
            all_targets.append(move_idx)
            valid_reward_idxs.append(i)

        if not all_logits:
            return {}

        valid_rewards = rewards[valid_reward_idxs].to(
            self._device
        )

        # Compute value loss and advantage (diagnostic)
        q_preds_t: Tensor = torch.stack(q_preds)  # [N]
        # detach: prevent Q-head gradient from flowing
        # through advantage into the policy loss
        advantage: Tensor = (
            valid_rewards - q_preds_t.detach()
        )  # [N]
        # Whiten advantages for diagnostic logging only.
        if advantage.numel() > 1:
            advantage = (
                advantage - advantage.mean()
            ) / (advantage.std() + 1e-8)
        mean_adv = advantage.mean().item()
        std_adv = (
            advantage.std().item()
            if advantage.numel() > 1
            else 0.0
        )
        value_loss = F.mse_loss(q_preds_t, valid_rewards)

        # --- RSBC loss (reward-signed behavioral cloning) ---
        rsbc_loss = self._compute_rsbc_loss(
            all_logits=all_logits,
            all_targets=all_targets,
            rewards=valid_rewards,
        )

        total_loss = (
            self._cfg.rl.lambda_rsbc * rsbc_loss
            + self._cfg.rl.lambda_value * value_loss
        )

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
            "rsbc_loss": rsbc_loss.item(),
            "n_plies": len(plies),
            "mean_reward": mean_reward,
            "value_loss": value_loss.item(),
            "mean_advantage": mean_adv,
            "std_advantage": std_adv,
            # deprecated — backward compat for Aim dashboard
            "pg_loss": 0.0,
            "ce_loss": 0.0,
            "awbc_loss": 0.0,
            "entropy_bonus": 0.0,
        }

    def train_epoch(
        self,
        pgn_path: Path,
        max_games: int = 0,
    ) -> dict[str, float]:
        """Train over all games in pgn_path for one epoch.

        Args:
            pgn_path: Path to .pgn or .pgn.zst file.
            max_games: Max games to load (0 = all).

        Returns:
            Dict with averaged loss metrics over all games.
        """
        games = _stream_pgn(pgn_path, max_games)
        total_loss = 0.0
        rsbc_loss_sum: float = 0.0
        reward_sum = 0.0
        value_loss_sum: float = 0.0
        mean_adv_sum: float = 0.0
        std_adv_sum: float = 0.0
        n_games = 0

        for gi, game in enumerate(games):
            metrics = self.train_game(game, game_idx=gi)
            if not metrics:
                continue
            total_loss += metrics["total_loss"]
            rsbc_loss_sum += metrics["rsbc_loss"]
            reward_sum += metrics["mean_reward"]
            value_loss_sum += metrics["value_loss"]
            mean_adv_sum += metrics["mean_advantage"]
            std_adv_sum += metrics["std_advantage"]
            n_games += 1
            self._tracker.track_scalars(
                {
                    "rsbc_loss": metrics["rsbc_loss"],
                    "total_loss": metrics["total_loss"],
                    "mean_reward": metrics[
                        "mean_reward"
                    ],
                    "value_loss": metrics["value_loss"],
                    "mean_advantage": metrics[
                        "mean_advantage"
                    ],
                    "std_advantage": metrics[
                        "std_advantage"
                    ],
                },
                step=self._global_step,
            )

        denom = max(n_games, 1)
        return {
            "total_loss": total_loss / denom,
            "rsbc_loss": rsbc_loss_sum / denom,
            "mean_reward": reward_sum / denom,
            "n_games": n_games,
            "value_loss": value_loss_sum / denom,
            "mean_advantage": mean_adv_sum / denom,
            "std_advantage": std_adv_sum / denom,
            # deprecated — backward compat
            "pg_loss": 0.0,
            "ce_loss": 0.0,
            "awbc_loss": 0.0,
            "entropy_bonus": 0.0,
        }

    def evaluate(
        self,
        pgn_path: Path,
        max_games: int = 0,
    ) -> dict[str, float]:
        """Evaluate CE loss and accuracy over pgn_path games.

        Args:
            pgn_path: Path to .pgn or .pgn.zst file.
            max_games: Max games to load (0 = all).

        Returns:
            Dict with val_loss and val_accuracy keys.
        """
        games = _stream_pgn(pgn_path, max_games)
        total_ce = 0.0
        correct = 0
        total = 0
        n_games = 0

        self._model.eval()
        with torch.no_grad():
            for game in games:
                plies = self._replayer.replay(game)
                train_white = (
                    self._cfg.rl.train_color == "white"
                )
                plies = [
                    p
                    for p in plies
                    if p.is_white_ply == train_white
                ]
                if not plies:
                    continue
                n_games += 1
                for ply in plies:
                    bt = ply.board_tokens.unsqueeze(0).to(
                        self._device
                    )
                    ct = ply.color_tokens.unsqueeze(0).to(
                        self._device
                    )
                    tt = ply.traj_tokens.unsqueeze(0).to(
                        self._device
                    )
                    prefix = (
                        ply.move_prefix.unsqueeze(0).to(
                            self._device
                        )
                    )
                    logits = self._model(
                        bt, ct, tt, prefix, None
                    )
                    last_logits = logits[0, -1]
                    try:
                        move_idx = (
                            self._move_tok.tokenize_move(
                                ply.move_uci
                            )
                        )
                    except KeyError:
                        continue
                    ce = F.cross_entropy(
                        last_logits.unsqueeze(0),
                        torch.tensor(
                            [move_idx],
                            device=self._device,
                        ),
                    )
                    total_ce += ce.item()
                    pred = last_logits.argmax().item()
                    correct += int(pred == move_idx)
                    total += 1

        denom = max(total, 1)
        return {
            "val_loss": total_ce / denom,
            "val_accuracy": correct / denom,
            "n_games": n_games,
        }

    def sample_visuals(
        self,
        pgn_path: Path,
        n_plies: int = 4,
        top_k: int = 8,
        *,
        train_accuracy: float | None = None,
    ) -> list[object]:
        """Generate matplotlib figures for sampled plies from the first game.

        Replays the first game in pgn_path, evenly samples up to n_plies
        positions, runs inference (no_grad), and returns rendered figures
        via rl_visualizer.render_rl_ply.

        Args:
            pgn_path: Path to .pgn or .pgn.zst file.
            n_plies: Number of positions to sample per call.
            top_k: Number of top predicted moves to show in chart.
            train_accuracy: Latest validation accuracy (0-1) for overlay.

        Returns:
            List of matplotlib Figure objects (may be empty on failure).
        """
        from chess_sim.training.rl_visualizer import render_rl_ply

        games = _stream_pgn(pgn_path, max_games=1)
        if not games:
            return []

        game = games[0]
        vocab = MoveVocab()

        # Replay via shared replayer and compute rewards via
        # shared reward function — no inline reimplementation.
        all_plies = self._replayer.replay(game)
        if not all_plies:
            return []
        rewards = self._reward_fn.compute(
            all_plies, self._cfg.rl
        )

        # Build (board_snapshot, ply, reward) triples by
        # re-walking the mainline to capture board copies.
        board = game.board()
        snapshots: list[
            tuple[chess.Board, object, float]
        ] = []
        for i, move in enumerate(game.mainline_moves()):
            board_copy = board.copy()
            board.push(move)
            snapshots.append(
                (board_copy, all_plies[i], float(rewards[i]))
            )

        if not snapshots:
            return []

        # Evenly sample n_plies positions
        indices = np.linspace(
            0, len(snapshots) - 1, min(n_plies, len(snapshots)),
            dtype=int,
        )
        figs: list[object] = []
        self._model.eval()
        with torch.no_grad():
            for idx in indices:
                board_snap, ply, reward = snapshots[int(idx)]
                bt = ply.board_tokens.unsqueeze(0).to(self._device)
                ct = ply.color_tokens.unsqueeze(0).to(self._device)
                tt = ply.traj_tokens.unsqueeze(0).to(self._device)
                prefix = ply.move_prefix.unsqueeze(0).to(self._device)

                last_logits, cls_emb, move_idx = (
                    self._encode_and_decode(
                        bt, ct, tt, prefix, ply.move_uci
                    )
                )
                probs = F.softmax(last_logits, dim=-1)
                log_p = torch.log(probs + 1e-10)
                entropy = -(probs * log_p).sum().item()

                top_indices = probs.topk(top_k).indices.tolist()
                top_pairs: list[tuple[str, float]] = []
                for vi in top_indices:
                    try:
                        uci = vocab.decode(vi)
                        top_pairs.append((uci, probs[vi].item()))
                    except (KeyError, IndexError):
                        continue

                # Opponent's last move from trajectory token roles 3/4.
                traj_list: list[int] = ply.traj_tokens.tolist()
                opp_from_sq: int | None = next(
                    (i - 1 for i, v in enumerate(traj_list) if v == 3),
                    None,
                )
                opp_to_sq: int | None = next(
                    (i - 1 for i, v in enumerate(traj_list) if v == 4),
                    None,
                )

                # Q-value reusing cls_emb — no extra forward pass.
                q_val: float | None = None
                if move_idx is not None:
                    midx_t = torch.tensor(
                        [move_idx],
                        dtype=torch.long,
                        device=self._device,
                    )
                    action_emb = self._model.move_token_emb(midx_t)
                    q_raw = self._model.value_head(
                        cls_emb.detach(), action_emb.detach()
                    ).item()
                    q_val = q_raw if abs(q_raw) <= 100 else None

                advantage: float | None = (
                    (reward - q_val) if q_val is not None else None
                )

                try:
                    fig = render_rl_ply(
                        board=board_snap,
                        actual_uci=ply.move_uci,
                        top_k=top_pairs,
                        reward=reward,
                        entropy=entropy,
                        ply_idx=int(idx),
                        opp_from_sq=opp_from_sq,
                        opp_to_sq=opp_to_sq,
                        train_accuracy=train_accuracy,
                        q_value=q_val,
                        advantage=advantage,
                        is_winner_ply=ply.is_winner_ply,
                    )
                    figs.append(fig)
                except Exception:
                    logger.warning(
                        "render_rl_ply failed at idx %d",
                        idx,
                        exc_info=True,
                    )

        return figs

    def _compute_awbc_loss(
        self,
        all_logits: list[Tensor],
        all_targets: list[int],
        advantage: Tensor,
    ) -> Tensor:
        """Compute advantage-weighted behavioral cloning loss.

        Weights per-ply CE by clamp(advantage, min=0) so only
        positive-advantage moves contribute imitation gradient.
        Normalizes weights to keep loss scale stable across
        game lengths.

        Args:
            all_logits: Per-ply decoder logits, each [vocab].
            all_targets: Per-ply teacher move indices (int).
            advantage: Per-ply whitened advantage, shape [N].

        Returns:
            Scalar AWBC loss tensor.

        Example:
            >>> loss = trainer._compute_awbc_loss(
            ...     logits, targets, adv)
        """
        n: int = len(all_logits)
        targets_t = torch.tensor(
            all_targets,
            dtype=torch.long,
            device=self._device,
        )
        per_ply_ce: Tensor = F.cross_entropy(
            torch.stack(all_logits),
            targets_t,
            label_smoothing=self._cfg.rl.label_smoothing,
            reduction="none",
        )  # [N]
        weights_raw: Tensor = advantage.clamp(min=0.0)
        weight_sum: Tensor = (
            weights_raw.sum() + self._cfg.rl.awbc_eps
        )
        # normalize so mean weight ~ 1 over positive-adv plies
        weights: Tensor = weights_raw / weight_sum * n
        return (weights * per_ply_ce).mean()

    def _compute_entropy_bonus(
        self,
        all_logits: list[Tensor],
    ) -> Tensor:
        """Compute mean policy entropy over all plies.

        H = -sum(p * log(p+eps)). Returns 0.0 tensor when
        lambda_entropy=0 to avoid unnecessary computation.

        Args:
            all_logits: Per-ply decoder logits, each [vocab].

        Returns:
            Scalar mean entropy tensor (caller negates).

        Example:
            >>> ent = trainer._compute_entropy_bonus(logits)
        """
        if self._cfg.rl.lambda_entropy == 0.0:
            return torch.zeros(1, device=self._device)
        logits_t: Tensor = torch.stack(all_logits)  # [N, V]
        probs: Tensor = F.softmax(logits_t, dim=-1)
        log_probs: Tensor = torch.log(
            probs + self._cfg.rl.awbc_eps
        )
        entropy_per_ply: Tensor = -(probs * log_probs).sum(
            dim=-1
        )  # [N]
        # Return negative mean entropy: minimizing total_loss
        # (which adds lambda_entropy * bonus) then maximizes
        # entropy, since d(total)/d(entropy) < 0.
        return -entropy_per_ply.mean()

    def _compute_rsbc_loss(
        self,
        all_logits: list[Tensor],
        all_targets: list[int],
        rewards: Tensor,
    ) -> Tensor:
        """Compute reward-signed behavioral cloning loss.

        Weights per-ply CE by per-game-normalized reward
        r_hat_t in [-1, 1]. Positive reward -> imitate
        (lower log-prob of teacher move). Negative reward
        -> avoid (raise log-prob of teacher move).
        No label smoothing: smoothing opposes anti-imitation
        on negative plies.

        Args:
            all_logits: Per-ply decoder logits, each [vocab].
            all_targets: Per-ply teacher move indices (int).
            rewards: Raw per-ply discounted rewards [N].

        Returns:
            Scalar RSBC loss tensor (may be negative when
            losing plies dominate).

        Example:
            >>> loss = trainer._compute_rsbc_loss(
            ...     logits, targets, rewards)
        """
        targets_t = torch.tensor(
            all_targets,
            dtype=torch.long,
            device=self._device,
        )
        # No label_smoothing: smoothing opposes anti-imitation
        # on negative-reward plies.
        per_ply_ce: Tensor = F.cross_entropy(
            torch.stack(all_logits),
            targets_t,
            label_smoothing=0.0,
            reduction="none",
        )  # [N]
        if self._cfg.rl.rsbc_normalize_per_game:
            max_abs = rewards.abs().max()
            r_hat: Tensor = rewards / (max_abs + 1e-8)
        else:
            r_hat = rewards
        return (r_hat * per_ply_ce).mean()

    def save_checkpoint(self, path: Path) -> None:
        """Save model, optimizer, and scheduler state.

        Args:
            path: Destination .pt file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self._model.state_dict(),
                "optimizer": self._opt.state_dict(),
                "scheduler": self._sched.state_dict(),
            },
            path,
        )
        logger.info("RL checkpoint saved to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint. Skips optimizer/scheduler if absent.

        Supports loading Phase1 checkpoints that only have a
        'model' key.

        Args:
            path: Source .pt file path.
        """
        ckpt = torch.load(
            path,
            map_location=self._device,
            weights_only=True,
        )
        self._model.load_state_dict(ckpt["model"])
        opt_state = ckpt.get("optimizer")
        if opt_state is not None:
            self._opt.load_state_dict(opt_state)
        sched_state = ckpt.get("scheduler")
        if sched_state is not None:
            self._sched.load_state_dict(sched_state)
        logger.info(
            "RL checkpoint loaded from %s", path
        )
