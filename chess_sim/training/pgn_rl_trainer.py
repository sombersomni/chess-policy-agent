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

from chess_sim.config import PGNRLConfig
from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.move_vocab import MoveVocab
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.model.chess_model import ChessModel
from chess_sim.tracking.noop_tracker import NoOpTracker
from chess_sim.tracking.protocol import MetricTracker
from chess_sim.training.pgn_replayer import PGNReplayer
from chess_sim.training.pgn_rl_reward_computer import (
    PGNRLRewardComputer,
)
from chess_sim.training.phase2_trainer import _make_trajectory_tokens

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

        self._opt = torch.optim.AdamW(
            self._model.parameters(),
            lr=cfg.rl.learning_rate,
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

    def _encode_and_decode(
        self,
        bt: torch.Tensor,
        ct: torch.Tensor,
        tt: torch.Tensor,
        prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared encode->decode path used by train_game and sample_visuals.

        Returns (last_logits [1, vocab], cls_emb [1, d_model]).
        """
        raise NotImplementedError("To be implemented")

    @property
    def model(self) -> ChessModel:
        """Expose the underlying ChessModel."""
        return self._model

    def train_game(
        self, game: chess.pgn.Game
    ) -> dict[str, float]:
        """Train on one complete PGN game.

        Args:
            game: A parsed PGN game object.

        Returns:
            Dict with loss metrics. Empty dict if game is
            skipped (unknown result, no valid plies).
        """
        plies = self._replayer.replay(game)
        train_white = self._cfg.rl.train_color == "white"
        plies = [
            p for p in plies if p.is_white_ply == train_white
        ]
        if not plies:
            return {}

        rewards = self._reward_fn.compute(
            plies, self._cfg.rl
        )
        self._model.train()

        log_probs: list[torch.Tensor] = []
        v_preds: list[torch.Tensor] = []  # noqa: F841
        all_logits: list[torch.Tensor] = []
        all_targets: list[int] = []
        valid_reward_idxs: list[int] = []

        for i, ply in enumerate(plies):
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

            logits = self._model(bt, ct, tt, prefix, None)
            last_logits = logits[0, -1]

            try:
                move_idx = self._move_tok.tokenize_move(
                    ply.move_uci
                )
            except KeyError:
                continue

            probs = F.softmax(last_logits, dim=-1)
            log_p = torch.log(probs + 1e-10)

            log_probs.append(log_p[move_idx])
            all_logits.append(last_logits)
            all_targets.append(move_idx)
            valid_reward_idxs.append(i)

        if not log_probs:
            return {}

        log_probs_t = torch.stack(log_probs)
        valid_rewards = rewards[valid_reward_idxs].to(
            self._device
        )

        pg_loss = -(log_probs_t * valid_rewards).sum()

        targets_t = torch.tensor(
            all_targets, dtype=torch.long, device=self._device
        )
        ce_loss = F.cross_entropy(
            torch.stack(all_logits),
            targets_t,
            label_smoothing=self._cfg.rl.label_smoothing,
        )

        total_loss = (
            pg_loss + self._cfg.rl.lambda_ce * ce_loss
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
            "pg_loss": pg_loss.item(),
            "ce_loss": ce_loss.item(),
            "n_plies": len(plies),
            "mean_reward": mean_reward,
            "value_loss": 0.0,
            "mean_advantage": 0.0,
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
        pg_loss = 0.0
        ce_loss = 0.0
        reward_sum = 0.0
        value_loss_sum: float = 0.0
        mean_adv_sum: float = 0.0
        n_games = 0

        for game in games:
            metrics = self.train_game(game)
            if not metrics:
                continue
            total_loss += metrics["total_loss"]
            pg_loss += metrics["pg_loss"]
            ce_loss += metrics["ce_loss"]
            reward_sum += metrics["mean_reward"]
            value_loss_sum += metrics["value_loss"]
            mean_adv_sum += metrics["mean_advantage"]
            n_games += 1
            self._tracker.track_scalars(
                {
                    "pg_loss": metrics["pg_loss"],
                    "ce_loss": metrics["ce_loss"],
                    "total_loss": metrics["total_loss"],
                    "mean_reward": metrics["mean_reward"],
                    "value_loss": metrics["value_loss"],
                    "mean_advantage": metrics[
                        "mean_advantage"
                    ],
                },
                step=self._global_step,
            )

        denom = max(n_games, 1)
        return {
            "total_loss": total_loss / denom,
            "pg_loss": pg_loss / denom,
            "ce_loss": ce_loss / denom,
            "mean_reward": reward_sum / denom,
            "n_games": n_games,
            "value_loss": value_loss_sum / denom,
            "mean_advantage": mean_adv_sum / denom,
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
    ) -> list[object]:
        """Generate matplotlib figures for sampled plies from the first game.

        Replays the first game in pgn_path, evenly samples up to n_plies
        positions, runs inference (no_grad), and returns rendered figures
        via rl_visualizer.render_rl_ply.

        Args:
            pgn_path: Path to .pgn or .pgn.zst file.
            n_plies: Number of positions to sample per call.
            top_k: Number of top predicted moves to show in chart.

        Returns:
            List of matplotlib Figure objects (may be empty on failure).
        """
        from chess_sim.training.rl_visualizer import render_rl_ply

        games = _stream_pgn(pgn_path, max_games=1)
        if not games:
            return []

        game = games[0]
        result = game.headers.get("Result", "*")
        _RESULT_MAP = {
            "1-0": (True, False),
            "0-1": (False, True),
            "1/2-1/2": (True, True),
        }
        if result not in _RESULT_MAP:
            return []

        white_wins, black_wins = _RESULT_MAP[result]
        is_draw = white_wins and black_wins
        board_tok = BoardTokenizer()
        move_tok = MoveTokenizer()
        vocab = MoveVocab()

        # Collect all (board_snapshot, ply_tuple) pairs
        board = game.board()
        move_history: list[chess.Move] = []
        snapshots: list[tuple[chess.Board, object]] = []
        T = sum(1 for _ in game.mainline_moves())

        for move in game.mainline_moves():
            is_white = board.turn == chess.WHITE
            is_winner = white_wins if is_white else black_wins

            board_copy = board.copy()
            tb = board_tok.tokenize(board, board.turn)
            bt = torch.tensor(tb.board_tokens, dtype=torch.long)
            ct = torch.tensor(tb.color_tokens, dtype=torch.long)
            traj = _make_trajectory_tokens(move_history)
            tt = torch.tensor(traj, dtype=torch.long)
            prior = [m.uci() for m in move_history]
            prefix_t = move_tok.tokenize_game(prior)[:-1]

            board.push(move)
            reward_base = (
                self._cfg.rl.draw_reward
                if is_draw
                else self._cfg.rl.win_reward
                if is_winner
                else self._cfg.rl.loss_reward
            )
            t = len(move_history)
            reward = reward_base * (
                self._cfg.rl.gamma ** max(T - 1 - t, 0)
            )

            from chess_sim.types import OfflinePlyTuple
            ply = OfflinePlyTuple(
                board_tokens=bt,
                color_tokens=ct,
                traj_tokens=tt,
                move_prefix=prefix_t,
                move_uci=move.uci(),
                is_winner_ply=is_winner,
                is_white_ply=is_white,
                is_draw_ply=is_draw,
            )
            snapshots.append((board_copy, ply, float(reward)))
            move_history.append(move)

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

                logits = self._model(bt, ct, tt, prefix, None)
                last_logits = logits[0, -1]
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

                fig = render_rl_ply(
                    board=board_snap,
                    actual_uci=ply.move_uci,
                    top_k=top_pairs,
                    reward=reward,
                    entropy=entropy,
                    ply_idx=int(idx),
                )
                figs.append(fig)

        return figs

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
