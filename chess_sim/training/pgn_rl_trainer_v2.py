"""PGNRLTrainerV2: outcome-only offline RL trainer (no value head).

v2 variant of PGNRLTrainer that uses only the outcome-weighted
cross-entropy (RSBC) loss. The value head is not called, leaving
a single-term loss: total_loss = lambda_rsbc * rsbc_loss.
"""
from __future__ import annotations

import io
import logging
import random
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
from chess_sim.data.structural_mask import StructuralMaskBuilder
from chess_sim.model.chess_model import ChessModel
from chess_sim.protocols import StructuralMaskable
from chess_sim.tracking.noop_tracker import NoOpTracker
from chess_sim.tracking.protocol import MetricTracker
from chess_sim.training.pgn_replayer import PGNReplayer
from chess_sim.training.pgn_rl_reward_computer import (
    PGNRLRewardComputer,
)
from chess_sim.types import OfflinePlyTuple

logger = logging.getLogger(__name__)


def _split_games_by_outcome(
    games: list[chess.pgn.Game],
    train_white: bool,
) -> tuple[
    list[chess.pgn.Game],
    list[chess.pgn.Game],
    list[chess.pgn.Game],
]:
    """Partition games into (win, loss, draw) for the trained side.

    Args:
        games: Parsed PGN games.
        train_white: True if training white, False for black.

    Returns:
        Tuple of (win_games, loss_games, draw_games).
    """
    win_result = "1-0" if train_white else "0-1"
    loss_result = "0-1" if train_white else "1-0"
    win_games: list[chess.pgn.Game] = []
    loss_games: list[chess.pgn.Game] = []
    draw_games: list[chess.pgn.Game] = []
    for g in games:
        result = g.headers.get("Result", "*")
        if result == win_result:
            win_games.append(g)
        elif result == loss_result:
            loss_games.append(g)
        elif result == "1/2-1/2":
            draw_games.append(g)
    return win_games, loss_games, draw_games


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


class PGNRLTrainerV2:
    """Outcome-only offline RL trainer on PGN master games.

    v2 variant: trains using only RSBC loss (outcome-weighted CE).
    The value head is not called — total_loss = lambda_rsbc * rsbc.
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

        # Single param group: all model params at one LR
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
        self._ply_step: int = 0

        # Structural mask: suppress logits for tokens whose
        # from-square has no player piece. Opt-in via config.
        self._struct_mask: StructuralMaskable | None = (
            StructuralMaskBuilder(self._device)
            if cfg.rl.use_structural_mask
            else None
        )

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
        move_idx | None).

        Args:
            bt: [1, 65] board tokens.
            ct: [1, 65] color tokens.
            tt: [1, 65] trajectory tokens.
            prefix: [1, T] decoder prefix tokens.
            move_uci: UCI string of the teacher's move.

        Returns:
            Tuple of (last_logits, cls_embedding, move_idx).
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

    def _pad_prefixes(
        self,
        prefixes: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Pad variable-length prefix tensors to a common length.

        Returns zero-padded tensor and boolean mask (True = PAD).

        Args:
            prefixes: List of T 1-D long tensors, each [L_i].

        Returns:
            Tuple of (padded [T, max_len], mask [T, max_len]).
        """
        max_len = max(p.size(0) for p in prefixes)
        t_count = len(prefixes)
        padded = torch.zeros(
            t_count,
            max_len,
            dtype=torch.long,
            device=self._device,
        )
        for i, p in enumerate(prefixes):
            padded[i, : p.size(0)] = p.to(self._device)
        mask = padded == 0
        for i, p in enumerate(prefixes):
            mask[i, : p.size(0)] = False
        return padded, mask

    def _encode_and_decode_batch(
        self,
        plies: list[OfflinePlyTuple],
        move_ucis: list[str],
    ) -> tuple[Tensor, Tensor, list[int | None]]:
        """Batched encode-decode over all T plies of one game.

        Args:
            plies: T OfflinePlyTuples from one game.
            move_ucis: T UCI strings, aligned with plies.

        Returns:
            Tuple of (last_logits [T, vocab], cls [T, d_model],
            move_idxs list[int | None] of length T).
        """
        move_idxs: list[int | None] = []
        for uci in move_ucis:
            try:
                move_idxs.append(
                    self._move_tok.tokenize_move(uci)
                )
            except KeyError:
                move_idxs.append(None)

        bt = torch.stack(
            [p.board_tokens for p in plies]
        ).to(self._device)
        ct = torch.stack(
            [p.color_tokens for p in plies]
        ).to(self._device)
        tt = torch.stack(
            [p.traj_tokens for p in plies]
        ).to(self._device)

        padded, mask = self._pad_prefixes(
            [p.move_prefix for p in plies]
        )

        enc_out = self._model.encoder.encode(bt, ct, tt)

        memory = torch.cat(
            [
                enc_out.cls_embedding.unsqueeze(1),
                enc_out.square_embeddings,
            ],
            dim=1,
        )

        dec_out = self._model.decoder.decode(
            padded, memory, mask
        )

        prefix_lens: list[int] = [
            p.move_prefix.size(0) for p in plies
        ]

        last_logits = torch.stack([
            dec_out.logits[i, prefix_lens[i] - 1, :]
            for i in range(len(plies))
        ])

        return (
            last_logits,
            enc_out.cls_embedding,
            move_idxs,
        )

    @property
    def model(self) -> ChessModel:
        """Expose the underlying ChessModel."""
        return self._model

    @property
    def current_lr(self) -> float:
        """Return the current learning rate (single param group)."""
        return self._opt.param_groups[0]["lr"]

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
            Board snapshots before each move for the trained side.
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

        Args:
            board: Board state before the move.
            move_uci: UCI string of the teacher's move.
            game_idx: Index of the current game in the epoch.
            is_winner_ply: True if this ply belongs to the winner.
            reward: Composite reward at this ply.
            last_logits: Raw logits [vocab_size] from decoder.
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
            except (
                ValueError,
                chess.InvalidMoveError,
                AssertionError,
            ):
                pred_san = pred_uci
        except (KeyError, IndexError):
            pred_uci, pred_san = "?", "?"
        hit = (
            "[HIT]"
            if move_idx is not None
            and top1_idx == move_idx
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
            "Board @ ply_step=%d game=%d side=%s "
            "reward=%.2f move=%s(%s) pred=%s(%s)%s "
            "top1=%.1f%%",
            self._ply_step,
            game_idx,
            side,
            reward,
            move_uci,
            san,
            pred_uci,
            pred_san,
            hit,
            top1_prob,
        )
        self._tracker.log_text(
            f"{header}\n{board.unicode()}",
            step=self._ply_step,
        )

    def train_game(
        self,
        game: chess.pgn.Game,
        game_idx: int = 0,
    ) -> dict[str, float]:
        """Train on one complete PGN game (RSBC loss only).

        Args:
            game: A parsed PGN game object.
            game_idx: Index of this game in the epoch.

        Returns:
            Dict with loss metrics. Empty dict if game skipped.
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

        # Outcome weights: winner=1.0, draw=draw_reward_norm,
        # loser=loser_ply_weight. Always positive.
        ply_weights = torch.tensor(
            [
                (
                    self._cfg.rl.draw_reward_norm
                    if plies[i].is_draw_ply
                    else 1.0
                    if plies[i].is_winner_ply
                    else self._cfg.rl.loser_ply_weight
                )
                for i in valid_mask
            ],
            dtype=torch.float32,
            device=self._device,
        )
        rsbc_loss = self._compute_rsbc_loss(
            all_logits=all_logits,
            all_targets=all_targets,
            weights=ply_weights,
            all_color_tokens=(
                all_color_tokens
                if self._struct_mask is not None
                else None
            ),
        )

        total_loss = self._cfg.rl.lambda_rsbc * rsbc_loss

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
            "n_games": 1,
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
        if self._cfg.rl.balance_outcomes:
            train_white = (
                self._cfg.rl.train_color == "white"
            )
            win_games, loss_games, draw_games = (
                _split_games_by_outcome(games, train_white)
            )
            if win_games and loss_games:
                n = min(len(win_games), len(loss_games))
                if len(win_games) > n:
                    win_games = random.sample(
                        win_games, n
                    )
                if len(loss_games) > n:
                    loss_games = random.sample(
                        loss_games, n
                    )
                draws = (
                    []
                    if self._cfg.rl.skip_draws
                    else draw_games
                )
                games = [
                    g
                    for pair in zip(win_games, loss_games)
                    for g in pair
                ] + draws
                dropped = abs(
                    len(win_games) - len(loss_games)
                )
                logger.info(
                    "Outcome balance: %d win + %d loss"
                    " + %d draw = %d games"
                    " (dropped %d from majority)",
                    n,
                    n,
                    len(draws),
                    2 * n + len(draws),
                    dropped,
                )
            else:
                logger.warning(
                    "balance_outcomes=True but only one "
                    "outcome class present (%d win, %d "
                    "loss) — skipping balance",
                    len(win_games),
                    len(loss_games),
                )
        total_loss = 0.0
        rsbc_loss_sum: float = 0.0
        reward_sum = 0.0
        n_games = 0

        for gi, game in enumerate(games):
            metrics = self.train_game(game, game_idx=gi)
            if not metrics:
                continue
            total_loss += metrics["total_loss"]
            rsbc_loss_sum += metrics["rsbc_loss"]
            reward_sum += metrics["mean_reward"]
            n_games += 1
            self._tracker.track_scalars(
                {
                    "rsbc_loss": metrics["rsbc_loss"],
                    "total_loss": metrics[
                        "total_loss"
                    ],
                    "mean_reward": metrics[
                        "mean_reward"
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
                    bt = ply.board_tokens.unsqueeze(
                        0
                    ).to(self._device)
                    ct = ply.color_tokens.unsqueeze(
                        0
                    ).to(self._device)
                    tt = ply.traj_tokens.unsqueeze(
                        0
                    ).to(self._device)
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
            "val_n_games": n_games,
        }

    def sample_visuals(
        self,
        pgn_path: Path,
        n_plies: int = 4,
        top_k: int = 8,
        *,
        train_accuracy: float | None = None,
    ) -> list[object]:
        """Render matplotlib figures for sampled plies.

        No Q-value/advantage computation (v2 has no value head).

        Args:
            pgn_path: Path to .pgn or .pgn.zst file.
            n_plies: Number of positions to sample per call.
            top_k: Number of top predicted moves to show.
            train_accuracy: Latest val accuracy for overlay.

        Returns:
            List of matplotlib Figure objects (may be empty).
        """
        from chess_sim.training.rl_visualizer import (
            render_rl_ply,
        )

        games = _stream_pgn(pgn_path, max_games=1)
        if not games:
            return []

        game = games[0]
        vocab = MoveVocab()

        all_plies = self._replayer.replay(game)
        if not all_plies:
            return []
        rewards = self._reward_fn.compute(
            all_plies, self._cfg.rl
        )

        board = game.board()
        snapshots: list[
            tuple[chess.Board, object, float]
        ] = []
        for i, move in enumerate(game.mainline_moves()):
            board_copy = board.copy()
            board.push(move)
            snapshots.append(
                (
                    board_copy,
                    all_plies[i],
                    float(rewards[i]),
                )
            )

        if not snapshots:
            return []

        indices = np.linspace(
            0,
            len(snapshots) - 1,
            min(n_plies, len(snapshots)),
            dtype=int,
        )
        figs: list[object] = []
        self._model.eval()
        with torch.no_grad():
            for idx in indices:
                board_snap, ply, reward = snapshots[
                    int(idx)
                ]
                bt = ply.board_tokens.unsqueeze(0).to(
                    self._device
                )
                ct = ply.color_tokens.unsqueeze(0).to(
                    self._device
                )
                tt = ply.traj_tokens.unsqueeze(0).to(
                    self._device
                )
                prefix = ply.move_prefix.unsqueeze(
                    0
                ).to(self._device)

                last_logits, _, move_idx = (
                    self._encode_and_decode(
                        bt, ct, tt, prefix, ply.move_uci
                    )
                )
                probs = F.softmax(last_logits, dim=-1)
                log_p = torch.log(probs + 1e-10)
                entropy = -(probs * log_p).sum().item()

                top_indices = (
                    probs.topk(top_k).indices.tolist()
                )
                top_pairs: list[tuple[str, float]] = []
                for vi in top_indices:
                    try:
                        uci = vocab.decode(vi)
                        top_pairs.append(
                            (uci, probs[vi].item())
                        )
                    except (KeyError, IndexError):
                        continue

                traj_list: list[int] = (
                    ply.traj_tokens.tolist()
                )
                opp_from_sq: int | None = next(
                    (
                        i - 1
                        for i, v in enumerate(traj_list)
                        if v == 3
                    ),
                    None,
                )
                opp_to_sq: int | None = next(
                    (
                        i - 1
                        for i, v in enumerate(traj_list)
                        if v == 4
                    ),
                    None,
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
                        q_value=None,
                        advantage=None,
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

    def _compute_rsbc_loss(
        self,
        all_logits: list[Tensor],
        all_targets: list[int],
        weights: Tensor,
        all_color_tokens: list[Tensor] | None = None,
    ) -> Tensor:
        """Outcome-weighted behavioral cloning loss.

        Minimizes CE for all plies (winner, draw, loser),
        scaled by a non-negative outcome weight in (0, 1].

        Args:
            all_logits: Per-ply decoder logits, each [vocab].
            all_targets: Per-ply teacher move indices (int).
            weights: Non-negative per-ply CE weights [N].
            all_color_tokens: Per-ply color tokens [65] each,
                or None when structural masking is disabled.

        Returns:
            Scalar weighted CE loss tensor (always >= 0).
        """
        targets_t = torch.tensor(
            all_targets,
            dtype=torch.long,
            device=self._device,
        )
        logits_t = torch.stack(all_logits)  # [N, V]

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
        return (weights * per_ply_ce).mean()

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
        logger.info("RL v2 checkpoint saved to %s", path)

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
            try:
                self._opt.load_state_dict(opt_state)
            except ValueError:
                logger.warning(
                    "Optimizer state mismatch (v1->v2 "
                    "param group change) — skipping "
                    "optimizer restore"
                )
        sched_state = ckpt.get("scheduler")
        if sched_state is not None:
            self._sched.load_state_dict(sched_state)
        logger.info(
            "RL v2 checkpoint loaded from %s", path
        )
