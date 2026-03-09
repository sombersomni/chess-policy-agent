"""SelfPlayLoop: orchestrates Phase 2 self-play RL training.

Implements the Trainable protocol where train_step corresponds to one
complete self-play episode. Drives ChessSimEnv, coordinates
EpisodeRecorder, calls RewardComputer, trains ValueHeads, computes
REINFORCE policy loss, and triggers EMAUpdater after each episode.

Checkpoint saves both player and EMA opponent state dicts for exact
resume.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import chess
import torch
import torch.nn as nn
from torch.optim import AdamW

from chess_sim.config import Phase2Config
from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.env.self_play_source import SelfPlaySource
from chess_sim.model.chess_model import ChessModel
from chess_sim.model.value_heads import ValueHeads
from chess_sim.training.ema_updater import EMAUpdater
from chess_sim.training.episode_recorder import EpisodeRecorder
from chess_sim.training.phase2_trainer import _make_trajectory_tokens
from chess_sim.training.reward_computer import RewardComputer
from chess_sim.types import PlyTuple

logger = logging.getLogger(__name__)

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _material_balance(board: chess.Board) -> float:
    """Return white material minus black material in pawn units."""
    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = _PIECE_VALUES.get(piece.piece_type, 0)
        score += val if piece.color == chess.WHITE else -val
    return score


class SelfPlayLoop:
    """Orchestrates the full Phase 2 self-play RL loop.

    Implements the Trainable protocol where train_step corresponds
    to one complete self-play episode. Drives ChessSimEnv, coordinates
    EpisodeRecorder, calls RewardComputer, trains ValueHeads, computes
    REINFORCE policy loss, and triggers EMAUpdater after each episode.

    Checkpoint saves both player and EMA opponent state dicts for
    exact resume.
    """

    def __init__(
        self,
        player: ChessModel,
        cfg: Phase2Config,
        device: str = "cpu",
    ) -> None:
        """Initialise the self-play loop.

        Args:
            player: The learning ChessModel (weights updated).
            cfg: Phase2Config with all RL hyperparameters.
            device: Torch device string, e.g. "cpu" or "cuda".
        """
        self._cfg = cfg
        self._device = torch.device(device)
        self._player = player.to(self._device)
        self._opponent = copy.deepcopy(player).to(
            self._device
        )
        for p in self._opponent.parameters():
            p.requires_grad_(False)
        # Detect d_model from encoder embedding layer
        d_model = self._player.encoder.embedding.d_model
        self._recorder = EpisodeRecorder()
        self._reward_fn = RewardComputer()
        self._value_heads = ValueHeads(d_model).to(
            self._device
        )
        self._ema = EMAUpdater(cfg.ema_alpha)
        self._opt = AdamW(
            list(self._player.parameters())
            + list(self._value_heads.parameters()),
            lr=3e-4,
        )
        self._board_tok = BoardTokenizer()
        self._move_tok = MoveTokenizer()

    def run(self, episodes: int) -> None:
        """Run self-play training for the given number of episodes.

        Each episode: play full game -> record plies -> compute
        rewards -> train value heads -> compute REINFORCE loss ->
        step optimizer -> update EMA opponent weights.

        Args:
            episodes: Total number of self-play games to run.
        """
        for ep in range(episodes):
            source = SelfPlaySource()
            board = source.reset()
            self._recorder.reset()
            move_history: list[chess.Move] = []
            step_count = 0
            while (
                not source.is_terminal()
                and step_count < self._cfg.max_episode_steps
            ):
                legal_ucis = source.legal_moves()
                if not legal_ucis:
                    break
                tb = self._board_tok.tokenize(
                    board, board.turn
                )
                bt = (
                    torch.tensor(
                        tb.board_tokens, dtype=torch.long
                    )
                    .unsqueeze(0)
                    .to(self._device)
                )
                ct = (
                    torch.tensor(
                        tb.color_tokens, dtype=torch.long
                    )
                    .unsqueeze(0)
                    .to(self._device)
                )
                traj = _make_trajectory_tokens(move_history)
                tt = (
                    torch.tensor(traj, dtype=torch.long)
                    .unsqueeze(0)
                    .to(self._device)
                )
                move_prefix = (
                    self._move_tok.tokenize_game(
                        [m.uci() for m in move_history]
                    )
                    .unsqueeze(0)
                    .to(self._device)
                )
                legal_mask = self._move_tok.build_legal_mask(
                    legal_ucis
                ).to(self._device)
                is_player_turn = board.turn == chess.WHITE
                mat_before = _material_balance(board)
                if is_player_turn:
                    ply = self._player_ply(
                        bt, ct, tt, move_prefix, legal_mask
                    )
                else:
                    ply = self._opponent_ply(
                        bt, ct, tt, move_prefix, legal_mask
                    )
                step_info = source.step(ply.move_uci)
                board = step_info.board
                mat_delta = _material_balance(board) - mat_before
                gave_check = (
                    1.0 if is_player_turn and board.is_check()
                    else 0.0
                )
                ply = ply._replace(
                    material_delta=mat_delta,
                    gave_check=gave_check,
                )
                self._recorder.record(ply)
                move_history.append(
                    chess.Move.from_uci(ply.move_uci)
                )
                step_count += 1
            outcome = self._compute_outcome(board)
            record = self._recorder.finalize(outcome)
            rewards = self._reward_fn.compute(
                record, self._cfg
            )
            player_plies = [
                p for p in record.plies if p.is_player_ply
            ]
            if player_plies and rewards.numel() > 0:
                self._update_params(player_plies, rewards)
            self._ema.step(self._player, self._opponent)
            logger.info(
                "Episode %d/%d: %d plies, outcome=%.1f",
                ep + 1,
                episodes,
                step_count,
                outcome,
            )

    def _player_ply(
        self,
        bt: torch.Tensor,
        ct: torch.Tensor,
        tt: torch.Tensor,
        move_prefix: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> PlyTuple:
        """Sample a move from the player model."""
        logits = self._player(bt, ct, tt, move_prefix, None)
        last_logits = logits[0, -1]
        with torch.no_grad():
            pre_mask_probs = torch.softmax(
                last_logits.detach(), dim=-1
            )
            illegal_mass = float(
                1.0 - pre_mask_probs[legal_mask].sum().item()
            )
        masked = last_logits.masked_fill(
            ~legal_mask, float("-inf")
        )
        probs = torch.softmax(masked, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().item()
        move_uci = self._move_tok._vocab.decode(
            action.item()
        )
        return PlyTuple(
            board_tokens=bt[0],
            color_tokens=ct[0],
            traj_tokens=tt[0],
            move_prefix=move_prefix[0],
            log_prob=log_prob,
            probs=probs.detach(),
            entropy=entropy,
            move_uci=move_uci,
            is_player_ply=True,
            illegal_mass=illegal_mass,
        )

    def _opponent_ply(
        self,
        bt: torch.Tensor,
        ct: torch.Tensor,
        tt: torch.Tensor,
        move_prefix: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> PlyTuple:
        """Sample a move from the opponent model (no grad)."""
        with torch.no_grad():
            logits = self._opponent(
                bt, ct, tt, move_prefix, None
            )
            last_logits = logits[0, -1]
            masked = last_logits.masked_fill(
                ~legal_mask, float("-inf")
            )
            probs = torch.softmax(masked, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().item()
            move_uci = self._move_tok._vocab.decode(
                action.item()
            )
        return PlyTuple(
            board_tokens=bt[0],
            color_tokens=ct[0],
            traj_tokens=tt[0],
            move_prefix=move_prefix[0],
            log_prob=log_prob,
            probs=probs.detach(),
            entropy=entropy,
            move_uci=move_uci,
            is_player_ply=False,
        )

    def _compute_outcome(self, board: chess.Board) -> float:
        """Derive outcome from the board state."""
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return self._cfg.win_reward
            elif result == "0-1":
                return self._cfg.loss_reward
        return self._cfg.draw_reward

    def _update_params(
        self,
        player_plies: list[PlyTuple],
        rewards: torch.Tensor,
    ) -> None:
        """Compute PG + value loss and step optimizer."""
        with torch.no_grad():
            cls_embs = torch.stack([
                self._player.encoder.encode(
                    p.board_tokens.unsqueeze(0).to(
                        self._device
                    ),
                    p.color_tokens.unsqueeze(0).to(
                        self._device
                    ),
                    p.traj_tokens.unsqueeze(0).to(
                        self._device
                    ),
                ).cls_embedding
                for p in player_plies
            ]).squeeze(1)
        vh_out = self._value_heads(cls_embs.detach())
        v_target = rewards.unsqueeze(1).to(self._device)
        mse_loss = nn.MSELoss()
        vh_loss = mse_loss(vh_out.v_win, v_target)
        log_probs = torch.stack(
            [p.log_prob for p in player_plies]
        ).to(self._device)
        pg_loss = -(
            log_probs * rewards.to(self._device)
        ).sum()
        total_loss = pg_loss + vh_loss
        self._opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self._player.parameters(), 1.0
        )
        self._opt.step()

    def save_checkpoint(self, path: Path) -> None:
        """Save player and EMA opponent state dicts to path.

        Args:
            path: Destination .pt file path.
        """
        torch.save(
            {
                "player": self._player.state_dict(),
                "opponent": self._opponent.state_dict(),
                "optimizer": self._opt.state_dict(),
                "value_heads": self._value_heads.state_dict(),
            },
            path,
        )
        logger.info("Phase2 checkpoint saved to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load player and EMA opponent state dicts from path.

        Args:
            path: Source .pt file path.
        """
        ckpt = torch.load(
            path, map_location=self._device
        )
        self._player.load_state_dict(ckpt["player"])
        self._opponent.load_state_dict(ckpt["opponent"])
        self._opt.load_state_dict(ckpt["optimizer"])
        self._value_heads.load_state_dict(
            ckpt["value_heads"]
        )
        logger.info(
            "Phase2 checkpoint loaded from %s", path
        )
