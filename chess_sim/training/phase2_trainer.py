"""Phase2Trainer: REINFORCE self-play trainer for the chess model.

Uses policy gradient (REINFORCE) to fine-tune the ChessModel through
self-play games. The model plays against itself, and the game outcome
(win/loss/draw) provides the reward signal for policy updates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chess
import torch
from torch import Tensor

from chess_sim.config import ChessModelV2Config
from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.model.chess_model import ChessModel
from chess_sim.types import SelfPlayGame

logger = logging.getLogger(__name__)


def _make_trajectory_tokens(
    move_history: list[chess.Move],
) -> list[int]:
    """Return trajectory_tokens len-65 from last two half-moves.

    Index 0 is CLS (always 0). Indices 1-64 map to squares a1-h8.
    Values: 0=none, 1=player prev loc, 2=player curr loc,
            3=opp prev loc, 4=opp curr loc.
    """
    tokens: list[int] = [0] * 65
    if len(move_history) >= 2:
        pl = move_history[-2]
        tokens[pl.from_square + 1] = 1
        tokens[pl.to_square + 1] = 2
    if len(move_history) >= 1:
        opp = move_history[-1]
        tokens[opp.from_square + 1] = 3
        tokens[opp.to_square + 1] = 4
    return tokens


class Phase2Trainer:
    """REINFORCE self-play trainer for the encoder-decoder chess model.

    Plays games against itself, computes REINFORCE policy gradients
    using game outcomes as rewards, and updates the model parameters.

    Owns:
      model:     ChessModel (encoder + decoder)
      optimizer: AdamW
      phase2_cfg: Phase2Config with reward values and hyperparams

    Example:
        >>> trainer = Phase2Trainer(device="cpu")
        >>> game = trainer.self_play_game(trainer.model)
        >>> loss = trainer.train_step(game)
    """

    def __init__(
        self,
        device: str = "cpu",
        v2_cfg: ChessModelV2Config | None = None,
    ) -> None:
        """Initialize model, optimizer, and phase2 config.

        Args:
            device: Torch device string. Use 'cpu' for tests.
            v2_cfg: Optional ChessModelV2Config. When None, uses defaults.

        Example:
            >>> trainer = Phase2Trainer(device="cpu")
        """
        cfg = v2_cfg or ChessModelV2Config()
        self.device = torch.device(device)
        self.model = ChessModel(
            cfg.model, cfg.decoder
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.trainer.learning_rate,
        )
        self.phase2_cfg = cfg.phase2

    def self_play_game(
        self, model: ChessModel
    ) -> SelfPlayGame:
        """Play a complete game of chess via self-play.

        The model alternates as white and black, sampling moves from
        its own policy distribution with legal-move masking. The game
        ends by checkmate, stalemate, draw rules, or max-move limit.

        Args:
            model: ChessModel instance to use for move generation.

        Returns:
            SelfPlayGame with move list, board states, and outcome.

        Example:
            >>> game = trainer.self_play_game(model)
            >>> isinstance(game.outcome, float)
            True
        """
        board = chess.Board()
        tok_board = BoardTokenizer()
        move_history: list[chess.Move] = []
        moves_uci: list[str] = []
        board_toks: list[Tensor] = []
        color_toks: list[Tensor] = []
        traj_toks: list[Tensor] = []

        model.eval()
        with torch.no_grad():
            while not board.is_game_over():
                tb = tok_board.tokenize(
                    board, board.turn
                )
                bt = torch.tensor(
                    tb.board_tokens, dtype=torch.long
                ).unsqueeze(0).to(self.device)
                ct = torch.tensor(
                    tb.color_tokens, dtype=torch.long
                ).unsqueeze(0).to(self.device)
                traj = _make_trajectory_tokens(move_history)
                tt = torch.tensor(
                    traj, dtype=torch.long
                ).unsqueeze(0).to(self.device)

                legal = [
                    m.uci() for m in board.legal_moves
                ]
                uci = model.predict_next_move(
                    bt, ct, tt, moves_uci, legal,
                )
                move = chess.Move.from_uci(uci)

                board_toks.append(bt.squeeze(0).cpu())
                color_toks.append(ct.squeeze(0).cpu())
                traj_toks.append(tt.squeeze(0).cpu())
                moves_uci.append(uci)
                move_history.append(move)
                board.push(move)

        result = board.result()
        if result == "1-0":
            outcome = self.phase2_cfg.win_reward
        elif result == "0-1":
            outcome = self.phase2_cfg.loss_reward
        else:
            outcome = self.phase2_cfg.draw_reward

        return SelfPlayGame(
            moves=moves_uci,
            board_tokens=board_toks,
            color_tokens=color_toks,
            trajectory_tokens=traj_toks,
            outcome=outcome,
        )

    def compute_rewards(
        self, game: SelfPlayGame
    ) -> Tensor:
        """Compute per-move reward signal from game outcome.

        Assigns the game outcome to each move. Novelty bonus is
        a stub (always 0) for now.

        Args:
            game: SelfPlayGame record.

        Returns:
            FloatTensor [num_moves] of per-move rewards.

        Example:
            >>> rewards = trainer.compute_rewards(game)
            >>> rewards.shape[0] == len(game.moves)
            True
        """
        T = len(game.moves)
        return torch.full(
            (T,), game.outcome, dtype=torch.float
        )

    def compute_log_probs(
        self, game: SelfPlayGame
    ) -> Tensor:
        """Compute log-probs of the moves actually taken.

        Feeds each board state and move history prefix through the
        model, then extracts the log-probability of the actual move.

        Args:
            game: SelfPlayGame record.

        Returns:
            FloatTensor [num_moves] of log-probabilities.

        Example:
            >>> log_probs = trainer.compute_log_probs(game)
            >>> log_probs.shape[0] == len(game.moves)
            True
        """
        tok = MoveTokenizer()
        log_probs: list[Tensor] = []
        self.model.train()

        for t, uci in enumerate(game.moves):
            bt = game.board_tokens[t].unsqueeze(0).to(
                self.device
            )
            ct = game.color_tokens[t].unsqueeze(0).to(
                self.device
            )
            tt = game.trajectory_tokens[t].unsqueeze(0).to(
                self.device
            )
            # Decoder input: SOS + moves[:t] (no EOS)
            prefix = tok.tokenize_game(
                game.moves[:t]
            ).unsqueeze(0).to(self.device)
            prefix = prefix[:, :-1]  # drop EOS

            logits = self.model(bt, ct, tt, prefix)
            log_p = torch.log_softmax(
                logits[0, -1], dim=-1
            )
            move_idx = tok.tokenize_move(uci)
            log_probs.append(log_p[move_idx])

        return torch.stack(log_probs)

    def train_step(self, game: SelfPlayGame) -> float:
        """Execute one REINFORCE gradient update from a self-play game.

        Computes log-probs of the moves taken, multiplies by discounted
        rewards, and performs a gradient step.

        Args:
            game: SelfPlayGame record from self_play_game().

        Returns:
            Scalar policy loss as a Python float.

        Example:
            >>> loss = trainer.train_step(game)
            >>> isinstance(loss, float)
            True
        """
        rewards = self.compute_rewards(game).to(self.device)
        baseline = rewards.mean()
        advantages = rewards - baseline
        log_probs = self.compute_log_probs(game)
        loss = -(advantages.detach() * log_probs).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_checkpoint(self, path: Path) -> None:
        """Save model state_dict to a checkpoint file.

        Args:
            path: Destination path for the .pt checkpoint file.

        Example:
            >>> trainer.save_checkpoint(Path("checkpoints/p2.pt"))
        """
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Phase2 checkpoint saved to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model state_dict from a checkpoint file.

        Args:
            path: Path to the .pt checkpoint file.

        Example:
            >>> trainer.load_checkpoint(Path("checkpoints/p2.pt"))
        """
        ckpt = torch.load(
            path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(
            "Phase2 checkpoint loaded from %s", path
        )
