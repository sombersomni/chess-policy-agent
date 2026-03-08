"""ChessModelAgent: adapts ChessModel into the Policy protocol.

Wraps ChessModel.predict_next_move() to provide:
  - select_action(): returns a MoveVocab integer index
  - top_n_predictions(): returns ranked MovePrediction list with probabilities

CPU-only inference is enforced (consistent with the GUI convention) unless
the caller explicitly passes device="cuda".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.move_vocab import MoveVocab
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.env import MovePrediction, Observation
from chess_sim.model.chess_model import ChessModel


def _compute_trajectory_tokens(move_history: list[str]) -> list[int]:
    """Compute trajectory tokens from the game's move history.

    Trajectory roles (channel 2 of the observation):
      0 = no trajectory
      1 = player's previous move FROM square
      2 = player's previous move TO square
      3 = opponent's previous move FROM square
      4 = opponent's previous move TO square

    The most recent move was played by the opponent (it's now the player's
    turn).  The second-most-recent move was played by the player.

    Args:
        move_history: Ordered UCI strings from game start to current position.

    Returns:
        List of 65 ints (index 0 = CLS, indices 1–64 = squares a1–h8).
    """
    import chess

    traj: list[int] = [0] * 65

    def _mark(uci: str, from_role: int, to_role: int) -> None:
        if len(uci) < 4:
            return
        try:
            from_sq = chess.parse_square(uci[:2]) + 1   # shift to 1-based
            to_sq = chess.parse_square(uci[2:4]) + 1
            traj[from_sq] = from_role
            traj[to_sq] = to_role
        except ValueError:
            pass

    # Second-to-last move = player's previous move
    if len(move_history) >= 2:
        _mark(move_history[-2], from_role=1, to_role=2)
    # Most recent move = opponent's previous move
    if len(move_history) >= 1:
        _mark(move_history[-1], from_role=3, to_role=4)

    return traj


class ChessModelAgent:
    """Adapts ChessModel into the Policy protocol for ChessSimEnv.

    Provides select_action() (returns a MoveVocab index) and
    top_n_predictions() (returns ranked MovePrediction objects).

    @torch.no_grad() is applied to all forward passes via the _forward
    internal method to ensure no gradient graphs are built during simulation.

    Attributes:
        _model: ChessModel in eval mode.
        _tokenizer: BoardTokenizer for converting boards to token arrays.
        _vocab: MoveVocab for encoding/decoding UCI strings.
        _move_tok: MoveTokenizer for building decoder input sequences.
        _temperature: Softmax temperature for sampling.
        _device: Torch device (CPU by default).

    Example:
        >>> agent = ChessModelAgent(model, BoardTokenizer(), MoveVocab())
        >>> action = agent.select_action(obs, ["e2e4", "d2d4"])
        >>> isinstance(action, int)
        True
    """

    def __init__(
        self,
        model: ChessModel,
        tokenizer: BoardTokenizer,
        vocab: MoveVocab,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        """Initialise the agent and place the model on device.

        Args:
            model: Pretrained ChessModel (encoder-decoder).
            tokenizer: BoardTokenizer for board → tokens conversion.
            vocab: MoveVocab for UCI ↔ index mapping.
            temperature: Softmax temperature (1.0 = no scaling).
            device: Torch device string (default "cpu").
        """
        self._model = model.to(device)
        self._model.eval()
        self._tokenizer = tokenizer
        self._vocab = vocab
        self._move_tok = MoveTokenizer()
        self._temperature = temperature
        self._device = torch.device(device)

    def select_action(
        self,
        obs: Observation,
        legal_moves: list[str],
    ) -> int:
        """Return the MoveVocab index of the agent's top prediction.

        Args:
            obs: Float32 array of shape (65, 3) — packed board observation.
            legal_moves: UCI strings of all legal moves in the current position.

        Returns:
            Integer MoveVocab index of the top-ranked legal move.
        """
        preds = self.top_n_predictions(obs, legal_moves, n=1)
        return self._vocab.encode(preds[0].move_uci)

    def top_n_predictions(
        self,
        obs: Observation,
        legal_moves: list[str],
        n: int = 3,
    ) -> list[MovePrediction]:
        """Return the top-N agent predictions with probabilities.

        Args:
            obs: Float32 array of shape (65, 3) — packed board observation.
            legal_moves: UCI strings of all legal moves in the current position.
            n: Number of predictions to return (clamped to len(legal_moves)).

        Returns:
            Ordered list of MovePrediction (rank 1 = highest probability).
        """
        logits = self._forward(obs)  # shape [V]
        probs = self._masked_probs(logits, legal_moves)
        top_n = min(n, len(legal_moves))
        top_indices = torch.topk(probs, top_n).indices.tolist()
        result: list[MovePrediction] = []
        for rank, idx in enumerate(top_indices, start=1):
            uci = self._vocab.decode(idx)
            prob = probs[idx].item()
            result.append(MovePrediction(move_uci=uci, probability=prob, rank=rank))
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward(self, obs: Observation) -> Tensor:
        """Run a single forward pass and return next-move logits.

        Unpacks the (65, 3) observation into the three token streams,
        builds a minimal decoder input (SOS only), and returns the logits
        at the last decoder position.

        Args:
            obs: Float32 numpy array of shape (65, 3).

        Returns:
            FloatTensor of shape [V] (move vocabulary logits).
        """
        board_t, color_t, traj_t = self._obs_to_tensors(obs)
        # Decoder input: just [SOS] for single-step next-move prediction.
        from chess_sim.data.move_vocab import SOS_IDX

        move_tok = torch.tensor([[SOS_IDX]], dtype=torch.long, device=self._device)
        logits = self._model(board_t, color_t, traj_t, move_tok)  # [1, 1, V]
        return logits[0, -1, :]  # [V]

    def _obs_to_tensors(
        self,
        obs: Observation,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Unpack a (65, 3) observation array into three [1, 65] LongTensors.

        Args:
            obs: Float32 numpy array of shape (65, 3).

        Returns:
            Tuple of (board_tokens, color_tokens, trajectory_tokens),
            each LongTensor of shape [1, 65] on self._device.
        """
        import numpy as np

        arr = obs.astype(np.int64)
        board_t = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(self._device)
        color_t = torch.from_numpy(arr[:, 1]).unsqueeze(0).to(self._device)
        traj_t = torch.from_numpy(arr[:, 2]).unsqueeze(0).to(self._device)
        return board_t, color_t, traj_t

    def _masked_probs(
        self,
        logits: Tensor,
        legal_moves: list[str],
    ) -> Tensor:
        """Apply legal-move mask and temperature, return softmax probabilities.

        Args:
            logits: Raw logits FloatTensor of shape [V].
            legal_moves: UCI strings of legal moves.

        Returns:
            Probability FloatTensor of shape [V].
        """
        legal_mask = self._move_tok.build_legal_mask(legal_moves).to(self._device)
        masked = logits.clone()
        masked[~legal_mask] = -1e9
        return F.softmax(masked / self._temperature, dim=-1)
