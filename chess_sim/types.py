"""NamedTuple data containers for the chess encoder pipeline.

All data structures are immutable, typed, and self-documenting.
Shape annotations use the convention [dim, ...] where B = batch size.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from torch import Tensor


class TokenizedBoard(NamedTuple):
    """Output of BoardTokenizer.tokenize().

    Attributes:
        board_tokens: Length-65 list. Index 0 is CLS token (value 0).
                      Indices 1-64 map to squares a1-h8 with piece-type values 0-7.
        color_tokens: Length-65 list. Index 0 is CLS (value 0).
                      Square indices: 0=empty, 1=player, 2=opponent.
    """

    board_tokens: list[int]
    color_tokens: list[int]


class ConditionedBoard(NamedTuple):
    """Output of tokenize_with_piece_cond().

    Contains board tokens with VALID_EMPTY/INVALID_EMPTY split
    based on piece-type reachability, plus the conditioning signal.

    Attributes:
        board_tokens: Length-65 list. Index 0=CLS(0). Indices 1-64:
                      piece types 0-7 for occupied; VALID_EMPTY(1) or
                      INVALID_EMPTY(8) for empty squares.
        color_tokens: Length-65 list (unchanged semantics).
        piece_type_idx: int 1-7 (chess.PieceType), the conditioning signal.
    """

    board_tokens: list[int]
    color_tokens: list[int]
    piece_type_idx: int


class EncoderOutput(NamedTuple):
    """Output of ChessEncoder.encode().

    Attributes:
        cls_embedding:     Float tensor [B, 256]. Global board representation.
        square_embeddings: Float tensor [B, 64, 256]. Per-square contextualized vectors.
    """

    cls_embedding: Tensor      # [B, 256]
    square_embeddings: Tensor  # [B, 64, 256]


class DecoderOutput(NamedTuple):
    """Output of MoveDecoder.forward().

    Attributes:
        logits: Float tensor [B, T, MOVE_VOCAB_SIZE]. Raw move logits per step.
    """

    logits: Tensor              # [B, T, MOVE_VOCAB_SIZE]


class ValueHeadOutput(NamedTuple):
    """Output from ValueHeads.forward.

    Both tensors have shape [B, 1].
    """

    v_win: Tensor
    v_surprise: Tensor


class OfflinePlyTuple(NamedTuple):
    """One half-move from PGN replay for offline RL training.

    All tokens captured BEFORE the move is applied.
    """

    board_tokens: Tensor
    color_tokens: Tensor
    traj_tokens: Tensor
    move_prefix: Tensor        # SOS + prior move vocab indices
    move_uci: str
    is_winner_ply: bool        # True = positive reward side
    is_white_ply: bool         # True when white is side-to-move
    is_draw_ply: bool          # True for all plies in a drawn game
    material_delta: float      # net material gain since last ply
    legal_move_ucis: list[str]  # all legal UCI moves before push
    capture_map: list[int]     # 64-elem: 1 if sq is legally capturable


class RLRewardRow(NamedTuple):
    """Immutable intermediate: one ply for HDF5 write, no torch tensors.

    Passed between PGNRewardPreprocessor replay and HDF5 write.
    All fields are plain Python / NumPy types.

    Attributes:
        board: (65, 3) float32 array — board/color/traj channels.
        color_tokens: (65,) int8 array — for structural mask.
        target_move: Vocab index of the teacher's move.
        game_id: Source game index.
        ply_idx: 0-indexed ply within game.
        outcome: +1 / 0 / -1 from side-to-move perspective.
        legal_mask: (1971,) bool array — true legal moves.
        src_square: 0-indexed from-square of the played move (0-63).
        capture_map: (64,) bool array — legally capturable squares.

    Example:
        >>> row = RLRewardRow(
        ...     np.zeros((65,3), dtype=np.float32),
        ...     np.zeros(65, dtype=np.int8),
        ...     42, 0, 0, 1,
        ...     np.zeros(1971, dtype=bool),
        ...     12,
        ...     np.zeros(64, dtype=bool),
        ... )
        >>> row.target_move
        42
    """

    board: np.ndarray          # (65, 3) float32
    color_tokens: np.ndarray   # (65,) int8
    target_move: int
    game_id: int
    ply_idx: int
    outcome: int               # +1 / 0 / -1
    legal_mask: np.ndarray     # (1971,) bool — true legal moves
    src_square: int            # 0-63, from-square of the played move
    capture_map: np.ndarray    # (64,) bool — legally capturable squares


class PlyRecord(NamedTuple):
    """One live-policy half-move for self-play gradient computation.

    Captures the board state and sampled action with its log-prob
    so REINFORCE can compute policy gradients after the game ends.

    Attributes:
        board_tokens: [1, 65] long tensor — piece-type tokens.
        color_tokens: [1, 65] long tensor — player/opponent.
        traj_tokens:  [1, 65] long tensor — trajectory roles.
        move_token:   Vocab index of the sampled move.
        log_prob:     Scalar tensor with requires_grad=True.
        is_white_ply: True when white is the side to move.

    Example:
        >>> ply = PlyRecord(bt, ct, tt, 42, lp, True)
        >>> ply.move_token
        42
    """

    board_tokens: Tensor    # [1, 65] long
    color_tokens: Tensor    # [1, 65] long
    traj_tokens: Tensor     # [1, 65] long
    move_token: int         # vocab index of sampled move
    log_prob: Tensor        # scalar; requires_grad=True
    is_white_ply: bool


class AuxLossOutput(NamedTuple):
    """Per-head auxiliary losses for joint encoder training.

    Attributes:
        capture_loss: BCE loss on per-square capture map.
        category_loss: CE loss on 7-class move category.
        phase_loss: CE loss on 3-class game phase.
    """

    capture_loss: Tensor
    category_loss: Tensor
    phase_loss: Tensor


class GameRecord(NamedTuple):
    """Completed self-play game result for the gradient step.

    Holds the trajectory of live-policy plies and the terminal
    outcome from the policy's perspective.

    Attributes:
        plies: List of PlyRecord for live-policy moves only.
        outcome: +1 win, 0 draw, -1 loss (policy perspective).
        n_ply: Total plies played (both sides).
        termination: Reason the game ended.

    Example:
        >>> game = GameRecord([], 1, 10, "checkmate")
        >>> game.outcome
        1
    """

    plies: list[PlyRecord]
    outcome: int            # +1 win, 0 draw, -1 loss
    n_ply: int
    termination: str        # checkmate|stalemate|50move|threefold|maxply
