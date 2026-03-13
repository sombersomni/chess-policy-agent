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


class RLRewardRow(NamedTuple):
    """Immutable intermediate: one ply for HDF5 write, no torch tensors.

    Passed between PGNRewardPreprocessor replay and HDF5 write.
    All fields are plain Python / NumPy types.

    Attributes:
        board: (65, 3) float32 array — board/color/traj channels.
        color_tokens: (65,) int8 array — for structural mask.
        target_move: Vocab index of the teacher's move.
        multiplier: Pre-normalized m_hat RSCE weight.
        loss_mode: +1 for imitation (CE), -1 for repulsion.
        game_id: Source game index.
        ply_idx: 0-indexed ply within game.
        outcome: +1 / 0 / -1 from side-to-move perspective.

    Example:
        >>> row = RLRewardRow(
        ...     np.zeros((65,3), dtype=np.float32),
        ...     np.zeros(65, dtype=np.int8),
        ...     42, 1.0, 1, 0, 0, 1,
        ... )
        >>> row.target_move
        42
    """

    board: np.ndarray          # (65, 3) float32
    color_tokens: np.ndarray   # (65,) int8
    target_move: int
    multiplier: float          # pre-normalized m_hat
    loss_mode: int             # +1 imitation, -1 repulsion
    game_id: int
    ply_idx: int
    outcome: int               # +1 / 0 / -1
    legal_mask: np.ndarray     # (1971,) bool — true legal moves
