"""NamedTuple data containers for the chess encoder pipeline.

All data structures are immutable, typed, and self-documenting.
Shape annotations use the convention [dim, ...] where B = batch size.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
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


class TrainingExample(NamedTuple):
    """A single preprocessed training example saved to disk.

    Each example represents one ply from the side-to-move's perspective.

    Attributes:
        board_tokens:       Length-65 list. Piece-type indices including CLS at position 0.
        color_tokens:       Length-65 list. Color indices including CLS at position 0.
        trajectory_tokens:  Length-65 list. Trajectory roles 0-4, CLS always 0.
                            0=none, 1=player prev loc, 2=player curr loc,
                            3=opp prev loc, 4=opp curr loc.
        src_sq:             0-63 index of the player's source square for this move.
        tgt_sq:             0-63 index of the player's target square for this move.
    """

    board_tokens: list[int]
    color_tokens: list[int]
    trajectory_tokens: list[int]
    src_sq: int
    tgt_sq: int


class ChessBatch(NamedTuple):
    """A batched training example served by DataLoader.

    Attributes:
        board_tokens:       torch.long tensor [B, 65]. Piece-type indices.
        color_tokens:       torch.long tensor [B, 65]. Color indices.
        trajectory_tokens:  torch.long tensor [B, 65]. Trajectory roles 0-4.
                            0=none, 1=player prev loc, 2=player curr loc,
                            3=opp prev loc, 4=opp curr loc.
        src_sq:             torch.long tensor [B]. Player source square indices 0-63.
        tgt_sq:             torch.long tensor [B]. Player target square indices 0-63.
    """

    board_tokens: Tensor        # [B, 65] long
    color_tokens: Tensor        # [B, 65] long
    trajectory_tokens: Tensor   # [B, 65] long
    src_sq: Tensor              # [B] long
    tgt_sq: Tensor              # [B] long


class EncoderOutput(NamedTuple):
    """Output of ChessEncoder.encode().

    Attributes:
        cls_embedding:     Float tensor [B, 256]. Global board representation.
        square_embeddings: Float tensor [B, 64, 256]. Per-square contextualized vectors.
    """

    cls_embedding: Tensor      # [B, 256]
    square_embeddings: Tensor  # [B, 64, 256]


class PredictionOutput(NamedTuple):
    """Output of PredictionHeads.predict().

    All logit tensors are raw (pre-softmax) scores over 64 board squares.

    Attributes:
        src_sq_logits:     [B, 64]. Player's source square logits.
        tgt_sq_logits:     [B, 64]. Player's target square logits.
    """

    src_sq_logits: Tensor      # [B, 64]
    tgt_sq_logits: Tensor      # [B, 64]


class LabelTensors(NamedTuple):
    """Ground-truth square labels for LossComputer.compute().

    Attributes:
        src_sq:     [B] long. Player source squares 0-63.
        tgt_sq:     [B] long. Player target squares 0-63.
    """

    src_sq: Tensor      # [B] long
    tgt_sq: Tensor      # [B] long


# ---------------------------------------------------------------------------
# v2 Encoder-Decoder types
# ---------------------------------------------------------------------------


class GameTurnSample(NamedTuple):
    """A single game-turn sample for the encoder-decoder model.

    Represents one ply: the board state plus the decoder's input/target
    move-token sequences up to that point.

    Attributes:
        board_tokens:      Length-65 long tensor. CLS + 64 squares.
        color_tokens:      Length-65 long tensor. Color indices.
        trajectory_tokens: Length-65 long tensor. Trajectory roles 0-4.
        move_tokens:       Length-T long tensor. Decoder input (SOS + prior moves).
        target_tokens:     Length-T long tensor. Shifted targets (m_1, ..., m_t).
        move_pad_mask:     Length-T bool tensor. True = PAD position.
    """

    board_tokens: Tensor        # [65] long
    color_tokens: Tensor        # [65] long
    trajectory_tokens: Tensor   # [65] long
    move_tokens: Tensor         # [T] long — decoder input (SOS + prior moves)
    target_tokens: Tensor       # [T] long — shifted targets
    move_pad_mask: Tensor       # [T] bool — True = PAD


class GameTurnBatch(NamedTuple):
    """A collated batch of game-turn samples for the encoder-decoder model.

    All tensors are padded to the maximum sequence length T in the batch.

    Attributes:
        board_tokens:      [B, 65] long. Piece-type indices.
        color_tokens:      [B, 65] long. Color indices.
        trajectory_tokens: [B, 65] long. Trajectory roles 0-4.
        move_tokens:       [B, T] long. Decoder input sequences.
        target_tokens:     [B, T] long. Shifted target sequences.
        move_pad_mask:     [B, T] bool. True at PAD positions.
    """

    board_tokens: Tensor        # [B, 65] long
    color_tokens: Tensor        # [B, 65] long
    trajectory_tokens: Tensor   # [B, 65] long
    move_tokens: Tensor         # [B, T] long
    target_tokens: Tensor       # [B, T] long
    move_pad_mask: Tensor       # [B, T] bool


class DecoderOutput(NamedTuple):
    """Output of MoveDecoder.forward().

    Attributes:
        logits: Float tensor [B, T, MOVE_VOCAB_SIZE]. Raw move logits per step.
    """

    logits: Tensor              # [B, T, MOVE_VOCAB_SIZE]


class SelfPlayGame(NamedTuple):
    """Record of a single self-play game for Phase 2 REINFORCE training.

    Attributes:
        moves:             UCI move strings in game order.
        board_tokens:      One [65] long tensor per move.
        color_tokens:      One [65] long tensor per move.
        trajectory_tokens: One [65] long tensor per move.
        outcome:           +1.0 for win, -1.0 for loss, 0.0 for draw.
    """

    moves: list[str]                    # UCI strings in game order
    board_tokens: list[Tensor]          # one [65] per move
    color_tokens: list[Tensor]          # one [65] per move
    trajectory_tokens: list[Tensor]     # one [65] per move
    outcome: float                      # +1.0 win / -1.0 loss / 0.0 draw
