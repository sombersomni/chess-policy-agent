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

    Attributes:
        board_tokens:    Length-65 list. Piece-type indices including CLS at position 0.
        color_tokens:    Length-65 list. Color indices including CLS at position 0.
        activity_tokens: Length-65 list. Activity scores 0-8, CLS always 0.
        src_sq:          0-63 index of the player's source square for this move.
        tgt_sq:          0-63 index of the player's target square for this move.
        opp_src_sq:      0-63 index of opponent's source square, or -1 if last move.
        opp_tgt_sq:      0-63 index of opponent's target square, or -1 if last move.
    """

    board_tokens: list[int]
    color_tokens: list[int]
    activity_tokens: list[int]
    src_sq: int
    tgt_sq: int
    opp_src_sq: int
    opp_tgt_sq: int


class ChessBatch(NamedTuple):
    """A batched training example served by DataLoader.

    Attributes:
        board_tokens:    torch.long tensor [B, 65]. Piece-type indices.
        color_tokens:    torch.long tensor [B, 65]. Color indices.
        activity_tokens: torch.long tensor [B, 65]. Activity scores 0-8.
        src_sq:          torch.long tensor [B]. Player source square indices 0-63.
        tgt_sq:          torch.long tensor [B]. Player target square indices 0-63.
        opp_src_sq:      torch.long tensor [B]. Opponent source indices; -1 = ignore.
        opp_tgt_sq:      torch.long tensor [B]. Opponent target indices; -1 = ignore.
    """

    board_tokens: Tensor     # [B, 65] long
    color_tokens: Tensor     # [B, 65] long
    activity_tokens: Tensor  # [B, 65] long
    src_sq: Tensor           # [B] long
    tgt_sq: Tensor           # [B] long
    opp_src_sq: Tensor       # [B] long
    opp_tgt_sq: Tensor       # [B] long


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
        opp_src_sq_logits: [B, 64]. Opponent's source square logits.
        opp_tgt_sq_logits: [B, 64]. Opponent's target square logits.
    """

    src_sq_logits: Tensor      # [B, 64]
    tgt_sq_logits: Tensor      # [B, 64]
    opp_src_sq_logits: Tensor  # [B, 64]
    opp_tgt_sq_logits: Tensor  # [B, 64]


class LabelTensors(NamedTuple):
    """Ground-truth square labels for LossComputer.compute().

    Attributes:
        src_sq:     [B] long. Player source squares 0-63.
        tgt_sq:     [B] long. Player target squares 0-63.
        opp_src_sq: [B] long. Opponent source squares 0-63, or -1 (ignore_index).
        opp_tgt_sq: [B] long. Opponent target squares 0-63, or -1 (ignore_index).
    """

    src_sq: Tensor      # [B] long
    tgt_sq: Tensor      # [B] long
    opp_src_sq: Tensor  # [B] long, -1 = ignore
    opp_tgt_sq: Tensor  # [B] long, -1 = ignore
