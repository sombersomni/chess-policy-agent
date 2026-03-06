"""Shared test utilities and fixtures for the chess encoder test suite.

All fixtures are CPU-only and deterministic. Use these helpers to avoid
duplicating setup logic across test files.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import chess
import chess.pgn
import torch
from torch import Tensor

from chess_sim.types import ChessBatch, LabelTensors, PredictionOutput, TrainingExample

# Sequence length constant: CLS + 64 squares.
SEQ_LEN: int = 65
D_MODEL: int = 256
N_SQUARES: int = 64


def make_synthetic_batch(batch_size: int = 4, device: str = "cpu") -> ChessBatch:
    """Return a random ChessBatch of long tensors for use in unit tests.

    All label values are in range [0, 63] (valid square indices). Opponent
    labels are also set to valid squares (no -1) to simplify loss testing.

    Args:
        batch_size: Number of examples in the batch. Default 4.
        device: Torch device string. Default 'cpu'.

    Returns:
        ChessBatch namedtuple with all tensors on the specified device.

    Example:
        >>> batch = make_synthetic_batch(4)
        >>> batch.board_tokens.shape
        torch.Size([4, 65])
    """
    return ChessBatch(
        board_tokens=torch.randint(0, 8, (batch_size, SEQ_LEN), dtype=torch.long, device=device),
        color_tokens=torch.randint(0, 3, (batch_size, SEQ_LEN), dtype=torch.long, device=device),
        activity_tokens=torch.randint(0, 9, (batch_size, SEQ_LEN), dtype=torch.long, device=device),
        src_sq=torch.randint(0, N_SQUARES, (batch_size,), dtype=torch.long, device=device),
        tgt_sq=torch.randint(0, N_SQUARES, (batch_size,), dtype=torch.long, device=device),
        opp_src_sq=torch.randint(0, N_SQUARES, (batch_size,), dtype=torch.long, device=device),
        opp_tgt_sq=torch.randint(0, N_SQUARES, (batch_size,), dtype=torch.long, device=device),
    )


def make_synthetic_batch_with_ignore(batch_size: int = 4, device: str = "cpu") -> ChessBatch:
    """Return a ChessBatch where opp labels contain -1 (ignore_index) for last move.

    Half the examples have -1 in opp_src_sq and opp_tgt_sq to simulate
    terminal board states. Used to test T18.

    Args:
        batch_size: Number of examples in the batch. Default 4.
        device: Torch device string. Default 'cpu'.

    Returns:
        ChessBatch with -1 values in opponent label tensors.

    Example:
        >>> batch = make_synthetic_batch_with_ignore(4)
        >>> (batch.opp_src_sq == -1).any().item()
        True
    """
    batch = make_synthetic_batch(batch_size, device)
    opp_src = batch.opp_src_sq.clone()
    opp_tgt = batch.opp_tgt_sq.clone()
    opp_src[batch_size // 2:] = -1
    opp_tgt[batch_size // 2:] = -1
    return ChessBatch(
        board_tokens=batch.board_tokens,
        color_tokens=batch.color_tokens,
        activity_tokens=batch.activity_tokens,
        src_sq=batch.src_sq,
        tgt_sq=batch.tgt_sq,
        opp_src_sq=opp_src,
        opp_tgt_sq=opp_tgt,
    )


def make_label_tensors(batch_size: int = 4, device: str = "cpu") -> LabelTensors:
    """Return random LabelTensors with all valid square indices (no -1).

    Args:
        batch_size: Number of examples. Default 4.
        device: Torch device string. Default 'cpu'.

    Returns:
        LabelTensors namedtuple with [B] long tensors in range [0, 63].
    """
    return LabelTensors(
        src_sq=torch.randint(0, N_SQUARES, (batch_size,), dtype=torch.long, device=device),
        tgt_sq=torch.randint(0, N_SQUARES, (batch_size,), dtype=torch.long, device=device),
        opp_src_sq=torch.randint(0, N_SQUARES, (batch_size,), dtype=torch.long, device=device),
        opp_tgt_sq=torch.randint(0, N_SQUARES, (batch_size,), dtype=torch.long, device=device),
    )


def make_prediction_output(batch_size: int = 4, device: str = "cpu") -> PredictionOutput:
    """Return random PredictionOutput logit tensors for loss testing.

    Args:
        batch_size: Number of examples. Default 4.
        device: Torch device string. Default 'cpu'.

    Returns:
        PredictionOutput with four [B, 64] float tensors.
    """
    return PredictionOutput(
        src_sq_logits=torch.randn(
            batch_size, N_SQUARES, device=device,
            requires_grad=True,
        ),
        tgt_sq_logits=torch.randn(
            batch_size, N_SQUARES, device=device,
            requires_grad=True,
        ),
        opp_src_sq_logits=torch.randn(
            batch_size, N_SQUARES, device=device,
            requires_grad=True,
        ),
        opp_tgt_sq_logits=torch.randn(
            batch_size, N_SQUARES, device=device,
            requires_grad=True,
        ),
    )


# ---------------------------------------------------------------------------
# PGN fixture
# ---------------------------------------------------------------------------

_SMALL_PGN = """\
[Event "Test"]
[Site "?"]
[Date "2024.01.01"]
[Round "?"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 1-0

[Event "Test"]
[Site "?"]
[Date "2024.01.01"]
[Round "?"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1

[Event "Test"]
[Site "?"]
[Date "2024.01.01"]
[Round "?"]
[White "Player5"]
[Black "Player6"]
[Result "1/2-1/2"]

1. e4 e5 2. Ke2 Ke7 1/2-1/2
"""


def make_small_pgn_fixture() -> str:
    """Return a valid PGN string containing exactly 3 simple games.

    Returns:
        PGN-formatted string suitable for testing StreamingPGNReader.

    Example:
        >>> pgn = make_small_pgn_fixture()
        >>> pgn.count("[Event")
        3
    """
    return _SMALL_PGN


def parse_pgn_games(pgn_text: str) -> list[chess.pgn.Game]:
    """Parse all games from a PGN string and return them as a list.

    Args:
        pgn_text: A PGN-formatted string containing one or more games.

    Returns:
        List of chess.pgn.Game objects.

    Example:
        >>> games = parse_pgn_games(make_small_pgn_fixture())
        >>> len(games)
        3
    """
    games: list[chess.pgn.Game] = []
    pgn_io = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        games.append(game)
    return games


def make_training_examples(n: int = 10) -> list[TrainingExample]:
    """Return n synthetic TrainingExample namedtuples for dataset testing.

    Args:
        n: Number of examples to generate. Default 10.

    Returns:
        List of TrainingExample namedtuples with random but valid values.

    Example:
        >>> examples = make_training_examples(10)
        >>> len(examples)
        10
    """
    examples: list[TrainingExample] = []
    for _ in range(n):
        board_tokens = [0] + [int(torch.randint(1, 8, (1,)).item()) for _ in range(64)]
        color_tokens = [0] + [int(torch.randint(0, 3, (1,)).item()) for _ in range(64)]
        activity_tokens = [0] * 65
        examples.append(TrainingExample(
            board_tokens=board_tokens,
            color_tokens=color_tokens,
            activity_tokens=activity_tokens,
            src_sq=int(torch.randint(0, 64, (1,)).item()),
            tgt_sq=int(torch.randint(0, 64, (1,)).item()),
            opp_src_sq=int(torch.randint(0, 64, (1,)).item()),
            opp_tgt_sq=int(torch.randint(0, 64, (1,)).item()),
        ))
    return examples


def make_initial_board_tokens() -> tuple[list[int], list[int]]:
    """Return known-good token lists for the chess starting position as White.

    Token layout (board_tokens):
      Index 0: 0 (CLS). Indices 1-64: piece types for squares a1-h8.
      a1(idx 1)=5(ROOK), b1(2)=3(KNIGHT), c1(3)=4(BISHOP), d1(4)=6(QUEEN),
      e1(5)=7(KING), f1(6)=4(BISHOP), g1(7)=3(KNIGHT), h1(8)=5(ROOK),
      a2-h2(9-16)=2(PAWN), a7-h7(49-56)=2(PAWN),
      a8(57)=5(ROOK), b8(58)=3(KNIGHT), c8(59)=4(BISHOP), d8(60)=6(QUEEN),
      e8(61)=7(KING), f8(62)=4(BISHOP), g8(63)=3(KNIGHT), h8(64)=5(ROOK).

    Color tokens (as White to move):
      Player (White) pieces: color=1. Opponent (Black) pieces: color=2. Empty: 0.

    Returns:
        Tuple (board_tokens, color_tokens), each of length 65.

    Example:
        >>> bt, ct = make_initial_board_tokens()
        >>> bt[0]   # CLS
        0
        >>> bt[1]   # a1 = White Rook
        5
        >>> ct[1]   # White's piece -> PLAYER
        1
    """
    # Build ground truth using python-chess for correctness.
    board = chess.Board()
    piece_map = {
        chess.PAWN: 2, chess.KNIGHT: 3, chess.BISHOP: 4,
        chess.ROOK: 5, chess.QUEEN: 6, chess.KING: 7,
    }
    board_tokens: list[int] = [0]  # CLS
    color_tokens: list[int] = [0]  # CLS
    for sq in range(64):  # a1=0 .. h8=63
        piece = board.piece_at(sq)
        if piece is None:
            board_tokens.append(1)  # EMPTY
            color_tokens.append(0)
        else:
            board_tokens.append(piece_map[piece.piece_type])
            # White is to move → White pieces are PLAYER, Black are OPPONENT.
            color_tokens.append(1 if piece.color == chess.WHITE else 2)
    return board_tokens, color_tokens
