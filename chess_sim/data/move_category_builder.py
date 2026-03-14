"""MoveCategoryBuilder: classify a UCI move into 7 categories.

Pure python-chess, no ML dependencies. Given a UCI move string
and the board state BEFORE the move is applied, returns an
integer 0-6 identifying the move category.

Priority order: PROMOTION > CASTLE > CAPTURE_* > QUIET.

Example:
    >>> import chess
    >>> build("e2e4", chess.Board())
    0
"""
from __future__ import annotations

import chess
import torch
from torch import Tensor

QUIET: int = 0
CAPTURE_PAWN: int = 1
CAPTURE_MINOR: int = 2
CAPTURE_ROOK: int = 3
CAPTURE_QUEEN: int = 4
CASTLE: int = 5
PROMOTION: int = 6

_CAPTURE_CATEGORY: dict[chess.PieceType, int] = {
    chess.PAWN: CAPTURE_PAWN,
    chess.KNIGHT: CAPTURE_MINOR,
    chess.BISHOP: CAPTURE_MINOR,
    chess.ROOK: CAPTURE_ROOK,
    chess.QUEEN: CAPTURE_QUEEN,
    chess.KING: QUIET,  # king capture shouldn't happen
}


def build(uci_move: str, board: chess.Board) -> int:
    """Classify a UCI move into one of 7 categories.

    The board must reflect the state BEFORE the move is applied.
    Priority: PROMOTION > CASTLE > CAPTURE_* > QUIET.

    Args:
        uci_move: UCI move string (e.g. "e2e4", "e7e8q").
        board: Board state before the move.

    Returns:
        Integer 0-6 indicating the move category.

    Example:
        >>> import chess
        >>> build("e1g1", chess.Board(
        ...     "r1bqkbnr/pppppppp/2n5/8/8/5N2/PPPPPPPP/"
        ...     "RNBQKB1R w KQkq - 2 2"))
        0
    """
    move = chess.Move.from_uci(uci_move)

    # Highest priority: promotion
    if move.promotion is not None:
        return PROMOTION

    # Second priority: castling
    if board.is_castling(move):
        return CASTLE

    # Third priority: captures (including en passant)
    captured = board.piece_at(move.to_square)
    if captured is not None:
        return _CAPTURE_CATEGORY.get(
            captured.piece_type, QUIET
        )

    # En passant: no piece on to_square but it's a capture
    if board.is_en_passant(move):
        return CAPTURE_PAWN

    return QUIET


# Castling UCIs for both sides — board-state-independent
_CASTLE_UCIS: frozenset[str] = frozenset(
    ["e1g1", "e1c1", "e8g8", "e8c8"]
)

# Piece type constants matching the board tensor encoding
# (0=CLS, 1=EMPTY, 2=PAWN, 3=KNIGHT, 4=BISHOP, 5=ROOK, 6=QUEEN, 7=KING)
_PIECE_TO_CAPTURE_CATEGORY: list[int] = [
    QUIET,          # 0 CLS  — unused
    QUIET,          # 1 EMPTY — shouldn't capture empty
    CAPTURE_PAWN,   # 2 PAWN
    CAPTURE_MINOR,  # 3 KNIGHT
    CAPTURE_MINOR,  # 4 BISHOP
    CAPTURE_ROOK,   # 5 ROOK
    CAPTURE_QUEEN,  # 6 QUEEN
    QUIET,          # 7 KING  — shouldn't happen
]


def _build_base_tables(
    vocab_size: int = 1971,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build static per-vocab LUTs: base_category, move_src, move_dst.

    Returns:
        base_category: LongTensor [vocab_size] — PROMOTION(6), CASTLE(5), or 0.
        move_src: LongTensor [vocab_size] — from-square (0-63), -1 for specials.
        move_dst: LongTensor [vocab_size] — to-square (0-63), -1 for specials.
    """
    from chess_sim.data.move_vocab import MoveVocab

    vocab = MoveVocab()
    base_cat = torch.zeros(vocab_size, dtype=torch.long)
    move_src = torch.full((vocab_size,), -1, dtype=torch.long)
    move_dst = torch.full((vocab_size,), -1, dtype=torch.long)

    for idx in range(3, vocab_size):
        uci = vocab.decode(idx)
        if uci is None:
            continue
        move_src[idx] = chess.Move.from_uci(uci).from_square
        move_dst[idx] = chess.Move.from_uci(uci).to_square
        if len(uci) == 5:          # promotion
            base_cat[idx] = PROMOTION
        elif uci in _CASTLE_UCIS:  # castling
            base_cat[idx] = CASTLE

    return base_cat, move_src, move_dst


# Module-level LUTs — built once on first import
_BASE_CATEGORY, _MOVE_SRC, _MOVE_DST = _build_base_tables()
_PIECE_CAT_LUT: Tensor = torch.tensor(
    _PIECE_TO_CAPTURE_CATEGORY, dtype=torch.long
)


def batch_build(
    target_moves: Tensor,
    board: Tensor,
) -> Tensor:
    """Classify a batch of moves using board tensor + static LUTs.

    Derives move category per sample using the board tensor to detect
    captures. Promotions and castling are determined from UCI string alone.

    Args:
        target_moves: LongTensor [B] — vocab indices of played moves.
        board: Float tensor [B, 65, 3]. Channel 0=piece_type (1=empty,
            2=PAWN, …), channel 1=color (0=empty, 1=player, 2=opponent).
            Index 0 is the CLS token; squares are indices 1-64.

    Returns:
        LongTensor [B] — category per sample (0-6).

    Example:
        >>> import torch
        >>> from chess_sim.data.move_vocab import MoveVocab
        >>> vocab = MoveVocab()
        >>> idx = torch.tensor([vocab.encode("e2e4")])
        >>> board = torch.ones(1, 65, 3)  # all pieces, dummy
        >>> batch_build(idx, board).shape
        torch.Size([1])
    """
    device = target_moves.device
    base_cat = _BASE_CATEGORY.to(device)
    move_dst = _MOVE_DST.to(device)
    piece_cat = _PIECE_CAT_LUT.to(device)

    # Start from base categories (handles promotion + castle)
    cats = base_cat[target_moves].clone()  # [B]

    # For moves with base_cat == 0 (not yet classified), check captures
    undecided = cats == 0  # [B] bool
    if not undecided.any():
        return cats

    move_src = _MOVE_SRC.to(device)

    # Source / destination square slots (sq+1 for CLS offset at index 0)
    dst_sqs = move_dst[target_moves]                    # [B]
    src_sqs = move_src[target_moves]                    # [B]
    dst_slots = (dst_sqs + 1).clamp(min=0)              # [B]
    src_slots = (src_sqs + 1).clamp(min=0)              # [B]

    b_idx = torch.arange(board.size(0), device=device)
    dst_color = board[b_idx, dst_slots, 1].long()       # [B] 0/1/2
    dst_piece = board[b_idx, dst_slots, 0].long()       # [B] 0-7
    src_piece = board[b_idx, src_slots, 0].long()       # [B] 0-7

    # Opponent piece at destination → regular capture
    is_capture = undecided & (dst_color == 2)
    cap_cats = piece_cat[dst_piece.clamp(min=0, max=7)]  # [B]
    cats = torch.where(is_capture, cap_cats, cats)

    # En passant: pawn (piece type 2) moves diagonally to empty square
    dst_file = dst_sqs % 8
    src_file = src_sqs % 8
    is_ep = (
        undecided
        & ~is_capture
        & (dst_color == 0)          # destination is empty
        & (src_piece == 2)          # moving piece is PAWN
        & (dst_file != src_file)    # diagonal move
    )
    cats = torch.where(is_ep, torch.full_like(cats, CAPTURE_PAWN), cats)

    return cats
