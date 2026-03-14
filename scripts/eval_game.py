"""Evaluate the model on a single unseen grandmaster game (greedy top-1).

Replays Kasparov vs Topalov, Wijk aan Zee 1999 ("Kasparov's Immortal")
position by position, predicting each move via argmax and comparing to
the move actually played. Reports per-move results and per-side accuracy.

Usage:
    python -m scripts.eval_game \
        --config configs/train_rl_v4_100k.yaml \
        --checkpoint checkpoints/chess_rl_v4_100k.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import torch

from chess_sim.config import load_pgn_rl_config
from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.move_vocab import SOS_IDX
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.env.agent_adapter import _compute_trajectory_tokens
from chess_sim.model.chess_model import ChessModel

# Kasparov vs Topalov, Wijk aan Zee 1999 — "Kasparov's Immortal"
# Definitively outside the 2013 Lichess online-rated-game training corpus.
_GAME_MOVES: list[str] = [
    "e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "c1e3", "f8g7",
    "d1d2", "c7c6", "f2f3", "b7b5", "g1e2", "b8d7", "e3h6", "g7h6",
    "d2h6", "c8b7", "a2a3", "e7e5", "e1c1", "d8e7", "c1b1", "a7a6",
    "e2c1", "e8c8", "c1b3", "e5d4", "d1d4", "c6c5", "d4d1", "d7b6",
    "g2g3", "c8b8", "b3a5", "b7a8", "f1h3", "d6d5", "h6f4", "b8a7",
    "h1e1", "d5d4", "c3d5", "b6d5", "e4d5", "e7d6", "d1d4", "c5d4",
    "e1e7", "a7b6", "f4d4", "b6a5", "b2b4", "a5a4", "d4c3", "d6d5",
    "e7a7", "a8b7", "a7b7", "d5c4", "c3f6", "a4a3", "f6a6", "a3b4",
    "c2c3", "b4c3", "a6a1", "c3d2", "a1b2", "d2d1", "h3f1", "d8d2",
    "b7d7", "d2d7", "f1c4", "b5c4", "b2h8", "d7d3", "h8a8",
]

_ILLEGAL_FILL: float = float("-inf")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Greedy move-prediction eval on Kasparov vs Topalov 1999."
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    return p


def _load_model(
    cfg_path: Path,
    ckpt_path: Path,
    device: torch.device,
) -> ChessModel:
    """Load ChessModel from config + checkpoint."""
    cfg = load_pgn_rl_config(cfg_path)
    model = ChessModel(cfg.model, cfg.decoder).to(device)
    ckpt = torch.load(
        ckpt_path, map_location=device, weights_only=True
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def _predict(
    model: ChessModel,
    board: chess.Board,
    move_history: list[str],
    tokenizer: BoardTokenizer,
    move_tok: MoveTokenizer,
    device: torch.device,
) -> tuple[str, float]:
    """Greedy top-1 prediction for the current position.

    Returns:
        Tuple of (predicted_uci, top_probability).
    """
    tb = tokenizer.tokenize(board, board.turn)
    traj = _compute_trajectory_tokens(move_history)

    bt = torch.tensor(
        tb.board_tokens, dtype=torch.long, device=device
    ).unsqueeze(0)
    ct = torch.tensor(
        tb.color_tokens, dtype=torch.long, device=device
    ).unsqueeze(0)
    tt = torch.tensor(
        traj, dtype=torch.long, device=device
    ).unsqueeze(0)
    prefix = torch.tensor(
        [[SOS_IDX]], dtype=torch.long, device=device
    )
    move_colors = torch.zeros(
        1, 1, dtype=torch.long, device=device
    )

    logits = model(bt, ct, tt, prefix, None, move_colors)
    last = logits[0, -1, :]  # [V]

    legal = [m.uci() for m in board.legal_moves]
    mask = move_tok.build_legal_mask(legal).to(device)
    last = last.masked_fill(~mask, _ILLEGAL_FILL)

    probs = torch.softmax(last, dim=-1)
    idx = int(last.argmax().item())
    return move_tok.decode(idx), float(probs[idx].item())  # type: ignore[arg-type]


def main() -> None:
    args = _build_parser().parse_args()
    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}\n")

    model = _load_model(cfg_path, ckpt_path, device)
    tokenizer = BoardTokenizer()
    move_tok = MoveTokenizer()

    board = chess.Board()
    move_history: list[str] = []

    white_correct = white_total = 0
    black_correct = black_total = 0

    print(
        f"{'Ply':>4}  {'Side':<6}  {'Actual':<8}  "
        f"{'Predicted':<10}  {'OK':>2}  {'p(pred)':>8}"
    )
    print("-" * 52)

    for ply, uci in enumerate(_GAME_MOVES, start=1):
        is_white = board.turn == chess.WHITE
        side = "White" if is_white else "Black"

        predicted, prob = _predict(
            model, board, move_history, tokenizer, move_tok, device
        )
        correct = predicted == uci
        mark = "✓" if correct else "✗"

        if is_white:
            white_total += 1
            white_correct += int(correct)
        else:
            black_total += 1
            black_correct += int(correct)

        print(
            f"{ply:>4}  {side:<6}  {uci:<8}  "
            f"{predicted:<10}  {mark:>2}  {prob:>7.1%}"
        )

        move = chess.Move.from_uci(uci)
        board.push(move)
        move_history.append(uci)

    total = white_total + black_total
    total_correct = white_correct + black_correct
    print("\n" + "=" * 52)
    print(
        f"White : {white_correct}/{white_total} "
        f"({white_correct/max(white_total,1):.1%})"
    )
    print(
        f"Black : {black_correct}/{black_total} "
        f"({black_correct/max(black_total,1):.1%})"
    )
    print(
        f"Overall: {total_correct}/{total} "
        f"({total_correct/max(total,1):.1%})"
    )


if __name__ == "__main__":
    main()
