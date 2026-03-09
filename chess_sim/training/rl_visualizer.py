"""RL training visualizer: chess board + move prediction rendering.

Produces matplotlib figures showing:
- Left panel: chess board with actual move highlighted in yellow/green,
  top-1 predicted target highlighted in blue (when wrong).
- Right panel: horizontal probability bar chart of top-K predicted
  moves with actual move marked in green.

Intended for use in PGNRLTrainer.sample_visuals() and logged to Aim
via tracker.track_image().
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import chess
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

matplotlib.use("Agg")

# Unicode chess piece symbols keyed by (piece_type, is_white)
_PIECE_UNICODE: dict[tuple[int, bool], str] = {
    (chess.PAWN, True): "\u2659",
    (chess.KNIGHT, True): "\u2658",
    (chess.BISHOP, True): "\u2657",
    (chess.ROOK, True): "\u2656",
    (chess.QUEEN, True): "\u2655",
    (chess.KING, True): "\u2654",
    (chess.PAWN, False): "\u265f",
    (chess.KNIGHT, False): "\u265e",
    (chess.BISHOP, False): "\u265d",
    (chess.ROOK, False): "\u265c",
    (chess.QUEEN, False): "\u265b",
    (chess.KING, False): "\u265a",
}

_LIGHT_SQ = "#F0D9B5"
_DARK_SQ = "#B58863"
_SRC_COLOR = "#F6F669"   # yellow  — actual move source
_TGT_COLOR = "#BACA44"   # green   — actual move target
_PRED_COLOR = "#88BBDD"  # blue    — top-1 predicted target (when wrong)


def render_rl_ply(
    board: chess.Board,
    actual_uci: str,
    top_k: list[tuple[str, float]],
    reward: float,
    entropy: float,
    ply_idx: int,
) -> "Figure":
    """Render a single ply as a board + prediction chart figure.

    Args:
        board: chess.Board state BEFORE the move is applied.
        actual_uci: UCI string of the move actually played.
        top_k: List of (uci, probability) pairs sorted descending.
            Probabilities should sum to ~1 over the top-K slice.
        reward: Scalar reward assigned to this ply.
        entropy: Shannon entropy (nats) of the full action distribution.
        ply_idx: Zero-based ply index within the game (for labelling).

    Returns:
        matplotlib Figure with two subplots (board left, chart right).
    """
    fig = plt.figure(figsize=(13, 5.5))
    ax_board = fig.add_subplot(1, 2, 1)
    ax_pred = fig.add_subplot(1, 2, 2)

    _draw_board(ax_board, board, actual_uci, top_k)
    _draw_predictions(ax_pred, actual_uci, top_k, reward, entropy)

    side = "White" if board.turn == chess.WHITE else "Black"
    fig.suptitle(
        f"Ply {ply_idx} ({side} to move) | "
        f"Reward={reward:+.3f} | H={entropy:.3f} nats",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _draw_board(
    ax: plt.Axes,
    board: chess.Board,
    actual_uci: str,
    top_k: list[tuple[str, float]],
) -> None:
    """Draw the 8x8 board with piece symbols and move highlights.

    Args:
        ax: Axes to draw onto.
        board: Board state before the move.
        actual_uci: UCI of the actual move.
        top_k: Top-K predictions for optional target highlight.
    """
    actual_move = chess.Move.from_uci(actual_uci)
    src_sq = actual_move.from_square
    tgt_sq = actual_move.to_square

    pred_tgt: int | None = None
    if top_k:
        try:
            pred_move = chess.Move.from_uci(top_k[0][0])
            pred_tgt = pred_move.to_square
        except (ValueError, IndexError):
            pass

    for rank in range(8):
        for file in range(8):
            sq = chess.square(file, rank)
            is_light = (rank + file) % 2 == 0
            color = _LIGHT_SQ if is_light else _DARK_SQ

            if sq == src_sq:
                color = _SRC_COLOR
            elif sq == tgt_sq:
                color = _TGT_COLOR
            elif pred_tgt is not None and sq == pred_tgt and sq != tgt_sq:
                color = _PRED_COLOR

            rect = mpatches.Rectangle(
                (file, rank), 1, 1,
                linewidth=0,
                facecolor=color,
            )
            ax.add_patch(rect)

            piece = board.piece_at(sq)
            if piece is not None:
                sym = _PIECE_UNICODE.get(
                    (piece.piece_type, piece.color), "?"
                )
                ax.text(
                    file + 0.5,
                    rank + 0.5,
                    sym,
                    ha="center",
                    va="center",
                    fontsize=17,
                    fontfamily="DejaVu Sans",
                )

    files = "abcdefgh"
    for i in range(8):
        ax.text(
            i + 0.5, -0.35, files[i],
            ha="center", va="center", fontsize=8, color="#555555",
        )
        ax.text(
            -0.35, i + 0.5, str(i + 1),
            ha="center", va="center", fontsize=8, color="#555555",
        )

    # Legend patches
    legend = [
        mpatches.Patch(color=_SRC_COLOR, label="Actual src"),
        mpatches.Patch(color=_TGT_COLOR, label="Actual tgt"),
        mpatches.Patch(color=_PRED_COLOR, label="Pred tgt (wrong)"),
    ]
    ax.legend(
        handles=legend, fontsize=7,
        loc="upper right", framealpha=0.8,
    )

    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Board  (actual: {actual_uci})", fontsize=10)


def _draw_predictions(
    ax: plt.Axes,
    actual_uci: str,
    top_k: list[tuple[str, float]],
    reward: float,
    entropy: float,
) -> None:
    """Draw a horizontal bar chart of top-K predicted move probabilities.

    Args:
        ax: Axes to draw onto.
        actual_uci: UCI of the actual move (highlighted green).
        top_k: List of (uci, probability) pairs, sorted descending.
        reward: Reward value shown in subtitle.
        entropy: Entropy value shown in subtitle.
    """
    if not top_k:
        ax.text(
            0.5, 0.5, "No predictions",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=10,
        )
        return

    moves = [m for m, _ in top_k]
    probs = np.array([p for _, p in top_k], dtype=float)
    colors = [
        "#4CAF50" if m == actual_uci else "#2196F3"
        for m in moves
    ]

    y_pos = np.arange(len(moves))
    ax.barh(
        y_pos, probs,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(moves, fontsize=9)
    ax.set_xlabel("Probability", fontsize=9)
    xlim = float(probs.max()) * 1.25 if probs.size > 0 else 1.0
    ax.set_xlim(0.0, max(xlim, 1e-6))
    ax.invert_yaxis()

    # Annotate each bar with its probability value
    for i, (prob, move) in enumerate(zip(probs, moves)):
        ax.text(
            prob + xlim * 0.01,
            i,
            f"{prob:.3f}",
            va="center",
            fontsize=8,
        )

    legend = [
        mpatches.Patch(color="#4CAF50", label="Actual move"),
        mpatches.Patch(color="#2196F3", label="Other predictions"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    ax.set_title(
        f"Top-{len(top_k)} Predictions\n"
        f"Reward={reward:+.3f}  H={entropy:.3f} nats",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
