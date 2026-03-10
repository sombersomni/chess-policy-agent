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
_SRC_COLOR = "#F6F669"      # yellow     — actual move source
_TGT_COLOR = "#BACA44"      # green      — actual move target
_PRED_COLOR = "#88BBDD"     # blue       — top-1 predicted target (when wrong)
_PRED_SRC_COLOR = "#FFA500"  # orange    — top-1 predicted source square
_OPP_FROM_COLOR = "#E87272"  # muted red — opponent's previous from-square
_OPP_TO_COLOR = "#C03030"    # dark red  — opponent's previous to-square

_PIECE_NAMES: dict[int, str] = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def _sq_name(board: chess.Board, sq: int) -> str:
    """Return piece letter at sq, or '?' if empty."""
    piece = board.piece_at(sq)
    return _PIECE_NAMES.get(piece.piece_type, "?") if piece else "?"


def render_rl_ply(
    board: chess.Board,
    actual_uci: str,
    top_k: list[tuple[str, float]],
    reward: float,
    entropy: float,
    ply_idx: int,
    *,
    opp_from_sq: int | None = None,
    opp_to_sq: int | None = None,
    train_accuracy: float | None = None,
    q_value: float | None = None,
    advantage: float | None = None,
    is_winner_ply: bool | None = None,
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
        opp_from_sq: Opponent's previous from-square (chess.Square int).
        opp_to_sq: Opponent's previous to-square (chess.Square int).
        train_accuracy: Latest validation accuracy (0-1) for stats box.
        q_value: Q-function estimate for the actual move.
        advantage: Raw advantage (reward - q_value).
        is_winner_ply: True if this ply belongs to the winning side.

    Returns:
        matplotlib Figure with two subplots (board left, chart right).
    """
    fig = plt.figure(figsize=(13, 5.5))
    ax_board = fig.add_subplot(1, 2, 1)
    ax_pred = fig.add_subplot(1, 2, 2)

    _draw_board(
        ax_board, board, actual_uci, top_k,
        opp_from_sq=opp_from_sq,
        opp_to_sq=opp_to_sq,
    )
    _draw_predictions(
        ax_pred, board, actual_uci, top_k, reward, entropy,
        train_accuracy=train_accuracy,
        q_value=q_value,
        advantage=advantage,
        is_winner_ply=is_winner_ply,
    )

    # Game phase from non-king piece count (rough heuristic).
    non_king = sum(
        1 for sq in chess.SQUARES
        if board.piece_at(sq) is not None
        and board.piece_at(sq).piece_type != chess.KING  # type: ignore[union-attr]
    )
    phase = (
        "opening" if non_king > 20
        else ("endgame" if non_king <= 8 else "mid")
    )
    side = "White" if board.turn == chess.WHITE else "Black"
    ply_tag = (
        (" [W]" if is_winner_ply else " [L]")
        if is_winner_ply is not None else ""
    )
    fig.suptitle(
        f"Ply {ply_idx} ({side}{ply_tag}, {phase}) | "
        f"R={reward:+.3f} | H={entropy:.3f}",
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
    *,
    opp_from_sq: int | None = None,
    opp_to_sq: int | None = None,
) -> None:
    """Draw the 8x8 board with piece symbols and move highlights.

    Args:
        ax: Axes to draw onto.
        board: Board state before the move.
        actual_uci: UCI of the actual move.
        top_k: Top-K predictions for optional target highlight.
        opp_from_sq: Opponent's previous from-square (chess.Square int).
        opp_to_sq: Opponent's previous to-square (chess.Square int).
    """
    actual_move = chess.Move.from_uci(actual_uci)
    src_sq = actual_move.from_square
    tgt_sq = actual_move.to_square

    # Build priority color map — last written wins; actual move is highest.
    sq_colors: dict[int, str] = {}
    if opp_from_sq is not None:
        sq_colors[opp_from_sq] = _OPP_FROM_COLOR
    if opp_to_sq is not None:
        sq_colors[opp_to_sq] = _OPP_TO_COLOR

    pred_src: int | None = None
    pred_tgt: int | None = None
    if top_k:
        try:
            pm = chess.Move.from_uci(top_k[0][0])
            pred_src = pm.from_square
            pred_tgt = pm.to_square
            if pred_src != src_sq:
                sq_colors[pred_src] = _PRED_SRC_COLOR
            if pred_tgt != tgt_sq:
                sq_colors[pred_tgt] = _PRED_COLOR
        except (ValueError, IndexError):
            pass

    sq_colors[src_sq] = _SRC_COLOR  # actual move always wins
    sq_colors[tgt_sq] = _TGT_COLOR

    for rank in range(8):
        for file in range(8):
            sq = chess.square(file, rank)
            is_light = (rank + file) % 2 == 0
            color = sq_colors.get(sq, _LIGHT_SQ if is_light else _DARK_SQ)

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

    # Build conditional legend — only show entries that appear on board.
    legend = [
        mpatches.Patch(color=_SRC_COLOR, label="Actual src"),
        mpatches.Patch(color=_TGT_COLOR, label="Actual tgt"),
    ]
    if pred_src is not None and pred_src != src_sq:
        legend.append(mpatches.Patch(color=_PRED_SRC_COLOR, label="Pred src"))
    if pred_tgt is not None and pred_tgt != tgt_sq:
        legend.append(mpatches.Patch(color=_PRED_COLOR, label="Pred tgt"))
    if opp_from_sq is not None:
        legend.append(mpatches.Patch(color=_OPP_FROM_COLOR, label="Opp from"))
    if opp_to_sq is not None:
        legend.append(mpatches.Patch(color=_OPP_TO_COLOR, label="Opp to"))
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
    board: chess.Board,
    actual_uci: str,
    top_k: list[tuple[str, float]],
    reward: float,
    entropy: float,
    *,
    train_accuracy: float | None = None,
    q_value: float | None = None,
    advantage: float | None = None,
    is_winner_ply: bool | None = None,
) -> None:
    """Draw a horizontal bar chart of top-K predicted move probabilities.

    Args:
        ax: Axes to draw onto.
        board: Board state (used to annotate bars with piece letters).
        actual_uci: UCI of the actual move (highlighted green).
        top_k: List of (uci, probability) pairs, sorted descending.
        reward: Reward value shown in subtitle.
        entropy: Entropy value shown in subtitle.
        train_accuracy: Latest validation accuracy (0-1) for text box.
        q_value: Q-function estimate for the actual move.
        advantage: Raw advantage (reward - q_value).
        is_winner_ply: True if this ply belongs to the winning side.
    """
    if not top_k:
        ax.text(
            0.5, 0.5, "No predictions",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=10,
        )
        return

    def _label(uci: str) -> str:
        """Prefix UCI with the moving piece letter."""
        try:
            m = chess.Move.from_uci(uci)
            return f"{_sq_name(board, m.from_square)} {uci}"
        except ValueError:
            return uci

    moves = [_label(m) for m, _ in top_k]
    raw_ucis = [m for m, _ in top_k]
    probs = np.array([p for _, p in top_k], dtype=float)
    colors = [
        "#4CAF50" if m == actual_uci else "#2196F3"
        for m in raw_ucis
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
    for i, prob in enumerate(probs):
        ax.text(
            prob + xlim * 0.01,
            i,
            f"{prob:.3f}",
            va="center",
            fontsize=8,
        )

    # Stats text box: accuracy, Q-value, advantage, winner/loser tag.
    stats: list[str] = []
    if train_accuracy is not None:
        stats.append(f"Train acc: {train_accuracy:.1%}")
    if q_value is not None:
        stats.append(f"Q-value:  {q_value:+.3f}")
    if advantage is not None:
        stats.append(f"Advantage:{advantage:+.3f}")
    if is_winner_ply is not None:
        stats.append("WINNER ply" if is_winner_ply else "loser ply")
    if stats:
        ax.text(
            0.02, 0.02,
            "\n".join(stats),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="lightyellow",
                edgecolor="#AAAAAA",
                alpha=0.85,
            ),
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
