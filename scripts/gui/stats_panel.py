"""StatsPanel: right-side per-ply metrics and piece embedding scatter."""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scripts.evaluate import StepResult
from scripts.gui import GameSource
from scripts.gui.formatters import _fmt_acc, _fmt_entropy, _fmt_loss


class StatsPanel(tk.Frame):
    """Right panel: ply info, metrics table, entropy bar, embedding scatter.

    Implements Renderable.

    Sub-widgets (top-to-bottom):
        _ply_label:      tk.Label   "Ply 3 -- Move: e7e5"
        _tree:           ttk.Treeview  4 rows x 4 cols (Head, Loss, Acc, Entropy)
        _entropy_canvas: tk.Canvas  horizontal entropy bar
        _fig_canvas:     FigureCanvasTkAgg  matplotlib piece embedding scatter

    Attributes:
        _controller: GameSource providing metrics per ply.
        _fig: matplotlib Figure for the scatter plot.
        _ax: matplotlib Axes for the scatter plot.
    """

    MAX_ENTROPY: float = math.log(64)   # ~4.158 nats -- uniform over 64 squares

    PIECE_LABELS: list[str] = [
        "CLS", "EMPTY", "PAWN", "KNIGHT",
        "BISHOP", "ROOK", "QUEEN", "KING",
    ]
    PIECE_COLORS: list[str] = [
        "grey", "lightgrey", "gold", "green",
        "blue", "red", "purple", "black",
    ]

    _HEADS: list[str] = ["src", "tgt"]

    def __init__(
        self,
        parent: tk.Widget,
        controller: GameSource,
    ) -> None:
        """Build all sub-widgets and draw the embedding scatter once.

        Args:
            parent: Parent Tk widget.
            controller: GameSource providing metrics and embeddings.
        """
        super().__init__(parent)
        self._controller = controller

        # Ply info label
        self._ply_label = tk.Label(self, text="Ply -- / --", font=("Helvetica", 12))
        self._ply_label.pack(pady=(8, 4))

        # Metrics treeview
        self._tree = ttk.Treeview(
            self,
            columns=("head", "loss", "acc", "entropy"),
            show="headings",
            height=2,
        )
        for col, width, label in [
            ("head", 80, "Head"),
            ("loss", 90, "Loss"),
            ("acc", 60, "Acc"),
            ("entropy", 90, "Entropy"),
        ]:
            self._tree.heading(col, text=label)
            self._tree.column(col, width=width, anchor=tk.CENTER)
        for head in self._HEADS:
            self._tree.insert("", tk.END, iid=head, values=(head, "--", "--", "--"))
        self._tree.pack(padx=8, pady=4)

        # Entropy bar
        entropy_frame = tk.Frame(self)
        entropy_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(entropy_frame, text="Mean Entropy:").pack(side=tk.LEFT)
        self._entropy_canvas = tk.Canvas(entropy_frame, height=20, bg="#DDDDDD")
        self._entropy_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Matplotlib embedding scatter
        self._fig = Figure(figsize=(4, 3), dpi=80)
        self._ax = self._fig.add_subplot(111)
        self._fig_canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._fig_canvas.get_tk_widget().pack(pady=4)

        self._draw_embedding_scatter()

    def render(self) -> None:
        """Read current ply from controller and refresh all sub-widgets."""
        result = self._controller.step_result_at(
            self._controller.current_ply
        )
        self._update_ply_info(result)
        self._update_metrics_table(result)
        self._update_entropy_bar(result)

    def _update_ply_info(self, result: StepResult) -> None:
        """Update the ply label with ply index and move UCI.

        Args:
            result: StepResult for the current ply.
        """
        self._ply_label.config(
            text=(
                f"Ply {result.ply + 1}"
                f" \u2014 Move: {result.move_uci}"
            )
        )

    def _update_metrics_table(self, result: StepResult) -> None:
        """Populate treeview rows with loss, accuracy, and entropy per head.

        Args:
            result: StepResult for the current ply.
        """
        rows = {
            "src": (
                result.loss_src,
                result.acc_src,
                result.entropy_src,
            ),
            "tgt": (
                result.loss_tgt,
                result.acc_tgt,
                result.entropy_tgt,
            ),
        }
        for head, (loss, acc, entropy) in rows.items():
            self._tree.item(
                head,
                values=(
                    head,
                    _fmt_loss(loss),
                    _fmt_acc(acc),
                    _fmt_entropy(entropy),
                ),
            )

    def _update_entropy_bar(self, result: StepResult) -> None:
        """Redraw entropy bar proportional to mean_entropy / MAX_ENTROPY.

        Args:
            result: StepResult for the current ply.
        """
        self._entropy_canvas.delete("all")
        w = self._entropy_canvas.winfo_width()
        if w <= 1:
            w = 400
        ratio = min(
            result.mean_entropy / self.MAX_ENTROPY, 1.0
        )
        bar_w = int(w * ratio)
        self._entropy_canvas.create_rectangle(
            0, 0, bar_w, 20,
            fill="#4CAF50", outline=""
        )
        label = (
            f"{result.mean_entropy:.3f}"
            f" / {self.MAX_ENTROPY:.3f}"
        )
        self._entropy_canvas.create_text(
            w // 2, 10, text=label,
            fill="black", font=("Arial", 9)
        )

    def _draw_embedding_scatter(self) -> None:
        """PCA-project piece_emb [8, 256] -> [8, 2] and draw scatter.

        Called once during __init__. Uses numpy SVD (no sklearn dep).
        """
        emb = self._controller.piece_embeddings()
        X = emb.astype(np.float64)
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(
            X_centered, full_matrices=False
        )
        coords = X_centered @ Vt[:2].T

        self._ax.clear()
        for i, (label, color) in enumerate(
            zip(self.PIECE_LABELS, self.PIECE_COLORS)
        ):
            self._ax.scatter(
                coords[i, 0], coords[i, 1],
                c=color, s=80, zorder=3,
            )
            self._ax.annotate(
                label, (coords[i, 0], coords[i, 1]),
                textcoords="offset points",
                xytext=(4, 4), fontsize=8,
            )
        self._ax.set_title(
            "Piece Embeddings (PCA)", fontsize=9
        )
        self._ax.set_xlabel("PC1", fontsize=8)
        self._ax.set_ylabel("PC2", fontsize=8)
        self._ax.tick_params(labelsize=7)
        self._fig.tight_layout()
        self._fig_canvas.draw()
