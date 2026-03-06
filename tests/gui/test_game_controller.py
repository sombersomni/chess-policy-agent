"""Tests GC01-GC05: GameController loads, serves boards and metrics."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import chess
import numpy as np

from scripts.evaluate import StepResult
from scripts.gui.game_controller import GameController

PGN_PATH = Path("data/games.pgn")
CKPT_PATH = Path("checkpoints/winner_run_01.pt")


class TestGameController(unittest.TestCase):
    """GC01-GC05: shared controller loaded once in setUpClass."""

    _controller: GameController

    @classmethod
    def setUpClass(cls) -> None:
        cls._controller = GameController()
        cls._controller.load(PGN_PATH, CKPT_PATH, game_index=0)

    def test_GC01_load_no_exception_total_plies_positive(self) -> None:
        """GC01: load() succeeds and total_plies() > 0."""
        self.assertGreater(self._controller.total_plies(), 0)

    def test_GC02_board_at_zero_is_starting_position(self) -> None:
        """GC02: board_at(0) returns starting position."""
        board = self._controller.board_at(0)
        self.assertEqual(board.fen(), chess.Board().fen())

    def test_GC03_board_at_all_plies_no_index_error(self) -> None:
        """GC03: board_at(ply) works for all plies; boards differ for ply > 0."""
        total = self._controller.total_plies()
        for ply in range(total):
            board = self._controller.board_at(ply)
            self.assertIsInstance(board, chess.Board)
        if total > 1:
            self.assertNotEqual(
                self._controller.board_at(0).fen(),
                self._controller.board_at(1).fen(),
            )

    def test_GC04_step_result_fields_finite(self) -> None:
        """GC04: step_result_at(0) returns StepResult with finite floats."""
        result = self._controller.step_result_at(0)
        self.assertIsInstance(result, StepResult)
        for field in (
            result.loss_src, result.loss_tgt,
            result.entropy_src, result.entropy_tgt, result.mean_entropy,
        ):
            self.assertTrue(math.isfinite(field), f"Non-finite field: {field}")

    def test_GC05_piece_embeddings_shape(self) -> None:
        """GC05: piece_embeddings() returns ndarray of shape (8, 256)."""
        emb = self._controller.piece_embeddings()
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (8, 256))


if __name__ == "__main__":
    unittest.main()
