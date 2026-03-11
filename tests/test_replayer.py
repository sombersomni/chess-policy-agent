"""Tests for PGNReplayer material delta tracking.

T-CR9, T-CR15: verify material_delta field is computed correctly
from the _material_of helper across various capture scenarios.
"""
from __future__ import annotations

import unittest

from chess_sim.training.pgn_replayer import PGNReplayer


class TestPGNReplayerMaterialDelta(unittest.TestCase):
    """Tests for material_delta field in OfflinePlyTuple."""

    def setUp(self) -> None:
        """Initialize replayer."""
        self.replayer = PGNReplayer()

    def test_non_capture_move_delta_zero(self) -> None:
        """T-CR15: Pawn push -> material_delta = 0.0."""
        raise NotImplementedError("To be implemented")

    def test_own_capture_positive_delta(self) -> None:
        """Knight capture -> material_delta = +3.0 on that ply."""
        raise NotImplementedError("To be implemented")

    def test_opponent_capture_negative_delta(self) -> None:
        """White ply after black took white's rook -> delta = -5.0."""
        raise NotImplementedError("To be implemented")

    def test_first_ply_delta_zero(self) -> None:
        """First ply of game -> material_delta = 0.0."""
        raise NotImplementedError("To be implemented")

    def test_en_passant_capture_delta(self) -> None:
        """T-CR9: En passant -> material_delta = +1.0."""
        raise NotImplementedError("To be implemented")


if __name__ == "__main__":
    unittest.main()
