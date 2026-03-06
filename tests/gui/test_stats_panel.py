"""Tests SP01-SP02: pure formatting helpers in stats_panel."""
from __future__ import annotations

import math
import unittest

from scripts.gui.formatters import _fmt_acc, _fmt_entropy, _fmt_loss


class TestFmtLoss(unittest.TestCase):
    """SP01: _fmt_loss formats to 4 decimal places."""

    def test_sp01_four_decimal_places(self) -> None:
        self.assertEqual(_fmt_loss(3.141592), "3.1416")

    def test_sp01_zero(self) -> None:
        self.assertEqual(_fmt_loss(0.0), "0.0000")

    def test_sp01_large_value(self) -> None:
        self.assertEqual(_fmt_loss(17.09876), "17.0988")


class TestFmtAcc(unittest.TestCase):
    """SP02: _fmt_acc maps 1/0/-1 to symbols."""

    def test_sp02_correct(self) -> None:
        self.assertEqual(_fmt_acc(1), "\u2713")

    def test_sp02_wrong(self) -> None:
        self.assertEqual(_fmt_acc(0), "\u2717")

    def test_sp02_ignored(self) -> None:
        self.assertEqual(_fmt_acc(-1), "\u2013")


class TestFmtEntropy(unittest.TestCase):
    """SP01-variant: _fmt_entropy formats to 4 decimal places."""

    def test_entropy_four_decimal_places(self) -> None:
        result = _fmt_entropy(math.log(64))
        self.assertEqual(result, "4.1589")


if __name__ == "__main__":
    unittest.main()
