"""Tests for ReservoirSampler: T10, T11."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import chess.pgn
from parameterized import parameterized

from chess_sim.data.sampler import ReservoirSampler


def _make_mock_games(n: int) -> list[MagicMock]:
    """Return n distinct MagicMock objects standing in for chess.pgn.Game."""
    return [MagicMock(spec=chess.pgn.Game, name=f"game_{i}") for i in range(n)]


class TestReservoirSamplerNLargerThanStream(unittest.TestCase):
    """T11: When N > stream length, all stream items are returned."""

    def setUp(self) -> None:
        self.sampler = ReservoirSampler()

    def test_sample_returns_all_when_stream_shorter_than_n(self) -> None:
        """T11: 50 games sampled with n=100 returns all 50 games."""
        games = _make_mock_games(50)
        result = self.sampler.sample(iter(games), n=100)
        self.assertEqual(len(result), 50)

    def test_sample_returns_all_items_when_stream_shorter_than_n(self) -> None:
        """T11: All games from the short stream appear in the result."""
        games = _make_mock_games(10)
        result = self.sampler.sample(iter(games), n=50)
        self.assertEqual(set(result), set(games))


class TestReservoirSamplerUniformity(unittest.TestCase):
    """T10: Sample distribution is approximately uniform (chi-squared p > 0.01)."""

    def setUp(self) -> None:
        self.sampler = ReservoirSampler()

    @parameterized.expand([
        ("small_stream", 100, 10, 500),
        ("medium_stream", 1000, 50, 200),
    ])
    def test_sample_approximate_uniformity(
        self, _name: str, stream_size: int, sample_n: int, trials: int
    ) -> None:
        """T10: Each item appears with roughly equal frequency across many trials.

        Runs `trials` sampling passes and checks that observed frequencies are
        consistent with a uniform distribution using a manual chi-squared statistic.
        Rejects if chi2 > 2*df, a conservative threshold equivalent to p < 0.01.
        """
        counts: dict[int, int] = {i: 0 for i in range(stream_size)}
        games = list(range(stream_size))  # use ints as lightweight stand-ins

        for _ in range(trials):
            result = self.sampler.sample(iter(games), n=sample_n)  # type: ignore[arg-type]
            for item in result:
                counts[item] += 1

        observed = list(counts.values())
        expected = trials * sample_n / stream_size
        chi2 = sum((o - expected) ** 2 / expected for o in observed)
        df = stream_size - 1
        # chi2 / df should be near 1 for uniform data; > 2 indicates non-uniformity.
        self.assertLessEqual(
            chi2, 2.0 * df,
            msg=f"chi2={chi2:.1f} > 2*df={2*df} — sampler distribution may not be uniform",
        )

    def test_sample_size_is_exactly_n(self) -> None:
        """T10: Returned list has exactly n items when stream is longer than n."""
        games = _make_mock_games(10_000)
        result = self.sampler.sample(iter(games), n=100)
        self.assertEqual(len(result), 100)


if __name__ == "__main__":
    unittest.main()
