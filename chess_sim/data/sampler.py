"""ReservoirSampler: O(N) uniform random sampling from a stream of unknown length.

Implements Vitter's Algorithm R. Suitable for sampling games from the
StreamingPGNReader without knowing the total game count in advance.
"""

from __future__ import annotations

from typing import Iterator

import chess.pgn

from chess_sim.protocols import Samplable


class ReservoirSampler:
    """Selects N games uniformly at random from a stream using O(N) memory.

    Implements the Samplable protocol using Vitter's Algorithm R.

    Algorithm sketch:
      - Fill the reservoir with the first N games.
      - For each subsequent game i (i >= N): generate j = randint(0, i).
        If j < N, replace reservoir[j] with the current game.

    The result is a uniformly random sample regardless of stream length.

    Example:
        >>> sampler = ReservoirSampler()
        >>> games = sampler.sample(reader.stream(path), n=1_000_000)
    """

    def sample(
        self, stream: Iterator[chess.pgn.Game], n: int
    ) -> list[chess.pgn.Game]:
        """Sample exactly n games uniformly at random from the stream.

        If the stream has fewer than n games, all games are returned.
        Each game in the stream has an equal probability of appearing in the sample.

        Args:
            stream: An iterator of chess.pgn.Game objects of unknown length.
            n: The desired sample size.

        Returns:
            A list of up to n games selected uniformly at random.

        Example:
            >>> sampler = ReservoirSampler()
            >>> result = sampler.sample(iter([game1, game2, game3]), n=2)
            >>> len(result)
            2
        """
        raise NotImplementedError("To be implemented")
        return []
