"""EpisodeRecorder: accumulates per-ply data during self-play episodes.

Implements the Recordable protocol. record() must be called for every
half-move (both player and opponent). finalize() applies episode-wide
softmax entropy normalization over player plies before sealing.
"""

from __future__ import annotations

from chess_sim.protocols import Recordable  # noqa: F401
from chess_sim.types import EpisodeRecord, PlyTuple


class EpisodeRecorder:
    """Accumulates per-ply data for both sides during an episode.

    Implements the Recordable protocol. record() is called for every
    half-move (both player and opponent plies). finalize() applies
    episode-wide softmax entropy normalization over player plies
    before sealing the record, so that normalized entropies sum to
    1.0 and are comparable across episodes of different lengths.
    """

    def __init__(self) -> None:
        """Initialize empty ply buffer."""
        self._plies: list[PlyTuple] = []

    def record(self, ply: PlyTuple) -> None:
        """Append ply to internal buffer.

        Call for every half-move, both player and opponent sides.

        Args:
            ply: PlyTuple for a single half-move.
        """
        raise NotImplementedError("To be implemented")

    def finalize(self, outcome: float) -> EpisodeRecord:
        """Seal the episode and return an EpisodeRecord.

        Applies softmax over raw Shannon entropies of player plies
        to produce normalized values in (0, 1) summing to 1.0, then
        returns an EpisodeRecord. Resets internal state after sealing.

        Args:
            outcome: +1.0 for win, -1.0 for loss, draw_reward for
                     draws.

        Returns:
            Sealed EpisodeRecord with all plies and normalized
            player entropies.
        """
        raise NotImplementedError("To be implemented")

    def reset(self) -> None:
        """Clear internal buffer for a new episode."""
        raise NotImplementedError("To be implemented")
