"""EpisodeRecorder: accumulates per-ply data during self-play episodes.

Implements the Recordable protocol. record() must be called for every
half-move (both player and opponent). finalize() applies episode-wide
softmax entropy normalization over player plies before sealing.
"""

from __future__ import annotations

import torch

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
        self._plies.append(ply)

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
        player_indices = [
            i for i, p in enumerate(self._plies)
            if p.is_player_ply
        ]
        if player_indices:
            raw_entropies = torch.tensor(
                [self._plies[i].entropy for i in player_indices],
                dtype=torch.float32,
            )
            norm_entropies = torch.softmax(raw_entropies, dim=0)
            new_plies = list(self._plies)
            for idx, norm_h in zip(
                player_indices, norm_entropies.tolist()
            ):
                old = new_plies[idx]
                new_plies[idx] = old._replace(entropy=norm_h)
        else:
            new_plies = list(self._plies)
        record = EpisodeRecord(
            plies=new_plies,
            outcome=outcome,
            total_plies=len(new_plies),
        )
        self.reset()
        return record

    def reset(self) -> None:
        """Clear internal buffer for a new episode."""
        self._plies = []
