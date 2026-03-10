"""EpisodeRecorder: accumulates per-ply data during self-play episodes.

Implements the Recordable protocol. record() must be called for every
half-move (both player and opponent). finalize() applies episode-wide
softmax entropy normalization over player plies before sealing.
"""

from __future__ import annotations

import torch

from chess_sim.types import EpisodeRecord, PlyTuple


def _apply_entropy_normalization(
    plies: list[PlyTuple],
    player_indices: list[int],
) -> list[PlyTuple]:
    """Softmax-normalize raw Shannon entropies at player_indices.

    Returns a new list with normalized entropy values replacing raw
    ones at player_indices. Normalized values sum to 1.0, enabling
    cross-episode comparison.
    """
    new_plies = list(plies)
    raw_entropies = torch.tensor(
        [new_plies[i].entropy for i in player_indices],
        dtype=torch.float32,
    )
    norm_entropies = torch.softmax(raw_entropies, dim=0)
    for idx, norm_h in zip(
        player_indices, norm_entropies.tolist()
    ):
        new_plies[idx] = new_plies[idx]._replace(entropy=norm_h)
    return new_plies


class EpisodeRecorder:
    """Accumulates per-ply data for both sides during an episode.

    Example:
        >>> rec = EpisodeRecorder()
        >>> rec.record(ply)
        >>> episode = rec.finalize(outcome=1.0)
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
            outcome: +1.0 for win, -1.0 for loss,
                     caller-defined reward for draws (e.g. 0.0).

        Returns:
            Sealed EpisodeRecord with all plies and normalized
            player entropies.
        """
        player_indices = [
            i for i, p in enumerate(self._plies)
            if p.is_player_ply
        ]
        if player_indices:
            new_plies = _apply_entropy_normalization(
                self._plies, player_indices
            )
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
