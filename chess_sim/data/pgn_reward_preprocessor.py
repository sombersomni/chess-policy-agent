"""PGNRewardPreprocessor: offline HDF5 generation for batched RSCE training.

Streams a PGN file, replays each game via PGNReplayer, computes
per-ply composite rewards via PGNRLRewardComputer, applies per-game
normalization (m_hat = m * N / sum(m)), encodes the board state,
and writes all rows to a chunked gzip-compressed HDF5 file.

HDF5 schema (datasets, all N-length along axis 0):
    board:        float32 (N, 65, 3) — board/color/traj channels
    color_tokens: int8    (N, 65)    — for structural mask building
    target_move:  int32   (N,)       — vocab index
    multiplier:   float32 (N,)       — pre-normalized m_hat per ply
    game_id:      int32   (N,)       — source game index
    ply_idx:      int16   (N,)       — 0-indexed ply within game
    outcome:      int8    (N,)       — +1 / 0 / -1
"""
from __future__ import annotations

from pathlib import Path

from torch import Tensor

from chess_sim.config import PGNRLConfig


def _encode_board(
    board_tokens: Tensor,
    color_tokens: Tensor,
    traj_tokens: Tensor,
) -> Tensor:
    """Stack board/color/traj channels into (65, 3) float32.

    Args:
        board_tokens: Long tensor of shape [65].
        color_tokens: Long tensor of shape [65].
        traj_tokens: Long tensor of shape [65].

    Returns:
        Float32 tensor of shape [65, 3].

    Example:
        >>> bt = torch.zeros(65, dtype=torch.long)
        >>> ct = torch.zeros(65, dtype=torch.long)
        >>> tt = torch.zeros(65, dtype=torch.long)
        >>> out = _encode_board(bt, ct, tt)
        >>> out.shape
        torch.Size([65, 3])
    """
    raise NotImplementedError("To be implemented")


class PGNRewardPreprocessor:
    """Offline preprocessor: PGN -> HDF5 with pre-normalized RSCE multipliers.

    Streams games from a PGN file, replays each via PGNReplayer,
    computes rewards via PGNRLRewardComputer, normalizes per-game
    multipliers offline, and writes to HDF5.

    Example:
        >>> pp = PGNRewardPreprocessor(cfg)
        >>> hdf5 = pp.generate(Path("games.pgn"), Path("out.h5"))
    """

    def __init__(
        self,
        cfg: PGNRLConfig,
        device: str = "cpu",
    ) -> None:
        """Initialize with config and optional device.

        Args:
            cfg: PGNRLConfig containing rl hyperparameters
                (lambda_outcome, lambda_material, rsce_r_ref,
                draw_reward_norm, rsbc_normalize_per_game).
            device: Torch device string for board encoding.

        Example:
            >>> pp = PGNRewardPreprocessor(PGNRLConfig())
        """
        raise NotImplementedError("To be implemented")

    def generate(
        self,
        pgn_path: Path,
        hdf5_path: Path,
        max_games: int = 0,
    ) -> Path:
        """Stream PGN, compute rewards, write HDF5. Returns path.

        Checks cache validity first; skips regeneration if HDF5
        attributes match current config. Writes chunked gzip HDF5
        with datasets: board, color_tokens, target_move,
        multiplier, game_id, ply_idx, outcome.

        Args:
            pgn_path: Path to .pgn or .pgn.zst file.
            hdf5_path: Destination HDF5 file path.
            max_games: Max games to process (0 = all).

        Returns:
            Path to the generated (or cached) HDF5 file.

        Example:
            >>> path = pp.generate(Path("g.pgn"), Path("o.h5"))
            >>> path.exists()
            True
        """
        raise NotImplementedError("To be implemented")

    def _is_cache_valid(
        self,
        hdf5_path: Path,
        pgn_path: Path,
        max_games: int,
    ) -> bool:
        """Check HDF5 attrs vs current config for cache hit.

        Reads stored attributes (pgn_checksum, max_games,
        lambda_outcome, lambda_material, rsce_r_ref,
        draw_reward_norm, rsbc_normalize_per_game) and compares
        against current config values.

        Args:
            hdf5_path: Path to existing HDF5 file.
            pgn_path: Path to PGN file for checksum.
            max_games: Max games used in generation.

        Returns:
            True if all attributes match; False otherwise.

        Example:
            >>> pp._is_cache_valid(Path("o.h5"), Path("g.pgn"), 0)
            False
        """
        raise NotImplementedError("To be implemented")

    def _compute_checksum(
        self,
        pgn_path: Path,
        max_games: int,
    ) -> str:
        """First-1MB + file-size checksum, like shard cache.

        Reads the first 1 MB of the PGN file, appends the file
        size and max_games as strings, and returns a hex SHA-256
        digest.

        Args:
            pgn_path: Path to PGN file.
            max_games: Max games param (part of cache key).

        Returns:
            Hex string SHA-256 checksum.

        Example:
            >>> cs = pp._compute_checksum(Path("g.pgn"), 1000)
            >>> len(cs)
            64
        """
        raise NotImplementedError("To be implemented")
