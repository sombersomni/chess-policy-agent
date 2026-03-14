"""PGNRewardPreprocessor: offline HDF5 generation for batched CE training.

Streams a PGN file, replays each game via PGNReplayer, encodes the
board state, and writes all rows to a chunked gzip-compressed HDF5
file.

HDF5 schema (datasets, all N-length along axis 0):
    board:        float32 (N, 65, 3) — board/color/traj channels
    color_tokens: int8    (N, 65)    — for structural mask building
    target_move:  int32   (N,)       — vocab index
    game_id:      int32   (N,)       — source game index
    ply_idx:      int16   (N,)       — 0-indexed ply within game
    outcome:      int8    (N,)       — +1 / 0 / -1
    legal_mask:   bool    (N, 1971)  — true legal moves per position
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

import chess
import chess.pgn
import h5py
import numpy as np
import torch
from torch import Tensor

from chess_sim.config import PGNRLConfig
from chess_sim.data.capture_map_builder import (
    build as build_capture_map,
)
from chess_sim.data.move_category_builder import (
    build as build_move_category,
)
from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.training.pgn_replayer import PGNReplayer
from chess_sim.types import RLRewardRow

logger = logging.getLogger(__name__)

# HDF5 dataset names and their (dtype, per-row shape).
_DATASETS: dict[str, tuple[str, tuple[int, ...]]] = {
    "board": ("float32", (65, 3)),
    "color_tokens": ("int8", (65,)),
    "target_move": ("int32", ()),
    "game_id": ("int32", ()),
    "ply_idx": ("int16", ()),
    "outcome": ("int8", ()),
    "legal_mask": ("bool", (1971,)),
}

_SCHEMA_VERSION: int = 4  # bump when HDF5 schema changes

# Aux head datasets: only created when use_aux_heads=True
_AUX_DATASETS: dict[str, tuple[str, tuple[int, ...]]] = {
    "capture_map": ("uint8", (64,)),
    "move_category": ("uint8", ()),
}


def _stream_pgn(
    pgn_path: Path,
    max_games: int = 0,
) -> list[chess.pgn.Game]:
    """Read games from a PGN file (plain or .zst).

    Args:
        pgn_path: Path to .pgn or .pgn.zst file.
        max_games: Maximum games to read (0 = all).

    Returns:
        List of parsed chess.pgn.Game objects.
    """
    games: list[chess.pgn.Game] = []
    if str(pgn_path).endswith(".zst"):
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        with open(pgn_path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_io = io.TextIOWrapper(
                    reader,
                    encoding="utf-8",
                    errors="replace",
                )
                while True:
                    game = chess.pgn.read_game(text_io)
                    if game is None:
                        break
                    games.append(game)
                    if 0 < max_games <= len(games):
                        break
    else:
        with open(pgn_path, "r") as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                games.append(game)
                if 0 < max_games <= len(games):
                    break
    return games


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
    return torch.stack(
        [board_tokens.float(),
         color_tokens.float(),
         traj_tokens.float()],
        dim=-1,
    )


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
        self._cfg = cfg
        self._device = device
        self._replayer = PGNReplayer()
        self._move_tok = MoveTokenizer()

    def generate(
        self,
        pgn_path: Path,
        hdf5_path: Path,
        max_games: int = 0,
    ) -> Path:
        """Stream PGN, replay games, write HDF5. Returns path.

        Checks cache validity first; skips regeneration if HDF5
        attributes match current config. Writes chunked gzip HDF5
        with datasets: board, color_tokens, target_move,
        game_id, ply_idx, outcome, legal_mask.

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
        if self._is_cache_valid(hdf5_path, pgn_path, max_games):
            logger.info(
                "HDF5 cache hit: %s — skipping regeneration",
                hdf5_path,
            )
            return hdf5_path

        games = _stream_pgn(pgn_path, max_games)
        rl = self._cfg.rl
        train_color = rl.train_color
        chunk_size = rl.hdf5_chunk_size

        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        use_aux = rl.use_aux_heads

        rows: list[RLRewardRow] = []
        aux_capture: list[list[int]] = []
        aux_category: list[int] = []
        cursor = 0

        with h5py.File(hdf5_path, "w") as hf:
            # Create resizable datasets
            all_ds = dict(_DATASETS)
            if use_aux:
                all_ds.update(_AUX_DATASETS)
            for name, (dtype, shape) in all_ds.items():
                full_shape = (0,) + shape
                max_shape = (None,) + shape
                chunk_shape = (chunk_size,) + shape
                hf.create_dataset(
                    name,
                    shape=full_shape,
                    maxshape=max_shape,
                    dtype=dtype,
                    chunks=chunk_shape,
                    compression="gzip",
                    compression_opts=4,
                )

            for game_idx, game in enumerate(games):
                if (
                    rl.skip_draws
                    and game.headers.get("Result")
                    == "1/2-1/2"
                ):
                    continue

                plies = self._replayer.replay(game)

                # Build board states for aux labels
                boards: list[chess.Board] | None = None
                if use_aux:
                    boards = self._replay_boards(game)

                if train_color != "both":
                    if boards is not None:
                        boards = [
                            b
                            for b, p in zip(boards, plies)
                            if p.is_white_ply
                            == (train_color == "white")
                        ]
                    plies = [
                        p for p in plies
                        if p.is_white_ply
                        == (train_color == "white")
                    ]
                if not plies:
                    continue
                if len(plies) > rl.max_plies_per_game:
                    continue

                for ply_i, ply in enumerate(plies):
                    try:
                        target_idx = (
                            self._move_tok.tokenize_move(
                                ply.move_uci
                            )
                        )
                    except KeyError:
                        continue

                    board_enc = _encode_board(
                        ply.board_tokens,
                        ply.color_tokens,
                        ply.traj_tokens,
                    )

                    if ply.is_draw_ply:
                        outcome = 0
                    elif ply.is_winner_ply:
                        outcome = 1
                    else:
                        outcome = -1

                    ct_np = (
                        ply.color_tokens.numpy()
                        .astype(np.int8)
                    )

                    legal_mask_t = (
                        self._move_tok.build_legal_mask(
                            ply.legal_move_ucis
                        )
                    )

                    rows.append(RLRewardRow(
                        board=board_enc.numpy(),
                        color_tokens=ct_np,
                        target_move=target_idx,
                        game_id=game_idx,
                        ply_idx=ply_i,
                        outcome=outcome,
                        legal_mask=legal_mask_t.numpy(),
                    ))

                    if use_aux and boards is not None:
                        brd = boards[ply_i]
                        aux_capture.append(
                            build_capture_map(
                                brd, brd.turn
                            )
                        )
                        aux_category.append(
                            build_move_category(
                                ply.move_uci, brd
                            )
                        )

                    if len(rows) >= chunk_size:
                        cursor = self._write_batch(
                            hf, rows, cursor,
                            aux_capture if use_aux
                            else None,
                            aux_category if use_aux
                            else None,
                        )
                        rows.clear()
                        aux_capture.clear()
                        aux_category.clear()

            # Flush remaining rows
            if rows:
                cursor = self._write_batch(
                    hf, rows, cursor,
                    aux_capture if use_aux else None,
                    aux_category if use_aux else None,
                )
                rows.clear()
                aux_capture.clear()
                aux_category.clear()

            # Store config attributes for cache validation
            checksum = self._compute_checksum(
                pgn_path, max_games
            )
            hf.attrs["checksum"] = checksum
            hf.attrs["schema_version"] = _SCHEMA_VERSION
            hf.attrs["max_games"] = max_games

        logger.info(
            "HDF5 generated: %s (%d rows)",
            hdf5_path,
            cursor,
        )
        return hdf5_path

    def _write_batch(
        self,
        hf: h5py.File,
        rows: list[RLRewardRow],
        cursor: int,
        capture_maps: list[list[int]] | None = None,
        categories: list[int] | None = None,
    ) -> int:
        """Write a batch of RLRewardRow to open HDF5 file.

        Resizes datasets and copies row data in bulk. When aux
        labels are provided, also writes capture_map and
        move_category datasets.

        Args:
            hf: Open h5py.File in write mode.
            rows: List of RLRewardRow to write.
            cursor: Current write position in the dataset.
            capture_maps: Optional list of 64-int capture maps.
            categories: Optional list of move category ints.

        Returns:
            New cursor position after writing.
        """
        n = len(rows)
        new_size = cursor + n
        has_aux = capture_maps is not None

        ds_names = list(_DATASETS)
        if has_aux:
            ds_names += list(_AUX_DATASETS)
        for name in ds_names:
            hf[name].resize(new_size, axis=0)

        boards = np.stack([r.board for r in rows])
        hf["board"][cursor:new_size] = boards

        ct = np.stack([r.color_tokens for r in rows])
        hf["color_tokens"][cursor:new_size] = ct

        hf["target_move"][cursor:new_size] = np.array(
            [r.target_move for r in rows], dtype=np.int32
        )
        hf["game_id"][cursor:new_size] = np.array(
            [r.game_id for r in rows], dtype=np.int32
        )
        hf["ply_idx"][cursor:new_size] = np.array(
            [r.ply_idx for r in rows], dtype=np.int16
        )
        hf["outcome"][cursor:new_size] = np.array(
            [r.outcome for r in rows], dtype=np.int8
        )
        hf["legal_mask"][cursor:new_size] = np.stack(
            [r.legal_mask for r in rows]
        )

        if has_aux and capture_maps and categories is not None:
            hf["capture_map"][cursor:new_size] = np.array(
                capture_maps, dtype=np.uint8
            )
            hf["move_category"][cursor:new_size] = np.array(
                categories, dtype=np.uint8
            )

        return new_size

    @staticmethod
    def _replay_boards(
        game: chess.pgn.Game,
    ) -> list[chess.Board]:
        """Replay game moves, returning board BEFORE each push.

        Args:
            game: Parsed PGN game.

        Returns:
            List of board copies, one per ply, captured
            before the move is applied.
        """
        board = game.board()
        boards: list[chess.Board] = []
        for move in game.mainline_moves():
            boards.append(board.copy())
            board.push(move)
        return boards

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
        if not hdf5_path.exists():
            return False

        try:
            with h5py.File(hdf5_path, "r") as hf:
                stored_checksum = hf.attrs.get("checksum")
                expected = self._compute_checksum(
                    pgn_path, max_games
                )
                if stored_checksum != expected:
                    return False

                if (
                    hf.attrs.get("schema_version")
                    != _SCHEMA_VERSION
                ):
                    return False

                stored_max = hf.attrs.get("max_games")
                if stored_max is None or stored_max != max_games:
                    return False

                return True
        except (OSError, KeyError):
            return False

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
        h = hashlib.sha256()
        with open(pgn_path, "rb") as f:
            h.update(f.read(1024 * 1024))
        h.update(str(pgn_path.stat().st_size).encode())
        h.update(str(max_games).encode())
        return h.hexdigest()
