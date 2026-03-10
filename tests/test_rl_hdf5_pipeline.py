"""Tests for the RL HDF5 preprocessing pipeline (T1-T12).

Each test exercises a specific component from the design doc.
Tests cover real behavior after stub implementations were filled in.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import chess.pgn
import h5py
from parameterized import parameterized

from chess_sim.config import (
    RLFilterConfig,
    RLOutputConfig,
    RLPreprocessConfig,
    load_rl_preprocess_config,
)
from chess_sim.data.rl_hdf5_dataset import (
    RLPlyHDF5Dataset,
    rl_hdf5_worker_init,
)
from chess_sim.preprocess.rl_parser import RLPlyParser
from chess_sim.preprocess.rl_preprocessor import (
    RLHdf5Preprocessor,
)
from chess_sim.preprocess.rl_reader import RLPGNReader
from chess_sim.preprocess.rl_validator import RLHdf5Validator
from chess_sim.preprocess.rl_writer import RLHdf5Writer
from chess_sim.types import RLPlyRecord


def _make_record(
    game_id: int = 0,
    ply_index: int = 0,
    move_uci: str = "e2e4",
    is_winner_ply: bool = True,
    is_white_ply: bool = True,
    is_draw_ply: bool = False,
    prefix_len: int = 1,
) -> RLPlyRecord:
    """Create a minimal RLPlyRecord for testing."""
    return RLPlyRecord(
        board_tokens=[0] * 65,
        color_tokens=[0] * 65,
        traj_tokens=[0] * 65,
        move_prefix=[1] + [100] * (prefix_len - 1),
        move_uci=move_uci,
        is_winner_ply=is_winner_ply,
        is_white_ply=is_white_ply,
        is_draw_ply=is_draw_ply,
        game_id=game_id,
        ply_index=ply_index,
    )


def _write_test_h5(
    path: Path,
    train_records: list[RLPlyRecord],
    val_records: list[RLPlyRecord],
    max_prefix_len: int = 512,
    train_color: str = "white",
) -> None:
    """Helper: write records to HDF5 via RLHdf5Writer."""
    writer = RLHdf5Writer(
        max_prefix_len, 1000, "gzip", 4, train_color
    )
    writer.open(path)
    if train_records:
        writer.write_batch(train_records, "train")
    if val_records:
        writer.write_batch(val_records, "val")
    writer.close()


def _make_pgn_game(
    moves_uci: list[str],
    result: str = "1-0",
) -> chess.pgn.Game:
    """Build a chess.pgn.Game from UCI move strings."""
    game = chess.pgn.Game()
    game.headers["Result"] = result
    board = game.board()
    node = game
    for uci in moves_uci:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    return game


# -------------------------------------------------------------------
# T1 -- Schema correctness (RLHdf5Validator)
# -------------------------------------------------------------------

class TestSchemaCorrectness(unittest.TestCase):
    """T1: Validator checks all required datasets and attrs."""

    def test_validate_good_file(self) -> None:
        """Validator passes on a correctly-written file."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.h5"
            recs = [_make_record(game_id=i) for i in range(5)]
            _write_test_h5(p, recs, [_make_record(game_id=99)])
            cfg = RLPreprocessConfig()
            RLHdf5Validator().validate(p, cfg)

    def test_validate_missing_file(self) -> None:
        """Validator raises FileNotFoundError for missing file."""
        cfg = RLPreprocessConfig()
        with self.assertRaises(FileNotFoundError):
            RLHdf5Validator().validate(
                Path("nonexistent.h5"), cfg
            )


# -------------------------------------------------------------------
# T2 -- Round-trip board/color/traj tokens
# -------------------------------------------------------------------

class TestTokenRoundTrip(unittest.TestCase):
    """T2: Write one record, read back via dataset[0]."""

    def test_round_trip(self) -> None:
        """Tokens survive write -> read cycle."""
        rec = _make_record()
        rec = rec._replace(
            board_tokens=[1, 2, 3] + [0] * 62,
            color_tokens=[0, 1, 2] + [0] * 62,
            traj_tokens=[0, 0, 4] + [0] * 62,
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rt.h5"
            _write_test_h5(p, [rec], [_make_record()])
            ds = RLPlyHDF5Dataset(p, "train")
            ply = ds[0]
            self.assertEqual(ply.board_tokens[0].item(), 1)
            self.assertEqual(ply.color_tokens[2].item(), 2)
            self.assertEqual(ply.traj_tokens[2].item(), 4)

    def test_dataset_file_not_found(self) -> None:
        """Dataset raises when HDF5 file does not exist."""
        with self.assertRaises(OSError):
            RLPlyHDF5Dataset(Path("fake.h5"), "train")


# -------------------------------------------------------------------
# T3 -- move_prefix padding round-trip
# -------------------------------------------------------------------

class TestPrefixPadding(unittest.TestCase):
    """T3: Prefix len 3 with max_prefix_len=10 -> shape (3,)."""

    def test_record_prefix_length(self) -> None:
        """RLPlyRecord stores correct prefix length."""
        rec = _make_record(prefix_len=3)
        self.assertEqual(len(rec.move_prefix), 3)

    def test_prefix_round_trip(self) -> None:
        """Prefix is padded on disk, sliced on read."""
        rec = _make_record(prefix_len=3)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "pfx.h5"
            _write_test_h5(
                p, [rec], [_make_record()],
                max_prefix_len=10,
            )
            ds = RLPlyHDF5Dataset(p, "train")
            ply = ds[0]
            self.assertEqual(ply.move_prefix.shape[0], 3)


# -------------------------------------------------------------------
# T4 -- move_prefix truncation
# -------------------------------------------------------------------

class TestPrefixTruncation(unittest.TestCase):
    """T4: Prefix > max_prefix_len is truncated; WARNING logged."""

    def test_long_prefix_record_created(self) -> None:
        """A record with prefix_len=600 is constructable."""
        rec = _make_record(prefix_len=600)
        self.assertEqual(len(rec.move_prefix), 600)

    def test_writer_truncates_long_prefix(self) -> None:
        """Writer truncates prefix > max_prefix_len."""
        rec = _make_record(prefix_len=600)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trunc.h5"
            writer = RLHdf5Writer(
                512, 1000, "gzip", 4, "white"
            )
            writer.open(p)
            writer.write_batch([rec], "train")
            writer.close()
            with h5py.File(str(p), "r") as f:
                pl = int(f["train/prefix_lengths"][0])
                self.assertEqual(pl, 512)


# -------------------------------------------------------------------
# T5 -- train_color filtering
# -------------------------------------------------------------------

class TestTrainColorFiltering(unittest.TestCase):
    """T5: Parser keeps only configured train_color plies."""

    def test_parser_init_valid(self) -> None:
        """Parser accepts 'white' and 'black'."""
        p1 = RLPlyParser(train_color="white")
        p2 = RLPlyParser(train_color="black")
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)

    def test_parser_init_invalid(self) -> None:
        """Parser rejects invalid train_color."""
        with self.assertRaises(ValueError):
            RLPlyParser(train_color="red")

    def test_filter_config_valid_colors(self) -> None:
        """RLFilterConfig accepts 'white' and 'black'."""
        cfg_w = RLFilterConfig(train_color="white")
        cfg_b = RLFilterConfig(train_color="black")
        self.assertEqual(cfg_w.train_color, "white")
        self.assertEqual(cfg_b.train_color, "black")

    def test_filter_config_invalid_color_raises(self) -> None:
        """RLFilterConfig rejects invalid train_color."""
        with self.assertRaises(ValueError):
            RLFilterConfig(train_color="red")

    def test_parser_filters_white_only(self) -> None:
        """White parser returns only white plies."""
        # Scholars mate: e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7
        moves = [
            "e2e4", "e7e5", "d1h5", "b8c6",
            "f1c4", "g8f6", "h5f7",
        ]
        game = _make_pgn_game(moves, "1-0")
        parser = RLPlyParser("white", min_moves=1)
        records = parser.parse_game(game, 0)
        # 7 half-moves => 4 white, 3 black
        self.assertEqual(len(records), 4)
        for r in records:
            self.assertTrue(r.is_white_ply)


# -------------------------------------------------------------------
# T6 -- move_uci round-trip
# -------------------------------------------------------------------

class TestMoveUciRoundTrip(unittest.TestCase):
    """T6: Write record with move_uci, read back intact."""

    @parameterized.expand([
        ("normal_move", "e2e4"),
        ("promotion", "e7e8q"),
        ("short_move", "a2a3"),
        ("castling", "e1g1"),
    ])
    def test_record_preserves_uci(
        self, name: str, uci: str
    ) -> None:
        """RLPlyRecord stores UCI string without mutation."""
        rec = _make_record(move_uci=uci)
        self.assertEqual(rec.move_uci, uci)

    def test_uci_hdf5_round_trip(self) -> None:
        """UCI string survives write -> read cycle."""
        rec = _make_record(move_uci="e7e8q")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "uci.h5"
            _write_test_h5(p, [rec], [_make_record()])
            ds = RLPlyHDF5Dataset(p, "train")
            self.assertEqual(ds[0].move_uci, "e7e8q")


# -------------------------------------------------------------------
# T7 -- Reward recomputation from stored flags
# -------------------------------------------------------------------

class TestRewardRecomputation(unittest.TestCase):
    """T7: OfflinePlyTuple from HDF5 is compatible with rewards."""

    def test_record_flags_are_bool(self) -> None:
        """RLPlyRecord flag fields are proper booleans."""
        rec = _make_record(
            is_winner_ply=True,
            is_draw_ply=False,
            is_white_ply=True,
        )
        self.assertIsInstance(rec.is_winner_ply, bool)
        self.assertIsInstance(rec.is_draw_ply, bool)
        self.assertIsInstance(rec.is_white_ply, bool)

    def test_dataset_flags_round_trip(self) -> None:
        """Flags read from HDF5 are booleans."""
        rec = _make_record(
            is_winner_ply=True,
            is_draw_ply=False,
            is_white_ply=True,
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "flags.h5"
            _write_test_h5(p, [rec], [_make_record()])
            ds = RLPlyHDF5Dataset(p, "train")
            ply = ds[0]
            self.assertIsInstance(ply.is_winner_ply, bool)
            self.assertTrue(ply.is_winner_ply)
            self.assertFalse(ply.is_draw_ply)
            self.assertTrue(ply.is_white_ply)


# -------------------------------------------------------------------
# T8 -- Empty game handling
# -------------------------------------------------------------------

class TestEmptyGameHandling(unittest.TestCase):
    """T8: Game with result '*' returns empty list from parser."""

    def test_star_result_returns_empty(self) -> None:
        """Parser returns [] for unfinished games."""
        game = _make_pgn_game(["e2e4", "e7e5"], result="*")
        parser = RLPlyParser("white", min_moves=1)
        self.assertEqual(parser.parse_game(game, 0), [])

    def test_short_game_returns_empty(self) -> None:
        """Parser returns [] when game has < min_moves."""
        game = _make_pgn_game(
            ["e2e4", "e7e5"], result="1-0"
        )
        parser = RLPlyParser("white", min_moves=10)
        self.assertEqual(parser.parse_game(game, 0), [])


# -------------------------------------------------------------------
# T9 -- Dataset length
# -------------------------------------------------------------------

class TestDatasetLength(unittest.TestCase):
    """T9: Write 50 train + 5 val -> correct len() for each."""

    def test_lengths(self) -> None:
        """Dataset reports correct len per split."""
        train = [_make_record(game_id=i) for i in range(50)]
        val = [_make_record(game_id=i) for i in range(5)]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "len.h5"
            _write_test_h5(p, train, val)
            ds_t = RLPlyHDF5Dataset(p, "train")
            ds_v = RLPlyHDF5Dataset(p, "val")
            self.assertEqual(len(ds_t), 50)
            self.assertEqual(len(ds_v), 5)


# -------------------------------------------------------------------
# T10 -- Out-of-range __getitem__
# -------------------------------------------------------------------

class TestOutOfRange(unittest.TestCase):
    """T10: dataset[-1] and dataset[len] raise IndexError."""

    def test_negative_index(self) -> None:
        """Negative index raises IndexError."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "oor.h5"
            _write_test_h5(
                p, [_make_record()], [_make_record()]
            )
            ds = RLPlyHDF5Dataset(p, "train")
            with self.assertRaises(IndexError):
                ds[-1]

    def test_over_index(self) -> None:
        """Index == len raises IndexError."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "oor2.h5"
            _write_test_h5(
                p, [_make_record()], [_make_record()]
            )
            ds = RLPlyHDF5Dataset(p, "train")
            with self.assertRaises(IndexError):
                ds[len(ds)]


# -------------------------------------------------------------------
# T11 -- Multi-worker DataLoader
# -------------------------------------------------------------------

class TestMultiWorkerDataLoader(unittest.TestCase):
    """T11: worker_init_fn re-opens HDF5 without errors."""

    def test_worker_init_no_worker_info(self) -> None:
        """rl_hdf5_worker_init is a no-op when not in worker."""
        # In main process, get_worker_info() returns None
        rl_hdf5_worker_init(0)  # should not raise


# -------------------------------------------------------------------
# T12 -- train_color attribute mismatch
# -------------------------------------------------------------------

class TestTrainColorMismatch(unittest.TestCase):
    """T12: Validator raises ValueError on color mismatch."""

    def test_color_mismatch_raises(self) -> None:
        """Validator detects train_color attr mismatch."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mismatch.h5"
            _write_test_h5(
                p,
                [_make_record()],
                [_make_record()],
                train_color="white",
            )
            config = RLPreprocessConfig(
                filter=RLFilterConfig(train_color="black")
            )
            with self.assertRaises(ValueError):
                RLHdf5Validator().validate(p, config)


# -------------------------------------------------------------------
# Config and type supplementary tests
# -------------------------------------------------------------------

class TestRLPreprocessConfig(unittest.TestCase):
    """Verify config defaults and YAML loader."""

    def test_default_output_config(self) -> None:
        """RLOutputConfig defaults match design spec."""
        cfg = RLOutputConfig()
        self.assertEqual(
            cfg.hdf5_path, "data/processed/chess_rl.h5"
        )
        self.assertEqual(cfg.chunk_size, 1000)
        self.assertEqual(cfg.compression, "gzip")
        self.assertEqual(cfg.compression_opts, 4)
        self.assertEqual(cfg.max_prefix_len, 512)

    def test_default_filter_config(self) -> None:
        """RLFilterConfig defaults match design spec."""
        cfg = RLFilterConfig()
        self.assertEqual(cfg.min_moves, 5)
        self.assertEqual(cfg.max_moves, 512)
        self.assertEqual(cfg.train_color, "white")

    def test_full_config_defaults(self) -> None:
        """RLPreprocessConfig assembles sub-configs correctly."""
        cfg = RLPreprocessConfig()
        self.assertIsInstance(cfg.output, RLOutputConfig)
        self.assertIsInstance(cfg.filter, RLFilterConfig)
        self.assertEqual(cfg.split.train, 0.95)
        self.assertEqual(cfg.processing.workers, 4)

    def test_load_config_from_yaml(self) -> None:
        """YAML loader parses preprocess_rl.yaml."""
        yaml_path = Path("configs/preprocess_rl.yaml")
        if not yaml_path.exists():
            self.skipTest("preprocess_rl.yaml not found")
        cfg = load_rl_preprocess_config(yaml_path)
        self.assertEqual(cfg.filter.train_color, "white")
        self.assertEqual(cfg.output.max_prefix_len, 512)

    def test_load_config_missing_file(self) -> None:
        """YAML loader raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_rl_preprocess_config(
                Path("nonexistent.yaml")
            )


class TestRLPlyRecordFields(unittest.TestCase):
    """Verify RLPlyRecord NamedTuple field count and names."""

    def test_field_count(self) -> None:
        """RLPlyRecord has exactly 10 fields."""
        self.assertEqual(len(RLPlyRecord._fields), 10)

    @parameterized.expand([
        ("board_tokens", 0),
        ("color_tokens", 1),
        ("traj_tokens", 2),
        ("move_prefix", 3),
        ("move_uci", 4),
        ("is_winner_ply", 5),
        ("is_white_ply", 6),
        ("is_draw_ply", 7),
        ("game_id", 8),
        ("ply_index", 9),
    ])
    def test_field_names(
        self, field_name: str, index: int
    ) -> None:
        """RLPlyRecord field {0} is at index {1}."""
        self.assertEqual(
            RLPlyRecord._fields[index], field_name
        )


class TestRLPGNReader(unittest.TestCase):
    """Verify RLPGNReader.stream behavior."""

    def test_stream_file_not_found(self) -> None:
        """Reader raises FileNotFoundError for missing file."""
        reader = RLPGNReader()
        with self.assertRaises(FileNotFoundError):
            list(reader.stream(Path("fake.pgn")))

    def test_stream_real_pgn(self) -> None:
        """Reader streams games from a real PGN file."""
        pgn_path = Path("data/games.pgn")
        if not pgn_path.exists():
            self.skipTest("data/games.pgn not found")
        reader = RLPGNReader()
        games = list(reader.stream(pgn_path, max_games=2))
        self.assertLessEqual(len(games), 2)
        self.assertGreater(len(games), 0)


class TestRLHdf5Preprocessor(unittest.TestCase):
    """Verify RLHdf5Preprocessor wiring."""

    def test_init_succeeds(self) -> None:
        """Preprocessor can be constructed with components."""
        reader = RLPGNReader()
        parser = RLPlyParser(train_color="white")
        writer = RLHdf5Writer(512, 1000, "gzip", 4, "white")
        validator = RLHdf5Validator()
        preprocessor = RLHdf5Preprocessor(
            reader, parser, writer, validator
        )
        self.assertIsNotNone(preprocessor)

    def test_init_with_mocks(self) -> None:
        """Preprocessor accepts mock dependencies."""
        preprocessor = RLHdf5Preprocessor(
            MagicMock(), MagicMock(),
            MagicMock(), MagicMock(),
        )
        self.assertIsNotNone(preprocessor)


if __name__ == "__main__":
    unittest.main()
