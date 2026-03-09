"""Tests for the RL HDF5 preprocessing pipeline (T1-T12).

Each test exercises a specific component from the design doc.
Stubs raise NotImplementedError; tests catch that to confirm
the stubs are importable and wired correctly.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from chess_sim.preprocess.rl_preprocessor import RLHdf5Preprocessor
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


# -------------------------------------------------------------------
# T1 — Schema correctness (RLHdf5Validator)
# -------------------------------------------------------------------

class TestSchemaCorrectness(unittest.TestCase):
    """T1: Validator checks all required datasets and attributes."""

    def test_validate_raises_not_implemented(self) -> None:
        """Validator stub raises NotImplementedError."""
        validator = RLHdf5Validator()
        config = RLPreprocessConfig()
        with self.assertRaises(NotImplementedError):
            validator.validate(Path("fake.h5"), config)


# -------------------------------------------------------------------
# T2 — Round-trip board/color/traj tokens
# -------------------------------------------------------------------

class TestTokenRoundTrip(unittest.TestCase):
    """T2: Write one record, read back via dataset.__getitem__(0)."""

    def test_writer_open_raises_not_implemented(self) -> None:
        """Writer.open stub raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            RLHdf5Writer(512, 1000, "gzip", 4, "white")

    def test_dataset_init_raises_not_implemented(self) -> None:
        """Dataset.__init__ stub raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            RLPlyHDF5Dataset(Path("fake.h5"), "train")


# -------------------------------------------------------------------
# T3 — move_prefix padding round-trip
# -------------------------------------------------------------------

class TestPrefixPadding(unittest.TestCase):
    """T3: Prefix len 3 with max_prefix_len=10 -> shape (3,)."""

    def test_record_prefix_length(self) -> None:
        """RLPlyRecord stores correct prefix length."""
        rec = _make_record(prefix_len=3)
        self.assertEqual(len(rec.move_prefix), 3)

    def test_writer_write_batch_raises_not_implemented(
        self,
    ) -> None:
        """Writer.write_batch stub raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            writer = RLHdf5Writer(10, 100, "gzip", 4, "white")
            writer.write_batch([_make_record(prefix_len=3)], "train")


# -------------------------------------------------------------------
# T4 — move_prefix truncation
# -------------------------------------------------------------------

class TestPrefixTruncation(unittest.TestCase):
    """T4: Prefix > max_prefix_len is truncated; WARNING logged."""

    def test_long_prefix_record_created(self) -> None:
        """A record with prefix_len=600 is constructable."""
        rec = _make_record(prefix_len=600)
        self.assertEqual(len(rec.move_prefix), 600)

    def test_writer_truncation_raises_not_implemented(
        self,
    ) -> None:
        """Writer must truncate; stub raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            writer = RLHdf5Writer(512, 1000, "gzip", 4, "white")
            writer.write_batch(
                [_make_record(prefix_len=600)], "train"
            )


# -------------------------------------------------------------------
# T5 — train_color filtering
# -------------------------------------------------------------------

class TestTrainColorFiltering(unittest.TestCase):
    """T5: Parser keeps only configured train_color plies."""

    def test_parser_init_raises_not_implemented(self) -> None:
        """Parser stub raises NotImplementedError on init."""
        with self.assertRaises(NotImplementedError):
            RLPlyParser(train_color="white")

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


# -------------------------------------------------------------------
# T6 — move_uci round-trip
# -------------------------------------------------------------------

class TestMoveUciRoundTrip(unittest.TestCase):
    """T6: Write record with move_uci='e7e8q', read back intact."""

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


# -------------------------------------------------------------------
# T7 — Reward recomputation from stored flags
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

    def test_dataset_getitem_raises_not_implemented(
        self,
    ) -> None:
        """Dataset.__getitem__ raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            RLPlyHDF5Dataset(Path("fake.h5"), "train")


# -------------------------------------------------------------------
# T8 — Empty game handling
# -------------------------------------------------------------------

class TestEmptyGameHandling(unittest.TestCase):
    """T8: Game with result '*' returns empty list from parser."""

    def test_parser_parse_game_raises_not_implemented(
        self,
    ) -> None:
        """Parser.parse_game stub raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            parser = RLPlyParser(train_color="white")
            # Would call parser.parse_game(game, 0)


# -------------------------------------------------------------------
# T9 — Dataset length
# -------------------------------------------------------------------

class TestDatasetLength(unittest.TestCase):
    """T9: Write 50 train + 5 val -> correct len() for each."""

    def test_dataset_init_raises_not_implemented(self) -> None:
        """Dataset init raises NotImplementedError (no file)."""
        with self.assertRaises(NotImplementedError):
            ds = RLPlyHDF5Dataset(Path("fake.h5"), "train")
            len(ds)


# -------------------------------------------------------------------
# T10 — Out-of-range __getitem__
# -------------------------------------------------------------------

class TestOutOfRange(unittest.TestCase):
    """T10: dataset[-1] and dataset[len] raise IndexError."""

    def test_dataset_init_raises_not_implemented(self) -> None:
        """Cannot test indexing until __init__ is implemented."""
        with self.assertRaises(NotImplementedError):
            RLPlyHDF5Dataset(Path("fake.h5"), "train")


# -------------------------------------------------------------------
# T11 — Multi-worker DataLoader
# -------------------------------------------------------------------

class TestMultiWorkerDataLoader(unittest.TestCase):
    """T11: worker_init_fn re-opens HDF5 without collisions."""

    def test_worker_init_raises_not_implemented(self) -> None:
        """rl_hdf5_worker_init stub raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            rl_hdf5_worker_init(0)


# -------------------------------------------------------------------
# T12 — train_color attribute mismatch
# -------------------------------------------------------------------

class TestTrainColorMismatch(unittest.TestCase):
    """T12: Validator raises ValueError on color mismatch."""

    def test_validator_raises_not_implemented(self) -> None:
        """Validator.validate stub raises NotImplementedError."""
        validator = RLHdf5Validator()
        config = RLPreprocessConfig(
            filter=RLFilterConfig(train_color="black")
        )
        with self.assertRaises(NotImplementedError):
            validator.validate(Path("fake.h5"), config)


# -------------------------------------------------------------------
# Config and type supplementary tests
# -------------------------------------------------------------------

class TestRLPreprocessConfig(unittest.TestCase):
    """Verify config defaults and YAML loader stub."""

    def test_default_output_config(self) -> None:
        """RLOutputConfig defaults match design spec."""
        cfg = RLOutputConfig()
        self.assertEqual(cfg.hdf5_path, "data/processed/chess_rl.h5")
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

    def test_load_config_raises_not_implemented(self) -> None:
        """YAML loader stub raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            load_rl_preprocess_config(Path("fake.yaml"))


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
        self.assertEqual(RLPlyRecord._fields[index], field_name)


class TestRLPGNReader(unittest.TestCase):
    """Verify RLPGNReader.stream stub is callable."""

    def test_stream_raises_not_implemented(self) -> None:
        """Reader.stream stub raises NotImplementedError."""
        reader = RLPGNReader()
        with self.assertRaises(NotImplementedError):
            list(reader.stream(Path("fake.pgn")))


class TestRLHdf5Preprocessor(unittest.TestCase):
    """Verify RLHdf5Preprocessor wiring."""

    def test_init_raises_not_implemented(self) -> None:
        """Preprocessor.__init__ stub raises NotImplementedError."""
        reader = RLPGNReader()
        with self.assertRaises(NotImplementedError):
            parser = RLPlyParser(train_color="white")

    def test_preprocessor_init_raises_not_implemented(
        self,
    ) -> None:
        """Cannot construct preprocessor when parser fails."""
        with self.assertRaises(NotImplementedError):
            preprocessor = RLHdf5Preprocessor(
                MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )


if __name__ == "__main__":
    unittest.main()
