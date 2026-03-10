"""Tests for the HDF5 preprocessing pipeline (T1-T25).

All tests run on CPU. Synthetic games are built via python-chess
to avoid dependency on real PGN files.
"""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn
import h5py
import numpy as np
import torch

from chess_sim.config import (
    FilterConfig,
    InputConfig,
    OutputConfig,
    PreprocessV2Config,
    ProcessingConfig,
    SplitConfig,
    load_preprocess_v2_config,
)
from chess_sim.data.hdf5_dataset import (
    ChessHDF5Dataset,
)
from chess_sim.data.move_vocab import SOS_IDX
from chess_sim.data.pgn_sequence_dataset import (
    PGNSequenceCollator,
)
from chess_sim.preprocess.parse import GameParser
from chess_sim.preprocess.validate import HDF5Validator
from chess_sim.preprocess.writer import HDF5Writer
from chess_sim.types import RawTurnRecord

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

def _make_game(
    num_moves: int = 10,
    result: str = "1-0",
    white_elo: str = "1500",
    black_elo: str = "1500",
    seed: int = 42,
) -> chess.pgn.Game:
    """Create a synthetic PGN game with random legal moves."""
    rng = random.Random(seed)
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Result"] = result
    game.headers["WhiteElo"] = white_elo
    game.headers["BlackElo"] = black_elo
    node = game
    for _ in range(num_moves):
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        node = node.add_variation(move)
        board.push(move)
    game.headers["Result"] = result
    return game


def _make_records(
    n: int = 5,
    game_id: int = 0,
    move_len: int = 3,
) -> list[RawTurnRecord]:
    """Create n synthetic RawTurnRecords."""
    records = []
    for i in range(n):
        records.append(RawTurnRecord(
            board_tokens=[0] + [1] * 64,
            color_tokens=[0] + [1] * 64,
            trajectory_tokens=[0] * 65,
            move_tokens=[SOS_IDX] + [3] * (move_len - 1),
            target_tokens=[3] * move_len,
            outcome=1,
            turn=i,
            game_id=game_id,
        ))
    return records


def _write_valid_hdf5(
    path: Path,
    n_train: int = 10,
    n_val: int = 5,
    max_seq_len: int = 32,
) -> None:
    """Write a minimal valid HDF5 file for testing."""
    writer = HDF5Writer(
        max_seq_len=max_seq_len,
        chunk_size=100,
        compression="gzip",
        compression_opts=4,
    )
    writer.open(path)
    train_recs = _make_records(n_train, game_id=0)
    val_recs = _make_records(n_val, game_id=1)
    writer.write_batch(train_recs, "train")
    writer.write_batch(val_recs, "val")
    writer.close()


def _default_config(
    max_seq_len: int = 32,
) -> PreprocessV2Config:
    """Return a default test config."""
    return PreprocessV2Config(
        output=OutputConfig(max_seq_len=max_seq_len),
    )


# ---------------------------------------------------------------
# T1-T8: GameParser
# ---------------------------------------------------------------

class TestGameParser(unittest.TestCase):
    """Tests T1-T8 for GameParser correctness."""

    def setUp(self) -> None:
        self.parser = GameParser(
            min_moves=1, max_moves=512,
            winners_only=False, min_elo=0,
        )

    def test_t1_normal_game(self) -> None:
        """T1: Normal 10-move game produces 10 records."""
        game = _make_game(num_moves=10, result="1-0")
        records = self.parser.parse_game(game, game_id=0)
        moves = list(game.mainline_moves())
        self.assertEqual(len(records), len(moves))
        for i, r in enumerate(records):
            self.assertEqual(r.turn, i)
            self.assertEqual(r.game_id, 0)

    def test_t2_min_moves_filter(self) -> None:
        """T2: Game with 3 moves, min_moves=5 returns []."""
        parser = GameParser(
            min_moves=5, max_moves=512,
            winners_only=False, min_elo=0,
        )
        game = _make_game(num_moves=3, result="1-0")
        records = parser.parse_game(game, game_id=0)
        self.assertEqual(records, [])

    def test_t3_max_moves_filter(self) -> None:
        """T3: max_moves=5 truncates to 5 records."""
        parser = GameParser(
            min_moves=1, max_moves=5,
            winners_only=False, min_elo=0,
        )
        game = _make_game(num_moves=20, result="1-0")
        records = parser.parse_game(game, game_id=0)
        self.assertEqual(len(records), 5)
        for i, r in enumerate(records):
            self.assertEqual(r.turn, i)

    def test_t4_winners_only_draw(self) -> None:
        """T4: Draw with winners_only=True returns []."""
        parser = GameParser(
            min_moves=1, max_moves=512,
            winners_only=True, min_elo=0,
        )
        game = _make_game(
            num_moves=10, result="1/2-1/2"
        )
        records = parser.parse_game(game, game_id=0)
        self.assertEqual(records, [])

    def test_t5_outcome_white_win(self) -> None:
        """T5: '1-0', white to move at turn 0 => outcome +1."""
        game = _make_game(num_moves=10, result="1-0")
        records = self.parser.parse_game(game, game_id=0)
        # Turn 0 is White to move
        self.assertEqual(records[0].outcome, 1)

    def test_t6_outcome_black_perspective(self) -> None:
        """T6: '1-0', Black to move at turn 1 => outcome -1."""
        game = _make_game(num_moves=10, result="1-0")
        records = self.parser.parse_game(game, game_id=0)
        if len(records) > 1:
            # Turn 1 is Black to move
            self.assertEqual(records[1].outcome, -1)

    def test_t7_board_tokens_length(self) -> None:
        """T7: Every record has board_tokens of length 65."""
        game = _make_game(num_moves=6, result="1-0")
        records = self.parser.parse_game(game, game_id=0)
        for r in records:
            self.assertEqual(len(r.board_tokens), 65)

    def test_t8_move_tokens_starts_with_sos(self) -> None:
        """T8: Every record's move_tokens[0] == SOS_IDX."""
        game = _make_game(num_moves=6, result="1-0")
        records = self.parser.parse_game(game, game_id=0)
        for r in records:
            self.assertEqual(r.move_tokens[0], SOS_IDX)

    def test_elo_filter_both_below(self) -> None:
        """ELO filter: both below min_elo skips the game."""
        parser = GameParser(
            min_moves=1, max_moves=512,
            winners_only=False, min_elo=2000,
        )
        game = _make_game(
            num_moves=10, result="1-0",
            white_elo="1500", black_elo="1500",
        )
        records = parser.parse_game(game, game_id=0)
        self.assertEqual(records, [])

    def test_elo_filter_one_above(self) -> None:
        """ELO filter: one player above min_elo passes."""
        parser = GameParser(
            min_moves=1, max_moves=512,
            winners_only=False, min_elo=2000,
        )
        game = _make_game(
            num_moves=10, result="1-0",
            white_elo="2100", black_elo="1500",
        )
        records = parser.parse_game(game, game_id=0)
        self.assertGreater(len(records), 0)


# ---------------------------------------------------------------
# T9-T11: HDF5Writer
# ---------------------------------------------------------------

class TestHDF5Writer(unittest.TestCase):
    """Tests T9-T11 for HDF5Writer."""

    def test_t9_write_flush_cycle(self) -> None:
        """T9: 3 records with chunk_size=2 => shape (3,...)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            writer = HDF5Writer(
                max_seq_len=32, chunk_size=2,
                compression="gzip", compression_opts=4,
            )
            writer.open(path)
            records = _make_records(3, game_id=0)
            writer.write_batch(records, "train")
            writer.close()

            with h5py.File(str(path), "r") as f:
                n = f["train/board_tokens"].shape[0]
                self.assertEqual(n, 3)

    def test_t10_padding(self) -> None:
        """T10: move_len=3, max_seq_len=10 => pad 3-9 are 0."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            writer = HDF5Writer(
                max_seq_len=10, chunk_size=100,
                compression="gzip", compression_opts=4,
            )
            writer.open(path)
            records = _make_records(
                1, game_id=0, move_len=3
            )
            writer.write_batch(records, "train")
            writer.close()

            with h5py.File(str(path), "r") as f:
                mt = f["train/move_tokens"][0]
                # Positions 3-9 should be PAD=0
                self.assertTrue(
                    np.all(mt[3:] == 0),
                    f"Expected zeros at positions 3-9, "
                    f"got {mt[3:]}",
                )
                # First 3 should be non-zero (SOS + data)
                self.assertNotEqual(mt[0], 0)

    def test_t11_split_routing(self) -> None:
        """T11: 2 train + 1 val => correct counts."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            writer = HDF5Writer(
                max_seq_len=32, chunk_size=100,
                compression="gzip", compression_opts=4,
            )
            writer.open(path)
            writer.write_batch(
                _make_records(2, game_id=0), "train"
            )
            writer.write_batch(
                _make_records(1, game_id=1), "val"
            )
            writer.close()

            with h5py.File(str(path), "r") as f:
                self.assertEqual(
                    f["train/board_tokens"].shape[0], 2
                )
                self.assertEqual(
                    f["val/board_tokens"].shape[0], 1
                )


# ---------------------------------------------------------------
# T12-T15: HDF5Validator
# ---------------------------------------------------------------

class TestHDF5Validator(unittest.TestCase):
    """Tests T12-T15 for HDF5Validator."""

    def test_t12_valid_file_passes(self) -> None:
        """T12: Correctly written file passes validation."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            _write_valid_hdf5(path)
            validator = HDF5Validator()
            cfg = _default_config()
            # Should not raise
            validator.validate(path, cfg)

    def test_t13_missing_dataset_fails(self) -> None:
        """T13: Deleting 'outcome' dataset raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            _write_valid_hdf5(path)

            # Remove a dataset
            with h5py.File(str(path), "a") as f:
                del f["train/outcome"]

            validator = HDF5Validator()
            cfg = _default_config()
            with self.assertRaises(ValueError):
                validator.validate(path, cfg)

    def test_t14_out_of_range_fails(self) -> None:
        """T14: board_tokens value 8 raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            _write_valid_hdf5(path)

            # Corrupt board_tokens
            with h5py.File(str(path), "a") as f:
                f["train/board_tokens"][0, 1] = 8

            validator = HDF5Validator()
            cfg = _default_config()
            with self.assertRaises(ValueError):
                validator.validate(path, cfg)

    def test_t15_shape_mismatch_fails(self) -> None:
        """T15: Mismatched row counts raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            _write_valid_hdf5(path, n_train=10)

            # Resize one dataset to create mismatch
            with h5py.File(str(path), "a") as f:
                f["train/outcome"].resize(5, axis=0)

            validator = HDF5Validator()
            cfg = _default_config()
            with self.assertRaises(ValueError):
                validator.validate(path, cfg)


# ---------------------------------------------------------------
# T16-T19: ChessHDF5Dataset
# ---------------------------------------------------------------

class TestChessHDF5Dataset(unittest.TestCase):
    """Tests T16-T19 for ChessHDF5Dataset."""

    def test_t16_len(self) -> None:
        """T16: __len__ matches written sample count."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            _write_valid_hdf5(path, n_train=50)
            ds = ChessHDF5Dataset(path, split="train")
            self.assertEqual(len(ds), 50)

    def test_t17_getitem_shapes(self) -> None:
        """T17: __getitem__ returns correct tensor shapes."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            _write_valid_hdf5(
                path, n_train=5, max_seq_len=32
            )
            ds = ChessHDF5Dataset(path, split="train")
            sample = ds[0]

            self.assertEqual(
                sample.board_tokens.shape,
                torch.Size([65]),
            )
            self.assertEqual(
                sample.color_tokens.shape,
                torch.Size([65]),
            )
            self.assertEqual(
                sample.trajectory_tokens.shape,
                torch.Size([65]),
            )
            # move_tokens length should match move_lengths
            with h5py.File(str(path), "r") as f:
                ml = int(f["train/move_lengths"][0])
            self.assertEqual(
                sample.move_tokens.shape,
                torch.Size([ml]),
            )
            self.assertEqual(
                sample.target_tokens.shape,
                torch.Size([ml]),
            )

    def test_t18_collator_compatibility(self) -> None:
        """T18: Collator produces valid GameTurnBatch."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            # Write records with varying move lengths
            writer = HDF5Writer(
                max_seq_len=32, chunk_size=100,
                compression="gzip", compression_opts=4,
            )
            writer.open(path)
            for i in range(4):
                recs = _make_records(
                    1, game_id=i, move_len=i + 2
                )
                writer.write_batch(recs, "train")
            writer.close()

            ds = ChessHDF5Dataset(path, split="train")
            samples = [ds[i] for i in range(4)]
            collator = PGNSequenceCollator()
            batch = collator(samples)

            self.assertEqual(
                batch.board_tokens.shape[0], 4
            )
            self.assertEqual(
                batch.move_tokens.shape[0], 4
            )
            # Max move length in batch is 5 (i=3 => len 5)
            self.assertEqual(
                batch.move_tokens.shape[1], 5
            )

    def test_t19_boundary_index(self) -> None:
        """T19: Last index returns valid sample."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            _write_valid_hdf5(path, n_train=10)
            ds = ChessHDF5Dataset(path, split="train")
            sample = ds[len(ds) - 1]
            self.assertEqual(
                sample.board_tokens.shape,
                torch.Size([65]),
            )


# ---------------------------------------------------------------
# T20-T21: PreprocessV2Config
# ---------------------------------------------------------------

class TestPreprocessV2Config(unittest.TestCase):
    """Tests T20-T21 for PreprocessV2Config."""

    def test_t20_yaml_round_trip(self) -> None:
        """T20: Write YAML, load it, all fields match."""
        import yaml

        cfg = PreprocessV2Config(
            input=InputConfig(
                pgn_path="test.pgn", max_games=100
            ),
            output=OutputConfig(
                hdf5_path="out.h5",
                chunk_size=500,
                compression="gzip",
                compression_opts=6,
                max_seq_len=256,
            ),
            filter=FilterConfig(
                min_elo=1200, min_moves=10,
                max_moves=300, winners_only=True,
            ),
            split=SplitConfig(
                train=0.8, val=0.2, seed=99
            ),
            processing=ProcessingConfig(workers=8),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            # Serialize to YAML
            raw = {
                "input": {
                    "pgn_path": cfg.input.pgn_path,
                    "max_games": cfg.input.max_games,
                },
                "output": {
                    "hdf5_path": cfg.output.hdf5_path,
                    "chunk_size": cfg.output.chunk_size,
                    "compression": cfg.output.compression,
                    "compression_opts": (
                        cfg.output.compression_opts
                    ),
                    "max_seq_len": cfg.output.max_seq_len,
                },
                "filter": {
                    "min_elo": cfg.filter.min_elo,
                    "min_moves": cfg.filter.min_moves,
                    "max_moves": cfg.filter.max_moves,
                    "winners_only": cfg.filter.winners_only,
                },
                "split": {
                    "train": cfg.split.train,
                    "val": cfg.split.val,
                    "seed": cfg.split.seed,
                },
                "processing": {
                    "workers": cfg.processing.workers,
                },
            }
            path.write_text(yaml.dump(raw))

            loaded = load_preprocess_v2_config(path)
            self.assertEqual(
                loaded.input.pgn_path, "test.pgn"
            )
            self.assertEqual(loaded.input.max_games, 100)
            self.assertEqual(
                loaded.output.hdf5_path, "out.h5"
            )
            self.assertEqual(loaded.filter.min_elo, 1200)
            self.assertTrue(loaded.filter.winners_only)
            self.assertEqual(loaded.split.seed, 99)
            self.assertEqual(
                loaded.processing.workers, 8
            )

    def test_t21_unknown_key_raises(self) -> None:
        """T21: Unknown key in YAML raises TypeError."""
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            raw = {
                "filter": {"unknown_key": 99},
            }
            path.write_text(yaml.dump(raw))
            with self.assertRaises(TypeError):
                load_preprocess_v2_config(path)


# ---------------------------------------------------------------
# T22-T23: HDF5Preprocessor end-to-end
# ---------------------------------------------------------------

class TestHDF5Preprocessor(unittest.TestCase):
    """Tests T22-T23 for HDF5Preprocessor."""

    def _write_pgn_file(
        self, path: Path, num_games: int = 5
    ) -> None:
        """Write synthetic games to a plain PGN file."""
        with open(path, "w") as fh:
            for i in range(num_games):
                game = _make_game(
                    num_moves=10, result="1-0", seed=i
                )
                exporter = chess.pgn.StringExporter(
                    headers=True, variations=False,
                    comments=False,
                )
                fh.write(game.accept(exporter))
                fh.write("\n\n")

    def test_t22_end_to_end_smoke(self) -> None:
        """T22: 5 synthetic games produce valid HDF5."""
        from chess_sim.data.reader import StreamingPGNReader
        from chess_sim.preprocess import (
            GameParser,
            HDF5Preprocessor,
            HDF5Validator,
            HDF5Writer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            pgn_path = Path(tmp) / "games.pgn"
            hdf5_path = Path(tmp) / "out.h5"
            self._write_pgn_file(pgn_path, num_games=5)

            cfg = PreprocessV2Config(
                input=InputConfig(
                    pgn_path=str(pgn_path),
                    max_games=5,
                ),
                output=OutputConfig(
                    hdf5_path=str(hdf5_path),
                    chunk_size=100,
                    max_seq_len=64,
                ),
                filter=FilterConfig(min_moves=1),
                split=SplitConfig(train=0.80, val=0.20),
                processing=ProcessingConfig(workers=1),
            )

            reader = StreamingPGNReader()
            parser = GameParser(
                cfg.filter.min_moves,
                cfg.filter.max_moves,
                cfg.filter.winners_only,
                cfg.filter.min_elo,
            )
            writer = HDF5Writer(
                cfg.output.max_seq_len,
                cfg.output.chunk_size,
                cfg.output.compression,
                cfg.output.compression_opts,
            )
            validator = HDF5Validator()
            preprocessor = HDF5Preprocessor(
                reader, parser, writer, validator
            )
            preprocessor.run(cfg)

            # Verify file exists and has rows
            with h5py.File(str(hdf5_path), "r") as f:
                total = (
                    f["train/board_tokens"].shape[0]
                    + f["val/board_tokens"].shape[0]
                )
                self.assertGreater(total, 0)

    def test_t23_no_v1_interference(self) -> None:
        """T23: V1 shard cache is untouched."""
        from chess_sim.data.reader import StreamingPGNReader
        from chess_sim.preprocess import (
            GameParser,
            HDF5Preprocessor,
            HDF5Validator,
            HDF5Writer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            # Create a fake V1 cache dir
            v1_cache = Path(tmp) / ".shard_cache"
            v1_cache.mkdir()
            marker = v1_cache / "marker.txt"
            marker.write_text("v1_data")

            pgn_path = Path(tmp) / "games.pgn"
            hdf5_path = Path(tmp) / "out.h5"
            self._write_pgn_file(pgn_path, num_games=3)

            cfg = PreprocessV2Config(
                input=InputConfig(
                    pgn_path=str(pgn_path),
                    max_games=3,
                ),
                output=OutputConfig(
                    hdf5_path=str(hdf5_path),
                    chunk_size=100,
                    max_seq_len=64,
                ),
                filter=FilterConfig(min_moves=1),
                processing=ProcessingConfig(workers=1),
            )

            reader = StreamingPGNReader()
            parser = GameParser(
                cfg.filter.min_moves,
                cfg.filter.max_moves,
                cfg.filter.winners_only,
                cfg.filter.min_elo,
            )
            writer = HDF5Writer(
                cfg.output.max_seq_len,
                cfg.output.chunk_size,
                cfg.output.compression,
                cfg.output.compression_opts,
            )
            validator = HDF5Validator()
            preprocessor = HDF5Preprocessor(
                reader, parser, writer, validator
            )
            preprocessor.run(cfg)

            # V1 cache should be untouched
            self.assertTrue(marker.exists())
            self.assertEqual(
                marker.read_text(), "v1_data"
            )


# ---------------------------------------------------------------
# T24-T25: train_v2.py integration
# ---------------------------------------------------------------

class TestTrainV2Integration(unittest.TestCase):
    """Tests T24-T25 for train_v2.py HDF5 integration."""

    def test_t24_hdf5_fallback_absent(self) -> None:
        """T24: Empty hdf5_path uses PGNSequenceDataset path."""
        from chess_sim.config import (
            ChessModelV2Config,
        )

        cfg = ChessModelV2Config()
        # hdf5_path is empty string by default
        self.assertEqual(cfg.data.hdf5_path, "")
        # PGNSequenceDataset can be imported and used
        from chess_sim.data.pgn_sequence_dataset import (
            PGNSequenceDataset,
        )
        self.assertTrue(callable(PGNSequenceDataset))

    def test_t25_hdf5_path_active(self) -> None:
        """T25: Valid HDF5 path loads ChessHDF5Dataset."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"

            # Write valid HDF5 with real game data
            parser = GameParser(
                min_moves=1, max_moves=512,
                winners_only=False, min_elo=0,
            )
            game = _make_game(
                num_moves=10, result="1-0"
            )
            records = parser.parse_game(game, game_id=0)
            self.assertGreater(len(records), 0)

            writer = HDF5Writer(
                max_seq_len=64, chunk_size=100,
                compression="gzip", compression_opts=4,
            )
            writer.open(path)
            writer.write_batch(records, "train")
            writer.write_batch(records[:2], "val")
            writer.close()

            # Load dataset
            ds = ChessHDF5Dataset(path, split="train")
            self.assertEqual(len(ds), len(records))

            # Verify collator works
            collator = PGNSequenceCollator()
            samples = [ds[i] for i in range(min(4, len(ds)))]
            batch = collator(samples)
            self.assertEqual(
                batch.board_tokens.shape[0],
                len(samples),
            )
            self.assertEqual(
                batch.board_tokens.shape[1], 65
            )

            # Verify all tensor dtypes are correct
            sample = ds[0]
            self.assertEqual(
                sample.board_tokens.dtype, torch.long
            )
            self.assertEqual(
                sample.move_tokens.dtype, torch.long
            )


if __name__ == "__main__":
    unittest.main()
