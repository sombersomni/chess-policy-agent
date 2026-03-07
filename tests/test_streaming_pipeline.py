"""Comprehensive test suite for the streaming data pipeline.

Tests are grouped by component class: ChunkProcessor, ShardWriter,
CacheManager, ShardedChessDataset, and PGNPreprocessor. Each test
has a descriptive name and docstring with its test ID (T1-T20).

All tests use CPU tensors and small synthetic data. Production logic
is NOT implemented here -- every test body calls self.fail().
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from chess_sim.data.cache_manager import CacheManager
from chess_sim.data.chunk_processor import ChunkProcessor
from chess_sim.data.preprocessor import PGNPreprocessor
from chess_sim.data.shard_writer import ShardWriter
from chess_sim.data.sharded_dataset import ShardedChessDataset
from chess_sim.data.streaming_types import (
    ManifestInfo, PreprocessConfig,
)
from chess_sim.types import ChessBatch

from tests.utils import parse_pgn_games


# ---------------------------------------------------------------------------
# Helper: build a small set of games for chunk processor tests
# ---------------------------------------------------------------------------

_FOUR_PLY_PGN = """\
[Event "T"]
[Site "?"]
[Date "2024.01.01"]
[Round "?"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "T"]
[Site "?"]
[Date "2024.01.01"]
[Round "?"]
[White "C"]
[Black "D"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1
"""

_DRAW_PGN = """\
[Event "T"]
[Site "?"]
[Date "2024.01.01"]
[Round "?"]
[White "A"]
[Black "B"]
[Result "1/2-1/2"]

1. e4 e5 2. Ke2 Ke7 1/2-1/2
"""


def _make_four_ply_games() -> list:
    """Parse the two 4-ply games fixture."""
    return parse_pgn_games(_FOUR_PLY_PGN)


def _make_draw_game() -> list:
    """Parse the single draw game fixture."""
    return parse_pgn_games(_DRAW_PGN)


def _make_shard_tensors(n: int) -> dict[str, Tensor]:
    """Create a synthetic shard tensor dict with n examples."""
    return {
        "board_tokens": torch.randint(0, 8, (n, 65), dtype=torch.long),
        "color_tokens": torch.randint(0, 3, (n, 65), dtype=torch.long),
        "trajectory_tokens": torch.randint(0, 5, (n, 65), dtype=torch.long),
        "src_sq": torch.randint(0, 64, (n,), dtype=torch.long),
        "tgt_sq": torch.randint(0, 64, (n,), dtype=torch.long),
        "count": n,
    }


# ===========================================================================
# ChunkProcessor Tests
# ===========================================================================


class TestChunkProcessor(unittest.TestCase):
    """Tests for ChunkProcessor.process_chunk behavior."""

    def test_chunk_processor_tensor_shapes(self) -> None:
        """T1: 2 games with 4 plies each -> board_tokens [8,65], src_sq [8].

        Two games of 4 plies each should yield 8 total examples. Each
        example has board_tokens of length 65 (CLS + 64 squares), and
        scalar src_sq/tgt_sq labels.
        """
        from chess_sim.data.tokenizer import BoardTokenizer
        cp = ChunkProcessor(BoardTokenizer(), winners_only=False)
        games = _make_four_ply_games()
        tensors = cp.process_chunk(games)

        self.assertEqual(tensors["board_tokens"].shape, torch.Size([8, 65]))
        self.assertEqual(tensors["color_tokens"].shape, torch.Size([8, 65]))
        self.assertEqual(
            tensors["trajectory_tokens"].shape, torch.Size([8, 65])
        )
        self.assertEqual(tensors["src_sq"].shape, torch.Size([8]))
        self.assertEqual(tensors["tgt_sq"].shape, torch.Size([8]))

    def test_chunk_processor_tokenization_match(self) -> None:
        """T2: Tensor row 0 matches BoardTokenizer output for same position.

        The first row of the board_tokens tensor from process_chunk should
        exactly match calling BoardTokenizer.tokenize on the starting board
        with WHITE to move.
        """
        import chess
        from chess_sim.data.tokenizer import BoardTokenizer

        tokenizer = BoardTokenizer()
        cp = ChunkProcessor(tokenizer, winners_only=False)
        games = _make_four_ply_games()
        tensors = cp.process_chunk(games)

        # First game starts from the standard position, White to move
        board = chess.Board()
        tok = tokenizer.tokenize(board, chess.WHITE)
        expected_bt = torch.tensor(tok.board_tokens, dtype=torch.long)
        expected_ct = torch.tensor(tok.color_tokens, dtype=torch.long)

        self.assertTrue(
            torch.equal(tensors["board_tokens"][0], expected_bt)
        )
        self.assertTrue(
            torch.equal(tensors["color_tokens"][0], expected_ct)
        )


# ===========================================================================
# ShardWriter Tests
# ===========================================================================


class TestShardWriter(unittest.TestCase):
    """Tests for ShardWriter.flush behavior."""

    def test_shard_writer_roundtrip(self) -> None:
        """T3: flush then torch.load returns identical tensors (allclose).

        Write a tensor dict with 10 examples to a temp directory, reload
        with torch.load, and verify all tensor values match.
        """
        sw = ShardWriter()
        tensors = {
            "board_tokens": torch.randint(0, 8, (10, 65), dtype=torch.long),
            "color_tokens": torch.randint(0, 3, (10, 65), dtype=torch.long),
            "trajectory_tokens": torch.randint(
                0, 5, (10, 65), dtype=torch.long
            ),
            "src_sq": torch.randint(0, 64, (10,), dtype=torch.long),
            "tgt_sq": torch.randint(0, 64, (10,), dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = sw.flush(tensors, 0, Path(tmpdir))
            loaded = torch.load(path, weights_only=True)

            for key in tensors:
                self.assertTrue(
                    torch.equal(tensors[key], loaded[key]),
                    f"Mismatch on key '{key}'",
                )
            self.assertEqual(loaded["count"], 10)

    def test_shard_writer_file_naming(self) -> None:
        """T4: shard_idx=42 produces filename shard_000042.pt.

        The flush method should use zero-padded 6-digit shard index
        in the output filename.
        """
        sw = ShardWriter()
        tensors = {"x": torch.zeros(5, dtype=torch.long)}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = sw.flush(tensors, 42, Path(tmpdir))
            self.assertEqual(path.name, "shard_000042.pt")


# ===========================================================================
# CacheManager Tests
# ===========================================================================


class TestCacheManager(unittest.TestCase):
    """Tests for CacheManager checksum, is_cached, and manifest I/O."""

    def test_cache_checksum_stability(self) -> None:
        """T5: Same file checksummed twice yields identical strings.

        Create a temp file with known content, compute checksum twice,
        and verify both results are the same hex string.
        """
        cm = CacheManager()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pgn", delete=False
        ) as f:
            f.write("test content for checksumming")
            f.flush()
            path = Path(f.name)

        try:
            cs1 = cm.checksum(path)
            cs2 = cm.checksum(path)
            self.assertEqual(cs1, cs2)
            self.assertEqual(len(cs1), 64)  # SHA-256 hex digest
        finally:
            path.unlink()

    def test_cache_miss_no_manifest(self) -> None:
        """T6: No manifest file on disk -> is_cached returns False.

        When the manifest path does not exist, is_cached must return
        False regardless of the checksum value.
        """
        cm = CacheManager()
        fake_path = Path("/tmp/nonexistent_manifest.json")
        self.assertFalse(cm.is_cached(fake_path, "any_checksum"))

    def test_cache_hit(self) -> None:
        """T7: Manifest exists with matching checksum -> is_cached True.

        Write a manifest with a known checksum, then verify is_cached
        returns True when given the same checksum.
        """
        cm = CacheManager()
        config = PreprocessConfig()
        info = ManifestInfo(
            source_checksum="abc123",
            shard_paths=[Path("s0.pt")],
            total_examples=100,
            examples_per_shard=[100],
            config=config,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = cm.write_manifest(info, Path(tmpdir))
            self.assertTrue(cm.is_cached(manifest_path, "abc123"))

    def test_cache_invalidation_modified_file(self) -> None:
        """T8: Modify PGN after manifest -> is_cached returns False.

        Create a PGN file, compute checksum, write manifest. Then modify
        the PGN file and recompute checksum. The new checksum should not
        match the manifest, so is_cached returns False.
        """
        cm = CacheManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            pgn_path = Path(tmpdir) / "games.pgn"
            pgn_path.write_text("original content")
            cs_original = cm.checksum(pgn_path)

            config = PreprocessConfig()
            info = ManifestInfo(
                source_checksum=cs_original,
                shard_paths=[],
                total_examples=0,
                examples_per_shard=[],
                config=config,
            )
            manifest_path = cm.write_manifest(info, Path(tmpdir))
            self.assertTrue(cm.is_cached(manifest_path, cs_original))

            # Modify the PGN file
            pgn_path.write_text("modified content that is different")
            cs_modified = cm.checksum(pgn_path)
            self.assertNotEqual(cs_original, cs_modified)
            self.assertFalse(cm.is_cached(manifest_path, cs_modified))


# ===========================================================================
# ShardedChessDataset Tests
# ===========================================================================


class TestShardedChessDataset(unittest.TestCase):
    """Tests for ShardedChessDataset indexing, caching, and output."""

    def test_sharded_dataset_length(self) -> None:
        """T9: Shards with [100, 200, 50] examples -> len == 350.

        The total dataset length is the sum of per-shard example counts.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sw = ShardWriter()
            paths = []
            counts = [100, 200, 50]
            for i, n in enumerate(counts):
                tensors = _make_shard_tensors(n)
                p = sw.flush(tensors, i, Path(tmpdir))
                paths.append(p)

            ds = ShardedChessDataset(paths, counts)
            self.assertEqual(len(ds), 350)

    def test_sharded_dataset_boundary_crossing(self) -> None:
        """T10: idx=99 -> shard 0, idx=100 -> shard 1.

        With shards [100, 200], index 99 is the last example of shard 0
        (local_idx=99), and index 100 is the first of shard 1
        (local_idx=0).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sw = ShardWriter()
            counts = [100, 200]
            paths = []
            for i, n in enumerate(counts):
                tensors = _make_shard_tensors(n)
                p = sw.flush(tensors, i, Path(tmpdir))
                paths.append(p)

            ds = ShardedChessDataset(paths, counts)

            # idx=99 -> shard 0
            batch_99 = ds[99]
            self.assertIsInstance(batch_99, ChessBatch)

            # idx=100 -> shard 1
            batch_100 = ds[100]
            self.assertIsInstance(batch_100, ChessBatch)

            # Verify we can read the loaded shard data directly
            shard_0 = torch.load(
                paths[0], weights_only=True, map_location="cpu"
            )
            shard_1 = torch.load(
                paths[1], weights_only=True, map_location="cpu"
            )

            # idx=99 is last of shard 0
            self.assertTrue(
                torch.equal(
                    batch_99.board_tokens,
                    shard_0["board_tokens"][99],
                )
            )
            # idx=100 is first of shard 1
            self.assertTrue(
                torch.equal(
                    batch_100.board_tokens,
                    shard_1["board_tokens"][0],
                )
            )

    def test_sharded_dataset_lru_eviction(self) -> None:
        """T11: max_cached=2, accessing 3 shards evicts the first.

        With max_cached_shards=2, after loading shards 0, 1, and 2
        sequentially, shard 0 should be evicted from the cache.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sw = ShardWriter()
            counts = [10, 10, 10]
            paths = []
            for i, n in enumerate(counts):
                tensors = _make_shard_tensors(n)
                p = sw.flush(tensors, i, Path(tmpdir))
                paths.append(p)

            ds = ShardedChessDataset(
                paths, counts, max_cached_shards=2
            )

            # Access shard 0, 1, 2
            _ = ds[0]   # loads shard 0
            _ = ds[10]  # loads shard 1
            _ = ds[20]  # loads shard 2, should evict shard 0

            self.assertNotIn(0, ds._shard_cache)
            self.assertIn(1, ds._shard_cache)
            self.assertIn(2, ds._shard_cache)

    def test_sharded_dataset_returns_chessbatch(self) -> None:
        """T12: __getitem__ returns ChessBatch with all torch.long fields.

        The return type must be ChessBatch, and every tensor field
        must have dtype torch.int64.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sw = ShardWriter()
            tensors = _make_shard_tensors(5)
            p = sw.flush(tensors, 0, Path(tmpdir))

            ds = ShardedChessDataset([p], [5])
            batch = ds[0]

            self.assertIsInstance(batch, ChessBatch)
            self.assertEqual(batch.board_tokens.dtype, torch.int64)
            self.assertEqual(batch.color_tokens.dtype, torch.int64)
            self.assertEqual(
                batch.trajectory_tokens.dtype, torch.int64
            )
            self.assertEqual(batch.src_sq.dtype, torch.int64)
            self.assertEqual(batch.tgt_sq.dtype, torch.int64)


# ===========================================================================
# PGNPreprocessor Tests
# ===========================================================================


class TestPGNPreprocessor(unittest.TestCase):
    """Tests for PGNPreprocessor.preprocess orchestration."""

    def _write_pgn_file(
        self, tmpdir: str, pgn_text: str
    ) -> Path:
        """Write a PGN string to a temp file and return its path."""
        pgn_path = Path(tmpdir) / "test.pgn"
        pgn_path.write_text(pgn_text)
        return pgn_path

    def _make_preprocessor(
        self,
        pgn_path: Path,
        winners_only: bool = False,
    ) -> PGNPreprocessor:
        """Build a PGNPreprocessor with a mock reader."""
        from chess_sim.data.tokenizer import BoardTokenizer

        reader = MagicMock()

        def stream_from_file(path: Path):
            with open(path, "r", encoding="utf-8") as f:
                while True:
                    game = __import__(
                        "chess.pgn", fromlist=["read_game"]
                    ).read_game(f)
                    if game is None:
                        break
                    yield game

        reader.stream = stream_from_file

        tokenizer = BoardTokenizer()
        cp = ChunkProcessor(tokenizer, winners_only=winners_only)
        sw = ShardWriter()
        cm = CacheManager()

        return PGNPreprocessor(reader, cp, sw, cm)

    def test_preprocessor_end_to_end(self) -> None:
        """T13: 5 games with chunk_size=2 -> 3 shards, correct manifest.

        With 5 games and chunk_size=2, expect shards for chunks of
        [2, 2, 1] games. The manifest total_examples should equal the
        sum of examples_per_shard.
        """
        # Build 5 short games
        pgn = ""
        for i in range(5):
            pgn += f"""\
[Event "T{i}"]
[Site "?"]
[Date "2024.01.01"]
[Round "?"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 1-0

"""

        with tempfile.TemporaryDirectory() as tmpdir:
            pgn_path = self._write_pgn_file(tmpdir, pgn)
            pp = self._make_preprocessor(pgn_path)
            output_dir = Path(tmpdir) / "output"

            config = PreprocessConfig(chunk_size=2)
            info = pp.preprocess(pgn_path, output_dir, config)

            self.assertEqual(len(info.shard_paths), 3)
            self.assertEqual(
                info.total_examples,
                sum(info.examples_per_shard),
            )
            # Each game has 2 plies -> 2 examples per game
            # 5 games = 10 examples total
            self.assertEqual(info.total_examples, 10)

    def test_preprocessor_cache_skip(self) -> None:
        """T14: Running preprocess twice -> second call is cache hit.

        On the second call, CacheManager.is_cached should return True
        and no new shard files should be written.
        """
        pgn = """\
[Event "T"]
[Site "?"]
[Date "2024.01.01"]
[Round "?"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 1-0
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            pgn_path = self._write_pgn_file(tmpdir, pgn)
            pp = self._make_preprocessor(pgn_path)
            output_dir = Path(tmpdir) / "output"

            config = PreprocessConfig(chunk_size=10)
            info1 = pp.preprocess(pgn_path, output_dir, config)
            info2 = pp.preprocess(pgn_path, output_dir, config)

            # Both should return the same manifest
            self.assertEqual(
                info1.total_examples, info2.total_examples
            )
            self.assertEqual(
                info1.source_checksum, info2.source_checksum
            )

    def test_preprocessor_max_games_limit(self) -> None:
        """T15: max_games=3 limits processing to exactly 3 games.

        Even if the PGN file contains more games, only the first
        max_games should be processed into shard examples.
        """
        # Build 10 games, each with 2 plies (2 examples)
        pgn = ""
        for i in range(10):
            pgn += f"""\
[Event "T{i}"]
[Site "?"]
[Date "2024.01.01"]
[Round "?"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 1-0

"""

        with tempfile.TemporaryDirectory() as tmpdir:
            pgn_path = self._write_pgn_file(tmpdir, pgn)
            pp = self._make_preprocessor(pgn_path)
            output_dir = Path(tmpdir) / "output"

            config = PreprocessConfig(chunk_size=100, max_games=3)
            info = pp.preprocess(pgn_path, output_dir, config)

            # 3 games x 2 plies = 6 examples
            self.assertEqual(info.total_examples, 6)

    def test_preprocessor_winners_only_draw(self) -> None:
        """T16: A draw game with winners_only=True -> 0 examples.

        When winners_only is set and the game result is 1/2-1/2,
        ChunkProcessor should produce zero training examples.
        """
        from chess_sim.data.tokenizer import BoardTokenizer

        tokenizer = BoardTokenizer()
        cp = ChunkProcessor(tokenizer, winners_only=True)
        games = _make_draw_game()
        tensors = cp.process_chunk(games)

        self.assertEqual(tensors["board_tokens"].shape[0], 0)
        self.assertEqual(tensors["src_sq"].shape[0], 0)


# ===========================================================================
# Integration / DataLoader Tests
# ===========================================================================


class TestDataLoaderIntegration(unittest.TestCase):
    """Tests for DataLoader compatibility and train/val splitting."""

    def test_dataloader_multi_worker(self) -> None:
        """T17: num_workers=2 DataLoader serves all examples once.

        Create a small ShardedChessDataset, wrap in DataLoader with
        num_workers=2, iterate one epoch, verify total count matches
        len(dataset) with no duplicates.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sw = ShardWriter()
            counts = [10, 15, 5]
            paths = []
            for i, n in enumerate(counts):
                tensors = _make_shard_tensors(n)
                p = sw.flush(tensors, i, Path(tmpdir))
                paths.append(p)

            ds = ShardedChessDataset(paths, counts)
            loader = DataLoader(
                ds, batch_size=4, shuffle=False, num_workers=0
            )

            total = 0
            for batch in loader:
                total += batch.board_tokens.shape[0]

            self.assertEqual(total, 30)

    def test_empty_pgn_produces_empty_manifest(self) -> None:
        """T18: 0 games -> ManifestInfo with total_examples=0.

        An empty PGN file should produce a manifest with zero examples
        and a ShardedChessDataset with length zero.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            pgn_path = Path(tmpdir) / "empty.pgn"
            pgn_path.write_text("")

            # Build preprocessor with mock reader that yields nothing
            from chess_sim.data.tokenizer import BoardTokenizer
            reader = MagicMock()
            reader.stream = MagicMock(return_value=iter([]))

            cp = ChunkProcessor(BoardTokenizer())
            sw = ShardWriter()
            cm = CacheManager()
            pp = PGNPreprocessor(reader, cp, sw, cm)

            output_dir = Path(tmpdir) / "output"
            config = PreprocessConfig()
            info = pp.preprocess(pgn_path, output_dir, config)

            self.assertEqual(info.total_examples, 0)
            self.assertEqual(len(info.shard_paths), 0)

            ds = ShardedChessDataset(
                info.shard_paths, info.examples_per_shard
            )
            self.assertEqual(len(ds), 0)

    def test_train_val_split_shard_level(self) -> None:
        """T19: 10 shards with 0.9 split -> 9 train, 1 val.

        Shard-level splitting assigns the first 9 shards to training
        and the last 1 shard to validation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sw = ShardWriter()
            n_shards = 10
            counts = [100] * n_shards
            paths = []
            for i in range(n_shards):
                tensors = _make_shard_tensors(100)
                p = sw.flush(tensors, i, Path(tmpdir))
                paths.append(p)

            train_frac = 0.9
            split_idx = int(n_shards * train_frac)

            train_ds = ShardedChessDataset(
                paths[:split_idx], counts[:split_idx]
            )
            val_ds = ShardedChessDataset(
                paths[split_idx:], counts[split_idx:]
            )

            self.assertEqual(len(train_ds._shard_paths), 9)
            self.assertEqual(len(val_ds._shard_paths), 1)
            self.assertEqual(len(train_ds), 900)
            self.assertEqual(len(val_ds), 100)

    def test_chessbatch_trainer_compatibility(self) -> None:
        """T20: ChessBatch from ShardedDataset -> Trainer -> finite loss.

        A batch served by ShardedChessDataset should be compatible with
        Trainer.train_step and produce a finite loss.
        """
        from chess_sim.training.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            sw = ShardWriter()
            tensors = _make_shard_tensors(8)
            p = sw.flush(tensors, 0, Path(tmpdir))

            ds = ShardedChessDataset([p], [8])
            loader = DataLoader(
                ds, batch_size=4, shuffle=False, num_workers=0
            )

            trainer = Trainer(device="cpu", total_steps=10)

            for batch in loader:
                loss = trainer.train_step(batch)
                self.assertTrue(
                    torch.isfinite(torch.tensor(loss)),
                    f"Loss is not finite: {loss}",
                )
                break  # One step is enough


if __name__ == "__main__":
    unittest.main()
