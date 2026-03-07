# setup-dev Agent Memory

## Scaffolding Completed: Streaming Data Pipeline (2026-03-06)

### Module Layout
```
chess_sim/
├── protocols.py              # +3 Protocols: Preprocessable, ShardWritable, Cacheable
├── data/streaming_types.py   # PreprocessConfig, ManifestInfo (frozen dataclasses)
├── data/chunk_processor.py   # ChunkProcessor — games to dense tensors
├── data/shard_writer.py      # ShardWriter — implements ShardWritable
├── data/cache_manager.py     # CacheManager — implements Cacheable
├── data/preprocessor.py      # PGNPreprocessor — implements Preprocessable
└── data/sharded_dataset.py   # ShardedChessDataset(Dataset) — LRU shard cache
```

### Test Suite
```
tests/test_streaming_pipeline.py  # T1-T20 (20 tests, all self.fail)
  - TestChunkProcessor:       T1 (shapes), T2 (tokenization match)
  - TestShardWriter:          T3 (roundtrip), T4 (file naming)
  - TestCacheManager:         T5 (checksum stability), T6 (miss), T7 (hit), T8 (invalidation)
  - TestShardedChessDataset:  T9 (length), T10 (boundary), T11 (LRU eviction), T12 (ChessBatch type)
  - TestPGNPreprocessor:      T13 (e2e), T14 (cache skip), T15 (max_games), T16 (winners_only draw)
  - TestDataLoaderIntegration: T17 (multi-worker), T18 (empty pgn), T19 (train/val split), T20 (trainer compat)
```

### Key Decisions
- `.gitignore` has `data/` which matches `chess_sim/data/` — must use `git add -f` for data subpackage files
- Reuses `game_to_examples` and `_make_trajectory_tokens` from `scripts/train_real.py`
- No new dependencies — only torch, json, hashlib, pathlib, bisect from stdlib
- Shard file naming: `shard_{idx:06d}.pt`
- ShardedChessDataset uses OrderedDict for LRU + bisect for O(log S) shard lookup

---

## Scaffolding Completed: GUI Chess Viewer (2026-03-06)

### Module Layout
```
scripts/gui/
├── __init__.py            # 3 Protocols: Renderable, Navigable, GameSource
├── formatters.py          # _fmt_loss, _fmt_acc, _fmt_entropy (tkinter-free)
├── game_controller.py     # GameController — implements GameSource (no tkinter)
├── board_panel.py         # BoardPanel(tk.Frame) — implements Renderable, Navigable
├── stats_panel.py         # StatsPanel(tk.Frame) — implements Renderable
└── viewer.py              # ChessViewer — root window wiring panels
```

### Test Suite
```
tests/gui/__init__.py
tests/gui/test_game_controller.py   # GC01-GC05 (requires checkpoint, skip in CI)
tests/gui/test_stats_panel.py       # SP01-SP02 (7 tests, pure formatting)
```

### Key Decisions
- Pure formatting helpers extracted to `scripts/gui/formatters.py` to avoid tkinter import in headless tests
- `GameController` has zero tkinter deps — fully testable without display
- `tkinter` not available in CI/venv — tests importing tk-dependent modules will fail
- `matplotlib>=3.7` added to requirements.txt for embedding scatter plot
- StepResult imported from `scripts.evaluate`, not re-defined

---

## Scaffolding Completed: Chess Encoder (2026-03-05)

### Module Layout
```
chess_sim/
├── protocols.py          # 6 Protocols: Tokenizable, Embeddable, Encodable, Predictable, Trainable, Samplable
├── types.py              # 6 NamedTuples: TokenizedBoard, TrainingExample, ChessBatch, EncoderOutput, PredictionOutput, LabelTensors
├── data/tokenizer.py     # BoardTokenizer — implements Tokenizable
├── data/dataset.py       # ChessDataset — torch.utils.data.Dataset
├── data/reader.py        # StreamingPGNReader — zstandard-based streaming
├── data/sampler.py       # ReservoirSampler — implements Samplable
├── model/embedding.py    # EmbeddingLayer (nn.Module) — implements Embeddable
├── model/encoder.py      # ChessEncoder (nn.Module) — implements Encodable
├── model/heads.py        # PredictionHeads (nn.Module) — implements Predictable
├── training/loss.py      # LossComputer — 4x CE, ignore_index=-1 for opp heads
└── training/trainer.py   # Trainer — implements Trainable + 3 decorator stubs
```

### Test Suite
```
tests/utils.py            # Shared fixtures: make_synthetic_batch, make_small_pgn_fixture, make_initial_board_tokens
tests/test_tokenizer.py   # T01, T02, T03, T13, T14, T15, T16
tests/test_embedding.py   # T04
tests/test_encoder.py     # T05, T09
tests/test_heads.py       # T06
tests/test_loss.py        # T07
tests/test_trainer.py     # T08, T19
tests/test_sampler.py     # T10, T11
tests/test_reader.py      # T12
tests/test_dataset.py     # T17, T18, T20
```
Total: 45 tests, all raising NotImplementedError — zero import errors.

### Key Patterns Established
- `board_tokens` index offset: CLS at 0, squares are python-chess square + 1 (a1=1..h8=64)
- `ignore_index=-1` for opponent heads (last move in game has no opponent response)
- All nn.Module stubs have `__init__` raising NotImplementedError (no `super().__init__()` before it)
- `@log_metrics`, `@device_aware`, `@timed` are functools.wraps-based decorator stubs
- Chi-squared uniformity test for T10 is implemented manually (no scipy dependency)
- Virtualenv at `.venv/`; run tests with `.venv/bin/python -m unittest discover -s tests -v`

### Dependencies
- torch>=2.0, python-chess>=1.9, zstandard>=0.21, parameterized (all in requirements.txt)
