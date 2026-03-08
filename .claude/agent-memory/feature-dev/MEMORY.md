# Feature Dev Agent Memory

## Project Structure
- Source: `chess_sim/` with subpackages `data/`, `model/`, `training/`
- Tests: `tests/` using `unittest` + `pytest` + `parameterized`
- Types: `chess_sim/types.py` (all NamedTuples: TokenizedBoard, TrainingExample, ChessBatch, EncoderOutput, PredictionOutput, LabelTensors, GameTurnSample, GameTurnBatch, DecoderOutput, SelfPlayGame)
- Protocols: `chess_sim/protocols.py` (Tokenizable, Embeddable, Encodable, Predictable, Trainable, Samplable, Scorable)
- Test utils: `tests/utils.py` (make_synthetic_batch, make_prediction_output, make_training_examples, etc.)

## Key Implementation Decisions
- **Pre-LN Transformer**: Must use `norm_first=True` in TransformerEncoderLayer. PyTorch 2.x SDPA on CPU causes zero gradients in deeper layers with Post-LN (default).
- **enable_nested_tensor=False**: Required for TransformerEncoder to avoid SDPA fast-path issues.
- **NaN handling in LossComputer**: When all opponent labels are -1 (ignored), CE returns nan. Replace nan with zeros_like before summing.
- **requires_grad in test utils**: `make_prediction_output` needs `requires_grad=True` on logit tensors for backward tests.
- **Eval mode for checkpoint tests**: Dropout makes outputs non-deterministic in train mode; checkpoint roundtrip tests need `.eval()`.
- **Gitignore `data/`**: The `.gitignore` has `data/` which matches `chess_sim/data/`. New files in `chess_sim/data/` require `git add -f`.

## Encoder v2 (4-stream)
- `EmbeddingLayer` has 4 streams: piece_emb(8,256), color_emb(3,256), square_emb(65,256), activity_emb(9,256)
- `embed()`, `encode()` signatures: `(board_tokens, color_tokens, activity_tokens)`
- `TrainingExample` and `ChessBatch` include `activity_tokens` field (between color_tokens and src_sq)
- `ActivityScorer` in `chess_sim/data/scorer.py`: computes per-square activity from move history
- Square emb: geometric sin/cos init; Piece emb: role-feature init (7-elem tiled to 256, scaled 0.02)
- Checkpoint migration: `strict=False` when loading old 3-stream checkpoints
- `trainer.load_checkpoint()` uses `strict=False` for encoder (handles missing `trajectory_emb` in old checkpoints)

## GUI Module (`scripts/gui/`)
- `formatters.py` - pure functions, no tkinter deps, tested headlessly
- `game_controller.py` - tkinter-free GameSource impl, tested with real data
- `board_panel.py`, `stats_panel.py`, `viewer.py` - tkinter widgets, no headless tests
- Tests: `tests/gui/test_stats_panel.py` (7), `tests/gui/test_game_controller.py` (5)
- Test data: `data/games.pgn` + `checkpoints/winner_run_01.pt`

## Dependencies
- `torch`, `python-chess`, `zstandard` (in requirements.txt)
- `pytest`, `parameterized` (test deps, installed in .venv)

## Virtual Environment
- `.venv/` at project root
- Activate: `source .venv/bin/activate`
- Run tests: `python -m pytest tests/ -v`

## Test Files
- `tests/test_tokenizer.py` - T01-T03; `tests/test_embedding.py` - T04
- `tests/test_encoder.py` - T05, T09; `tests/test_heads.py` - T06
- `tests/test_loss.py` - T07; `tests/test_trainer.py` - T08, T19
- `tests/test_dataset.py` - T17, T18, T20
- `tests/test_reader.py`, `tests/test_sampler.py` - data pipeline
- `tests/test_evaluate.py` - TEV01-TEV14
- `tests/test_chess_encoder.py` - T26-T40: Encoder v2 tests
- `tests/test_streaming_pipeline.py` - T1-T20: Streaming pipeline

## Streaming Data Pipeline
- Types: `chess_sim/data/streaming_types.py` (PreprocessConfig, ManifestInfo)
- `chunk_processor.py` reuses `game_to_examples` from `scripts.train_real`
- `shard_writer.py` saves shard_{:06d}.pt via torch.save (includes "count" key)
- `cache_manager.py` SHA-256 of first 1MB + file size; manifest.json I/O
- `preprocessor.py` orchestrates: stream -> chunk -> shard -> manifest
- `sharded_dataset.py` bisect shard lookup, OrderedDict LRU cache
- `train_real.py` uses _PlainPGNReader adapter for .pgn/.zst support
- Shard files loaded with `weights_only=True`

## HDF5 Preprocessing Pipeline
- `chess_sim/preprocess/` package: `parse.py`, `writer.py`, `validate.py`, `preprocess.py`
- `chess_sim/data/hdf5_dataset.py`: `ChessHDF5Dataset` + `hdf5_worker_init`
- `chess_sim/data/tokenizer_utils.py`: extracted `make_trajectory_tokens` (shared)
- Config: `PreprocessV2Config` + sub-configs in `chess_sim/config.py`
- `DataConfig.hdf5_path` field; `scripts/train_v2.py` branches on it
- Tests: `tests/test_preprocess_v2.py` — T1-T25 (27 tests)
- `scripts/preprocess.py` + `configs/preprocess_v2.yaml`
- Protocols: HDF5Parseable, HDF5Writable, HDF5Readable, HDF5Validatable, HDF5Preprocessable
- New type: `RawTurnRecord` in `chess_sim/types.py`

## ChessModel v2 (Encoder-Decoder)
- `MoveVocab` in `chess_sim/data/move_vocab.py`: 1968 piece-reachable UCI moves + 3 special (PAD=0, SOS=1, EOS=2) = 1971 total
- Vocab enumerates only piece-reachable moves (not all 64*63 pairs): knight, king, rook/queen lines, bishop/queen diags, pawn moves with promotions
- `MoveTokenizer` in `chess_sim/data/move_tokenizer.py`: wraps MoveVocab, provides tokenize_game (SOS+moves+EOS) and build_legal_mask
- `MoveEmbedding` in `chess_sim/model/move_embedding.py`: token_emb + pos_emb, uses DecoderConfig
- `MoveDecoder` in `chess_sim/model/decoder.py`: TransformerDecoder + causal mask + output projection
- `ChessModel` in `chess_sim/model/chess_model.py`: encoder-decoder assembly, memory = CLS+squares [B,65,d_model]
- `PGNSequenceDataset` in `chess_sim/data/pgn_sequence_dataset.py`: per-ply samples, PGNSequenceCollator for padding
- `Phase1Trainer` in `chess_sim/training/phase1_trainer.py`: CE loss with teacher forcing, uses _to_device helper (not device_aware decorator)
- `Phase2Trainer` in `chess_sim/training/phase2_trainer.py`: REINFORCE self-play with policy gradient
- Config: `DecoderConfig` and `Phase2Config` in `chess_sim/config.py`; `ChessModelV2Config` = root config
- Tests: `tests/test_v2_skeleton.py` - TV01-TV17 (42 tests)
