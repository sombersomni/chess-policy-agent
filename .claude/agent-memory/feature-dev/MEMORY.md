# Feature Dev Agent Memory

## Functional Utilities
- `chess_sim/functional.py`: reusable tensor ops (torch.nn.functional pattern)
  - `entropy_from_logits(logits)` -> per-position entropy via `Categorical.entropy()`
  - `mean_entropy(logits, mask)` -> masked mean entropy as float
  - Tests: `tests/test_functional.py` (9 tests)
- `_mean_entropy` in `phase1_trainer.py` is a thin backward-compat wrapper

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

## Phase 2 Self-Play RL Components
- `chess_sim/training/ema_updater.py`: EMA sync player->opponent (Updatable protocol)
- `chess_sim/training/episode_recorder.py`: accumulates PlyTuples, softmax-normalizes player entropies in finalize()
- `chess_sim/training/reward_computer.py`: temporal advantage + surprise reward per player ply
  - temporal: `outcome * gamma^t` (earlier plies get higher reward, t=0 is max)
  - surprise: `entropy * correct * reward_sign`; loss trajectory flips `correct` sign
- `chess_sim/model/value_heads.py`: two Linear heads (v_win, v_surprise) on detached CLS embedding
- `chess_sim/env/self_play_source.py`: mutable chess.Board implementing SimSource protocol
- `chess_sim/training/self_play_loop.py`: orchestrates episodes, PG loss + value MSE + EMA update
  - d_model detected via `player.encoder.embedding.d_model`
  - `_make_trajectory_tokens` imported from `phase2_trainer.py`
  - `MoveVocab.decode(idx)` maps action index -> UCI string
- `scripts/train_phase2.py`: CLI entry point for Phase 2 training
- Tests: `tests/test_phase2_self_play.py` - TC01-TC15 (15 tests)
- Key gotcha: when testing reward correctness, `dummy_probs` argmax must match `MoveTokenizer.tokenize_move(uci)` index

## Offline RL PGN Trainer (Phase 3)
- `chess_sim/training/training_utils.py`: shared `material_balance`, `l1_normalize`
- `chess_sim/training/pgn_replayer.py`: PGNReplayer -> list[OfflinePlyTuple] (Replayable protocol)
- `chess_sim/training/pgn_rl_reward_computer.py`: temporal gamma^(T-1-t) + material + check shaping (OfflineComputable protocol)
- `chess_sim/training/pgn_rl_trainer.py`: PGNRLTrainer with RSBC loss (reward-signed behavioral cloning)
- Config: `RLConfig`, `PGNRLConfig`, `load_pgn_rl_config` in `chess_sim/config.py`
- Types: `OfflinePlyTuple` in `chess_sim/types.py`
- Protocols: `Replayable`, `OfflineComputable` in `chess_sim/protocols.py`
- Script: `scripts/train_rl.py` + `configs/train_rl.yaml`
- Tests: `tests/test_pgn_rl_trainer.py` - T1-T12 (12 tests)
- Key design: trains BOTH sides simultaneously (not just player plies like SelfPlayLoop)
- Temporal discount: gamma^(T-1-t) so LAST ply gets gamma^0=1.0 (opposite of RewardComputer buggy direction)
- LR schedule: identical SequentialLR (warmup -> constant -> cosine) from Phase1Trainer
- **RSBC loss**: `_compute_rsbc_loss` replaces AWBC in `train_game`; formula: `mean(r_hat * CE)` where `r_hat = r / max(|r|)` per game
  - `label_smoothing=0.0` hardcoded (smoothing opposes anti-imitation on negative plies)
  - AWBC and entropy_bonus methods retained for backward compat but not called
  - Config: `lambda_rsbc=1.0`, `rsbc_normalize_per_game=True`, `lambda_awbc=0.0` (deprecated)
  - Tests: `TestPGNRLTrainerRSBC` (15 tests) + `TestRLConfigRSBC` (3 tests) in `tests/test_trainer.py` and `tests/test_config.py`
  - Design doc: `docs/rsbc_loss_redesign.md`

## RL HDF5 Preprocessing Pipeline
- `chess_sim/preprocess/rl_reader.py`: RLPGNReader — streams .pgn/.zst via StreamingPGNReader delegation
- `chess_sim/preprocess/rl_parser.py`: RLPlyParser — filters by train_color, returns list[RLPlyRecord] (plain lists, no tensors)
- `chess_sim/preprocess/rl_writer.py`: RLHdf5Writer — 11 datasets per split, move_uci as S5, prefix zero-padded
- `chess_sim/preprocess/rl_validator.py`: RLHdf5Validator — schema + value range + train_color attr checks
- `chess_sim/preprocess/rl_preprocessor.py`: RLHdf5Preprocessor — serial/parallel orchestration with _rl_parse_worker
- `chess_sim/data/rl_hdf5_dataset.py`: RLPlyHDF5Dataset -> OfflinePlyTuple, rl_hdf5_worker_init for multi-worker
- Config: `RLPreprocessConfig` (RLOutputConfig, RLFilterConfig) + `load_rl_preprocess_config` in `chess_sim/config.py`
- Script: `scripts/preprocess_rl.py` + `configs/preprocess_rl.yaml`
- Tests: `tests/test_rl_hdf5_pipeline.py` — 47 tests
- HDF5 schema: version="rl_1.0", 11 datasets (board_tokens, color_tokens, traj_tokens, move_prefix, prefix_lengths, move_uci, is_winner_ply, is_white_ply, is_draw_ply, game_id, ply_index)

## RSCE V4 Batched Training Pipeline
- `chess_sim/data/pgn_reward_preprocessor.py`: PGNRewardPreprocessor — PGN -> HDF5 with pre-normalized RSCE multipliers
  - `_encode_board(bt, ct, tt)` -> (65,3) float32 via torch.stack
  - HDF5 schema: board(N,65,3), color_tokens(N,65), target_move(N), multiplier(N), game_id(N), ply_idx(N), outcome(N)
  - Cache: SHA-256(first 1MB + file_size + max_games) + config attrs in HDF5 attributes
  - Per-game RSCE normalization: m = exp(-(R - r_ref)), m_hat = m * N / sum(m).clamp(1e-8)
- `chess_sim/data/chess_rl_dataset.py`: ChessRLDataset — torch Dataset wrapping HDF5
  - Lazy HDF5 open: `_h5 = None` until first `__getitem__` (h5py not fork-safe)
  - Val split: last N% of unique game_ids (by sorted order)
- `chess_sim/training/pgn_rl_trainer_v4.py`: PGNRLTrainerV4 — batched minibatch RSCE
  - Standalone (no V2/V3 inheritance); SOS-only prefix [B,1] for decoder
  - Board channels split from (65,3): bt=ch0.long(), ct=ch1.long(), tt=ch2.long()
- `scripts/train_rl_v4.py`: entry point; preprocess -> build datasets -> epoch loop
- Config: `RLConfig` added: hdf5_path, batch_size, num_workers, val_split_fraction, hdf5_chunk_size
- Types: `RLRewardRow` in `chess_sim/types.py`
- Tests: `tests/test_rsce_v4.py` — TC01-TC15 (15 tests)
- Key gotcha: synthetic HDF5 board must use valid indices (0-7, 0-2, 0-4), not random floats

## Phase 1 Auxiliary Heads
- `chess_sim/data/capture_map_builder.py`: pure python-chess, `build(board, turn) -> list[int]` (64 binary)
- `chess_sim/data/move_category_builder.py`: pure python-chess, `build(uci, board) -> int` (0-6: QUIET, CAPTURE_PAWN/MINOR/ROOK/QUEEN, CASTLE, PROMOTION)
- `chess_sim/model/auxiliary_heads.py`: `AuxiliaryHeads(nn.Module)` — capture_target_head(D,1), move_category_head(D,7), phase_head(D,3)
  - CLS heads use `cls_emb.detach()` by default (ablation: remove detach for gradient flow)
  - Returns `AuxLossOutput` namedtuple (capture_loss, category_loss, phase_loss)
- `compute_phase_labels(ply_idx, board)` in `pgn_rl_trainer_v4.py`: Opening(0) ply<20, Endgame(1) material<=15, Midgame(2) otherwise
- Config: `RLConfig` gains `use_aux_heads=False`, `lambda_capture=0.5`, `lambda_category=0.2`, `lambda_phase=0.05`
- `ChessRLDataset.__getitem__` returns 8-tuple (was 5): adds capture_map, move_category, ply_idx
- `PGNRewardPreprocessor` writes `capture_map(N,64)` and `move_category(N)` when `use_aux_heads=True`
- Tests: `tests/data/test_capture_map_builder.py` (7), `tests/data/test_move_category_builder.py` (9), `tests/test_aux_heads.py` (14)

## RL Self-Play Fine-Tuner
- `chess_sim/training/rl_finetune_trainer.py`: RLFinetuneTrainer + play_game + compute_returns + _log_prob_of_move
  - Three model copies: _policy (trained), _ref (frozen KL anchor), _shadow (EMA opponent)
  - REINFORCE with RunningMeanBaseline + KL divergence penalty
  - Policy plays White, shadow plays Black; terminal-only rewards
  - compute_returns: G_t = gamma^(T-1-t) * outcome (last ply = gamma^0)
  - _log_prob_of_move: SOS-only prefix, legal mask, log_softmax
- `scripts/finetune_rl.py`: CLI entry point, follows train_rl_v4.py pattern
- Config: `FinetuneConfig`, `FinetuneRLConfig`, `load_finetune_rl_config` in `chess_sim/config.py`
- Types: `PlyRecord`, `GameRecord` in `chess_sim/types.py`
- Tests: `tests/test_rl_finetune_trainer.py` — 27 tests (TC01-TC20)
- `_compute_trajectory_tokens` imported from `chess_sim/env/agent_adapter.py`
