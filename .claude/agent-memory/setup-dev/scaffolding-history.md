# Scaffolding History (Detailed)

## Chess Encoder (2026-03-05)
- Module: protocols.py (6 Protocols), types.py (6 NamedTuples), data/*, model/*, training/*
- Tests: tests/test_tokenizer.py, test_embedding.py, test_encoder.py, test_heads.py, test_loss.py, test_trainer.py, test_sampler.py, test_reader.py, test_dataset.py — 45 tests total
- Patterns: CLS at index 0, ignore_index=-1, nn.Module stubs raise in __init__, decorators use functools.wraps

## GUI Chess Viewer (2026-03-06)
- Module: scripts/gui/ (3 Protocols, formatters, game_controller, board_panel, stats_panel, viewer)
- Tests: tests/gui/test_game_controller.py, test_stats_panel.py
- Key: Pure formatting extracted to avoid tkinter in tests; GameController has zero tkinter deps

## Streaming Data Pipeline (2026-03-06)
- Module: protocols.py (+3 Protocols), data/streaming_types.py, chunk_processor.py, shard_writer.py, cache_manager.py, preprocessor.py, sharded_dataset.py
- Tests: tests/test_streaming_pipeline.py — 20 tests
- Key: ShardedChessDataset uses OrderedDict LRU + bisect for O(log S) lookup

## ChessModel v2 Encoder-Decoder (2026-03-07)
- Module: protocols.py (+3 Protocols), types.py (+4 NamedTuples), config.py (+3 dataclasses), data/move_vocab.py, move_tokenizer.py, pgn_sequence_dataset.py, model/move_embedding.py, decoder.py, chess_model.py, training/phase1_trainer.py, phase2_trainer.py
- Tests: tests/test_v2_skeleton.py — 42 tests
- Key: MoveVocab 3 special + ~1968 moves, DecoderConfig d_model=256

## Aim Experiment Tracking (2026-03-07)
- Module: config.py (+AimConfig), tracking/{__init__,protocol,aim_tracker,noop_tracker,factory}.py, training/phase1_trainer.py (+tracker)
- Tests: tests/test_aim_tracking.py — 15 tests
- Key: NoOpTracker uses pass (production fallback), aim>=3.17 optional

## Phase 2 Self-Play RL (2026-03-08)
- Module: config.py (Phase2Config), types.py (+PlyTuple, EpisodeRecord, ValueHeadOutput), protocols.py (+Recordable, Computable, Updatable), env/self_play_source.py, model/value_heads.py, training/{episode_recorder,reward_computer,ema_updater,self_play_loop}.py
- Tests: tests/test_phase2_self_play.py — 15 tests
- Key: ValueHeads.__init__ calls super().__init__() then creates heads (NOT raising NIE)
