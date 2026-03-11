# setup-dev Agent Memory

## Key Patterns
- `.gitignore` has `data/` and `env/` catch-all — must use `git add -f` for `chess_sim/data/*` and `chess_sim/env/*`
- Virtualenv at `.venv/`; run with `source .venv/bin/activate && python -m ...`
- All nn.Module stubs raise NotImplementedError in `__init__` (consistent pattern)
- Append to existing config.py, types.py, protocols.py — never modify existing code
- Config pattern: `@dataclass` with defaults, `__post_init__` for validation, `load_*_config()` loader
- Test pattern: `unittest.TestCase` + `parameterized.expand`, assert NotImplementedError for stubs
- No new deps without justification; ruff line limit 88 chars

## Detailed History
See [scaffolding-history.md](scaffolding-history.md) for all prior scaffolding details.

---

## Scaffolding Completed: Composite Reward Redesign (2026-03-10)

### Changes
- `chess_sim/types.py` — `OfflinePlyTuple`: removed default from `is_draw_ply`, added `material_delta: float`
- `chess_sim/training/pgn_replayer.py` — `_PIECE_VALUES`, `_material_of()` stub, `last_material` tracking in `replay()`
- `chess_sim/training/pgn_rl_reward_computer.py` — `compute()` now NIE stub with composite formula
- `chess_sim/config.py` — RLConfig: removed gamma/win_reward/loss_reward/draw_reward; added lambda_outcome/lambda_material/draw_reward_norm
- `chess_sim/training/pgn_rl_trainer.py` — sample_visuals DRY fix via replayer+reward_fn delegation
- `chess_sim/data/rl_hdf5_dataset.py` — reads material_delta from HDF5 (defaults 0.0)
- `tests/test_reward_computer.py` — NEW: 10 NIE stubs (T-CR1 to T-CR10)
- `tests/test_replayer.py` — NEW: 5 NIE stubs (material delta scenarios)
- `tests/test_config.py` — 6 new composite reward config tests (all PASS)

---

## Scaffolding Completed: RSBC Test Skeletons (2026-03-10)

### Changes
- `tests/test_trainer.py` — `TestPGNRLTrainerRSBC` (15 skipTest stubs) added after `TestPGNRLTrainerAWBC`
- `tests/test_config.py` — `TestRLConfigRSBC` (3 skipTest stubs) added before `if __name__`
- No production code changes — config/trainer stubs already existed

### Test Results: 18 skipped (all `skipTest("TODO: implement")`)
- Pre-existing failure in `TestPGNRLTrainerAWBC::test_awbc_loss_no_advantage_collapse` — unrelated

---

## Scaffolding Completed: AWBC Loss Redesign (2026-03-10)

### Changes
- `chess_sim/config.py` — `RLConfig` gains `lambda_awbc=1.0`, `lambda_entropy=0.0`, `awbc_eps=1e-8`; `lambda_ce` default `0.0` (deprecated); validation in `__post_init__`
- `chess_sim/training/pgn_rl_trainer.py` — `train_game()`: `pg_loss+ce_loss` -> `awbc_loss+entropy_bonus`; removed `log_probs`; added `_compute_awbc_loss()` + `_compute_entropy_bonus()` stubs (NIE); `train_epoch()` accumulators updated; return dicts: `awbc_loss`/`entropy_bonus` + deprecated `pg_loss: 0.0`/`ce_loss: 0.0`
- `configs/train_rl.yaml` + `configs/train_rl_10k.yaml` — new AWBC fields; `lambda_ce: 0.0`
- `tests/test_trainer.py` — `TestPGNRLTrainerAWBC` (16 test stubs, all NIE)
- `tests/test_pgn_rl_trainer.py` — T12 assertion updated `lambda_ce==0.5` -> `0.0`
- Design doc at `docs/awbc_loss_redesign.md`

---

## Scaffolding Completed: RL HDF5 Pipeline (2026-03-09)

### Module Layout
```
chess_sim/
├── config.py                          # +RLOutputConfig, RLFilterConfig, RLPreprocessConfig, load_rl_preprocess_config
├── types.py                           # +RLPlyRecord NamedTuple (10 fields, no torch dep)
├── preprocess/rl_reader.py            # RLPGNReader — stream() yields chess.pgn.Game
├── preprocess/rl_parser.py            # RLPlyParser — parse_game() -> list[RLPlyRecord]
├── preprocess/rl_writer.py            # RLHdf5Writer — open/write_batch/flush/close
├── preprocess/rl_validator.py         # RLHdf5Validator — validate() raises ValueError
├── preprocess/rl_preprocessor.py      # RLHdf5Preprocessor — orchestrates full pipeline
├── preprocess/__init__.py             # +5 RL exports
├── data/rl_hdf5_dataset.py            # RLPlyHDF5Dataset(Dataset[OfflinePlyTuple]) + rl_hdf5_worker_init
configs/preprocess_rl.yaml             # Default RL preprocess YAML
scripts/preprocess_rl.py               # Entry-point stub
```

### Test Suite (39 tests, all PASS — stubs assert NotImplementedError, configs test real validation)
```
tests/test_rl_hdf5_pipeline.py
  - T1:  TestSchemaCorrectness — validator raises NIE
  - T2:  TestTokenRoundTrip — writer/dataset init raises NIE
  - T3:  TestPrefixPadding — record prefix len + writer NIE
  - T4:  TestPrefixTruncation — long prefix constructable + writer NIE
  - T5:  TestTrainColorFiltering — parser NIE + filter config validation (white/black/invalid)
  - T6:  TestMoveUciRoundTrip — 4 parameterized UCI string checks
  - T7:  TestRewardRecomputation — flag types + dataset NIE
  - T8:  TestEmptyGameHandling — parser init NIE
  - T9:  TestDatasetLength — dataset init NIE
  - T10: TestOutOfRange — dataset init NIE
  - T11: TestMultiWorkerDataLoader — worker_init NIE
  - T12: TestTrainColorMismatch — validator NIE with black filter
  + TestRLPreprocessConfig (4): defaults, loader NIE
  + TestRLPlyRecordFields (11): field count + 10 parameterized name checks
  + TestRLPGNReader (1): stream NIE
  + TestRLHdf5Preprocessor (2): parser NIE, preprocessor init NIE
```

### Key Decisions
- `git add -f` needed for chess_sim/data/rl_hdf5_dataset.py (gitignore `data/` pattern)
- RLFilterConfig.__post_init__ validates train_color in ("white", "black")
- RLPreprocessConfig reuses InputConfig, SplitConfig, ProcessingConfig from existing pipeline
- load_rl_preprocess_config raises NotImplementedError (stub)
- RLPlyRecord has 10 fields, no torch dependency — pure Python/lists
- HDF5 schema: board/color/traj [N,65] uint8, move_prefix [N,max_prefix_len] uint16, move_uci S5 ASCII
- Design doc at docs/rl_hdf5_pipeline.md

---

## Scaffolding Completed: ReturnValueHead Critic (2026-03-09)

### Changes
- `chess_sim/model/value_heads.py` — full replacement: `ValueHeads` -> `ReturnValueHead` (two-layer MLP stub, forward raises NIE)
- `chess_sim/model/chess_model.py` — added `self.value_head: ReturnValueHead` in `__init__`
- `chess_sim/config.py` — added `lambda_value: float = 1.0` to `RLConfig` with `__post_init__` validation
- `configs/train_rl.yaml` — added `lambda_value: 1.0` under `rl:` block
- `chess_sim/training/pgn_rl_trainer.py` — added `_encode_and_decode` stub (NIE), `v_preds` list, `value_loss`/`mean_advantage` placeholders (0.0) in return dicts
- `tests/test_pgn_rl_trainer.py` — T15-T23 (9 new tests), new `TestReturnValueHead` class

### Test Results: 24 pass, 1 fail (T20 — NIE from forward stub, expected)
- Design doc at docs/return_value_head.md

---

## Scaffolding Completed: ActionConditionedValueHead Q-function (2026-03-09)

### Changes
- `chess_sim/model/value_heads.py` — `ReturnValueHead` -> `ActionConditionedValueHead` (concat fusion MLP, 2*d_model input)
- `chess_sim/model/chess_model.py` — import updated, `@property move_token_emb` stub added (NIE)
- `chess_sim/training/pgn_rl_trainer.py` — `_encode_and_decode` gains `move_uci` param, returns `tuple[Tensor, Tensor, int | None]` (NIE); `v_preds` -> `q_preds`; ply loop updated with action_emb + Q-head
- `tests/test_pgn_rl_trainer.py` — `TestReturnValueHead` -> `TestActionConditionedValueHead`
- `tests/test_value_head_q.py` — T24-T37 (14 tests), all fail with NIE

### Test Results: 14 tests, all ERROR (NotImplementedError — expected)
- Design doc at docs/action_conditioned_value_head.md
