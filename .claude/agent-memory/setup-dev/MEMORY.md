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

## Scaffolding Completed: Candidate Piece Conditioning (2026-03-14)

### Changes
- `chess_sim/data/candidate_reachability_mapper.py` — NEW: `CandidateReachabilityMapper`, `compute_valid_empty_mask()`, `build_candidate_board_tokens()` stubs (all NIE)
- `chess_sim/data/piece_type_move_lut.py` — NEW: `PieceTypeMoveLUT` stub (mirrors SrcMoveLUT pattern, all NIE)
- `chess_sim/model/embedding.py` — `PIECE_VOCAB_SIZE` 8->9, `INVALID_EMPTY_IDX=8`, `piece_type_cond_emb` (zero-init nn.Embedding(8, d_model)), `piece_type_tokens` param added to `embed()`/`forward()`
- `chess_sim/model/encoder.py` — `piece_type_tokens` pass-through to embedding
- `chess_sim/model/chess_model.py` — `piece_type` param in `forward()` and `predict_next_move()`, PieceTypeMoveLUT filtering
- `chess_sim/config.py` — `ModelConfig.use_candidate_conditioning: bool = False`
- `chess_sim/types.py` — `ConditionedBoard` NamedTuple (board_tokens, color_tokens, piece_type_idx)

### Test Results: 37 pass, 14 skip
- Design doc at `docs/design_candidate_piece_conditioning.md`

---

## Prior Scaffolding (see scaffolding-history.md for details)
- RL Self-Play Fine-Tuner (2026-03-13) — 27 pass
- Dual-Direction RSCE Loss (2026-03-11) — 25 pass, 5 fail (expected NIE)
- RSCE V4 Batched Pipeline (2026-03-11) — 15 pass
- RSCE Trainer v3 (2026-03-11) — 15 skipped
- Composite Reward Redesign (2026-03-10) — 10+5 NIE stubs, 6 config pass
- AWBC Loss Redesign (2026-03-10) — 16 NIE stubs
- RL HDF5 Pipeline (2026-03-09) — 39 pass
- ReturnValueHead / ActionConditionedValueHead (2026-03-09) — value head stubs
- Phase 2 Self-Play RL (2026-03-08) — 15 tests
