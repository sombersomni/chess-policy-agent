# setup-dev Agent Memory

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
