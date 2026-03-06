# Feature Dev Agent Memory

## Project Structure
- Source: `chess_sim/` with subpackages `data/`, `model/`, `training/`
- Tests: `tests/` using `unittest` + `pytest` + `parameterized`
- Types: `chess_sim/types.py` (all NamedTuples: TokenizedBoard, TrainingExample, ChessBatch, EncoderOutput, PredictionOutput, LabelTensors)
- Protocols: `chess_sim/protocols.py` (Tokenizable, Embeddable, Encodable, Predictable, Trainable, Samplable)
- Test utils: `tests/utils.py` (make_synthetic_batch, make_prediction_output, make_training_examples, etc.)

## Key Implementation Decisions
- **Pre-LN Transformer**: Must use `norm_first=True` in TransformerEncoderLayer. PyTorch 2.x SDPA on CPU causes zero gradients in deeper layers with Post-LN (default).
- **enable_nested_tensor=False**: Required for TransformerEncoder to avoid SDPA fast-path issues.
- **NaN handling in LossComputer**: When all opponent labels are -1 (ignored), CE returns nan. Replace nan with zeros_like before summing.
- **requires_grad in test utils**: `make_prediction_output` needs `requires_grad=True` on logit tensors for backward tests.
- **Eval mode for checkpoint tests**: Dropout makes outputs non-deterministic in train mode; checkpoint roundtrip tests need `.eval()`.

## Dependencies
- `torch`, `python-chess`, `zstandard` (in requirements.txt)
- `pytest`, `parameterized` (test deps, installed in .venv)

## Virtual Environment
- `.venv/` at project root
- Activate: `source .venv/bin/activate`
- Run tests: `python -m pytest tests/ -v`
