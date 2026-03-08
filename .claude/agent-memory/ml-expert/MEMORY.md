# ML Expert Memory

## Project Structure
- `chess_sim/model/chess_model.py` -- ChessModel v2 (encoder-decoder), ~10.1M params
- `chess_sim/training/phase1_trainer.py` -- Phase1Trainer (AdamW + warmup + cosine + CE)
- `chess_sim/data/pgn_sequence_dataset.py` -- PGNSequenceDataset + PGNSequenceCollator
- `chess_sim/config.py` -- ChessModelV2Config, load_v2_config()
- `scripts/train_v2.py` -- Phase 1 training (supports real PGN via --pgn and synthetic)
- Virtual env: `venv/` at project root

## v2 Architecture
- d_model=256, encoder 6-layer, decoder 4-layer, 8-head, vocab=1971

## Training Run: 10k Real Games (2026-03-07)
- **Config**: `configs/train_v2_10k.yaml`
- **Data**: 10k lichess games -> 651,314 samples, bs=64, ~9159 steps/epoch
- **LR**: LinearLR warmup 500 steps + CosineAnnealingLR, base LR 3e-4
- **Results (epochs 1-4/10, training still running)**:

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| 1     | 3.159     | 1.534    | 62.8%   |
| 2     | 1.759     | 0.892    | 77.3%   |
| 3     | 1.357     | 0.639    | 83.5%   |
| 4     | 1.132     | 0.502    | 87.0%   |

- Smooth convergence, no overfitting, ~9 min/epoch on CPU
- Checkpoint: `checkpoints/chess_v2_10k.pt` (saved at end)

## Scheduler Fix (applied)
- Old: CosineAnnealingLR(T_max=total_steps) -- LR hit 0 at end
- New: SequentialLR(LinearLR warmup + CosineAnnealingLR)

## Key Patterns
- `log_metrics` decorator logs per-step loss/LR (verbose output)
- PGNSequenceDataset loads all into memory (~5.4GB for 10k games)
- Use PYTHONUNBUFFERED=1 for real-time log output in future runs
- Config merge: YAML -> dataclass, CLI overrides with None-default args
