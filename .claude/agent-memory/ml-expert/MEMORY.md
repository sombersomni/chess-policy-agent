# ML Expert Memory

## Project Structure
- `chess_sim/model/chess_model.py` -- ChessModel v2 (encoder-decoder), ~10.1M params
- `chess_sim/training/phase1_trainer.py` -- Phase1Trainer (AdamW + warmup + cosine + CE)
- `chess_sim/data/pgn_sequence_dataset.py` -- PGNSequenceDataset + PGNSequenceCollator
- `chess_sim/config.py` -- ChessModelV2Config, load_v2_config()
- `scripts/train_v2.py` -- Phase 1 training (supports real PGN via --pgn and synthetic)
- `scripts/evaluate_v2.py` -- v2 evaluation script (val_loss, val_accuracy, mean_entropy)
- Virtual env: `.venv/` at project root

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

## Training Run: 1k Real Games (2026-03-08)
- **Config**: `configs/train_v2_1k.yaml`
- **Data**: 1k lichess games -> 61,169 samples (55,052 train / 6,117 val), bs=64, 861 steps/epoch
- **LR**: LinearLR warmup 200 steps + CosineAnnealingLR, base LR 3e-4
- **Device**: CUDA, ~48s/epoch, total ~8min

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| 1     | 4.299     | 2.068    | 59.4%   |
| 2     | 1.686     | 0.647    | 85.7%   |
| 3     | 0.830     | 0.346    | 91.0%   |
| 4     | 0.529     | 0.254    | 92.7%   |
| 5     | 0.391     | 0.206    | 93.7%   |
| 6     | 0.314     | 0.176    | 94.5%   |
| 7     | 0.268     | 0.157    | 95.0%   |
| 8     | 0.239     | 0.144    | 95.4%   |
| 9     | 0.223     | 0.139    | 95.6%   |
| 10    | 0.216     | 0.137    | 95.6%   |

- Val loss still lower than train loss (no overfitting)
- Accuracy plateaus ~95.6% by epoch 9-10
- Checkpoint: `checkpoints/chess_v2_1k.pt`
- LR decayed to ~0 by final epoch (cosine T_max = total_steps)

## Eval: chess_v2_50games.pt on data/games.pgn (2026-03-08)
- val_loss=7.17, val_accuracy=0.52%, mean_entropy=5.92 nats
- 50 games, 4898 samples, batch_size=64, device=cuda
- Trained on 50 synthetic games only; near-random on real PGN data
- H/H_max = 5.92/7.59 = 78% -- low confidence but still some structure

## Scheduler Fix (applied)
- Old: CosineAnnealingLR(T_max=total_steps) -- LR hit 0 at end
- New: SequentialLR(LinearLR warmup + CosineAnnealingLR)

## Bug Fix: Phase1Trainer.evaluate() variable-length concat (2026-03-08)
- torch.cat on all_logits failed when batches had different seq lens T
- Fix: compute entropy per-batch with weighted sum, avoid cat entirely

## Deep Inspection: chess_v2_1k.pt (2026-03-08)
- **Script**: `scripts/inspect_v2.py` + `configs/inspect_v2.yaml`
- **Run**: `source .venv/bin/activate && python -m scripts.inspect_v2`
- **Virtual env**: `.venv/` (not `venv/`)
- **Findings**:
  - Weight sparsity <0.4% everywhere (no dead weights)
  - Piece embedding cosine sims: ROOK-QUEEN=0.81, BISHOP-ROOK=0.80, BISHOP-QUEEN=0.78
  - PAWN anti-correlated with all pieces (-0.74 to -0.86)
  - QUEEN L2 norm=2.09 (highest), EMPTY=0.23 (lowest) -- norm ~ piece value
  - PLAYER vs OPPONENT nearly orthogonal (cos=-0.03)
  - Entropy by phase: early=0.76, mid=0.98, late=1.62 nats
  - Model very confident on common patterns (e2e4 70%, Nc3/Bg7/O-O 99%+)
- **Critical hook trick**: Must call `torch.backends.mha.set_fastpath_enabled(False)` when monkey-patching encoder self_attn.forward in no_grad mode, otherwise `torch._transformer_encoder_layer_fwd` fastpath bypasses `self_attn.forward` entirely

## Key Patterns
- `log_metrics` decorator logs per-step loss/LR (verbose output)
- PGNSequenceDataset loads all into memory (~5.4GB for 10k games)
- Use PYTHONUNBUFFERED=1 for real-time log output in future runs
- Config merge: YAML -> dataclass, CLI overrides with None-default args
- load_v2_config ignores unknown YAML keys (uses .get()); safe to add eval section
