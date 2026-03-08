# ML Expert Memory

## Project Structure
- `chess_sim/model/chess_model.py` -- ChessModel v2 (encoder-decoder), ~10.1M params
- `chess_sim/training/phase1_trainer.py` -- Phase1Trainer (AdamW + warmup + cosine + CE)
- `chess_sim/data/pgn_sequence_dataset.py` -- PGNSequenceDataset + PGNSequenceCollator
- `chess_sim/config.py` -- ChessModelV2Config, load_v2_config()
- `scripts/train_v2.py` -- Phase 1 training (supports real PGN via --pgn and synthetic)
- `scripts/evaluate_v2.py` -- v2 evaluation script (val_loss, val_accuracy, mean_entropy)
- Virtual env: `.venv/` at project root

## v2 Architecture (current: d_model=128)
- d_model=128, encoder 6-layer, decoder 4-layer, 8-head, ff=512, vocab=1971, ~2.83M params
- Previous: d_model=256, ff=1024, ~10.1M params

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

## Training Run: 1k Games, d_model=256, 10 epochs (2026-03-08, SUPERSEDED)
- 10.1M params, ~61k samples, final: train_loss=0.216, val_loss=0.137, val_acc=95.6%

## Training Run: 1k Games, d_model=128, 20 epochs (2026-03-08, CURRENT)
- **Config**: `configs/train_v2_1k.yaml`
- **Data**: 1k lichess games -> 60,060 samples (54,054 train / 6,006 val), bs=64, winners_only
- **Params**: 2,830,899 (~3.6x fewer than d_model=256)
- **Device**: CUDA (RTX 4070 SUPER), ~30s/epoch, total ~10min

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| 1     | 5.179     | 3.854    | 26.7%   |
| 5     | 1.604     | 0.819    | 82.2%   |
| 10    | 0.844     | 0.336    | 92.1%   |
| 15    | 0.622     | 0.232    | 94.5%   |
| 20    | 0.575     | 0.215    | 94.9%   |

- **Eval**: val_loss=0.229, val_acc=95.42%, mean_entropy=0.213 nats
- Val loss still lower than train loss (no overfitting)
- Accuracy plateau ~94.9% vs 95.6% for d_model=256 (only -0.7% with 3.6x fewer params)
- Checkpoint: `checkpoints/chess_v2_1k.pt`

## Scaling Insight: d_model=128 vs 256 on 1k games
- 3.6x param reduction (10.1M -> 2.8M) costs only ~0.7% accuracy
- Smaller model needs ~2x more epochs to converge but trains faster per epoch
- For 1k games, d_model=128 is likely capacity-appropriate; d_model=256 was overparameterized

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

## Bug Fix: evaluate_v2.py max_games (2026-03-08)
- Was hardcoded to `max_games=0` (loads ALL games from file)
- Fixed to use `cfg.data.max_games` from YAML config

## Key Patterns
- `log_metrics` decorator logs per-step loss/LR (verbose output)
- PGNSequenceDataset loads all into memory (~5.4GB for 10k games)
- Use PYTHONUNBUFFERED=1 for real-time log output in future runs
- Config merge: YAML -> dataclass, CLI overrides with None-default args
- load_v2_config ignores unknown YAML keys (uses .get()); safe to add eval section
