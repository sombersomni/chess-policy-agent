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

## Training Run: 1k Games, d_model=128, 20 epochs (2026-03-08, CURRENT, with gap tracking)
- **Config**: `configs/train_v2_1k.yaml`
- **Data**: 1k lichess games -> 60,060 samples (54,054 train / 6,006 val), bs=64, winners_only
- **Params**: 2,830,899 (~3.6x fewer than d_model=256)
- **Device**: CUDA (RTX 4070 SUPER), ~30s/epoch, total ~10.5min

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Gap (val-train) |
|-------|-----------|-----------|----------|---------|-----------------|
| 1     | 5.179     | 13.7%     | 3.855    | 26.7%   | +13.0%          |
| 5     | 1.604     | 64.2%     | 0.817    | 82.2%   | +18.1%          |
| 10    | 0.842     | 79.0%     | 0.337    | 92.2%   | +13.2%          |
| 15    | 0.621     | 83.9%     | 0.230    | 94.6%   | +10.7%          |
| 20    | 0.573     | 85.1%     | 0.214    | 95.0%   | +9.9%           |

- **Gap trend**: peaks at epoch 3 (+20.5%), then monotonically decreases to +9.9%
- **Positive gap** (val > train) is due to dropout/regularization active during train metric computation
- No overfitting; gap stabilizes at epochs 18-20
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

## Offline RL v4 Batched RSCE (chess_rl_v4.pt, 2026-03-11)
- **Script**: `scripts/train_rl_v4.py` + `configs/train_rl_v4.yaml`
- **Key files**: `chess_sim/training/pgn_rl_trainer_v4.py`, `chess_sim/data/chess_rl_dataset.py`, `chess_sim/data/pgn_reward_preprocessor.py`
- **Data**: 10k lichess games -> 328,267 HDF5 rows (295,335 train / 32,932 val), bs=512
- **Params**: 2,847,412 (~2.85M)
- **Speed**: ~2 min/epoch on CUDA, ~40 min total for 20 epochs
- **LR**: warmup 5% + constant to 50% + cosine decay, base 1e-4, min 1e-5
- **20-epoch results**:
  - Epoch 1: total=2.89, imit=5.28, repul=0.001, val_loss=4.64, val_acc=7.0%
  - Epoch 10: total=2.15, imit=3.88, repul=0.066, val_loss=3.81, val_acc=16.9%
  - Epoch 20: total=1.99, imit=3.56, repul=0.084, val_loss=3.56, val_acc=19.5%
- **frac_repulsion**: constant 0.45 (45% loser plies)
- **repulsion_top1_avoidance**: DECLINED from 93% to 81% -- winner/loser move overlap on common moves
- **Key insight**: repulsion loss (~0.08) is negligible vs imitation (~3.56), ~44x gap
- **Recommendations**: increase rsce_repulsion_weight to 5-10x, add lambda_ce>0, train 40+ epochs
- **Checkpoint**: `checkpoints/chess_rl_v4.pt`
- **HDF5 cache**: `data/chess_rl_v4.h5` -- must delete if schema changes (not auto-detected)
