"""Deep forensic inspection of a trained ChessModel v2 checkpoint.

Modules:
  A) Weight Sparsity Inspector — dead weights, effective rank via SVD
  B) Entropy Distribution by Game Phase — early/mid/late ply buckets
  C) Attention Hook Analyst — monkey-patch encoder self-attn for weights
  D) Embedding Space Probe — piece/color cosine similarity matrix
  E) Game Inference Trace — per-ply top-5 predictions with legal masking
"""

from __future__ import annotations

import argparse
import functools
import io
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from chess_sim.config import ChessModelV2Config, DecoderConfig, ModelConfig
from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.move_vocab import MoveVocab, PAD_IDX, SOS_IDX, EOS_IDX
from chess_sim.data.pgn_sequence_dataset import (
    PGNSequenceCollator,
    PGNSequenceDataset,
)
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.data.tokenizer_utils import make_trajectory_tokens
from chess_sim.functional import entropy_from_logits
from chess_sim.model.chess_model import ChessModel
from chess_sim.types import GameTurnBatch

# Piece token names (index = piece token value)
PIECE_NAMES: list[str] = [
    "CLS", "EMPTY", "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"
]

# File/rank labels for algebraic notation
FILES: list[str] = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS: list[str] = ["1", "2", "3", "4", "5", "6", "7", "8"]


def sq_idx_to_algebraic(sq_idx: int) -> str:
    """Convert 1-based square index to algebraic notation (sq_idx in [1,64])."""
    file = (sq_idx - 1) % 8
    rank = (sq_idx - 1) // 8
    return f"{FILES[file]}{RANKS[rank]}"


def _separator(title: str) -> None:
    """Print a section separator with title."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


# ── Module A: Weight Sparsity Inspector ──────────────────────────────────


def inspect_weight_sparsity(model: nn.Module) -> None:
    """Inspect weight matrices for sparsity, L1 density, and effective rank."""
    _separator("MODULE A: Weight Sparsity Inspector")

    rows: list[tuple[str, float, float, float | None]] = []
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        w = param.detach().float()
        sparsity = (w.abs() < 1e-4).float().mean().item() * 100
        l1_per_unit = w.abs().mean().item()

        # Effective rank for near-square matrices (ratio <= 4:1)
        eff_rank: float | None = None
        rows_w, cols_w = w.shape[0], w.shape[1]
        ratio = max(rows_w, cols_w) / max(min(rows_w, cols_w), 1)
        if ratio <= 4.0 and min(rows_w, cols_w) >= 2:
            try:
                # Reshape to 2D for SVD if needed
                w2d = w.reshape(rows_w, -1) if w.ndim > 2 else w
                sigma = torch.linalg.svdvals(w2d)
                eff_rank = (sigma.sum() / sigma.max()).item()
            except RuntimeError:
                pass

        rows.append((name, sparsity, l1_per_unit, eff_rank))

    # Sort by sparsity descending
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"{'Layer':<55} {'Sparsity%':>9} {'L1/unit':>9} {'EffRank':>9}")
    print("-" * 86)
    for name, sparsity, l1, eff_rank in rows:
        rank_str = f"{eff_rank:.1f}" if eff_rank is not None else "n/a"
        print(f"{name:<55} {sparsity:>8.2f}% {l1:>9.5f} {rank_str:>9}")

    total_params = sum(p.numel() for p in model.parameters())
    total_2d = sum(
        p.numel() for p in model.parameters() if p.ndim >= 2
    )
    print(f"\nTotal params: {total_params:,} | 2D weight params: {total_2d:,}")


# ── Module B: Entropy Distribution by Game Phase ─────────────────────────


def inspect_entropy_distribution(
    model: nn.Module,
    dataset: PGNSequenceDataset,
    device: str,
    batch_size: int = 32,
) -> None:
    """Analyze prediction entropy bucketed by game phase (ply number)."""
    _separator("MODULE B: Entropy Distribution by Game Phase")

    collator = PGNSequenceCollator()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator,
    )

    # Buckets: early=[1-10], mid=[11-30], late=[31+]
    buckets: dict[str, list[float]] = {
        "early (ply 1-10)": [],
        "mid (ply 11-30)": [],
        "late (ply 31+)": [],
    }

    model.eval()
    dev = torch.device(device)
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, dev)
            logits = model(
                batch.board_tokens,
                batch.color_tokens,
                batch.trajectory_tokens,
                batch.move_tokens,
                batch.move_pad_mask,
            )
            # Entropy per position: [B, T]
            h = entropy_from_logits(logits)
            mask = batch.target_tokens != PAD_IDX
            B = h.shape[0]

            for b_idx in range(B):
                # Ply number = length of non-pad move tokens for this sample
                # move_tokens is SOS + prior moves, so T_valid = non-pad count
                valid = (~batch.move_pad_mask[b_idx]).sum().item()
                ply = int(valid)  # approximate ply from decoder seq length

                # Average entropy across valid positions for this sample
                valid_mask = mask[b_idx]
                if valid_mask.any():
                    avg_h = h[b_idx][valid_mask].mean().item()
                else:
                    continue

                if ply <= 10:
                    buckets["early (ply 1-10)"].append(avg_h)
                elif ply <= 30:
                    buckets["mid (ply 11-30)"].append(avg_h)
                else:
                    buckets["late (ply 31+)"].append(avg_h)

                sample_idx += 1

    print(f"{'Phase':<20} {'Count':>7} {'Mean H':>9} {'Std H':>9} {'Min H':>9} {'Max H':>9}")
    print("-" * 68)
    for phase, values in buckets.items():
        if values:
            t = torch.tensor(values)
            print(
                f"{phase:<20} {len(values):>7} "
                f"{t.mean().item():>9.4f} {t.std().item():>9.4f} "
                f"{t.min().item():>9.4f} {t.max().item():>9.4f}"
            )
        else:
            print(f"{phase:<20} {'(no samples)':>7}")

    all_h = []
    for v in buckets.values():
        all_h.extend(v)
    if all_h:
        t = torch.tensor(all_h)
        print(f"\nOverall: {len(all_h)} samples, mean H = {t.mean().item():.4f} nats")


# ── Module C: Attention Hook Analyst ─────────────────────────────────────


def inspect_attention_patterns(
    model: nn.Module,
    dataset: PGNSequenceDataset,
    device: str,
    n_positions: int = 20,
) -> None:
    """Monkey-patch encoder self-attention to capture attention weights."""
    _separator("MODULE C: Attention Hook Analyst")

    encoder = model.encoder  # type: ignore[attr-defined]
    layers = encoder.transformer.layers
    n_layers = len(layers)

    # Store originals and captured weights
    originals: list[Any] = []
    captured: dict[int, Tensor] = {}

    # Monkey-patch each layer's self_attn.forward
    for layer_idx, layer in enumerate(layers):
        orig = layer.self_attn.forward
        originals.append(orig)

        def make_patched(orig_fn: Any, idx: int) -> Any:
            @functools.wraps(orig_fn)
            def patched_forward(*args: Any, **kwargs: Any) -> Any:
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                out = orig_fn(*args, **kwargs)
                # out = (attn_output, attn_weights) where weights: [B, H, S, S]
                if isinstance(out, tuple) and len(out) == 2:
                    captured[idx] = out[1].detach()
                return out
            return patched_forward

        layer.self_attn.forward = make_patched(orig, layer_idx)

    model.eval()
    dev = torch.device(device)
    collator = PGNSequenceCollator()

    # Take a small subset of positions
    subset_size = min(n_positions, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    loader = DataLoader(
        subset, batch_size=1, shuffle=False, collate_fn=collator,
    )

    print(f"Analyzing attention over {subset_size} positions, {n_layers} layers")
    print()

    # Disable MHA fastpath so self_attn.forward is actually called
    prev_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)

    positions_analyzed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if positions_analyzed >= n_positions:
                break
            batch = _to_device(batch, dev)
            captured.clear()

            # Forward pass triggers the patched attention
            _ = model(
                batch.board_tokens,
                batch.color_tokens,
                batch.trajectory_tokens,
                batch.move_tokens,
                batch.move_pad_mask,
            )

            # Get piece names for this board
            board_tokens = batch.board_tokens[0].cpu()  # [65]

            print(f"--- Position {batch_idx + 1} ---")

            # Show attention for a few interesting source squares
            # Find occupied squares
            occupied = []
            for sq_idx in range(1, 65):
                piece_tok = board_tokens[sq_idx].item()
                if piece_tok > 1:  # not CLS or EMPTY
                    occupied.append((sq_idx, PIECE_NAMES[piece_tok]))

            # Show top attention destinations for first few occupied pieces
            shown = 0
            for sq_idx, piece_name in occupied[:6]:
                sq_alg = sq_idx_to_algebraic(sq_idx)
                for layer_idx in [0, n_layers // 2, n_layers - 1]:
                    if layer_idx not in captured:
                        continue
                    attn = captured[layer_idx]  # [1, H, 65, 65]
                    n_heads = attn.shape[1]
                    for head_idx in range(n_heads):
                        # Attention from this source square to all destinations
                        weights = attn[0, head_idx, sq_idx, :]  # [65]
                        top3 = weights.topk(3)
                        dests = []
                        for k in range(3):
                            dst_idx = top3.indices[k].item()
                            dst_w = top3.values[k].item()
                            if dst_idx == 0:
                                dst_label = "CLS"
                            else:
                                dst_alg = sq_idx_to_algebraic(dst_idx)
                                dst_piece = PIECE_NAMES[board_tokens[dst_idx].item()]
                                dst_label = f"{dst_piece}@{dst_alg}"
                            dests.append(f"{dst_label}({dst_w:.3f})")
                        print(
                            f"  L{layer_idx}/H{head_idx}: "
                            f"{piece_name}@{sq_alg} -> [{', '.join(dests)}]"
                        )
                shown += 1
                if shown >= 4:
                    break
            print()
            positions_analyzed += 1

    # Restore originals and fastpath setting
    for layer_idx, layer in enumerate(layers):
        layer.self_attn.forward = originals[layer_idx]
    torch.backends.mha.set_fastpath_enabled(prev_fastpath)

    print(f"(Restored original forwards for {n_layers} layers)")


# ── Module D: Embedding Space Probe ──────────────────────────────────────


def inspect_embedding_space(model: nn.Module) -> None:
    """Analyze learned piece and color embedding similarities."""
    _separator("MODULE D: Embedding Space Probe")

    embedding = model.encoder.embedding  # type: ignore[attr-defined]

    # Piece embeddings: [8, d_model]
    piece_emb = embedding.piece_emb.weight.detach().float()
    n_pieces = piece_emb.shape[0]

    print("Piece Embedding Cosine Similarity Matrix:")
    print(f"{'':>8}", end="")
    for j in range(n_pieces):
        print(f"{PIECE_NAMES[j]:>8}", end="")
    print()

    high_sim_pairs: list[tuple[str, str, float]] = []

    for i in range(n_pieces):
        print(f"{PIECE_NAMES[i]:>8}", end="")
        for j in range(n_pieces):
            sim = F.cosine_similarity(
                piece_emb[i].unsqueeze(0),
                piece_emb[j].unsqueeze(0),
            ).item()
            print(f"{sim:>8.3f}", end="")
            if i < j and sim > 0.7:
                high_sim_pairs.append((PIECE_NAMES[i], PIECE_NAMES[j], sim))
        print()

    if high_sim_pairs:
        print("\nHigh similarity pairs (cosine > 0.7):")
        for a, b, sim in sorted(high_sim_pairs, key=lambda x: -x[2]):
            print(f"  {a} <-> {b}: {sim:.4f}")
    else:
        print("\nNo piece pairs with cosine similarity > 0.7")

    # Piece embedding norms
    print("\nPiece embedding L2 norms:")
    for i in range(n_pieces):
        norm = piece_emb[i].norm().item()
        print(f"  {PIECE_NAMES[i]:>8}: {norm:.4f}")

    # Color embeddings: [3, d_model]
    color_emb = embedding.color_emb.weight.detach().float()
    color_names = ["CLS/EMPTY", "PLAYER", "OPPONENT"]

    print("\nColor Embedding Cosine Similarity:")
    print(f"{'':>12}", end="")
    for j in range(3):
        print(f"{color_names[j]:>12}", end="")
    print()
    for i in range(3):
        print(f"{color_names[i]:>12}", end="")
        for j in range(3):
            sim = F.cosine_similarity(
                color_emb[i].unsqueeze(0),
                color_emb[j].unsqueeze(0),
            ).item()
            print(f"{sim:>12.4f}", end="")
        print()

    player_opp_sim = F.cosine_similarity(
        color_emb[1].unsqueeze(0), color_emb[2].unsqueeze(0),
    ).item()
    print(f"\nPLAYER vs OPPONENT cosine similarity: {player_opp_sim:.4f}")

    # Trajectory embeddings: [5, d_model]
    traj_emb = embedding.trajectory_emb.weight.detach().float()
    traj_names = ["none", "pl_prev", "pl_curr", "opp_prev", "opp_curr"]
    print("\nTrajectory Embedding L2 norms:")
    for i in range(5):
        norm = traj_emb[i].norm().item()
        print(f"  {traj_names[i]:>10}: {norm:.4f}")


# ── Module E: Game Inference Trace ───────────────────────────────────────


def trace_game_inference(
    model: nn.Module,
    pgn_path: str,
    n_games: int,
    device: str,
    temperature: float = 1.0,
) -> None:
    """Trace model predictions move-by-move through actual games."""
    _separator("MODULE E: Game Inference Trace")

    board_tok = BoardTokenizer()
    move_tok = MoveTokenizer()
    move_vocab = MoveVocab()
    dev = torch.device(device)

    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        games_traced = 0
        while games_traced < n_games:
            game = chess.pgn.read_game(fh)
            if game is None:
                break

            board = game.board()
            moves = list(game.mainline_moves())
            if not moves:
                continue

            games_traced += 1
            white = game.headers.get("White", "?")
            black = game.headers.get("Black", "?")
            result = game.headers.get("Result", "?")
            print(f"\n--- Game {games_traced}: {white} vs {black} [{result}] ---")
            print(f"{'Ply':>4} {'Move':>8} {'Top1':>8} {'P(top1)':>8} "
                  f"{'Top2':>8} {'Top3':>8} {'Rank':>5} {'H':>7} {'Correct':>8}")
            print("-" * 76)

            move_history: list[chess.Move] = []
            uci_history: list[str] = []

            model.eval()
            for ply_idx, move in enumerate(moves):
                # Tokenize board state
                tb = board_tok.tokenize(board, board.turn)
                traj = make_trajectory_tokens(move_history)

                board_tokens = torch.tensor(
                    tb.board_tokens, dtype=torch.long,
                ).unsqueeze(0).to(dev)
                color_tokens = torch.tensor(
                    tb.color_tokens, dtype=torch.long,
                ).unsqueeze(0).to(dev)
                trajectory_tokens = torch.tensor(
                    traj, dtype=torch.long,
                ).unsqueeze(0).to(dev)

                # Decoder input: SOS + prior moves
                input_ids = [SOS_IDX] + [
                    move_tok.tokenize_move(u) for u in uci_history
                ]
                move_tokens = torch.tensor(
                    input_ids, dtype=torch.long,
                ).unsqueeze(0).to(dev)

                with torch.no_grad():
                    logits = model(
                        board_tokens, color_tokens,
                        trajectory_tokens, move_tokens,
                    )

                # Logits at last position: [V]
                next_logits = logits[0, -1, :].clone()

                # Build legal move mask
                legal_uci = [m.uci() for m in board.legal_moves]
                legal_mask = move_tok.build_legal_mask(legal_uci).to(dev)

                # Apply legal mask
                masked_logits = next_logits.clone()
                masked_logits[~legal_mask] = -1e9

                # Probabilities and entropy
                probs = torch.softmax(masked_logits / temperature, dim=-1)
                h = -(probs * (probs + 1e-12).log()).sum().item()

                # Top-5 predictions
                top5 = probs.topk(5)
                top5_uci: list[str] = []
                top5_probs: list[float] = []
                for k in range(5):
                    idx = top5.indices[k].item()
                    p = top5.values[k].item()
                    try:
                        uci_str = move_vocab.decode(idx)
                    except KeyError:
                        uci_str = f"?{idx}"
                    top5_uci.append(uci_str)
                    top5_probs.append(p)

                # Actual move
                actual_uci = move.uci()
                actual_idx = move_tok.tokenize_move(actual_uci)

                # Rank of actual move in sorted predictions
                sorted_indices = probs.argsort(descending=True)
                rank = (sorted_indices == actual_idx).nonzero(as_tuple=True)[0]
                rank_val = rank[0].item() + 1 if len(rank) > 0 else -1

                correct = "YES" if top5_uci[0] == actual_uci else ""

                # Print up to ply 40, then summarize
                if ply_idx < 40:
                    print(
                        f"{ply_idx + 1:>4} {actual_uci:>8} "
                        f"{top5_uci[0]:>8} {top5_probs[0]:>7.3f} "
                        f"{top5_uci[1]:>8} {top5_uci[2]:>8} "
                        f"{rank_val:>5} {h:>7.3f} {correct:>8}"
                    )

                # Update history
                uci_history.append(actual_uci)
                move_history.append(move)
                board.push(move)

            # Summary stats
            print(f"\n  Total plies: {len(moves)}")


# ── Utility ──────────────────────────────────────────────────────────────


def _to_device(
    batch: GameTurnBatch, device: torch.device,
) -> GameTurnBatch:
    """Move all tensor fields in a GameTurnBatch to device."""
    return GameTurnBatch(
        *(
            t.to(device) if isinstance(t, Tensor) else t
            for t in batch
        )
    )


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu",
) -> ChessModel:
    """Load ChessModel from a .pt checkpoint file."""
    dev = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model = ChessModel()
    model.load_state_dict(ckpt["model"])
    model.to(dev)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Device: {device}")
    return model


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep forensic inspection of a ChessModel v2 checkpoint",
    )
    parser.add_argument(
        "--config", default="configs/inspect_v2.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pgn", default=None)
    parser.add_argument("--n_games", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    # Load YAML config
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # CLI overrides
    checkpoint = args.checkpoint or cfg.get("checkpoint", "checkpoints/chess_v2_1k.pt")
    pgn = args.pgn or cfg.get("pgn", "data/games.pgn")
    n_games = args.n_games if args.n_games is not None else cfg.get("n_games", 5)
    device = args.device or cfg.get("device", "cpu")
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32)
    temperature = args.temperature if args.temperature is not None else cfg.get("temperature", 1.0)

    print("=" * 72)
    print("  ChessModel v2 Deep Inspection")
    print("=" * 72)
    print(f"  Checkpoint: {checkpoint}")
    print(f"  PGN:        {pgn}")
    print(f"  Games:      {n_games}")
    print(f"  Device:     {device}")
    print(f"  Batch size: {batch_size}")

    # Load model
    model = load_model_from_checkpoint(checkpoint, device)

    # Module A: Weight Sparsity
    inspect_weight_sparsity(model)

    # Module D: Embedding Space (no data needed)
    inspect_embedding_space(model)

    # Module B: Entropy Distribution (needs dataset)
    print("\nLoading dataset for entropy analysis...")
    dataset = PGNSequenceDataset(pgn_path=pgn, max_games=n_games)
    inspect_entropy_distribution(model, dataset, device, batch_size)

    # Module C: Attention Patterns
    inspect_attention_patterns(model, dataset, device, n_positions=10)

    # Module E: Game Inference Trace
    trace_game_inference(model, pgn, n_games, device, temperature)

    print("\n" + "=" * 72)
    print("  Inspection complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
