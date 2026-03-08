"""scripts/simulate.py: terminal chess simulation CLI.

Three operating modes
---------------------
pgn     -- replay a single PGN game move-by-move (no agent required).
random  -- generate a random-legal-move game (no agent required).
agent   -- load a checkpoint, show model predictions before each move, then
           reveal the game's actual move (PGN ground truth) or random move.

Usage
-----
    python -m scripts.simulate --config configs/simulate.yaml --mode pgn
    python -m scripts.simulate --mode random --tick-rate 0.3
    python -m scripts.simulate --mode agent \\
        --pgn data/lichess_db_standard_rated_2013-01.pgn.zst \\
        --checkpoint checkpoints/chess_v2_1k.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from chess_sim.config import (
    DecoderConfig,
    ModelConfig,
    SimulateConfig,
    load_simulate_config,
)
from chess_sim.data.move_vocab import MoveVocab
from chess_sim.data.tokenizer import BoardTokenizer
from chess_sim.env import Policy
from chess_sim.env.chess_sim_env import ChessSimEnv
from chess_sim.env.sources import PGNSource, RandomSource
from chess_sim.env.terminal_renderer import TerminalRenderer

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _infer_configs_from_state(
    state: dict,
    model_cfg: ModelConfig,
    decoder_cfg: DecoderConfig,
) -> tuple[ModelConfig, DecoderConfig]:
    """Override ModelConfig/DecoderConfig with dimensions read from state dict.

    Args:
        state: Model state dict loaded from checkpoint.
        model_cfg: Baseline encoder config (n_heads/dropout kept from here).
        decoder_cfg: Baseline decoder config (n_heads/dropout/etc kept).

    Returns:
        Tuple of (ModelConfig, DecoderConfig) matching the checkpoint.
    """
    from dataclasses import replace

    enc_d = state["encoder.embedding.piece_emb.weight"].shape[1]
    enc_ff = state["encoder.transformer.layers.0.linear1.weight"].shape[0]
    enc_n = sum(
        1 for k in state if k.startswith("encoder.transformer.layers.")
        and k.endswith(".norm1.weight")
    )
    dec_d = state["decoder.move_embedding.token_emb.weight"].shape[1]
    dec_ff = state["decoder.transformer.layers.0.linear1.weight"].shape[0]
    dec_n = sum(
        1 for k in state if k.startswith("decoder.transformer.layers.")
        and k.endswith(".norm1.weight")
    )
    model_cfg = replace(
        model_cfg, d_model=enc_d, n_layers=enc_n, dim_feedforward=enc_ff
    )
    decoder_cfg = replace(
        decoder_cfg, d_model=dec_d, n_layers=dec_n, dim_feedforward=dec_ff
    )
    return model_cfg, decoder_cfg


def _load_agent(
    checkpoint: str,
    model_cfg: ModelConfig,
    decoder_cfg: DecoderConfig,
    tokenizer: BoardTokenizer,
    vocab: MoveVocab,
) -> Policy:
    """Load ChessModel from checkpoint and wrap in ChessModelAgent.

    Architecture dimensions are inferred from the checkpoint's state dict,
    so the config does not need to match exactly.

    Args:
        checkpoint: Path to .pt checkpoint file.
        model_cfg: Encoder architecture config (n_heads/dropout used as-is).
        decoder_cfg: Decoder architecture config (n_heads/dropout used as-is).
        tokenizer: BoardTokenizer instance.
        vocab: MoveVocab instance.

    Returns:
        ChessModelAgent implementing the Policy protocol.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
    """
    from chess_sim.env.agent_adapter import ChessModelAgent
    from chess_sim.model.chess_model import ChessModel

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model_cfg, decoder_cfg = _infer_configs_from_state(state, model_cfg, decoder_cfg)

    model = ChessModel(model_cfg, decoder_cfg)
    model.load_state_dict(state)
    model.eval()

    return ChessModelAgent(model, tokenizer, vocab, device="cpu")


def build_env(cfg: SimulateConfig) -> tuple[ChessSimEnv, Optional[Policy]]:
    """Construct ChessSimEnv and optional Policy from config.

    Args:
        cfg: Fully populated SimulateConfig.

    Returns:
        Tuple of (ChessSimEnv, Policy or None).

    Raises:
        ValueError: If mode is "agent" but no checkpoint is provided.
        ValueError: If mode is "pgn" or "agent" but no pgn path is provided.
        ValueError: If mode is unrecognised.
    """
    tokenizer = BoardTokenizer()
    vocab = MoveVocab()
    renderer = TerminalRenderer(use_unicode=cfg.use_unicode)
    policy: Optional[Policy] = None

    if cfg.mode == "random":
        source = RandomSource(max_plies=cfg.max_plies)

    elif cfg.mode in ("pgn", "agent"):
        if not cfg.pgn:
            raise ValueError("--pgn is required for modes 'pgn' and 'agent'.")
        source = PGNSource(
            pgn_path=Path(cfg.pgn),
            game_index=cfg.game_index,
        )
        if cfg.mode == "agent":
            if not cfg.checkpoint:
                raise ValueError(
                    "--checkpoint is required for mode 'agent'."
                )
            policy = _load_agent(
                cfg.checkpoint,
                cfg.model,
                cfg.decoder,
                tokenizer,
                vocab,
            )
    else:
        raise ValueError(
            f"Unknown mode {cfg.mode!r}. Choose from: pgn, random, agent."
        )

    env = ChessSimEnv(
        source=source,
        tokenizer=tokenizer,
        vocab=vocab,
        renderer=renderer,
        render_mode="terminal",
    )
    return env, policy


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


def run_loop(
    env: ChessSimEnv,
    policy: Optional[Policy],
    cfg: SimulateConfig,
) -> None:
    """Run the main simulation loop until the game ends.

    For each ply:
      1. If policy is available, compute and display predictions.
      2. Advance the game by one step.
      3. Render the board state to the terminal.
      4. Sleep for tick_rate seconds.

    Args:
        env: Initialised ChessSimEnv.
        policy: Optional Policy for agent prediction mode.
        cfg: SimulateConfig for tick_rate and top_n.
    """
    obs, info = env.reset()
    env.render()
    time.sleep(cfg.tick_rate)

    done = False
    while not done:
        legal = env.legal_uci_moves()
        if not legal:
            break

        predictions = []
        if policy is not None:
            predictions = policy.top_n_predictions(obs, legal, n=cfg.top_n)  # type: ignore[attr-defined]
            env.set_predictions(predictions)
            env.render()
            time.sleep(cfg.tick_rate * 0.6)  # brief pause to show predictions

        obs, reward, terminated, truncated, info = env.step(0)  # 0 = dummy for PGN
        done = terminated or truncated

        # Update predictions with actual-move context before final render.
        env.set_predictions(predictions)
        env.render()
        time.sleep(cfg.tick_rate)

    _print_outcome(env, reward)


def _print_outcome(env: ChessSimEnv, final_reward: float) -> None:
    """Print the game outcome to stdout.

    Args:
        env: The completed ChessSimEnv.
        final_reward: Reward returned by the last step.
    """
    source = env._source  # type: ignore[attr-defined]
    board = source._board if hasattr(source, "_board") else None
    if board is not None and board.is_game_over():
        result = board.result()
        print(f"\nGame over — result: {result}  (reward: {final_reward:+.1f})")
    else:
        print(f"\nSimulation ended at ply {source.current_ply()}.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="simulate",
        description="Terminal chess simulation with optional agent prediction.",
    )
    parser.add_argument(
        "--config", default="", help="Path to configs/simulate.yaml."
    )
    parser.add_argument(
        "--mode",
        choices=["pgn", "random", "agent"],
        help="Simulation mode (overrides YAML).",
    )
    parser.add_argument("--pgn", default=None, help="Path to PGN or PGN.zst file.")
    parser.add_argument("--game-index", type=int, default=None, dest="game_index")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint.")
    parser.add_argument(
        "--tick-rate",
        type=float,
        default=None,
        dest="tick_rate",
        help="Seconds between plies.",
    )
    parser.add_argument(
        "--max-plies", type=int, default=None, dest="max_plies"
    )
    parser.add_argument(
        "--top-n", type=int, default=None, dest="top_n"
    )
    parser.add_argument(
        "--no-unicode",
        action="store_true",
        dest="no_unicode",
        help="Use ASCII piece symbols instead of Unicode.",
    )
    return parser.parse_args(argv)


def _merge_config(
    args: argparse.Namespace,
) -> SimulateConfig:
    """Merge YAML config with CLI overrides.

    CLI args take precedence over YAML values when they are not None.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Merged SimulateConfig.
    """
    if args.config:
        cfg = load_simulate_config(Path(args.config))
    else:
        cfg = SimulateConfig()

    if args.mode is not None:
        cfg.mode = args.mode
    if args.pgn is not None:
        cfg.pgn = args.pgn
    if args.game_index is not None:
        cfg.game_index = args.game_index
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    if args.tick_rate is not None:
        cfg.tick_rate = args.tick_rate
    if args.max_plies is not None:
        cfg.max_plies = args.max_plies
    if args.top_n is not None:
        cfg.top_n = args.top_n
    if args.no_unicode:
        cfg.use_unicode = False

    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse args, build environment, and run simulation.

    Args:
        argv: Argument list; defaults to sys.argv[1:] when None.
    """
    args = _parse_args(argv)
    cfg = _merge_config(args)

    print(f"[simulate] mode={cfg.mode}  pgn={cfg.pgn or '─'}  "
          f"checkpoint={cfg.checkpoint or '─'}  tick={cfg.tick_rate}s")

    try:
        env, policy = build_env(cfg)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[simulate] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    run_loop(env, policy, cfg)


if __name__ == "__main__":
    main()
