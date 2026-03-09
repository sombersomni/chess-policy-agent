"""Entry-point for Phase 2 self-play RL training.

Usage:
    python -m scripts.train_phase2 --config configs/phase2.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chess_sim.config import load_v2_config
from chess_sim.model.chess_model import ChessModel
from chess_sim.training.self_play_loop import SelfPlayLoop


def main() -> None:
    """Parse args, load config, and run the self-play loop."""
    parser = argparse.ArgumentParser(
        description="Phase 2 self-play RL training"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2.yaml"),
    )
    parser.add_argument(
        "--episodes", type=int, default=100
    )
    args = parser.parse_args()
    cfg = load_v2_config(args.config)
    player = ChessModel(cfg.model, cfg.decoder)
    if cfg.phase2.pretrained_ckpt:
        ckpt = torch.load(
            cfg.phase2.pretrained_ckpt,
            map_location="cpu",
            weights_only=True,
        )
        player.load_state_dict(ckpt["model"])
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    loop = SelfPlayLoop(
        player=player, cfg=cfg.phase2, device=device
    )
    loop.run(args.episodes)


if __name__ == "__main__":
    main()
