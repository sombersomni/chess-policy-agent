"""Typed configuration dataclasses for training and evaluation.

Load from YAML via load_train_config() / load_eval_config(),
or construct directly in tests and scripts.

CLI args override YAML values — see scripts/train_real.py and
scripts/evaluate.py for the merge logic.

Example:
    >>> cfg = load_train_config(Path("configs/train_50k.yaml"))
    >>> cfg.trainer.learning_rate
    0.0003
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-configs (mirror the top-level YAML sections exactly)
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Data pipeline and loader settings."""

    pgn: str = ""
    num_games: int = 20
    max_games: int = 0
    winners_only: bool = False
    chunk_size: int = 1024
    num_workers: int = 2
    train_frac: float = 0.9
    batch_size: int = 128


@dataclass
class ModelConfig:
    """Transformer architecture hyperparameters."""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1


@dataclass
class TrainerConfig:
    """Optimizer, scheduler, and checkpoint settings."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    patience: int = 3
    epochs: int = 10
    resume: str = ""
    checkpoint: str = ""


@dataclass
class EvalConfig:
    """Evaluation-specific settings."""

    checkpoint: str = ""
    game_index: int = 0
    top_n: int = 3


# ---------------------------------------------------------------------------
# Top-level configs (one per script)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Root config for scripts/train_real.py."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


@dataclass
class EvaluateConfig:
    """Root config for scripts/evaluate.py."""

    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------

def load_train_config(path: Path) -> TrainConfig:
    """Load TrainConfig from a YAML file.

    Unknown keys raise TypeError immediately (no silent misconfiguration).

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully populated TrainConfig.

    Raises:
        FileNotFoundError: If path does not exist.
        TypeError: If YAML contains unknown keys for any section.

    Example:
        >>> cfg = load_train_config(Path("configs/train_50k.yaml"))
        >>> cfg.trainer.epochs
        10
    """
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return TrainConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        trainer=TrainerConfig(**raw.get("trainer", {})),
    )


def load_eval_config(path: Path) -> EvaluateConfig:
    """Load EvaluateConfig from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully populated EvaluateConfig.

    Raises:
        FileNotFoundError: If path does not exist.
        TypeError: If YAML contains unknown keys for any section.

    Example:
        >>> cfg = load_eval_config(Path("configs/evaluate.yaml"))
        >>> cfg.eval.game_index
        0
    """
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return EvaluateConfig(
        data=DataConfig(**raw.get("data", {})),
        eval=EvalConfig(**raw.get("eval", {})),
    )
