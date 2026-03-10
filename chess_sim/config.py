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
    winners_side: bool = False
    chunk_size: int = 1024
    num_workers: int = 2
    train_frac: float = 0.9
    batch_size: int = 128
    hdf5_path: str = ""


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
    warmup_fraction: float = 0.05
    decay_start_fraction: float = 0.5
    min_lr: float = 1e-5
    label_smoothing: float = 0.0
    gradient_clip: float = 1.0
    patience: int = 3
    epochs: int = 10
    resume: str = ""
    checkpoint: str = ""

    def __post_init__(self) -> None:
        """Validate schedule fractions are logically ordered."""
        if not (0 < self.warmup_fraction < self.decay_start_fraction < 1):
            raise ValueError(
                f"Must satisfy 0 < warmup_fraction ({self.warmup_fraction}) "
                f"< decay_start_fraction ({self.decay_start_fraction}) < 1"
            )


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


# ---------------------------------------------------------------------------
# v2 Encoder-Decoder configs
# ---------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    """Transformer decoder architecture hyperparameters."""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512
    move_vocab_size: int = 1971


@dataclass
class Phase2Config:
    """Hyperparameters for Phase 2 self-play reinforcement learning."""

    ema_alpha: float = 0.995
    gamma: float = 0.99
    lambda_surprise: float = 0.5
    draw_reward: float = 0.1
    episodes_per_update: int = 1
    max_episode_steps: int = 200
    win_reward: float = 1.0
    loss_reward: float = -1.0
    pretrained_ckpt: str = ""
    lambda_material: float = 0.01
    lambda_illegal: float = -0.5
    lambda_check: float = 0.1
    truncation_reward: float = -0.05

    def __post_init__(self) -> None:
        """Validate hyperparameter ranges."""
        if not (0 < self.ema_alpha < 1):
            raise ValueError(
                f"ema_alpha must be in (0, 1), got {self.ema_alpha}"
            )
        if not (0 < self.gamma <= 1):
            raise ValueError(
                f"gamma must be in (0, 1], got {self.gamma}"
            )
        if self.lambda_surprise < 0:
            raise ValueError(
                "lambda_surprise must be >= 0, "
                f"got {self.lambda_surprise}"
            )
        if self.episodes_per_update < 1:
            raise ValueError(
                "episodes_per_update must be >= 1, "
                f"got {self.episodes_per_update}"
            )
        if self.lambda_material < 0:
            raise ValueError(
                "lambda_material must be >= 0, "
                f"got {self.lambda_material}"
            )
        if self.lambda_illegal > 0:
            raise ValueError(
                "lambda_illegal must be <= 0, "
                f"got {self.lambda_illegal}"
            )
        if self.lambda_check < 0:
            raise ValueError(
                "lambda_check must be >= 0, "
                f"got {self.lambda_check}"
            )


@dataclass
class AimConfig:
    """Configuration for aim experiment tracking.

    When enabled=False (default), the training pipeline uses NoOpTracker
    and aim is never imported. This keeps aim as an optional dependency.

    Example:
        >>> cfg = AimConfig(enabled=True, experiment_name="sweep_lr")
        >>> cfg.log_every_n_steps
        50
    """

    enabled: bool = False
    experiment_name: str = "chess_v2"
    repo: str = ".aim"
    log_every_n_steps: int = 50


@dataclass
class ChessModelV2Config:
    """Root config for the v2 encoder-decoder chess model."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    aim: AimConfig = field(default_factory=AimConfig)


def load_v2_config(path: Path) -> ChessModelV2Config:
    """Load ChessModelV2Config from a YAML file.

    Parses data, model, decoder, trainer, and phase2 sections.
    Unknown keys raise TypeError immediately.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully populated ChessModelV2Config.

    Raises:
        FileNotFoundError: If path does not exist.
        TypeError: If YAML contains unknown keys for any section.

    Example:
        >>> cfg = load_v2_config(Path("configs/train_v2.yaml"))
        >>> cfg.decoder.n_layers
        4
    """
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return ChessModelV2Config(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        decoder=DecoderConfig(**raw.get("decoder", {})),
        trainer=TrainerConfig(**raw.get("trainer", {})),
        phase2=Phase2Config(**raw.get("phase2", {})),
        aim=AimConfig(**raw.get("aim", {})),
    )


# ---------------------------------------------------------------------------
# HDF5 preprocess pipeline configs
# ---------------------------------------------------------------------------

@dataclass
class InputConfig:
    """PGN input settings for preprocessing."""

    pgn_path: str = ""
    max_games: int = 0  # 0 = all


@dataclass
class OutputConfig:
    """HDF5 output settings for preprocessing."""

    hdf5_path: str = "data/processed/chess_dataset.h5"
    chunk_size: int = 1000
    compression: str = "gzip"
    compression_opts: int = 4
    max_seq_len: int = 512


@dataclass
class FilterConfig:
    """Game filtering settings for preprocessing."""

    min_elo: int = 0
    min_moves: int = 5
    max_moves: int = 512
    winners_only: bool = False


@dataclass
class SplitConfig:
    """Train/val split settings."""

    train: float = 0.95
    val: float = 0.05
    seed: int = 42


@dataclass
class ProcessingConfig:
    """Multiprocessing settings for preprocessing."""

    workers: int = 4


@dataclass
class PreprocessV2Config:
    """Root config for scripts/preprocess.py."""

    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(
        default_factory=OutputConfig
    )
    filter: FilterConfig = field(
        default_factory=FilterConfig
    )
    split: SplitConfig = field(default_factory=SplitConfig)
    processing: ProcessingConfig = field(
        default_factory=ProcessingConfig
    )


# ---------------------------------------------------------------------------
# Simulation config
# ---------------------------------------------------------------------------


@dataclass
class SimulateConfig:
    """Root config for scripts/simulate.py.

    Controls simulation mode, data sources, model architecture, and display.

    Attributes:
        mode: "pgn" | "random" | "agent".
        pgn: Path to PGN or PGN.zst file (required for pgn/agent modes).
        game_index: Zero-based game index within the PGN file.
        checkpoint: Path to .pt model checkpoint (required for agent mode).
        tick_rate: Seconds to pause between plies.
        top_n: Number of agent predictions to display.
        max_plies: Maximum plies for random mode before truncation.
        use_unicode: Use Unicode piece symbols when True.
        model: Encoder architecture hyperparameters.
        decoder: Decoder architecture hyperparameters.
    """

    mode: str = "pgn"
    pgn: str = ""
    game_index: int = 0
    checkpoint: str = ""
    tick_rate: float = 0.5
    top_n: int = 3
    max_plies: int = 200
    use_unicode: bool = True
    winners_side: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


def load_simulate_config(path: Path) -> SimulateConfig:
    """Load SimulateConfig from a YAML file.

    Unknown keys raise TypeError immediately (no silent misconfiguration).

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully populated SimulateConfig.

    Raises:
        FileNotFoundError: If path does not exist.
        TypeError: If YAML contains unknown keys for any section.

    Example:
        >>> cfg = load_simulate_config(Path("configs/simulate.yaml"))
        >>> cfg.mode
        'pgn'
    """
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    model_raw = raw.pop("model", {})
    decoder_raw = raw.pop("decoder", {})
    return SimulateConfig(
        model=ModelConfig(**model_raw),
        decoder=DecoderConfig(**decoder_raw),
        **raw,
    )


# ---------------------------------------------------------------------------
# Offline RL PGN configs
# ---------------------------------------------------------------------------


@dataclass
class RLConfig:
    """Hyperparameters for offline RL training on PGN games."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    decay_start_fraction: float = 0.5
    min_lr: float = 1e-5
    gradient_clip: float = 1.0
    epochs: int = 20
    seed: int = 42
    skip_draws: bool = False
    checkpoint: str = ""
    resume: str = ""
    gamma: float = 0.99
    win_reward: float = 10.0
    loss_reward: float = -10.0
    draw_reward: float = 2.0
    lambda_ce: float = 0.5
    lambda_value: float = 1.0
    label_smoothing: float = 0.1
    train_color: str = "white"
    value_lr_multiplier: float = 5.0

    def __post_init__(self) -> None:
        """Validate RL hyperparameter ranges."""
        if self.train_color not in ("white", "black"):
            raise ValueError(
                "train_color must be 'white' or 'black'"
                f", got '{self.train_color}'"
            )
        if self.lambda_ce < 0:
            raise ValueError(
                f"lambda_ce must be >= 0, got {self.lambda_ce}"
            )
        if self.lambda_value < 0:
            raise ValueError(
                f"lambda_value must be >= 0, "
                f"got {self.lambda_value}"
            )
        if not (0 < self.gamma <= 1):
            raise ValueError(
                f"gamma must be in (0, 1], got {self.gamma}"
            )
        if self.win_reward <= 0:
            raise ValueError(
                f"win_reward must be > 0, got {self.win_reward}"
            )
        if self.loss_reward >= 0:
            raise ValueError(
                "loss_reward must be < 0, "
                f"got {self.loss_reward}"
            )
        if not (
            self.loss_reward < self.draw_reward < self.win_reward
        ):
            raise ValueError(
                "draw_reward must satisfy "
                "loss_reward < draw_reward < win_reward, "
                f"got {self.draw_reward}"
            )
        if self.warmup_fraction >= self.decay_start_fraction:
            raise ValueError(
                "warmup_fraction must be < decay_start_fraction"
                f", got {self.warmup_fraction} >= "
                f"{self.decay_start_fraction}"
            )


@dataclass
class PGNRLConfig:
    """Root config for offline RL PGN training."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decoder: DecoderConfig = field(
        default_factory=DecoderConfig
    )
    rl: RLConfig = field(default_factory=RLConfig)
    aim: AimConfig = field(default_factory=AimConfig)


def load_pgn_rl_config(path: Path) -> PGNRLConfig:
    """Load PGNRLConfig from a YAML file.

    Parses data, model, decoder, rl, and aim sections.
    Unknown keys raise TypeError immediately.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully populated PGNRLConfig.

    Raises:
        FileNotFoundError: If path does not exist.
        TypeError: If YAML contains unknown keys.

    Example:
        >>> cfg = load_pgn_rl_config(Path("configs/train_rl.yaml"))
        >>> cfg.rl.gamma
        0.99
    """
    raw: dict[str, Any] = (
        yaml.safe_load(path.read_text()) or {}
    )
    return PGNRLConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        decoder=DecoderConfig(**raw.get("decoder", {})),
        rl=RLConfig(**raw.get("rl", {})),
        aim=AimConfig(**raw.get("aim", {})),
    )


def load_preprocess_v2_config(
    path: Path,
) -> PreprocessV2Config:
    """Load PreprocessV2Config from a YAML file.

    Unknown keys raise TypeError immediately.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully populated PreprocessV2Config.

    Raises:
        FileNotFoundError: If path does not exist.
        TypeError: If YAML contains unknown keys.

    Example:
        >>> cfg = load_preprocess_v2_config(
        ...     Path("configs/preprocess_v2.yaml")
        ... )
        >>> cfg.output.hdf5_path
        'data/processed/chess_dataset.h5'
    """
    raw: dict[str, Any] = (
        yaml.safe_load(path.read_text()) or {}
    )
    return PreprocessV2Config(
        input=InputConfig(**raw.get("input", {})),
        output=OutputConfig(**raw.get("output", {})),
        filter=FilterConfig(**raw.get("filter", {})),
        split=SplitConfig(**raw.get("split", {})),
        processing=ProcessingConfig(
            **raw.get("processing", {})
        ),
    )


# ---------------------------------------------------------------------------
# RL HDF5 preprocess pipeline configs
# ---------------------------------------------------------------------------


@dataclass
class RLOutputConfig:
    """HDF5 output settings for RL preprocessing.

    Controls the on-disk layout: chunk size, compression, and
    the padded width of the move_prefix dataset.

    Example:
        >>> RLOutputConfig(hdf5_path="data/chess_rl.h5").chunk_size
        1000
    """

    hdf5_path: str = "data/processed/chess_rl.h5"
    chunk_size: int = 1000
    compression: str = "gzip"
    compression_opts: int = 4
    max_prefix_len: int = 512


@dataclass
class RLFilterConfig:
    """Game filtering settings for RL preprocessing.

    train_color selects which side's plies to keep; the other
    side is discarded at write time.

    Example:
        >>> RLFilterConfig(train_color="black").min_moves
        5
    """

    min_moves: int = 5
    max_moves: int = 512
    train_color: str = "white"

    def __post_init__(self) -> None:
        """Validate train_color is 'white' or 'black'."""
        if self.train_color not in ("white", "black"):
            raise ValueError(
                "train_color must be 'white' or 'black'"
                f", got '{self.train_color}'"
            )


@dataclass
class RLPreprocessConfig:
    """Root config for scripts/preprocess_rl.py.

    Reuses InputConfig, SplitConfig, and ProcessingConfig from
    the existing preprocess pipeline. Adds RL-specific output
    and filter sections.

    Example:
        >>> cfg = RLPreprocessConfig()
        >>> cfg.output.max_prefix_len
        512
    """

    input: InputConfig = field(default_factory=InputConfig)
    output: RLOutputConfig = field(
        default_factory=RLOutputConfig
    )
    filter: RLFilterConfig = field(
        default_factory=RLFilterConfig
    )
    split: SplitConfig = field(default_factory=SplitConfig)
    processing: ProcessingConfig = field(
        default_factory=ProcessingConfig
    )


def load_rl_preprocess_config(
    path: Path,
) -> RLPreprocessConfig:
    """Load RLPreprocessConfig from a YAML file.

    Parses input, output, filter, split, and processing sections.
    Unknown keys raise TypeError immediately.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully populated RLPreprocessConfig.

    Raises:
        FileNotFoundError: If path does not exist.
        TypeError: If YAML contains unknown keys.

    Example:
        >>> cfg = load_rl_preprocess_config(
        ...     Path("configs/preprocess_rl.yaml")
        ... )
        >>> cfg.filter.train_color
        'white'
    """
    raw: dict[str, Any] = (
        yaml.safe_load(path.read_text()) or {}
    )
    return RLPreprocessConfig(
        input=InputConfig(**raw.get("input", {})),
        output=RLOutputConfig(**raw.get("output", {})),
        filter=RLFilterConfig(**raw.get("filter", {})),
        split=SplitConfig(**raw.get("split", {})),
        processing=ProcessingConfig(
            **raw.get("processing", {})
        ),
    )
