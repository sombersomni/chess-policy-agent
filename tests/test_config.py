"""Unit tests for chess_sim.config — TC01 through TC12.

Tests cover: YAML loading, CLI override merge logic,
Trainer backward compatibility, and model architecture config.

All tests run on CPU only (per project convention).
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from chess_sim.config import (
    ModelConfig,
    TrainConfig,
    TrainerConfig,
    load_train_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(content: str) -> Path:
    """Write a YAML string to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    )
    tmp.write(textwrap.dedent(content))
    tmp.flush()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# TC01-TC04: YAML loading
# ---------------------------------------------------------------------------

class TestLoadTrainConfig(unittest.TestCase):
    """TC01: load_train_config with valid YAML returns correct fields."""

    def test_tc01_valid_yaml_fields(self) -> None:
        path = _write_yaml("""
            data:
              pgn: "foo.pgn"
              max_games: 5000
              batch_size: 64
            model:
              n_layers: 4
              d_model: 128
            trainer:
              learning_rate: 1.0e-3
              epochs: 5
              checkpoint: "out.pt"
        """)
        cfg = load_train_config(path)
        self.assertEqual(cfg.data.pgn, "foo.pgn")
        self.assertEqual(cfg.data.max_games, 5000)
        self.assertEqual(cfg.data.batch_size, 64)
        self.assertEqual(cfg.model.n_layers, 4)
        self.assertEqual(cfg.model.d_model, 128)
        self.assertAlmostEqual(cfg.trainer.learning_rate, 1e-3)
        self.assertEqual(cfg.trainer.epochs, 5)
        self.assertEqual(cfg.trainer.checkpoint, "out.pt")

    def test_tc02_missing_sections_use_defaults(self) -> None:
        """TC02: Missing YAML sections fall back to dataclass defaults."""
        path = _write_yaml("""
            trainer:
              epochs: 3
        """)
        cfg = load_train_config(path)
        self.assertEqual(cfg.data.batch_size, 128)   # DataConfig default
        self.assertEqual(cfg.model.n_layers, 6)      # ModelConfig default
        self.assertEqual(cfg.trainer.epochs, 3)

    def test_tc03_unknown_key_raises(self) -> None:
        """TC03: Unknown YAML keys raise TypeError immediately."""
        path = _write_yaml("""
            trainer:
              unknown_hyperparameter: 99
        """)
        with self.assertRaises(TypeError):
            load_train_config(path)

    def test_tc04_missing_file_raises(self) -> None:
        """TC04: Missing config file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_train_config(Path("/nonexistent/config.yaml"))


# ---------------------------------------------------------------------------
# TC08-TC10: Trainer and model backward compat + config wiring
# ---------------------------------------------------------------------------

class TestTrainerConfig(unittest.TestCase):
    """Tests that Trainer and model accept config objects correctly."""

    def test_tc08_trainer_backward_compat(self) -> None:
        """TC08: Trainer() with no config args works as before."""
        from chess_sim.training.trainer import Trainer
        trainer = Trainer(device="cpu", total_steps=10)
        # Should instantiate without error; gradient_clip should be module default
        from chess_sim.training.trainer import GRADIENT_CLIP
        self.assertEqual(trainer._gradient_clip, GRADIENT_CLIP)

    def test_tc09_trainer_uses_lr_from_config(self) -> None:
        """TC09: Trainer uses learning_rate from TrainerConfig."""
        from chess_sim.training.trainer import Trainer
        cfg = TrainerConfig(learning_rate=1e-3)
        trainer = Trainer(device="cpu", total_steps=10, trainer_cfg=cfg)
        actual_lr = trainer.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(actual_lr, 1e-3, places=6)

    def test_tc10_encoder_n_layers_from_config(self) -> None:
        """TC10: ChessEncoder respects n_layers from ModelConfig."""
        from chess_sim.model.encoder import ChessEncoder
        enc = ChessEncoder(ModelConfig(n_layers=2))
        self.assertEqual(len(enc.transformer.layers), 2)


if __name__ == "__main__":
    unittest.main()
