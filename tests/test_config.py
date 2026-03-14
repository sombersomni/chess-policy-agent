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

    def test_tc10_encoder_n_layers_from_config(self) -> None:
        """TC10: ChessEncoder respects n_layers from ModelConfig."""
        from chess_sim.model.encoder import ChessEncoder
        enc = ChessEncoder(ModelConfig(n_layers=2))
        self.assertEqual(len(enc.transformer.layers), 2)

    def test_tc12c_invalid_fractions_raise(self) -> None:
        """TC12c: TrainerConfig raises when warmup_fraction >= decay_start_fraction."""
        with self.assertRaises(ValueError):
            TrainerConfig(warmup_fraction=0.5, decay_start_fraction=0.3)

    def test_tc14_warmup_steps_yaml_raises(self) -> None:
        """TC14: YAML with deprecated warmup_steps raises TypeError."""
        import tempfile
        import textwrap
        from pathlib import Path

        from chess_sim.config import load_v2_config
        yaml_str = textwrap.dedent("""\
            trainer:
              warmup_steps: 500
        """)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_str)
            tmp = Path(f.name)
        with self.assertRaises(TypeError):
            load_v2_config(tmp)


if __name__ == "__main__":
    unittest.main()
