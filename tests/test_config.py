"""Unit tests for chess_sim.config — TC01 through TC12.

Tests cover: YAML loading, CLI override merge logic,
Trainer backward compatibility, and model architecture config.

All tests run on CPU only (per project convention).
"""

from __future__ import annotations

import argparse
import tempfile
import textwrap
import unittest
from pathlib import Path

from chess_sim.config import (
    DataConfig,
    EvalConfig,
    EvaluateConfig,
    ModelConfig,
    TrainConfig,
    TrainerConfig,
    load_eval_config,
    load_train_config,
)
from scripts.train_real import _merge_train_config
from scripts.evaluate import _merge_eval_config


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


def _make_train_args(**kwargs: object) -> argparse.Namespace:
    """Build a Namespace with all train args defaulting to None."""
    defaults = dict(
        config=None, pgn=None, num_games=None,
        max_games=None, winners_only=None,
        batch_size=None, epochs=None,
        checkpoint=None, resume=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_eval_args(**kwargs: object) -> argparse.Namespace:
    """Build a Namespace with all eval args defaulting to None."""
    defaults = dict(
        config=None, checkpoint=None, pgn=None,
        game_index=None, top_n=None, winners_only=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


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
# TC05-TC07: CLI override merge logic
# ---------------------------------------------------------------------------

class TestMergeTrainConfig(unittest.TestCase):
    """Tests for _merge_train_config."""

    def test_tc05_cli_epochs_overrides_yaml(self) -> None:
        """TC05: --epochs CLI arg overrides trainer.epochs from YAML."""
        cfg = TrainConfig(trainer=TrainerConfig(epochs=10))
        args = _make_train_args(epochs=3)
        result = _merge_train_config(args, cfg)
        self.assertEqual(result.trainer.epochs, 3)

    def test_tc06_absent_cli_does_not_overwrite_yaml(self) -> None:
        """TC06: CLI args left as None do not overwrite YAML values."""
        cfg = TrainConfig(
            data=DataConfig(pgn="myfile.pgn", batch_size=64),
            trainer=TrainerConfig(epochs=20),
        )
        args = _make_train_args()  # all None
        result = _merge_train_config(args, cfg)
        self.assertEqual(result.data.pgn, "myfile.pgn")
        self.assertEqual(result.data.batch_size, 64)
        self.assertEqual(result.trainer.epochs, 20)

    def test_tc07_winners_only_flag_sets_true(self) -> None:
        """TC07: --winners-only flag sets data.winners_only=True."""
        cfg = TrainConfig(data=DataConfig(winners_only=False))
        args = _make_train_args(winners_only=True)
        result = _merge_train_config(args, cfg)
        self.assertTrue(result.data.winners_only)


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


# ---------------------------------------------------------------------------
# TC11-TC12: evaluate.py backward compat
# ---------------------------------------------------------------------------

class TestMergeEvalConfig(unittest.TestCase):
    """Tests for _merge_eval_config."""

    def test_tc11_checkpoint_override(self) -> None:
        """TC11: --checkpoint CLI arg overrides eval.checkpoint from config."""
        cfg = EvaluateConfig(eval=EvalConfig(checkpoint="old.pt"))
        args = _make_eval_args(checkpoint="new.pt")
        result = _merge_eval_config(args, cfg)
        self.assertEqual(result.eval.checkpoint, "new.pt")

    def test_tc12_defaults_produce_valid_eval_config(self) -> None:
        """TC12: EvaluateConfig defaults are well-formed."""
        cfg = EvaluateConfig()
        self.assertEqual(cfg.eval.game_index, 0)
        self.assertEqual(cfg.eval.top_n, 3)
        self.assertFalse(cfg.data.winners_only)


if __name__ == "__main__":
    unittest.main()
