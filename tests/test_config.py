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

    def test_tc11_warmup_fraction_computed_correctly(self) -> None:
        """TC11: Phase1Trainer first milestone equals warmup_steps."""
        from chess_sim.config import ChessModelV2Config
        from chess_sim.training.phase1_trainer import Phase1Trainer
        cfg = ChessModelV2Config()
        cfg.trainer.warmup_fraction = 0.05
        cfg.trainer.decay_start_fraction = 0.5
        trainer = Phase1Trainer(device="cpu", total_steps=1000, v2_cfg=cfg)
        warmup_milestone = trainer.scheduler._milestones[0]
        self.assertEqual(warmup_milestone, 50)  # int(0.05 * 1000)

    def test_tc12_min_lr_wired_to_cosine(self) -> None:
        """TC12: Phase1Trainer passes min_lr as eta_min to CosineAnnealingLR."""
        from chess_sim.config import ChessModelV2Config
        from chess_sim.training.phase1_trainer import Phase1Trainer
        cfg = ChessModelV2Config()
        cfg.trainer.min_lr = 1e-6
        trainer = Phase1Trainer(device="cpu", total_steps=1000, v2_cfg=cfg)
        # 3-phase schedule: [warmup, constant, cosine] — cosine is index 2
        eta_min = trainer.scheduler._schedulers[2].eta_min
        self.assertAlmostEqual(eta_min, 1e-6, places=10)

    def test_tc12b_decay_start_milestone_correct(self) -> None:
        """TC12b: Phase1Trainer second milestone equals decay_start step."""
        from chess_sim.config import ChessModelV2Config
        from chess_sim.training.phase1_trainer import Phase1Trainer
        cfg = ChessModelV2Config()
        cfg.trainer.warmup_fraction = 0.05
        cfg.trainer.decay_start_fraction = 0.5
        trainer = Phase1Trainer(device="cpu", total_steps=1000, v2_cfg=cfg)
        decay_milestone = trainer.scheduler._milestones[1]
        self.assertEqual(decay_milestone, 500)  # int(0.5 * 1000)

    def test_tc12c_invalid_fractions_raise(self) -> None:
        """TC12c: TrainerConfig raises when warmup_fraction >= decay_start_fraction."""
        from chess_sim.config import TrainerConfig
        with self.assertRaises(ValueError):
            TrainerConfig(warmup_fraction=0.5, decay_start_fraction=0.3)

    def test_tc13_label_smoothing_wired_to_criterion(self) -> None:
        """TC13: Phase1Trainer passes label_smoothing to CrossEntropyLoss."""
        from chess_sim.config import ChessModelV2Config
        from chess_sim.training.phase1_trainer import Phase1Trainer
        cfg = ChessModelV2Config()
        cfg.trainer.label_smoothing = 0.1
        trainer = Phase1Trainer(device="cpu", total_steps=1000, v2_cfg=cfg)
        self.assertAlmostEqual(
            trainer.criterion.label_smoothing, 0.1, places=6
        )

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


class TestRLConfigCompositeReward(unittest.TestCase):
    """Tests for RLConfig composite reward fields."""

    def test_lambda_outcome_default(self) -> None:
        """lambda_outcome defaults to 1.0."""
        from chess_sim.config import RLConfig
        self.assertEqual(RLConfig().lambda_outcome, 1.0)

    def test_lambda_material_default(self) -> None:
        """lambda_material defaults to 0.1."""
        from chess_sim.config import RLConfig
        self.assertEqual(RLConfig().lambda_material, 0.1)

    def test_draw_reward_norm_default(self) -> None:
        """draw_reward_norm defaults to 0.0."""
        from chess_sim.config import RLConfig
        self.assertEqual(RLConfig().draw_reward_norm, 0.0)

    def test_lambda_outcome_negative_raises(self) -> None:
        """T-CR11: lambda_outcome < 0 raises ValueError."""
        from chess_sim.config import RLConfig
        with self.assertRaises(ValueError):
            RLConfig(lambda_outcome=-0.1)

    def test_lambda_material_negative_raises(self) -> None:
        """T-CR12: lambda_material < 0 raises ValueError."""
        from chess_sim.config import RLConfig
        with self.assertRaises(ValueError):
            RLConfig(lambda_material=-0.5)

    def test_draw_reward_norm_out_of_range_raises(
        self,
    ) -> None:
        """T-CR13: draw_reward_norm outside [-1, 1] raises."""
        from chess_sim.config import RLConfig
        with self.assertRaises(ValueError):
            RLConfig(draw_reward_norm=1.5)
        with self.assertRaises(ValueError):
            RLConfig(draw_reward_norm=-1.5)


class TestRLConfigRSBC(unittest.TestCase):
    """Tests for RLConfig RSBC fields."""

    def test_lambda_rsbc_default_is_one(self) -> None:
        """T-R8: RLConfig() has lambda_rsbc == 1.0."""
        from chess_sim.config import RLConfig
        self.assertEqual(RLConfig().lambda_rsbc, 1.0)

    def test_lambda_rsbc_negative_raises_value_error(
        self,
    ) -> None:
        """T-R9: lambda_rsbc < 0 raises ValueError."""
        from chess_sim.config import RLConfig
        with self.assertRaises(ValueError):
            RLConfig(lambda_rsbc=-0.1)

    def test_rsbc_normalize_per_game_default_true(
        self,
    ) -> None:
        """T-R10: rsbc_normalize_per_game defaults True."""
        from chess_sim.config import RLConfig
        self.assertTrue(RLConfig().rsbc_normalize_per_game)


if __name__ == "__main__":
    unittest.main()
