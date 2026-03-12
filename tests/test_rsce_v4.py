"""Tests for RSCE V4 batched training pipeline.

TC01-TC05: PGNRewardPreprocessor (checksum, generate, cache)
TC06-TC10: ChessRLDataset (len, getitem, splits, multipliers)
TC11-TC13: PGNRLTrainerV4 (train_step, train_epoch, evaluate)
TC14:      Checkpoint round-trip
TC15:      Lazy HDF5 handle opening
TC16-TC30: Dual-direction RSCE loss (rsce_dual_loss, config,
           preprocessor loss_mode, warmup, evaluate avoidance)
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from chess_sim.config import (
    DecoderConfig,
    ModelConfig,
    PGNRLConfig,
    RLConfig,
)
from chess_sim.data.chess_rl_dataset import ChessRLDataset
from chess_sim.data.pgn_reward_preprocessor import (
    PGNRewardPreprocessor,
)


def _write_synthetic_hdf5(
    path: Path,
    n_rows: int = 200,
    n_games: int = 10,
    multipliers: np.ndarray | None = None,
) -> None:
    """Write a minimal HDF5 matching the RSCE v4 schema."""
    with h5py.File(path, "w") as hf:
        # Channels: piece[0-7], color[0-2], traj[0-4]
        ch0 = np.random.randint(0, 8, (n_rows, 65))
        ch1 = np.random.randint(0, 3, (n_rows, 65))
        ch2 = np.random.randint(0, 5, (n_rows, 65))
        board = np.stack(
            [ch0, ch1, ch2], axis=-1
        ).astype(np.float32)
        hf.create_dataset("board", data=board)

        ct = np.random.randint(
            0, 3, (n_rows, 65), dtype=np.int8
        )
        hf.create_dataset("color_tokens", data=ct)

        # Keep targets within legal range [3, 19] to match legal_mask
        tm = np.random.randint(
            3, 20, (n_rows,), dtype=np.int32
        )
        hf.create_dataset("target_move", data=tm)

        if multipliers is None:
            multipliers = np.ones(
                n_rows, dtype=np.float32
            )
        hf.create_dataset("multiplier", data=multipliers)

        # Assign game_ids evenly across rows
        game_ids = np.repeat(
            np.arange(n_games, dtype=np.int32),
            n_rows // n_games,
        )
        # Handle remainder rows
        remainder = n_rows - len(game_ids)
        if remainder > 0:
            game_ids = np.concatenate([
                game_ids,
                np.full(
                    remainder,
                    n_games - 1,
                    dtype=np.int32,
                ),
            ])
        hf.create_dataset("game_id", data=game_ids)

        ply_idx = np.tile(
            np.arange(
                n_rows // n_games, dtype=np.int16
            ),
            n_games,
        )[:n_rows]
        hf.create_dataset("ply_idx", data=ply_idx)

        outcome = np.random.choice(
            [-1, 0, 1], size=n_rows
        ).astype(np.int8)
        hf.create_dataset("outcome", data=outcome)

        # loss_mode: +1 for winners/draws, -1 for losers
        loss_mode = np.where(
            outcome >= 0, 1, -1
        ).astype(np.int8)
        hf.create_dataset("loss_mode", data=loss_mode)

        legal_mask = np.zeros(
            (n_rows, 1971), dtype=bool
        )
        # Mark a few tokens legal so targets are reachable
        legal_mask[:, 3:20] = True
        hf.create_dataset("legal_mask", data=legal_mask)


def _make_minimal_pgn(path: Path) -> None:
    """Write a tiny PGN file with 3 short games."""
    pgn = (
        '[Event "Test"]\n[Result "1-0"]\n\n'
        "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 "
        "4. Qxf7# 1-0\n\n"
        '[Event "Test"]\n[Result "0-1"]\n\n'
        "1. f3 e5 2. g4 Qh4# 0-1\n\n"
        '[Event "Test"]\n[Result "1/2-1/2"]\n\n'
        "1. e4 e5 2. Ke2 Ke7 1/2-1/2\n\n"
    )
    path.write_text(pgn)


def _make_cfg(**rl_overrides: object) -> PGNRLConfig:
    """Build a minimal PGNRLConfig suitable for testing."""
    defaults: dict[str, object] = {
        "batch_size": 4,
        "num_workers": 0,
        "val_split_fraction": 0.1,
        "hdf5_chunk_size": 64,
        "use_structural_mask": False,
        "warmup_fraction": 0.05,
        "decay_start_fraction": 0.5,
        "epochs": 1,
        "balance_outcomes": False,
    }
    defaults.update(rl_overrides)
    return PGNRLConfig(
        model=ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        ),
        decoder=DecoderConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            max_seq_len=64,
            move_vocab_size=1971,
        ),
        rl=RLConfig(**defaults),
    )


class TestPGNRewardPreprocessor(unittest.TestCase):
    """Tests for PGNRewardPreprocessor behavior."""

    def test_tc01_checksum_deterministic(
        self,
    ) -> None:
        """_compute_checksum returns same value for same inputs."""
        with tempfile.TemporaryDirectory() as td:
            pgn_path = Path(td) / "games.pgn"
            _make_minimal_pgn(pgn_path)
            cfg = _make_cfg()
            pp = PGNRewardPreprocessor(cfg)
            cs1 = pp._compute_checksum(pgn_path, 0)
            cs2 = pp._compute_checksum(pgn_path, 0)
            self.assertEqual(cs1, cs2)
            self.assertEqual(len(cs1), 64)

    def test_tc02_checksum_differs_for_max_games(
        self,
    ) -> None:
        """_compute_checksum differs when max_games changes."""
        with tempfile.TemporaryDirectory() as td:
            pgn_path = Path(td) / "games.pgn"
            _make_minimal_pgn(pgn_path)
            cfg = _make_cfg()
            pp = PGNRewardPreprocessor(cfg)
            cs_0 = pp._compute_checksum(pgn_path, 0)
            cs_5 = pp._compute_checksum(pgn_path, 5)
            self.assertNotEqual(cs_0, cs_5)

    def test_tc03_generate_creates_correct_datasets(
        self,
    ) -> None:
        """generate() creates HDF5 with all 9 required datasets."""
        with tempfile.TemporaryDirectory() as td:
            pgn_path = Path(td) / "games.pgn"
            _make_minimal_pgn(pgn_path)
            hdf5_path = Path(td) / "out.h5"

            cfg = _make_cfg()
            pp = PGNRewardPreprocessor(cfg)
            result = pp.generate(
                pgn_path, hdf5_path, max_games=3
            )
            self.assertTrue(result.exists())

            expected_keys = {
                "board",
                "color_tokens",
                "target_move",
                "multiplier",
                "game_id",
                "ply_idx",
                "outcome",
                "loss_mode",
                "legal_mask",
            }
            with h5py.File(hdf5_path, "r") as hf:
                self.assertEqual(
                    set(hf.keys()), expected_keys
                )
                n = hf["board"].shape[0]
                self.assertGreater(n, 0)
                self.assertEqual(
                    hf["board"].shape, (n, 65, 3)
                )
                self.assertEqual(
                    hf["color_tokens"].shape, (n, 65)
                )

    def test_tc04_generate_idempotent(self) -> None:
        """Calling generate() twice does not change the file."""
        with tempfile.TemporaryDirectory() as td:
            pgn_path = Path(td) / "games.pgn"
            _make_minimal_pgn(pgn_path)
            hdf5_path = Path(td) / "out.h5"

            cfg = _make_cfg()
            pp = PGNRewardPreprocessor(cfg)
            pp.generate(
                pgn_path, hdf5_path, max_games=3
            )
            mtime1 = hdf5_path.stat().st_mtime

            # Second call should be a cache hit
            pp.generate(
                pgn_path, hdf5_path, max_games=3
            )
            mtime2 = hdf5_path.stat().st_mtime
            self.assertEqual(mtime1, mtime2)

    def test_tc05_cache_invalid_on_config_change(
        self,
    ) -> None:
        """_is_cache_valid returns False when config differs."""
        with tempfile.TemporaryDirectory() as td:
            pgn_path = Path(td) / "games.pgn"
            _make_minimal_pgn(pgn_path)
            hdf5_path = Path(td) / "out.h5"

            cfg1 = _make_cfg(lambda_outcome=1.0)
            pp1 = PGNRewardPreprocessor(cfg1)
            pp1.generate(
                pgn_path, hdf5_path, max_games=3
            )

            # Change lambda_outcome
            cfg2 = _make_cfg(lambda_outcome=2.0)
            pp2 = PGNRewardPreprocessor(cfg2)
            self.assertFalse(
                pp2._is_cache_valid(
                    hdf5_path, pgn_path, 3
                )
            )


class TestChessRLDataset(unittest.TestCase):
    """Tests for ChessRLDataset splits and indexing."""

    def test_tc06_len_train_split(self) -> None:
        """__len__ returns correct count for train split."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=200, n_games=10
            )
            ds = ChessRLDataset(
                path,
                val_split_fraction=0.1,
                split="train",
            )
            try:
                # 10 games, val=last 1, train=9 games
                # 9 games * 20 rows = 180
                self.assertEqual(len(ds), 180)
            finally:
                ds.close()

    def test_tc07_len_val_split_nonoverlapping(
        self,
    ) -> None:
        """__len__ returns correct count for val split."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=200, n_games=10
            )
            train_ds = ChessRLDataset(
                path,
                val_split_fraction=0.1,
                split="train",
            )
            val_ds = ChessRLDataset(
                path,
                val_split_fraction=0.1,
                split="val",
            )
            try:
                # Total must equal 200
                self.assertEqual(
                    len(train_ds) + len(val_ds), 200
                )
                self.assertGreater(len(val_ds), 0)
            finally:
                train_ds.close()
                val_ds.close()

    def test_tc08_getitem_shapes_and_types(
        self,
    ) -> None:
        """__getitem__ returns correct 7-tuple shapes."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2
            )
            ds = ChessRLDataset(
                path,
                val_split_fraction=0.1,
                split="train",
            )
            try:
                (
                    board, tgt, mult,
                    ct, outcome, loss_mode, legal_mask,
                ) = ds[0]
                self.assertEqual(board.shape, (65, 3))
                self.assertEqual(
                    board.dtype, torch.float32
                )
                self.assertIsInstance(tgt, int)
                self.assertIsInstance(mult, float)
                self.assertEqual(ct.shape, (65,))
                self.assertEqual(
                    ct.dtype, torch.int64
                )
                self.assertIsInstance(outcome, int)
                self.assertIn(outcome, (-1, 0, 1))
                self.assertIsInstance(loss_mode, int)
                self.assertIn(loss_mode, (-1, +1))
                self.assertEqual(
                    legal_mask.shape, (1971,)
                )
                self.assertEqual(
                    legal_mask.dtype, torch.bool
                )
            finally:
                ds.close()

    def test_tc09_splits_nonoverlapping_by_game_id(
        self,
    ) -> None:
        """Train + val splits are non-overlapping by game_id."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=200, n_games=10
            )
            train_ds = ChessRLDataset(
                path,
                val_split_fraction=0.2,
                split="train",
            )
            val_ds = ChessRLDataset(
                path,
                val_split_fraction=0.2,
                split="val",
            )
            try:
                train_gids = train_ds._game_ids_set
                val_gids = val_ds._game_ids_set
                self.assertEqual(
                    len(train_gids & val_gids), 0,
                    "Train and val game_ids must "
                    "not overlap",
                )
            finally:
                train_ds.close()
                val_ds.close()

    def test_tc10_multipliers_positive_mean_one(
        self,
    ) -> None:
        """Multipliers from generate() are positive, mean~1."""
        with tempfile.TemporaryDirectory() as td:
            pgn_path = Path(td) / "games.pgn"
            _make_minimal_pgn(pgn_path)
            hdf5_path = Path(td) / "out.h5"

            cfg = _make_cfg(
                rsbc_normalize_per_game=True,
            )
            pp = PGNRewardPreprocessor(cfg)
            pp.generate(
                pgn_path, hdf5_path, max_games=3
            )

            with h5py.File(hdf5_path, "r") as hf:
                multipliers = hf["multiplier"][:]
                game_ids = hf["game_id"][:]

                # All multipliers must be positive
                self.assertTrue(
                    (multipliers > 0).all(),
                    "All multipliers must be positive",
                )

                # Per-game mean should be ~1.0
                for gid in np.unique(game_ids):
                    mask = game_ids == gid
                    game_mult = multipliers[mask]
                    mean_val = float(
                        game_mult.mean()
                    )
                    self.assertAlmostEqual(
                        mean_val,
                        1.0,
                        places=4,
                        msg=(
                            f"game {gid}: mean "
                            f"multiplier = {mean_val}"
                            ", expected ~1.0"
                        ),
                    )


class TestPGNRLTrainerV4(unittest.TestCase):
    """Tests for PGNRLTrainerV4 training and evaluation."""

    def _make_trainer(
        self,
        **rl_kw: object,
    ) -> tuple:
        from chess_sim.training.pgn_rl_trainer_v4 import (
            PGNRLTrainerV4,
        )
        cfg = _make_cfg(**rl_kw)
        trainer = PGNRLTrainerV4(
            cfg, device="cpu", total_steps=100
        )
        return trainer, cfg

    def test_tc11_train_step_finite_loss(
        self,
    ) -> None:
        """_train_step returns finite loss for random batch."""
        trainer, _ = self._make_trainer()
        B = 4
        ch0 = torch.randint(0, 8, (B, 65)).float()
        ch1 = torch.randint(0, 3, (B, 65)).float()
        ch2 = torch.randint(0, 5, (B, 65)).float()
        board = torch.stack(
            [ch0, ch1, ch2], dim=-1
        )
        # Keep targets within legal range [3, 19] so CE is finite
        targets = torch.randint(3, 20, (B,))
        mults = torch.ones(B)
        ct = torch.randint(
            0, 3, (B, 65), dtype=torch.long
        )
        loss_modes = torch.ones(
            B, dtype=torch.int8
        )

        legal_mask = torch.zeros(B, 1971, dtype=torch.bool)
        legal_mask[:, 3:20] = True  # ensure targets reachable

        result = trainer._train_step(
            board, targets, ct, legal_mask
        )
        self.assertIn("loss", result)
        self.assertTrue(
            np.isfinite(result["loss"]),
            f"Loss must be finite, got "
            f"{result['loss']}",
        )
        self.assertGreaterEqual(result["loss"], 0.0)
        self.assertEqual(result["n_total"], B)
        self.assertIn("grad_norm", result)
        self.assertTrue(
            np.isfinite(result["grad_norm"]),
            f"grad_norm must be finite, got "
            f"{result['grad_norm']}",
        )
        self.assertGreaterEqual(
            result["grad_norm"], 0.0
        )

    def test_tc12_train_epoch_returns_all_keys(
        self,
    ) -> None:
        """train_epoch returns all required dict keys."""
        trainer, _ = self._make_trainer()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2
            )
            ds = ChessRLDataset(
                path,
                val_split_fraction=0.1,
                split="train",
            )
            try:
                metrics = trainer.train_epoch(ds)
                expected_keys = {
                    "total_loss",
                    "train_accuracy",
                    "n_samples",
                    "n_games",
                }
                self.assertEqual(
                    set(metrics.keys()),
                    expected_keys,
                )
                self.assertGreater(
                    metrics["n_samples"], 0
                )
            finally:
                ds.close()

    def test_tc13_evaluate_returns_val_metrics(
        self,
    ) -> None:
        """evaluate returns val metrics incl avoidance."""
        trainer, _ = self._make_trainer()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2
            )
            ds = ChessRLDataset(
                path,
                val_split_fraction=0.5,
                split="val",
            )
            try:
                metrics = trainer.evaluate(ds)
                self.assertIn("val_loss", metrics)
                self.assertIn(
                    "val_accuracy", metrics
                )
                self.assertIn(
                    "val_n_samples", metrics
                )
                self.assertIn(
                    "acc_winners", metrics
                )
                self.assertIn(
                    "acc_losers", metrics
                )
                self.assertIn(
                    "acc_draws", metrics
                )
                self.assertTrue(
                    np.isfinite(
                        metrics["val_loss"]
                    )
                )
                self.assertGreaterEqual(
                    metrics["val_accuracy"],
                    0.0,
                )
                self.assertLessEqual(
                    metrics["val_accuracy"],
                    1.0,
                )
                for key in (
                    "val_accuracy",
                    "acc_winners",
                    "acc_losers",
                    "acc_draws",
                ):
                    self.assertGreaterEqual(
                        metrics[key],
                        0.0,
                        f"{key} must be >= 0",
                    )
                    self.assertLessEqual(
                        metrics[key],
                        1.0,
                        f"{key} must be <= 1",
                    )
            finally:
                ds.close()

    def test_tc14_checkpoint_roundtrip(self) -> None:
        """save + load preserves weights and step."""
        trainer, cfg = self._make_trainer()
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "ckpt.pt"

            trainer._global_step = 99
            trainer.save_checkpoint(ckpt_path)

            orig_w = (
                trainer.model.encoder.embedding
                .piece_emb.weight.data.clone()
            )

            with torch.no_grad():
                (
                    trainer.model.encoder
                    .embedding.piece_emb
                    .weight.data.fill_(999.0)
                )
            trainer._global_step = 0

            corrupted_w = (
                trainer.model.encoder.embedding
                .piece_emb.weight.data.clone()
            )
            self.assertFalse(
                torch.equal(orig_w, corrupted_w)
            )

            trainer.load_checkpoint(ckpt_path)
            restored_w = (
                trainer.model.encoder.embedding
                .piece_emb.weight.data.clone()
            )
            self.assertTrue(
                torch.equal(orig_w, restored_w),
                "Weights should be restored",
            )
            self.assertEqual(
                trainer._global_step,
                99,
                "_global_step must be restored",
            )

    def test_tc15_lazy_hdf5_handle(self) -> None:
        """ChessRLDataset opens h5py lazily."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2
            )
            ds = ChessRLDataset(
                path,
                val_split_fraction=0.1,
                split="train",
            )
            self.assertIsNone(
                ds._h5,
                "_h5 must be None before first "
                "__getitem__",
            )

            _ = ds[0]
            self.assertIsNotNone(ds._h5)
            ds.close()


class TestRSCEDualLoss(unittest.TestCase):
    """Tests for rsce_dual_loss and dual-direction changes."""

    def test_tc16_all_imitation_batch(self) -> None:
        """All-imitation: repulsion loss 0, frac 0."""
        from chess_sim.training.pgn_rl_trainer_v4 import (
            rsce_dual_loss,
        )
        B, V = 4, 1971
        logits = torch.randn(B, V)
        targets = torch.randint(3, V, (B,))
        mults = torch.ones(B)
        modes = torch.ones(B, dtype=torch.int8)
        loss, metrics = rsce_dual_loss(
            logits, targets, mults, modes
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(
            metrics["loss_repulsion"], 0.0
        )
        self.assertAlmostEqual(
            metrics["frac_repulsion"], 0.0
        )

    def test_tc17_all_repulsion_batch(self) -> None:
        """All-repulsion: imitation loss 0, frac 1."""
        from chess_sim.training.pgn_rl_trainer_v4 import (
            rsce_dual_loss,
        )
        B, V = 4, 1971
        logits = torch.randn(B, V)
        targets = torch.randint(3, V, (B,))
        mults = torch.ones(B)
        modes = -torch.ones(B, dtype=torch.int8)
        loss, metrics = rsce_dual_loss(
            logits, targets, mults, modes
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(
            metrics["loss_imitation"], 0.0
        )
        self.assertAlmostEqual(
            metrics["frac_repulsion"], 1.0
        )

    def test_tc18_mixed_batch(self) -> None:
        """Mixed: both branches contribute, frac 0.5."""
        from chess_sim.training.pgn_rl_trainer_v4 import (
            rsce_dual_loss,
        )
        B, V = 4, 1971
        logits = torch.randn(B, V)
        targets = torch.randint(3, V, (B,))
        mults = torch.ones(B)
        modes = torch.tensor(
            [1, -1, 1, -1], dtype=torch.int8
        )
        loss, metrics = rsce_dual_loss(
            logits, targets, mults, modes
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(
            metrics["loss_imitation"], 0.0
        )
        self.assertGreater(
            metrics["loss_repulsion"], 0.0
        )
        self.assertAlmostEqual(
            metrics["frac_repulsion"], 0.5
        )

    def test_tc19_numerical_stability_high_conf(
        self,
    ) -> None:
        """Repulsion with p near 1.0 produces finite loss."""
        from chess_sim.training.pgn_rl_trainer_v4 import (
            rsce_dual_loss,
        )
        B, V = 1, 1971
        logits = torch.zeros(B, V)
        target_idx = 100
        logits[0, target_idx] = 50.0
        targets = torch.tensor([target_idx])
        mults = torch.ones(B)
        modes = torch.tensor(
            [-1], dtype=torch.int8
        )
        loss, _ = rsce_dual_loss(
            logits, targets, mults, modes
        )
        self.assertTrue(
            torch.isfinite(loss),
            f"Loss must be finite, got {loss}",
        )

    def test_tc20_repulsion_weight_zero(self) -> None:
        """repulsion_weight=0 zeroes repulsion."""
        from chess_sim.training.pgn_rl_trainer_v4 import (
            rsce_dual_loss,
        )
        B, V = 4, 1971
        logits = torch.randn(B, V)
        targets = torch.randint(3, V, (B,))
        mults = torch.ones(B)
        modes = torch.tensor(
            [1, -1, 1, -1], dtype=torch.int8
        )
        loss, metrics = rsce_dual_loss(
            logits, targets, mults, modes,
            repulsion_weight=0.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(
            metrics["loss_repulsion"], 0.0,
            places=6,
        )

    def test_tc21_preprocessor_loss_mode_values(
        self,
    ) -> None:
        """Preprocessor stores correct loss_mode per outcome."""
        with tempfile.TemporaryDirectory() as td:
            pgn_path = Path(td) / "games.pgn"
            _make_minimal_pgn(pgn_path)
            hdf5_path = Path(td) / "out.h5"

            cfg = _make_cfg(
                draw_reward_norm=0.5,
                rsce_r_ref=0.0,
            )
            pp = PGNRewardPreprocessor(cfg)
            pp.generate(
                pgn_path, hdf5_path, max_games=3
            )

            with h5py.File(hdf5_path, "r") as hf:
                outcomes = hf["outcome"][:]
                loss_modes = hf["loss_mode"][:]

                # Winners -> imitation (+1)
                winner_mask = outcomes == 1
                if winner_mask.any():
                    self.assertTrue(
                        (loss_modes[winner_mask] == 1)
                        .all()
                    )
                # Losers -> repulsion (-1)
                loser_mask = outcomes == -1
                if loser_mask.any():
                    self.assertTrue(
                        (loss_modes[loser_mask] == -1)
                        .all()
                    )
                # Draws -> imitation (+1)
                # (draw_reward_norm=0.5 > r_ref=0.0)
                draw_mask = outcomes == 0
                if draw_mask.any():
                    self.assertTrue(
                        (loss_modes[draw_mask] == 1)
                        .all()
                    )

    def test_tc22_preprocessor_multiplier_sign(
        self,
    ) -> None:
        """Preprocessor stores corrected multiplier sign."""
        with tempfile.TemporaryDirectory() as td:
            pgn_path = Path(td) / "games.pgn"
            _make_minimal_pgn(pgn_path)
            hdf5_path = Path(td) / "out.h5"

            cfg = _make_cfg(
                rsbc_normalize_per_game=False,
                rsce_r_ref=0.0,
            )
            pp = PGNRewardPreprocessor(cfg)
            pp.generate(
                pgn_path, hdf5_path, max_games=3
            )

            with h5py.File(hdf5_path, "r") as hf:
                mults = hf["multiplier"][:]
                outcomes = hf["outcome"][:]

                # With exp(+(R - r_ref)):
                # Winner plies (R > 0) -> m > 1
                winner_mask = outcomes == 1
                if winner_mask.any():
                    self.assertTrue(
                        (mults[winner_mask] > 1.0)
                        .all(),
                        "Winner multipliers should "
                        "be > 1.0 after sign flip",
                    )
                # Loser plies (R < 0) -> m < 1
                loser_mask = outcomes == -1
                if loser_mask.any():
                    self.assertTrue(
                        (mults[loser_mask] < 1.0)
                        .all(),
                        "Loser multipliers should "
                        "be < 1.0 after sign flip",
                    )

    def test_tc23_warmup_step_zero(self) -> None:
        """warmup>0 at step 0: effective repul weight 0."""
        from chess_sim.training.pgn_rl_trainer_v4 import (
            PGNRLTrainerV4,
        )
        cfg = _make_cfg(
            rsce_repulsion_weight=2.0,
            rsce_repulsion_warmup=0.2,
        )
        trainer = PGNRLTrainerV4(
            cfg, device="cpu", total_steps=100
        )
        trainer._global_step = 0
        eff_w = trainer._effective_repulsion_weight()
        self.assertAlmostEqual(eff_w, 0.0)

    def test_tc24_warmup_at_warmup_steps(
        self,
    ) -> None:
        """At step=warmup_steps, full weight applies."""
        from chess_sim.training.pgn_rl_trainer_v4 import (
            PGNRLTrainerV4,
        )
        cfg = _make_cfg(
            rsce_repulsion_weight=2.0,
            rsce_repulsion_warmup=0.2,
        )
        trainer = PGNRLTrainerV4(
            cfg, device="cpu", total_steps=100
        )
        # warmup_steps = 0.2 * 100 = 20
        trainer._global_step = 20
        eff_w = trainer._effective_repulsion_weight()
        self.assertAlmostEqual(eff_w, 2.0)

    def test_tc25_warmup_disabled(self) -> None:
        """warmup=0 gives full weight at all steps."""
        from chess_sim.training.pgn_rl_trainer_v4 import (
            PGNRLTrainerV4,
        )
        cfg = _make_cfg(
            rsce_repulsion_weight=2.0,
            rsce_repulsion_warmup=0.0,
        )
        trainer = PGNRLTrainerV4(
            cfg, device="cpu", total_steps=100
        )
        trainer._global_step = 0
        eff_w = trainer._effective_repulsion_weight()
        self.assertAlmostEqual(eff_w, 2.0)

    def test_tc26_config_negative_repulsion_weight(
        self,
    ) -> None:
        """Negative rsce_repulsion_weight raises ValueError."""
        with self.assertRaises(ValueError):
            RLConfig(rsce_repulsion_weight=-0.5)

    def test_tc27_config_warmup_ge_one(self) -> None:
        """rsce_repulsion_warmup >= 1.0 raises ValueError."""
        with self.assertRaises(ValueError):
            RLConfig(rsce_repulsion_warmup=1.0)

    def test_tc28_evaluate_repulsion_avoidance(
        self,
    ) -> None:
        """evaluate returns val_accuracy in [0,1]."""
        from chess_sim.training.pgn_rl_trainer_v4 import (
            PGNRLTrainerV4,
        )
        cfg = _make_cfg()
        trainer = PGNRLTrainerV4(
            cfg, device="cpu", total_steps=100
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2
            )
            ds = ChessRLDataset(
                path,
                val_split_fraction=0.5,
                split="val",
            )
            try:
                metrics = trainer.evaluate(ds)
                acc = metrics["val_accuracy"]
                self.assertGreaterEqual(acc, 0.0)
                self.assertLessEqual(acc, 1.0)
            finally:
                ds.close()

    def test_tc29_synthetic_hdf5_has_loss_mode(
        self,
    ) -> None:
        """_write_synthetic_hdf5 produces loss_mode."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2
            )
            with h5py.File(path, "r") as hf:
                self.assertIn(
                    "loss_mode", hf.keys()
                )
                vals = hf["loss_mode"][:]
                for v in vals:
                    self.assertIn(int(v), (-1, +1))

    def test_tc30_trainer_unpacks_7_tuple(
        self,
    ) -> None:
        """train_epoch unpacks 7-tuple without error."""
        from chess_sim.training.pgn_rl_trainer_v4 import (
            PGNRLTrainerV4,
        )
        cfg = _make_cfg()
        trainer = PGNRLTrainerV4(
            cfg, device="cpu", total_steps=100
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2
            )
            ds = ChessRLDataset(
                path,
                val_split_fraction=0.1,
                split="train",
            )
            try:
                metrics = trainer.train_epoch(ds)
                self.assertIn(
                    "total_loss", metrics
                )
                self.assertIn(
                    "train_accuracy", metrics
                )
                self.assertIn(
                    "n_samples", metrics
                )
            finally:
                ds.close()


if __name__ == "__main__":
    unittest.main()
