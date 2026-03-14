"""Tests for Phase 1 auxiliary heads integration.

T1-T3: AuxiliaryHeads module (shapes, losses, detach)
T4-T5: compute_phase_labels
T6-T8: Trainer integration (train_step with aux, epoch keys, eval)
T9: ChessRLDataset returns src_square field
T10: Missing src_square in HDF5 raises KeyError
T11: use_aux_heads=False: no aux keys in metrics
T12: Preprocessor writes src_square, not capture_map/move_category
T13: Both preprocessor modes omit capture_map and move_category
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
from chess_sim.model.auxiliary_heads import AuxiliaryHeads
from chess_sim.training.pgn_rl_trainer_v4 import (
    PGNRLTrainerV4,
    compute_phase_labels,
)


def _make_cfg(
    use_aux: bool = True, **rl_kw: object
) -> PGNRLConfig:
    """Build a test PGNRLConfig with tiny model dims."""
    defaults: dict[str, object] = {
        "batch_size": 4,
        "num_workers": 0,
        "val_split_fraction": 0.1,
        "hdf5_chunk_size": 64,
        "use_structural_mask": False,
        "warmup_fraction": 0.05,
        "decay_start_fraction": 0.5,
        "epochs": 1,
        "use_aux_heads": use_aux,
    }
    defaults.update(rl_kw)
    return PGNRLConfig(
        model=ModelConfig(
            d_model=32, n_heads=2, n_layers=1,
            dim_feedforward=64, dropout=0.0,
        ),
        decoder=DecoderConfig(
            d_model=32, n_heads=2, n_layers=1,
            dim_feedforward=64, dropout=0.0,
            max_seq_len=64, move_vocab_size=1971,
        ),
        rl=RLConfig(**defaults),
    )


def _write_synthetic_hdf5(
    path: Path,
    n_rows: int = 200,
    n_games: int = 10,
    with_src_square: bool = True,  # False only used by T10 to test KeyError
) -> None:
    """Write a minimal HDF5 matching the v5 schema."""
    with h5py.File(path, "w") as hf:
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

        tm = np.random.randint(
            3, 20, (n_rows,), dtype=np.int32
        )
        hf.create_dataset("target_move", data=tm)

        game_ids = np.repeat(
            np.arange(n_games, dtype=np.int32),
            n_rows // n_games,
        )
        remainder = n_rows - len(game_ids)
        if remainder > 0:
            game_ids = np.concatenate([
                game_ids,
                np.full(
                    remainder, n_games - 1,
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

        legal_mask = np.zeros(
            (n_rows, 1971), dtype=bool
        )
        legal_mask[:, 3:20] = True
        hf.create_dataset("legal_mask", data=legal_mask)

        if with_src_square:
            src_sq = np.random.randint(
                0, 64, (n_rows,), dtype=np.int8
            )
            hf.create_dataset("src_square", data=src_sq)


class TestAuxiliaryHeads(unittest.TestCase):
    """Tests for the AuxiliaryHeads nn.Module."""

    def test_t1_output_shapes(self) -> None:
        """Forward returns three scalar losses."""
        heads = AuxiliaryHeads(d_model=32)
        B = 4
        sq = torch.randn(B, 64, 32)
        cls = torch.randn(B, 32)
        cap_gt = torch.zeros(B, 64)
        cat_gt = torch.zeros(B, dtype=torch.long)
        ph_gt = torch.zeros(B, dtype=torch.long)

        out = heads(sq, cls, cap_gt, cat_gt, ph_gt)
        self.assertEqual(out.capture_loss.shape, ())
        self.assertEqual(out.category_loss.shape, ())
        self.assertEqual(out.phase_loss.shape, ())

    def test_t2_losses_are_finite(self) -> None:
        """All three losses are finite and non-negative."""
        heads = AuxiliaryHeads(d_model=32)
        B = 8
        sq = torch.randn(B, 64, 32)
        cls = torch.randn(B, 32)
        cap_gt = torch.randint(0, 2, (B, 64)).float()
        cat_gt = torch.randint(0, 7, (B,))
        ph_gt = torch.randint(0, 3, (B,))

        out = heads(sq, cls, cap_gt, cat_gt, ph_gt)
        for name in ("capture_loss", "category_loss",
                      "phase_loss"):
            val = getattr(out, name).item()
            self.assertTrue(
                np.isfinite(val),
                f"{name} not finite: {val}",
            )
            self.assertGreaterEqual(val, 0.0)

    def test_t3_cls_detached_no_grad_flow(self) -> None:
        """CLS heads do not propagate grads into cls_emb."""
        heads = AuxiliaryHeads(d_model=16)
        B = 2
        sq = torch.randn(B, 64, 16, requires_grad=True)
        cls = torch.randn(B, 16, requires_grad=True)
        cap_gt = torch.zeros(B, 64)
        cat_gt = torch.zeros(B, dtype=torch.long)
        ph_gt = torch.zeros(B, dtype=torch.long)

        out = heads(sq, cls, cap_gt, cat_gt, ph_gt)
        # Only category + phase depend on cls (detached)
        total = out.category_loss + out.phase_loss
        total.backward()
        # CLS should NOT have gradient (detached)
        self.assertIsNone(cls.grad)
        # Square emb SHOULD have gradient (capture head)
        sq.grad = None
        out2 = heads(sq, cls, cap_gt, cat_gt, ph_gt)
        out2.capture_loss.backward()
        self.assertIsNotNone(sq.grad)


class TestComputePhaseLabels(unittest.TestCase):
    """Tests for compute_phase_labels()."""

    def test_t4_opening_ply_below_20(self) -> None:
        """Ply < 20 -> Opening(0)."""
        B = 4
        ply_idx = torch.tensor([0, 5, 10, 19])
        board = torch.zeros(B, 65, 3)
        for sq in [1, 2]:  # knight
            board[:, sq + 1, 0] = 3
        for sq in [3, 4]:  # bishop
            board[:, sq + 1, 0] = 4
        for sq in [5, 6]:  # rook
            board[:, sq + 1, 0] = 5
        board[:, 8, 0] = 6  # queen

        labels = compute_phase_labels(ply_idx, board)
        for i in range(B):
            self.assertEqual(
                labels[i].item(), 0,
                f"Ply {ply_idx[i]} should be opening",
            )

    def test_t5_endgame_low_material(self) -> None:
        """Low non-pawn material -> Endgame(1)."""
        B = 2
        ply_idx = torch.tensor([30, 50])
        board = torch.zeros(B, 65, 3)
        board[:, 1:9, 0] = 2  # pawns
        board[:, 9, 0] = 7    # king

        labels = compute_phase_labels(ply_idx, board)
        for i in range(B):
            self.assertEqual(
                labels[i].item(), 1,
                "Low material should be endgame",
            )

    def test_t5b_midgame(self) -> None:
        """High material + ply >= 20 -> Midgame(2)."""
        B = 1
        ply_idx = torch.tensor([25])
        board = torch.zeros(B, 65, 3)
        for sq in range(1, 5):
            board[:, sq, 0] = 5  # rook
        board[:, 5, 0] = 6  # queen
        board[:, 6, 0] = 6  # queen

        labels = compute_phase_labels(ply_idx, board)
        self.assertEqual(labels[0].item(), 2)


class TestTrainerAuxIntegration(unittest.TestCase):
    """Tests for PGNRLTrainerV4 with aux heads enabled."""

    def test_t6_train_step_with_aux_returns_aux_keys(
        self,
    ) -> None:
        """_train_step with aux heads returns aux loss keys."""
        cfg = _make_cfg(use_aux=True)
        trainer = PGNRLTrainerV4(
            cfg, device="cpu", total_steps=100
        )
        B = 4
        # piece type: 1=empty, 2=pawn, 3=knight...
        ch0 = torch.randint(1, 8, (B, 65)).float()
        # color: 0=empty, 1=player, 2=opponent
        ch1 = torch.randint(0, 3, (B, 65)).float()
        ch2 = torch.randint(0, 5, (B, 65)).float()
        board = torch.stack([ch0, ch1, ch2], dim=-1)
        targets = torch.randint(3, 20, (B,))
        ct = torch.randint(0, 3, (B, 65), dtype=torch.long)
        legal_mask = torch.zeros(
            B, 1971, dtype=torch.bool
        )
        legal_mask[:, 3:20] = True
        src_square = torch.randint(0, 64, (B,))
        ply_idx = torch.randint(0, 40, (B,))

        result = trainer._train_step(
            board, targets, legal_mask,
            src_square, ply_idx,
        )
        self.assertIn("capture_loss", result)
        self.assertIn("category_loss", result)
        self.assertIn("phase_loss", result)
        for k in ("capture_loss", "category_loss",
                   "phase_loss"):
            self.assertTrue(
                np.isfinite(result[k]),
                f"{k} not finite",
            )

    def test_t7_train_epoch_with_aux(self) -> None:
        """train_epoch with aux heads returns aux metrics."""
        cfg = _make_cfg(use_aux=True)
        trainer = PGNRLTrainerV4(
            cfg, device="cpu", total_steps=100
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2,
            )
            ds = ChessRLDataset(
                path, val_split_fraction=0.1,
                split="train",
            )
            try:
                metrics = trainer.train_epoch(ds)
                self.assertIn("capture_loss", metrics)
                self.assertIn("category_loss", metrics)
                self.assertIn("phase_loss", metrics)
            finally:
                ds.close()

    def test_t8_evaluate_with_aux(self) -> None:
        """evaluate with aux heads returns aux metrics."""
        cfg = _make_cfg(use_aux=True)
        trainer = PGNRLTrainerV4(
            cfg, device="cpu", total_steps=100
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2,
            )
            ds = ChessRLDataset(
                path, val_split_fraction=0.5,
                split="val",
            )
            try:
                metrics = trainer.evaluate(ds)
                self.assertIn("val_loss", metrics)
                self.assertIn("capture_loss", metrics)
                self.assertIn("category_loss", metrics)
                self.assertIn("phase_loss", metrics)
            finally:
                ds.close()

    def test_t9_dataset_returns_src_square(self) -> None:
        """ChessRLDataset returns src_square in 7-tuple."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2,
                with_src_square=True,
            )
            ds = ChessRLDataset(
                path, val_split_fraction=0.1,
                split="train",
            )
            sample = ds[0]
            self.assertEqual(len(sample), 7)
            # src_square is element 5 in the 7-tuple
            src_sq = sample[5]
            self.assertIsInstance(src_sq, int)
            self.assertGreaterEqual(src_sq, 0)
            self.assertLess(src_sq, 64)
            ds.close()

    def test_t10_dataset_without_src_square_raises(
        self,
    ) -> None:
        """HDF5 missing src_square raises KeyError at load time."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2,
                with_src_square=False,
            )
            with self.assertRaises(KeyError):
                ChessRLDataset(
                    path, val_split_fraction=0.1,
                    split="train",
                )

    def test_t11_no_aux_heads_backward_compat(
        self,
    ) -> None:
        """use_aux_heads=False: no aux keys in metrics."""
        cfg = _make_cfg(use_aux=False)
        trainer = PGNRLTrainerV4(
            cfg, device="cpu", total_steps=100
        )
        self.assertIsNone(trainer.aux_heads)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.h5"
            _write_synthetic_hdf5(
                path, n_rows=20, n_games=2,
            )
            ds = ChessRLDataset(
                path, val_split_fraction=0.1,
                split="train",
            )
            try:
                metrics = trainer.train_epoch(ds)
                self.assertNotIn(
                    "capture_loss", metrics
                )
                self.assertNotIn(
                    "category_loss", metrics
                )
                self.assertNotIn(
                    "phase_loss", metrics
                )
            finally:
                ds.close()

    def test_t12_preprocessor_writes_src_square(
        self,
    ) -> None:
        """PGNRewardPreprocessor writes src_square, not aux."""
        from chess_sim.data.pgn_reward_preprocessor import (  # noqa: E501
            PGNRewardPreprocessor,
        )

        cfg = _make_cfg(use_aux=True)
        pp = PGNRewardPreprocessor(cfg)
        with tempfile.TemporaryDirectory() as td:
            pgn_path = Path(td) / "games.pgn"
            pgn_path.write_text(
                '[Event "T"]\n[Result "1-0"]\n\n'
                "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 "
                "4. Qxf7# 1-0\n\n"
            )
            hdf5_path = Path(td) / "out.h5"
            pp.generate(
                pgn_path, hdf5_path, max_games=1
            )

            with h5py.File(hdf5_path, "r") as hf:
                self.assertIn("src_square", hf)
                self.assertNotIn("capture_map", hf)
                self.assertNotIn("move_category", hf)
                n = hf["board"].shape[0]
                self.assertEqual(
                    hf["src_square"].shape, (n,)
                )
                src_vals = hf["src_square"][:]
                self.assertTrue(
                    np.all(src_vals >= 0)
                )
                self.assertTrue(
                    np.all(src_vals < 64)
                )

    def test_t13_preprocessor_no_aux_in_either_mode(
        self,
    ) -> None:
        """Both use_aux=True/False omit capture_map and move_category."""
        from chess_sim.data.pgn_reward_preprocessor import (  # noqa: E501
            PGNRewardPreprocessor,
        )

        for use_aux in (True, False):
            cfg = _make_cfg(use_aux=use_aux)
            pp = PGNRewardPreprocessor(cfg)
            with tempfile.TemporaryDirectory() as td:
                pgn_path = Path(td) / "games.pgn"
                pgn_path.write_text(
                    '[Event "T"]\n[Result "1-0"]\n\n'
                    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 "
                    "4. Qxf7# 1-0\n\n"
                )
                hdf5_path = Path(td) / "out.h5"
                pp.generate(
                    pgn_path, hdf5_path, max_games=1
                )

                with h5py.File(hdf5_path, "r") as hf:
                    self.assertNotIn(
                        "capture_map", hf,
                        f"capture_map present when "
                        f"use_aux={use_aux}",
                    )
                    self.assertNotIn(
                        "move_category", hf,
                        f"move_category present when "
                        f"use_aux={use_aux}",
                    )


if __name__ == "__main__":
    unittest.main()
