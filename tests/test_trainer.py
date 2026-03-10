"""Tests for Trainer: T08, T19."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from chess_sim.training.trainer import Trainer
from chess_sim.types import EncoderOutput
from tests.utils import make_synthetic_batch


class TestTrainerLossDecreases(unittest.TestCase):
    """T08: Loss must decrease after one gradient update."""

    def setUp(self) -> None:
        self.trainer = Trainer(device="cpu")
        torch.manual_seed(0)

    def test_loss_decreases_after_one_step(self) -> None:
        """T08: second forward pass after one train_step yields a lower loss."""
        batch = make_synthetic_batch(batch_size=4, device="cpu")
        loss_before = self.trainer.train_step(batch)
        loss_after = self.trainer.train_step(batch)
        self.assertLess(loss_after, loss_before,
                        msg="Loss did not decrease after one gradient step")


class TestTrainerCheckpoint(unittest.TestCase):
    """T19: Checkpoint save/load round-trip preserves model outputs."""

    def setUp(self) -> None:
        self.trainer = Trainer(device="cpu")
        torch.manual_seed(42)

    def test_checkpoint_roundtrip_preserves_outputs(self) -> None:
        """T19: Outputs are identical before/after checkpoint reload."""
        bt = torch.randint(0, 8, (2, 65))
        ct = torch.randint(0, 3, (2, 65))
        at = torch.zeros(2, 65, dtype=torch.long)

        self.trainer.encoder.eval()
        with torch.no_grad():
            out_before: EncoderOutput = (
                self.trainer.encoder.encode(bt, ct, at)
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            self.trainer.save_checkpoint(ckpt_path)
            fresh_trainer = Trainer(device="cpu")
            fresh_trainer.load_checkpoint(ckpt_path)

        fresh_trainer.encoder.eval()
        with torch.no_grad():
            out_after: EncoderOutput = (
                fresh_trainer.encoder.encode(bt, ct, at)
            )

        self.assertTrue(
            torch.allclose(
                out_before.cls_embedding,
                out_after.cls_embedding,
            ),
            msg="cls_embedding differs after reload",
        )
        self.assertTrue(
            torch.allclose(
                out_before.square_embeddings,
                out_after.square_embeddings,
            ),
            msg="square_embeddings differ after reload",
        )


if __name__ == "__main__":
    unittest.main()
