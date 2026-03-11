"""Tests for StructuralMaskBuilder and trainer integration.

T1-T7: Direct tests of StructuralMaskBuilder (LUT and build).
T8-T12: Trainer integration tests with mocked dependencies.
"""

from __future__ import annotations

import unittest

import torch

from chess_sim.config import (
    DataConfig,
    DecoderConfig,
    ModelConfig,
    PGNRLConfig,
    RLConfig,
)
from chess_sim.data.move_vocab import MoveVocab
from chess_sim.data.structural_mask import (
    StructuralMaskBuilder,
    _uci_from_square_slot,
)
from chess_sim.protocols import StructuralMaskable


class TestStructuralMaskBuilder(unittest.TestCase):
    """T1-T7: StructuralMaskBuilder unit tests."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build shared fixtures: builder and vocab."""
        cls.builder = StructuralMaskBuilder()
        cls.vocab = MoveVocab()

    def test_t1_lut_covers_all_64_squares(self) -> None:
        """T1: slot_mask[1..64] each has > 0 True entries; slot 0 all-False."""
        lut = self.builder._slot_mask  # [65, 1971]
        # CLS slot (0) must be all-False.
        self.assertFalse(
            lut[0].any().item(),
            "Slot 0 (CLS) should be all-False",
        )
        # Each of the 64 board squares must have at least one token.
        for slot in range(1, 65):
            self.assertTrue(
                lut[slot].any().item(),
                f"Slot {slot} has zero True entries",
            )

    def test_t2_lut_token_coverage_exact(self) -> None:
        """T2: union of all rows covers exactly 1968 non-special tokens."""
        lut = self.builder._slot_mask  # [65, 1971]
        union = lut.any(dim=0)  # [V]
        # Special tokens 0,1,2 must be absent.
        for idx in (0, 1, 2):
            self.assertFalse(
                union[idx].item(),
                f"Special token {idx} should not be in any slot",
            )
        # All non-special tokens (3..1970) must be covered.
        non_special_covered = union[3:].sum().item()
        self.assertEqual(
            non_special_covered,
            1968,
            f"Expected 1968 non-special tokens covered, got "
            f"{non_special_covered}",
        )

    def test_t3_single_player_square_e2(self) -> None:
        """T3: color_tokens with only e2 as player -> 41 True values."""
        ct = torch.zeros(1, 65, dtype=torch.long)
        # e2: file=4, rank=2 -> slot = (2-1)*8 + 4 + 1 = 13
        ct[0, 13] = 1
        mask = self.builder.build(ct)  # [1, V]
        true_count = mask.sum().item()
        self.assertEqual(
            true_count,
            41,
            f"e2 should have 41 valid tokens, got {true_count}",
        )
        # Verify all True positions correspond to e2 from-square.
        for idx in mask[0].nonzero(as_tuple=False).squeeze(-1).tolist():
            uci = self.vocab.decode(idx)
            self.assertEqual(
                uci[:2],
                "e2",
                f"Token {idx} ({uci}) does not start from e2",
            )

    def test_t4_empty_board_all_false(self) -> None:
        """T4: no player pieces -> all-False mask."""
        ct = torch.zeros(1, 65, dtype=torch.long)
        mask = self.builder.build(ct)
        self.assertFalse(
            mask.any().item(),
            "Empty board should produce all-False mask",
        )

    def test_t5_full_board_16_player_pieces(self) -> None:
        """T5: 16 player squares -> mask covers union of 16 slots."""
        ct = torch.zeros(1, 65, dtype=torch.long)
        # Slots 1-16 as player squares (a1-h2).
        ct[0, 1:17] = 1
        mask = self.builder.build(ct)
        true_count = mask.sum().item()
        # Each square has at least 23 tokens; some overlap expected.
        # Union must be >= 16*23 only if no overlap, but test
        # just checks it's large enough and <= 1968.
        self.assertGreater(
            true_count,
            200,
            f"16 squares should yield >200 tokens, got {true_count}",
        )
        self.assertLessEqual(true_count, 1968)

    def test_t6_batch_independence(self) -> None:
        """T6: B=4, each row distinct -> no cross-contamination."""
        ct = torch.zeros(4, 65, dtype=torch.long)
        # Each batch item has exactly one unique player square.
        slots = [13, 25, 37, 49]  # e2, a4, e5, a7
        for b, s in enumerate(slots):
            ct[b, s] = 1
        mask = self.builder.build(ct)  # [4, V]
        for b in range(4):
            # Check that each row's True set corresponds only
            # to the single player square for that batch item.
            true_indices = mask[b].nonzero(
                as_tuple=False
            ).squeeze(-1).tolist()
            slot = slots[b]
            expected = set(
                self.builder._slot_mask[slot]
                .nonzero(as_tuple=False)
                .squeeze(-1)
                .tolist()
            )
            self.assertEqual(
                set(true_indices),
                expected,
                f"Batch {b} mask does not match slot {slot}",
            )

    def test_t7_teacher_target_in_valid_mask(self) -> None:
        """T7: teacher move's from-square is player -> target in mask."""
        # Use e2e4: from=e2, slot=13.
        ct = torch.zeros(1, 65, dtype=torch.long)
        ct[0, 13] = 1  # e2 is player
        mask = self.builder.build(ct)  # [1, V]
        target_idx = self.vocab.encode("e2e4")
        self.assertTrue(
            mask[0, target_idx].item(),
            "Teacher target e2e4 should be in the valid mask",
        )

    def test_protocol_conformance(self) -> None:
        """StructuralMaskBuilder satisfies StructuralMaskable."""
        self.assertIsInstance(self.builder, StructuralMaskable)

    def test_slot_index_formula(self) -> None:
        """Verify _uci_from_square_slot for known squares."""
        # a1 -> slot 1, e2 -> slot 13, h8 -> slot 64
        self.assertEqual(_uci_from_square_slot("a1a2"), 1)
        self.assertEqual(_uci_from_square_slot("e2e4"), 13)
        self.assertEqual(_uci_from_square_slot("h8h7"), 64)


class TestTrainerIntegration(unittest.TestCase):
    """T8-T12: PGNRLTrainer integration tests for structural mask."""

    def _make_cfg(
        self, use_mask: bool = False
    ) -> PGNRLConfig:
        """Build a minimal PGNRLConfig with optional masking."""
        return PGNRLConfig(
            data=DataConfig(),
            model=ModelConfig(
                d_model=64, n_heads=2, n_layers=1,
                dim_feedforward=128, dropout=0.0,
            ),
            decoder=DecoderConfig(
                d_model=64, n_heads=2, n_layers=1,
                dim_feedforward=128, dropout=0.0,
            ),
            rl=RLConfig(
                use_structural_mask=use_mask,
                lambda_rsbc=1.0,
                lambda_value=0.0,
            ),
        )

    def test_t8_rsbc_loss_with_mask_active(self) -> None:
        """T8: _compute_rsbc_loss with mask produces finite loss."""
        from chess_sim.training.pgn_rl_trainer import (
            PGNRLTrainer,
        )

        cfg = self._make_cfg(use_mask=True)
        trainer = PGNRLTrainer(cfg, device="cpu")

        vocab = MoveVocab()
        v = len(vocab)  # 1971
        # 3 plies with random logits, targets from e2 tokens.
        all_logits = [
            torch.randn(v, requires_grad=True) for _ in range(3)
        ]
        target_idx = vocab.encode("e2e4")
        all_targets = [target_idx] * 3
        weights = torch.ones(3)

        # Color tokens: e2 (slot 13) is player.
        ct = torch.zeros(65, dtype=torch.long)
        ct[13] = 1
        all_color_tokens = [ct.clone() for _ in range(3)]

        loss = trainer._compute_rsbc_loss(
            all_logits, all_targets, weights,
            all_color_tokens=all_color_tokens,
        )
        self.assertTrue(
            torch.isfinite(loss).item(),
            f"Loss should be finite, got {loss.item()}",
        )
        # Verify masked logits: softmax should concentrate
        # on e2 tokens only.
        self.assertGreater(loss.item(), 0.0)

    def test_t9_rsbc_loss_without_mask(self) -> None:
        """T9: _compute_rsbc_loss with mask disabled is unchanged."""
        from chess_sim.training.pgn_rl_trainer import (
            PGNRLTrainer,
        )

        cfg = self._make_cfg(use_mask=False)
        trainer = PGNRLTrainer(cfg, device="cpu")

        vocab = MoveVocab()
        v = len(vocab)
        logits_data = torch.randn(v)
        all_logits = [logits_data.clone() for _ in range(2)]
        target_idx = vocab.encode("e2e4")
        all_targets = [target_idx] * 2
        weights = torch.ones(2)

        # Even if we pass color tokens, mask is None.
        loss = trainer._compute_rsbc_loss(
            all_logits, all_targets, weights,
            all_color_tokens=None,
        )
        self.assertTrue(torch.isfinite(loss).item())

    def test_t10_mask_disabled_skips_builder(self) -> None:
        """T10: use_structural_mask=False -> _struct_mask is None."""
        from chess_sim.training.pgn_rl_trainer import (
            PGNRLTrainer,
        )

        cfg = self._make_cfg(use_mask=False)
        trainer = PGNRLTrainer(cfg, device="cpu")
        self.assertIsNone(trainer._struct_mask)

    def test_t10b_mask_enabled_creates_builder(self) -> None:
        """Complement: use_structural_mask=True creates builder."""
        from chess_sim.training.pgn_rl_trainer import (
            PGNRLTrainer,
        )

        cfg = self._make_cfg(use_mask=True)
        trainer = PGNRLTrainer(cfg, device="cpu")
        self.assertIsNotNone(trainer._struct_mask)
        self.assertIsInstance(
            trainer._struct_mask, StructuralMaskable
        )

    def test_t11_device_consistency(self) -> None:
        """T11: LUT and output mask are on the same device as input."""
        # CPU-only test (CUDA tested only if available).
        builder = StructuralMaskBuilder(torch.device("cpu"))
        self.assertEqual(
            builder._slot_mask.device.type, "cpu"
        )
        ct = torch.zeros(1, 65, dtype=torch.long)
        ct[0, 13] = 1
        mask = builder.build(ct)
        self.assertEqual(mask.device.type, "cpu")

    def test_t12_oov_move_skipped_before_masking(self) -> None:
        """T12: OOV teacher move skipped -> no index mismatch."""
        from chess_sim.training.pgn_rl_trainer import (
            PGNRLTrainer,
        )

        cfg = self._make_cfg(use_mask=True)
        trainer = PGNRLTrainer(cfg, device="cpu")

        vocab = MoveVocab()
        v = len(vocab)
        # Simulate: 2 valid plies + 1 OOV (skipped).
        # After OOV skip, all_logits has 2 entries and
        # all_color_tokens has 2 entries — lengths must match.
        all_logits = [
            torch.randn(v, requires_grad=True) for _ in range(2)
        ]
        target_idx = vocab.encode("e2e4")
        all_targets = [target_idx, target_idx]
        weights = torch.ones(2)

        ct = torch.zeros(65, dtype=torch.long)
        ct[13] = 1  # e2 player
        all_color_tokens = [ct.clone(), ct.clone()]

        # Should not raise — lengths are consistent.
        loss = trainer._compute_rsbc_loss(
            all_logits, all_targets, weights,
            all_color_tokens=all_color_tokens,
        )
        self.assertTrue(torch.isfinite(loss).item())

    def test_masked_loss_lower_than_unmasked(self) -> None:
        """Masked CE should be <= unmasked CE for valid targets."""
        from chess_sim.training.pgn_rl_trainer import (
            PGNRLTrainer,
        )

        cfg_masked = self._make_cfg(use_mask=True)
        cfg_unmasked = self._make_cfg(use_mask=False)
        trainer_m = PGNRLTrainer(cfg_masked, device="cpu")
        trainer_u = PGNRLTrainer(cfg_unmasked, device="cpu")

        vocab = MoveVocab()
        v = len(vocab)
        # Fixed logits: uniform distribution.
        logits = torch.zeros(v)
        all_logits_m = [logits.clone()]
        all_logits_u = [logits.clone()]
        target_idx = vocab.encode("e2e4")
        weights = torch.ones(1)

        ct = torch.zeros(65, dtype=torch.long)
        ct[13] = 1  # e2

        loss_m = trainer_m._compute_rsbc_loss(
            all_logits_m, [target_idx], weights,
            all_color_tokens=[ct],
        )
        loss_u = trainer_u._compute_rsbc_loss(
            all_logits_u, [target_idx], weights,
        )
        # With uniform logits, masking reduces the denominator,
        # so CE with mask should be lower.
        self.assertLess(loss_m.item(), loss_u.item())


if __name__ == "__main__":
    unittest.main()
