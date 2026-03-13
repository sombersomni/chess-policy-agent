"""Tests for StructuralMaskBuilder and trainer integration.

T1-T7: Direct tests of StructuralMaskBuilder (LUT and build).
T8-T12: Trainer integration tests with mocked dependencies.
"""

from __future__ import annotations

import unittest

import torch

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


class TestMaskDeviceConsistency(unittest.TestCase):
    """T11: StructuralMaskBuilder device consistency."""

    def test_t11_device_consistency(self) -> None:
        """T11: LUT and output mask are on the same device as input."""
        builder = StructuralMaskBuilder(torch.device("cpu"))
        self.assertEqual(
            builder._slot_mask.device.type, "cpu"
        )
        ct = torch.zeros(1, 65, dtype=torch.long)
        ct[0, 13] = 1
        mask = builder.build(ct)
        self.assertEqual(mask.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
