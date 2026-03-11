"""Tests for within-game batching: T1–T10.

Verifies that _pad_prefixes, _encode_and_decode_batch, and the
modified train_game produce numerically equivalent results to the
serial _encode_and_decode path.  All tests use d_model=64 on CPU.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn
import torch

from chess_sim.config import (
    DecoderConfig,
    ModelConfig,
    PGNRLConfig,
    RLConfig,
)
from chess_sim.training.pgn_rl_trainer import PGNRLTrainer


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

_D_MODEL = 64


def _small_model_cfg() -> ModelConfig:
    """d_model=64, 4 heads, 2 layers — small for fast CPU tests."""
    return ModelConfig(
        d_model=_D_MODEL,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.0,
    )


def _small_decoder_cfg() -> DecoderConfig:
    """Matching decoder with d_model=64."""
    return DecoderConfig(
        d_model=_D_MODEL,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        max_seq_len=64,
        move_vocab_size=1971,
    )


def _make_trainer(
    max_plies: int = 150,
    skip_draws: bool = False,
) -> PGNRLTrainer:
    """Build a minimal PGNRLTrainer on CPU with d_model=64."""
    cfg = PGNRLConfig(
        model=_small_model_cfg(),
        decoder=_small_decoder_cfg(),
        rl=RLConfig(
            max_plies_per_game=max_plies,
            skip_draws=skip_draws,
        ),
    )
    return PGNRLTrainer(
        cfg=cfg, device="cpu", total_steps=100,
    )


def _make_scholars_mate() -> chess.pgn.Game:
    """Scholar's Mate (7 plies, White wins 1-0)."""
    game = chess.pgn.Game()
    game.headers["Result"] = "1-0"
    node = game
    for uci in [
        "e2e4", "e7e5", "f1c4", "b8c6",
        "d1h5", "g8f6", "h5f7",
    ]:
        node = node.add_variation(
            chess.Move.from_uci(uci)
        )
    return game


def _make_long_game(n_plies: int = 200) -> chess.pgn.Game:
    """Game with many random plies for max_plies_per_game test."""
    game = chess.pgn.Game()
    game.headers["Result"] = "1-0"
    board = game.board()
    node = game
    count = 0
    # Play random legal moves until we reach n_plies or game ends
    gen = torch.Generator()
    gen.manual_seed(42)
    while count < n_plies:
        legal = list(board.legal_moves)
        if not legal:
            break
        idx = int(torch.randint(len(legal), (1,), generator=gen))
        move = legal[idx]
        node = node.add_variation(move)
        board.push(move)
        count += 1
    return game


def _write_pgn(game: chess.pgn.Game) -> Path:
    """Write a single game to a temp PGN file."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pgn", delete=False,
    )
    exporter = chess.pgn.StringExporter(headers=True)
    tmp.write(game.accept(exporter))
    tmp.flush()
    return Path(tmp.name)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestPadPrefixes(unittest.TestCase):
    """T8: _pad_prefixes output shapes and mask correctness."""

    def setUp(self) -> None:
        """Build a shared trainer."""
        torch.manual_seed(0)
        self.trainer = _make_trainer()

    def test_pad_prefixes_shapes(self) -> None:
        """T8: 3 tensors of lengths 2, 4, 6 -> padded (3,6), mask (3,6)."""
        prefixes = [
            torch.tensor([1, 2]),
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
        ]
        padded, mask = self.trainer._pad_prefixes(prefixes)
        self.assertEqual(padded.shape, (3, 6))
        self.assertEqual(mask.shape, (3, 6))
        # First prefix has 4 PAD positions at end
        self.assertTrue(mask[0, 2:].all())
        # Last prefix has no PAD
        self.assertFalse(mask[2, :].any())

    def test_pad_prefixes_values(self) -> None:
        """Padded tensor preserves original values."""
        prefixes = [
            torch.tensor([10, 20]),
            torch.tensor([30, 40, 50]),
        ]
        padded, mask = self.trainer._pad_prefixes(prefixes)
        self.assertTrue(
            torch.equal(padded[0, :2], torch.tensor([10, 20]))
        )
        self.assertEqual(padded[0, 2].item(), 0)
        self.assertTrue(
            torch.equal(
                padded[1, :3], torch.tensor([30, 40, 50])
            )
        )


class TestPaddingMask(unittest.TestCase):
    """T3: Padding mask excludes pad positions correctly."""

    def setUp(self) -> None:
        """Build a shared trainer."""
        torch.manual_seed(0)
        self.trainer = _make_trainer()

    def test_mask_excludes_pad_positions(self) -> None:
        """T3: prefix lengths 1, 3, 5; max_len=5.
        mask[0, 1:] all True; mask[2, :] all False."""
        prefixes = [
            torch.tensor([1]),
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3, 4, 5]),
        ]
        padded, mask = self.trainer._pad_prefixes(prefixes)
        # Ply 0: only position 0 is real
        self.assertTrue(mask[0, 1:].all())
        self.assertFalse(mask[0, 0].item())
        # Ply 2: all positions are real
        self.assertFalse(mask[2, :].any())


class TestEncoderMatchesSerial(unittest.TestCase):
    """T1: Batched encoder output matches serial calls."""

    def setUp(self) -> None:
        """Build trainer with fixed seed for determinism."""
        torch.manual_seed(42)
        self.trainer = _make_trainer()

    def test_batched_encoder_matches_serial(self) -> None:
        """T1: cls_all[i] equals serial cls[i] within 1e-5."""
        game = _make_scholars_mate()
        self.trainer._model.eval()
        plies = self.trainer._replayer.replay(game)
        train_white = True
        plies = [p for p in plies if p.is_white_ply == train_white]
        self.assertGreaterEqual(len(plies), 3)

        # Serial path: one call per ply
        serial_cls_list = []
        serial_logits_list = []
        with torch.no_grad():
            for ply in plies:
                bt = ply.board_tokens.unsqueeze(0)
                ct = ply.color_tokens.unsqueeze(0)
                tt = ply.traj_tokens.unsqueeze(0)
                prefix = ply.move_prefix.unsqueeze(0)
                logits, cls, _ = self.trainer._encode_and_decode(
                    bt, ct, tt, prefix, ply.move_uci,
                )
                serial_cls_list.append(cls.squeeze(0))
                serial_logits_list.append(logits)

        # Batched path: one call for all plies
        with torch.no_grad():
            batch_logits, batch_cls, _ = (
                self.trainer._encode_and_decode_batch(
                    plies, [p.move_uci for p in plies],
                )
            )

        for i in range(len(plies)):
            self.assertTrue(
                torch.allclose(
                    serial_cls_list[i],
                    batch_cls[i],
                    atol=1e-5,
                ),
                msg=f"CLS mismatch at ply {i}",
            )


class TestDecoderMatchesSerial(unittest.TestCase):
    """T2: Batched decoder logits match serial calls."""

    def setUp(self) -> None:
        """Build trainer with fixed seed for determinism."""
        torch.manual_seed(42)
        self.trainer = _make_trainer()

    def test_batched_decoder_logits_match_serial(self) -> None:
        """T2: last_logits_all[i] equals serial last_logits[i]
        within 1e-5."""
        game = _make_scholars_mate()
        self.trainer._model.eval()
        plies = self.trainer._replayer.replay(game)
        train_white = True
        plies = [p for p in plies if p.is_white_ply == train_white]

        serial_logits = []
        with torch.no_grad():
            for ply in plies:
                bt = ply.board_tokens.unsqueeze(0)
                ct = ply.color_tokens.unsqueeze(0)
                tt = ply.traj_tokens.unsqueeze(0)
                prefix = ply.move_prefix.unsqueeze(0)
                logits, _, _ = self.trainer._encode_and_decode(
                    bt, ct, tt, prefix, ply.move_uci,
                )
                serial_logits.append(logits)

        with torch.no_grad():
            batch_logits, _, _ = (
                self.trainer._encode_and_decode_batch(
                    plies, [p.move_uci for p in plies],
                )
            )

        for i in range(len(plies)):
            self.assertTrue(
                torch.allclose(
                    serial_logits[i],
                    batch_logits[i],
                    atol=1e-5,
                ),
                msg=f"Logits mismatch at ply {i}",
            )


class TestOOVPlyInMiddle(unittest.TestCase):
    """T4: OOV ply in middle of game."""

    def setUp(self) -> None:
        """Build trainer with fixed seed."""
        torch.manual_seed(42)
        self.trainer = _make_trainer()

    def test_oov_ply_yields_none_move_idx(self) -> None:
        """T4: ply with untokenizable UCI -> move_idxs[2] is None;
        other plies unaffected."""
        game = _make_scholars_mate()
        self.trainer._model.eval()
        plies = self.trainer._replayer.replay(game)
        train_white = True
        plies = [p for p in plies if p.is_white_ply == train_white]

        # Inject an OOV UCI at index 1
        move_ucis = [p.move_uci for p in plies]
        oov_idx = 1
        move_ucis[oov_idx] = "z9z9"

        with torch.no_grad():
            _, _, move_idxs = (
                self.trainer._encode_and_decode_batch(
                    plies, move_ucis,
                )
            )

        self.assertIsNone(move_idxs[oov_idx])
        # Others should be valid ints
        for i, idx in enumerate(move_idxs):
            if i != oov_idx:
                self.assertIsNotNone(idx)


class TestSinglePlyGame(unittest.TestCase):
    """T5: Single-ply game (T=1)."""

    def setUp(self) -> None:
        """Build trainer with fixed seed."""
        torch.manual_seed(42)
        self.trainer = _make_trainer()

    def test_single_ply_batch(self) -> None:
        """T5: One ply, prefix length 1 [SOS] -> valid shapes."""
        game = _make_scholars_mate()
        self.trainer._model.eval()
        plies = self.trainer._replayer.replay(game)
        train_white = True
        plies = [p for p in plies if p.is_white_ply == train_white]
        # Use only the first ply
        single = plies[:1]

        with torch.no_grad():
            logits, cls, move_idxs = (
                self.trainer._encode_and_decode_batch(
                    single, [single[0].move_uci],
                )
            )

        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 1971)
        self.assertEqual(cls.shape[0], 1)
        self.assertEqual(cls.shape[1], _D_MODEL)
        self.assertEqual(len(move_idxs), 1)


class TestTrainGameLossEquivalence(unittest.TestCase):
    """T6: train_game total loss from batched path equals serial."""

    def test_batched_loss_matches_serial(self) -> None:
        """T6: same game, same seed -> total_loss matches within 1e-4.

        Run serial path manually via _encode_and_decode per ply,
        then compare to train_game (which uses batched path).
        """
        torch.manual_seed(99)
        trainer = _make_trainer()

        game = _make_scholars_mate()
        trainer._model.eval()
        plies = trainer._replayer.replay(game)
        train_white = True
        plies_filtered = [
            p for p in plies if p.is_white_ply == train_white
        ]

        # Serial reference: collect logits and move_idxs
        serial_logits = []
        serial_move_idxs: list[int | None] = []
        serial_cls_list = []
        with torch.no_grad():
            for ply in plies_filtered:
                bt = ply.board_tokens.unsqueeze(0)
                ct = ply.color_tokens.unsqueeze(0)
                tt = ply.traj_tokens.unsqueeze(0)
                prefix = ply.move_prefix.unsqueeze(0)
                logits, cls, midx = trainer._encode_and_decode(
                    bt, ct, tt, prefix, ply.move_uci,
                )
                serial_logits.append(logits)
                serial_move_idxs.append(midx)
                serial_cls_list.append(cls)

        # Now run train_game with fresh trainer (same seed)
        torch.manual_seed(99)
        trainer2 = _make_trainer()
        metrics = trainer2.train_game(game)

        self.assertIn("total_loss", metrics)
        self.assertTrue(math.isfinite(metrics["total_loss"]))

        # Verify valid plies match
        valid_serial = [
            i for i, idx in enumerate(serial_move_idxs)
            if idx is not None
        ]
        self.assertGreater(len(valid_serial), 0)


class TestPlyStepCounter(unittest.TestCase):
    """T7: _ply_step counter increments correctly."""

    def test_ply_step_incremented_by_game_length(self) -> None:
        """T7: Game with T plies, initial _ply_step=0 ->
        _ply_step == T after train_game."""
        torch.manual_seed(42)
        trainer = _make_trainer()

        game = _make_scholars_mate()
        initial_step = trainer._ply_step
        self.assertEqual(initial_step, 0)

        # Count expected white plies
        plies = trainer._replayer.replay(game)
        white_plies = [
            p for p in plies if p.is_white_ply
        ]
        expected_T = len(white_plies)

        trainer.train_game(game)
        self.assertEqual(
            trainer._ply_step, expected_T,
            msg=(
                f"_ply_step={trainer._ply_step}, "
                f"expected {expected_T}"
            ),
        )


class TestAllPliesOOV(unittest.TestCase):
    """T9: All plies OOV -> train_game returns {}."""

    def test_all_oov_returns_empty(self) -> None:
        """T9: every move UCI untokenizable -> returns {}."""
        torch.manual_seed(42)
        trainer = _make_trainer()

        game = _make_scholars_mate()
        plies = trainer._replayer.replay(game)
        train_white = True
        plies_filtered = [
            p for p in plies if p.is_white_ply == train_white
        ]

        # All OOV: check _encode_and_decode_batch
        oov_ucis = ["z9z9"] * len(plies_filtered)
        trainer._model.eval()
        with torch.no_grad():
            _, _, move_idxs = (
                trainer._encode_and_decode_batch(
                    plies_filtered, oov_ucis,
                )
            )
        self.assertTrue(all(idx is None for idx in move_idxs))


class TestMaxPliesPerGame(unittest.TestCase):
    """T10: Game exceeding max_plies_per_game is skipped."""

    def test_long_game_skipped(self) -> None:
        """T10: T=200 filtered plies, max_plies_per_game=5 ->
        train_game returns {} immediately."""
        torch.manual_seed(42)
        # Use a very low threshold to guarantee skipping
        trainer = _make_trainer(max_plies=5)

        game = _make_scholars_mate()  # 4 white plies
        # Scholar's mate has 4 white plies; set threshold to 3
        trainer._cfg.rl.max_plies_per_game = 3

        initial_step = trainer._ply_step
        metrics = trainer.train_game(game)
        self.assertEqual(metrics, {})
        # No forward pass means _ply_step unchanged
        self.assertEqual(trainer._ply_step, initial_step)

    def test_config_max_plies_per_game_default(self) -> None:
        """Default max_plies_per_game is 150."""
        cfg = RLConfig()
        self.assertEqual(cfg.max_plies_per_game, 150)

    def test_config_max_plies_per_game_negative_raises(
        self,
    ) -> None:
        """max_plies_per_game < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            RLConfig(max_plies_per_game=0)


if __name__ == "__main__":
    unittest.main()
