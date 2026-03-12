"""Test suite for ChessModel v2 encoder-decoder skeleton.

All tests verify structural contracts (shapes, field names, config defaults,
roundtrip invariants). Every test uses CPU only. Stubs that raise
NotImplementedError are expected — tests catch and re-raise with self.fail
where appropriate to signal "not yet implemented" vs. "broken contract".

Test IDs:
  TV01 - MoveVocab size
  TV02 - MoveVocab encode/decode roundtrip
  TV03 - MoveVocab special tokens
  TV04 - MoveVocab promotion included
  TV05 - MoveTokenizer.tokenize_game shape
  TV06 - MoveTokenizer.tokenize_game SOS/EOS
  TV07 - MoveTokenizer.build_legal_mask shape
  TV08 - MoveTokenizer.build_legal_mask values
  TV09 - MoveEmbedding output shape
  TV10 - MoveDecoder._causal_mask shape
  TV11 - MoveDecoder._causal_mask upper-triangular
  TV12 - MoveDecoder.forward output shape
  TV13 - ChessModel.forward output shape
  TV14 - GameTurnSample fields
  TV15 - DecoderConfig defaults
  TV16 - Phase2Config defaults
  TV17 - load_v2_config roundtrip
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from parameterized import parameterized

from chess_sim.config import (
    ChessModelV2Config,
    DecoderConfig,
    Phase2Config,
    load_v2_config,
)
from chess_sim.data.move_tokenizer import MoveTokenizer
from chess_sim.data.move_vocab import EOS_IDX, PAD_IDX, SOS_IDX, MoveVocab
from chess_sim.model.chess_model import ChessModel
from chess_sim.model.decoder import MoveDecoder
from chess_sim.model.move_embedding import MoveEmbedding
from chess_sim.types import (
    DecoderOutput,
    GameTurnBatch,
    GameTurnSample,
    SelfPlayGame,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE: str = "cpu"
D_MODEL: int = 256
MOVE_VOCAB_SIZE: int = 1971


def _make_board_tokens(batch_size: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random board, color, trajectory tokens for testing."""
    bt = torch.randint(0, 8, (batch_size, 65), dtype=torch.long)
    ct = torch.randint(0, 3, (batch_size, 65), dtype=torch.long)
    tt = torch.zeros(batch_size, 65, dtype=torch.long)
    return bt, ct, tt


# ===========================================================================
# TV01-TV04: MoveVocab
# ===========================================================================


class TestMoveVocab(unittest.TestCase):
    """Tests for MoveVocab construction and lookup."""

    def test_tv01_vocab_size(self) -> None:
        """TV01: Vocab size should be ~1971 (3 special + ~1968 moves)."""
        try:
            vocab = MoveVocab()
        except NotImplementedError:
            self.fail("TV01: MoveVocab.__init__ not implemented")
        self.assertEqual(len(vocab), MOVE_VOCAB_SIZE)

    @parameterized.expand([
        ("standard_move", "e2e4"),
        ("knight_move", "g1f3"),
        ("long_move", "a1h8"),
    ])
    def test_tv02_encode_decode_roundtrip(self, name: str, uci: str) -> None:
        """TV02: encode then decode should return the original UCI string."""
        try:
            vocab = MoveVocab()
        except NotImplementedError:
            self.fail("TV02: MoveVocab.__init__ not implemented")
        try:
            idx = vocab.encode(uci)
            result = vocab.decode(idx)
        except NotImplementedError:
            self.fail("TV02: encode/decode not implemented")
        self.assertEqual(result, uci)

    def test_tv03_special_tokens(self) -> None:
        """TV03: PAD=0, SOS=1, EOS=2."""
        self.assertEqual(PAD_IDX, 0)
        self.assertEqual(SOS_IDX, 1)
        self.assertEqual(EOS_IDX, 2)

    @parameterized.expand([
        ("queen_promo", "e7e8q"),
        ("rook_promo", "e7e8r"),
        ("bishop_promo", "e7e8b"),
        ("knight_promo", "e7e8n"),
    ])
    def test_tv04_promotion_included(self, name: str, uci: str) -> None:
        """TV04: Promotion moves should be in the vocabulary."""
        try:
            vocab = MoveVocab()
        except NotImplementedError:
            self.fail("TV04: MoveVocab.__init__ not implemented")
        try:
            self.assertIn(uci, vocab)
        except NotImplementedError:
            self.fail("TV04: __contains__ not implemented")


# ===========================================================================
# TV05-TV08: MoveTokenizer
# ===========================================================================


class TestMoveTokenizer(unittest.TestCase):
    """Tests for MoveTokenizer game tokenization and legal masking."""

    def test_tv05_tokenize_game_shape(self) -> None:
        """TV05: tokenize_game returns LongTensor [T+2] for T moves."""
        try:
            tok = MoveTokenizer()
        except NotImplementedError:
            self.fail("TV05: MoveTokenizer.__init__ not implemented")
        moves = ["e2e4", "e7e5", "g1f3"]
        try:
            result = tok.tokenize_game(moves)
        except NotImplementedError:
            self.fail("TV05: tokenize_game not implemented")
        self.assertEqual(result.dtype, torch.long)
        self.assertEqual(result.shape, torch.Size([len(moves) + 2]))

    def test_tv06_tokenize_game_sos_eos(self) -> None:
        """TV06: First token == SOS_IDX, last token == EOS_IDX."""
        try:
            tok = MoveTokenizer()
        except NotImplementedError:
            self.fail("TV06: MoveTokenizer.__init__ not implemented")
        try:
            result = tok.tokenize_game(["e2e4", "e7e5"])
        except NotImplementedError:
            self.fail("TV06: tokenize_game not implemented")
        self.assertEqual(result[0].item(), SOS_IDX)
        self.assertEqual(result[-1].item(), EOS_IDX)

    def test_tv07_build_legal_mask_shape(self) -> None:
        """TV07: build_legal_mask returns BoolTensor [VOCAB_SIZE]."""
        try:
            tok = MoveTokenizer()
        except NotImplementedError:
            self.fail("TV07: MoveTokenizer.__init__ not implemented")
        try:
            mask = tok.build_legal_mask(["e2e4", "d2d4"])
        except NotImplementedError:
            self.fail("TV07: build_legal_mask not implemented")
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.shape, torch.Size([MOVE_VOCAB_SIZE]))

    def test_tv08_build_legal_mask_values(self) -> None:
        """TV08: Legal moves are True, special tokens are False."""
        try:
            tok = MoveTokenizer()
        except NotImplementedError:
            self.fail("TV08: MoveTokenizer.__init__ not implemented")
        try:
            mask = tok.build_legal_mask(["e2e4", "d2d4"])
        except NotImplementedError:
            self.fail("TV08: build_legal_mask not implemented")
        # Special tokens should be False
        self.assertFalse(mask[PAD_IDX].item())
        self.assertFalse(mask[SOS_IDX].item())
        self.assertFalse(mask[EOS_IDX].item())
        # Exactly 2 legal moves should be True
        self.assertEqual(mask.sum().item(), 2)


# ===========================================================================
# TV09: MoveEmbedding
# ===========================================================================


class TestMoveEmbedding(unittest.TestCase):
    """Tests for MoveEmbedding output shape."""

    def test_tv09_output_shape(self) -> None:
        """TV09: MoveEmbedding forward returns [B, T, d_model]."""
        B, T = 4, 20
        try:
            emb = MoveEmbedding()
        except NotImplementedError:
            self.fail("TV09: MoveEmbedding.__init__ not implemented")
        tokens = torch.randint(0, MOVE_VOCAB_SIZE, (B, T), dtype=torch.long)
        colors = torch.zeros(B, T, dtype=torch.long)
        try:
            out = emb(tokens, colors)
        except NotImplementedError:
            self.fail("TV09: embed_moves not implemented")
        self.assertEqual(out.shape, torch.Size([B, T, D_MODEL]))


# ===========================================================================
# TV10-TV12: MoveDecoder
# ===========================================================================


class TestMoveDecoder(unittest.TestCase):
    """Tests for MoveDecoder causal mask and forward pass."""

    def test_tv10_causal_mask_shape(self) -> None:
        """TV10: _causal_mask returns BoolTensor [T, T]."""
        T = 10
        try:
            dec = MoveDecoder()
        except NotImplementedError:
            self.fail("TV10: MoveDecoder.__init__ not implemented")
        try:
            mask = dec._causal_mask(T)
        except NotImplementedError:
            self.fail("TV10: _causal_mask not implemented")
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.shape, torch.Size([T, T]))

    def test_tv11_causal_mask_upper_triangular(self) -> None:
        """TV11: mask[0,1]==True (future masked), mask[1,0]==False (past visible)."""
        try:
            dec = MoveDecoder()
        except NotImplementedError:
            self.fail("TV11: MoveDecoder.__init__ not implemented")
        try:
            mask = dec._causal_mask(4)
        except NotImplementedError:
            self.fail("TV11: _causal_mask not implemented")
        # Position 0 cannot attend to position 1 (future)
        self.assertTrue(mask[0, 1].item())
        # Position 1 can attend to position 0 (past)
        self.assertFalse(mask[1, 0].item())
        # Diagonal: position can attend to itself
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(mask[3, 3].item())

    def test_tv12_forward_output_shape(self) -> None:
        """TV12: MoveDecoder.forward returns logits [B, T, VOCAB_SIZE]."""
        B, T = 4, 20
        try:
            dec = MoveDecoder()
        except NotImplementedError:
            self.fail("TV12: MoveDecoder.__init__ not implemented")
        tokens = torch.randint(0, MOVE_VOCAB_SIZE, (B, T), dtype=torch.long)
        memory = torch.randn(B, 65, D_MODEL)
        try:
            out = dec(tokens, memory)
        except NotImplementedError:
            self.fail("TV12: decode not implemented")
        self.assertIsInstance(out, DecoderOutput)
        self.assertEqual(out.logits.shape, torch.Size([B, T, MOVE_VOCAB_SIZE]))


# ===========================================================================
# TV13: ChessModel
# ===========================================================================


class TestChessModel(unittest.TestCase):
    """Tests for ChessModel end-to-end forward pass."""

    def test_tv13_forward_output_shape(self) -> None:
        """TV13: ChessModel.forward returns [B, T, VOCAB_SIZE]."""
        B, T = 4, 20
        try:
            model = ChessModel()
        except NotImplementedError:
            self.fail("TV13: ChessModel.__init__ not implemented")
        bt, ct, tt = _make_board_tokens(B)
        mt = torch.randint(0, MOVE_VOCAB_SIZE, (B, T), dtype=torch.long)
        try:
            logits = model(bt, ct, tt, mt)
        except NotImplementedError:
            self.fail("TV13: forward not implemented")
        self.assertEqual(logits.shape, torch.Size([B, T, MOVE_VOCAB_SIZE]))


# ===========================================================================
# TV14: GameTurnSample fields
# ===========================================================================


class TestGameTurnSample(unittest.TestCase):
    """Tests for GameTurnSample NamedTuple fields."""

    def test_tv14_fields_present(self) -> None:
        """TV14: GameTurnSample has correct field names."""
        expected_fields = (
            "board_tokens",
            "color_tokens",
            "trajectory_tokens",
            "move_tokens",
            "target_tokens",
            "move_pad_mask",
        )
        self.assertEqual(GameTurnSample._fields, expected_fields)

    def test_tv14_can_construct(self) -> None:
        """TV14: GameTurnSample can be constructed with tensors."""
        sample = GameTurnSample(
            board_tokens=torch.zeros(65, dtype=torch.long),
            color_tokens=torch.zeros(65, dtype=torch.long),
            trajectory_tokens=torch.zeros(65, dtype=torch.long),
            move_tokens=torch.zeros(5, dtype=torch.long),
            target_tokens=torch.zeros(5, dtype=torch.long),
            move_pad_mask=torch.zeros(5, dtype=torch.bool),
        )
        self.assertEqual(sample.board_tokens.shape, torch.Size([65]))
        self.assertEqual(sample.move_tokens.shape, torch.Size([5]))

    def test_tv14_game_turn_batch_fields(self) -> None:
        """TV14: GameTurnBatch has correct field names."""
        expected_fields = (
            "board_tokens",
            "color_tokens",
            "trajectory_tokens",
            "move_tokens",
            "target_tokens",
            "move_pad_mask",
        )
        self.assertEqual(GameTurnBatch._fields, expected_fields)

    def test_tv14_decoder_output_fields(self) -> None:
        """TV14: DecoderOutput has correct field names."""
        self.assertEqual(DecoderOutput._fields, ("logits",))

    def test_tv14_self_play_game_fields(self) -> None:
        """TV14: SelfPlayGame has correct field names."""
        expected_fields = (
            "moves",
            "board_tokens",
            "color_tokens",
            "trajectory_tokens",
            "outcome",
        )
        self.assertEqual(SelfPlayGame._fields, expected_fields)


# ===========================================================================
# TV15-TV16: Config defaults
# ===========================================================================


class TestDecoderConfig(unittest.TestCase):
    """Tests for DecoderConfig default values."""

    @parameterized.expand([
        ("d_model", 256),
        ("n_heads", 8),
        ("n_layers", 4),
        ("dim_feedforward", 1024),
        ("dropout", 0.1),
        ("max_seq_len", 512),
        ("move_vocab_size", 1971),
    ])
    def test_tv15_defaults(self, field_name: str, expected: object) -> None:
        """TV15: DecoderConfig defaults match design doc."""
        cfg = DecoderConfig()
        self.assertEqual(getattr(cfg, field_name), expected)


class TestPhase2Config(unittest.TestCase):
    """Tests for Phase2Config default values."""

    @parameterized.expand([
        ("win_reward", 1.0),
        ("loss_reward", -1.0),
        ("draw_reward", 0.0),
        ("delta_novelty", 0.05),
        ("delta_decay", 0.999),
        ("top_k_human", 5),
        ("min_win_rate", 0.55),
        ("self_play_games", 100),
        ("pretrained_ckpt", ""),
    ])
    def test_tv16_defaults(self, field_name: str, expected: object) -> None:
        """TV16: Phase2Config defaults match design doc."""
        cfg = Phase2Config()
        self.assertEqual(getattr(cfg, field_name), expected)


# ===========================================================================
# TV17: load_v2_config
# ===========================================================================


class TestLoadV2Config(unittest.TestCase):
    """Tests for load_v2_config YAML roundtrip."""

    def test_tv17_load_from_yaml(self) -> None:
        """TV17: load_v2_config loads a valid YAML without error."""
        yaml_content = """\
data:
  pgn: "test.pgn"
  batch_size: 64
model:
  d_model: 256
  n_layers: 6
decoder:
  n_layers: 4
  max_seq_len: 512
trainer:
  epochs: 5
phase2:
  win_reward: 1.0
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_v2_config(Path(f.name))

        self.assertIsInstance(cfg, ChessModelV2Config)
        self.assertEqual(cfg.data.pgn, "test.pgn")
        self.assertEqual(cfg.data.batch_size, 64)
        self.assertEqual(cfg.model.d_model, 256)
        self.assertEqual(cfg.model.n_layers, 6)
        self.assertEqual(cfg.decoder.n_layers, 4)
        self.assertEqual(cfg.decoder.max_seq_len, 512)
        self.assertEqual(cfg.trainer.epochs, 5)
        self.assertEqual(cfg.phase2.win_reward, 1.0)

    def test_tv17_load_empty_yaml(self) -> None:
        """TV17: load_v2_config with empty YAML uses all defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()
            cfg = load_v2_config(Path(f.name))

        self.assertIsInstance(cfg, ChessModelV2Config)
        self.assertEqual(cfg.decoder.d_model, 256)
        self.assertEqual(cfg.phase2.win_reward, 1.0)

    def test_tv17_unknown_key_raises(self) -> None:
        """TV17: Unknown YAML keys should raise TypeError."""
        yaml_content = """\
decoder:
  unknown_key: 42
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            with self.assertRaises(TypeError):
                load_v2_config(Path(f.name))


if __name__ == "__main__":
    unittest.main()
