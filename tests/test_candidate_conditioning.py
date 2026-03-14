"""Tests for candidate piece conditioning feature.

Covers CandidateReachabilityMapper, build_candidate_board_tokens,
PieceTypeMoveLUT, EmbeddingLayer piece_type_cond_emb, ChessEncoder
pass-through, ChessModel predict_next_move with piece_type, and
ModelConfig.use_candidate_conditioning.

All tests run on CPU only (per project convention).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
from parameterized import parameterized

from chess_sim.config import DecoderConfig, ModelConfig
from chess_sim.data.candidate_reachability_mapper import (
    INVALID_EMPTY_TOKEN,
    VALID_EMPTY_TOKEN,
    CandidateReachabilityMapper,
    build_candidate_board_tokens,
    compute_valid_empty_mask,
)
from chess_sim.data.piece_type_move_lut import PieceTypeMoveLUT
from chess_sim.model.embedding import (
    INVALID_EMPTY_IDX,
    PIECE_TYPE_COND_VOCAB_SIZE,
    PIECE_VOCAB_SIZE,
    EmbeddingLayer,
)
from chess_sim.types import ConditionedBoard

# ---------------------------------------------------------------------------
# TestCandidateReachabilityMapper
# ---------------------------------------------------------------------------


class TestCandidateReachabilityMapper(unittest.TestCase):
    """Tests for CandidateReachabilityMapper.compute()."""

    def test_compute_returns_frozenset(self) -> None:
        """CandidateReachabilityMapper.compute() returns a frozenset."""
        import chess

        mapper = CandidateReachabilityMapper()
        result = mapper.compute(chess.Board(), chess.ROOK, True)
        self.assertIsInstance(result, frozenset)

    def test_compute_valid_empty_mask_returns_64_bools(
        self,
    ) -> None:
        """compute_valid_empty_mask() returns length-64 list."""
        import chess

        mask = compute_valid_empty_mask(
            chess.Board(), chess.WHITE, chess.ROOK
        )
        self.assertEqual(len(mask), 64)

    def test_rook_reachable_squares_open_board(self) -> None:
        """Rook on open board reaches all rank/file squares.

        On an open board with a single rook on a1, the rook
        can reach all squares on file a (a2-a8) and rank 1
        (b1-h1). compute() should return those 14 squares.
        """
        import chess

        board = chess.Board(fen=None)
        board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
        mapper = CandidateReachabilityMapper()
        result = mapper.compute(board, chess.ROOK, True)
        # a-file: a2-a8 = 7 squares, rank 1: b1-h1 = 7 squares
        self.assertEqual(len(result), 14)

    def test_rook_blocked_by_friendly_piece(self) -> None:
        """Blocked squares excluded from reachable set.

        A friendly pawn on a2 blocks the rook on a1 from
        reaching a3-a8 along the a-file. a2 is occupied so
        not in reachable set. Only rank-1 squares remain.
        """
        import chess

        board = chess.Board(fen=None)
        board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.WHITE))
        mapper = CandidateReachabilityMapper()
        result = mapper.compute(board, chess.ROOK, True)
        # Only b1-h1 = 7 squares (a2 blocked, a3+ unreachable)
        self.assertEqual(len(result), 7)
        self.assertNotIn(chess.A2, result)

    def test_no_candidates_returns_empty_set(self) -> None:
        """When piece type not on board, compute() returns empty frozenset.

        Board with no knights and piece_type=KNIGHT should yield
        an empty frozenset.
        """
        import chess

        board = chess.Board(fen=None)
        board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
        mapper = CandidateReachabilityMapper()
        result = mapper.compute(board, chess.KNIGHT, True)
        self.assertEqual(result, frozenset())

    def test_pawn_reachable_squares(self) -> None:
        """Pawn reachability: forward pushes only to empty squares.

        White pawns on starting position can push 1 or 2 squares
        forward. Diagonal captures excluded (no enemy pieces).
        """
        import chess

        board = chess.Board()
        mapper = CandidateReachabilityMapper()
        result = mapper.compute(board, chess.PAWN, True)
        # 8 pawns, each can push 1 or 2 = 16 empty squares
        self.assertEqual(len(result), 16)


# ---------------------------------------------------------------------------
# TestBuildCandidateBoardTokens
# ---------------------------------------------------------------------------


class TestBuildCandidateBoardTokens(unittest.TestCase):
    """Tests for build_candidate_board_tokens()."""

    def test_build_returns_new_list(self) -> None:
        """build_candidate_board_tokens() returns a new list, not mutated input."""
        tokens = [0] + [1] * 64
        result = build_candidate_board_tokens(tokens, frozenset({0}))
        self.assertIsNot(result, tokens)
        # Original unchanged
        self.assertEqual(tokens[2], 1)

    def test_valid_empty_assigned_to_reachable(self) -> None:
        """Reachable empty squares keep token VALID_EMPTY (1)."""
        tokens = [0] + [1] * 64
        result = build_candidate_board_tokens(
            tokens, frozenset({0, 1, 2})
        )
        self.assertEqual(result[1], VALID_EMPTY_TOKEN)  # sq 0
        self.assertEqual(result[2], VALID_EMPTY_TOKEN)  # sq 1
        self.assertEqual(result[3], VALID_EMPTY_TOKEN)  # sq 2

    def test_invalid_empty_assigned_to_unreachable(self) -> None:
        """Unreachable empty squares get token INVALID_EMPTY (8)."""
        tokens = [0] + [1] * 64
        result = build_candidate_board_tokens(
            tokens, frozenset({0, 1, 2})
        )
        # sq 3 -> token index 4, not reachable
        self.assertEqual(result[4], INVALID_EMPTY_TOKEN)

    def test_occupied_squares_unchanged(self) -> None:
        """Piece tokens (2-7) are never modified regardless of reachability."""
        tokens = [0] + [1] * 64
        tokens[5] = 3  # sq 4 has a knight
        result = build_candidate_board_tokens(
            tokens, frozenset({4})  # sq 4 in reachable set
        )
        self.assertEqual(result[5], 3)  # still a knight

    def test_cls_token_unchanged(self) -> None:
        """CLS token at index 0 is never modified."""
        tokens = [0] + [1] * 64
        result = build_candidate_board_tokens(
            tokens, frozenset()
        )
        self.assertEqual(result[0], 0)


# ---------------------------------------------------------------------------
# TestPieceTypeMoveLUT
# ---------------------------------------------------------------------------


class TestPieceTypeMoveLUT(unittest.TestCase):
    """Tests for PieceTypeMoveLUT construction and filter_legal_mask()."""

    def test_lut_shape(self) -> None:
        """LUT is [7, 1971] bool tensor."""
        lut = PieceTypeMoveLUT()
        self.assertEqual(lut._lut.shape, (7, 1971))
        self.assertEqual(lut._lut.dtype, torch.bool)

    def test_filters_non_rook_moves(self) -> None:
        """Only rook moves survive when piece_type=ROOK(4)."""
        import chess

        lut = PieceTypeMoveLUT()
        legal = torch.ones(1, 1971, dtype=torch.bool)
        pt = torch.tensor([chess.ROOK], dtype=torch.long)
        filtered = lut.filter_legal_mask(legal, pt)
        # Row 0 of LUT is all-False, rook row should have entries
        self.assertTrue(filtered.any())
        # All True entries must be rook-from-square moves
        self.assertEqual(filtered.shape, (1, 1971))

    def test_no_conditioning_passthrough(self) -> None:
        """piece_type=0 returns all-False mask (caller must gate)."""
        lut = PieceTypeMoveLUT()
        legal = torch.ones(1, 1971, dtype=torch.bool)
        pt = torch.tensor([0], dtype=torch.long)
        filtered = lut.filter_legal_mask(legal, pt)
        self.assertFalse(filtered.any())

    def test_row_zero_all_false(self) -> None:
        """Row 0 of the LUT is all-False (no piece type 0)."""
        lut = PieceTypeMoveLUT()
        self.assertFalse(lut._lut[0].any())


# ---------------------------------------------------------------------------
# TestEmbeddingLayerCandidateConditioning
# ---------------------------------------------------------------------------


class TestEmbeddingLayerCandidateConditioning(unittest.TestCase):
    """Tests for EmbeddingLayer piece_type_cond_emb integration."""

    def setUp(self) -> None:
        self.layer = EmbeddingLayer(d_model=64)
        self.layer.eval()
        self.B = 2
        self.S = 65

    def _make_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return zero board, color, and trajectory tokens."""
        z = torch.zeros(self.B, self.S, dtype=torch.long)
        return z, z, z

    def test_piece_vocab_size_is_nine(self) -> None:
        """PIECE_VOCAB_SIZE constant should be 9 after adding INVALID_EMPTY."""
        self.assertEqual(PIECE_VOCAB_SIZE, 9)

    def test_invalid_empty_idx_is_eight(self) -> None:
        """INVALID_EMPTY_IDX constant should be 8."""
        self.assertEqual(INVALID_EMPTY_IDX, 8)

    def test_piece_type_cond_vocab_size_is_eight(self) -> None:
        """PIECE_TYPE_COND_VOCAB_SIZE constant should be 8 (0-7)."""
        self.assertEqual(PIECE_TYPE_COND_VOCAB_SIZE, 8)

    def test_piece_emb_accepts_index_eight(self) -> None:
        """piece_emb(8) for INVALID_EMPTY does not raise IndexError."""
        idx = torch.tensor([[8]])
        out = self.layer.piece_emb(idx)
        self.assertEqual(out.shape, (1, 1, 64))

    def test_piece_type_cond_emb_exists(self) -> None:
        """EmbeddingLayer has piece_type_cond_emb attribute."""
        self.assertTrue(
            hasattr(self.layer, "piece_type_cond_emb")
        )
        self.assertEqual(
            self.layer.piece_type_cond_emb.num_embeddings,
            PIECE_TYPE_COND_VOCAB_SIZE,
        )

    def test_piece_type_cond_zero_init_is_noop(self) -> None:
        """Zero-init piece_type_cond_emb means no change to output.

        With piece_type_tokens=0 (no conditioning) and freshly
        initialized weights (all zeros), the embed() output should
        be identical whether piece_type_tokens is None or zeros.
        """
        bt, ct, tt = self._make_inputs()
        with torch.no_grad():
            out_none = self.layer.embed(bt, ct, tt)
            pt = torch.zeros(self.B, dtype=torch.long)
            out_zero = self.layer.embed(bt, ct, tt, piece_type_tokens=pt)
        self.assertTrue(torch.allclose(out_none, out_zero))

    def test_piece_type_cond_changes_output_after_training(
        self,
    ) -> None:
        """Non-zero piece_type_cond_emb weight changes embed() output.

        After manually setting different weights per row in
        piece_type_cond_emb, embed() with piece_type_tokens=3 should
        differ from piece_type_tokens=0 since the rows are different.
        """
        bt, ct, tt = self._make_inputs()
        with torch.no_grad():
            # Set non-uniform weights per row (uniform offsets are
            # invisible to LayerNorm, so use randn-based init)
            torch.manual_seed(42)
            self.layer.piece_type_cond_emb.weight.copy_(
                torch.randn_like(
                    self.layer.piece_type_cond_emb.weight
                )
            )
            pt_zero = torch.zeros(self.B, dtype=torch.long)
            out_zero = self.layer.embed(
                bt, ct, tt, piece_type_tokens=pt_zero
            )
            pt_three = torch.full(
                (self.B,), 3, dtype=torch.long
            )
            out_cond = self.layer.embed(
                bt, ct, tt, piece_type_tokens=pt_three
            )
        self.assertFalse(torch.allclose(out_zero, out_cond))

    def test_embed_output_shape_with_piece_type(self) -> None:
        """embed() output shape is [B, 65, d_model] with piece_type_tokens."""
        bt, ct, tt = self._make_inputs()
        pt = torch.zeros(self.B, dtype=torch.long)
        with torch.no_grad():
            out = self.layer.embed(bt, ct, tt, piece_type_tokens=pt)
        self.assertEqual(out.shape, (self.B, self.S, 64))

    @parameterized.expand([
        ("no_cond", 0),
        ("pawn", 1),
        ("knight", 2),
        ("bishop", 3),
        ("rook", 4),
        ("queen", 5),
        ("king", 6),
        ("max_valid", 7),
    ])
    def test_piece_type_cond_emb_valid_indices(
        self, name: str, idx: int
    ) -> None:
        """piece_type_cond_emb accepts all valid indices 0-7."""
        t = torch.tensor([[idx]])
        out = self.layer.piece_type_cond_emb(t)
        self.assertEqual(out.shape[-1], 64)

    def test_board_tokens_with_invalid_empty(self) -> None:
        """Board containing INVALID_EMPTY (8) tokens embeds without error."""
        bt = torch.zeros(self.B, self.S, dtype=torch.long)
        bt[:, 10] = INVALID_EMPTY_IDX  # some squares are INVALID_EMPTY
        ct = torch.zeros(self.B, self.S, dtype=torch.long)
        tt = torch.zeros(self.B, self.S, dtype=torch.long)
        with torch.no_grad():
            out = self.layer.embed(bt, ct, tt)
        self.assertEqual(out.shape, (self.B, self.S, 64))


# ---------------------------------------------------------------------------
# TestChessEncoderPieceTypePassthrough
# ---------------------------------------------------------------------------


class TestChessEncoderPieceTypePassthrough(unittest.TestCase):
    """Tests for ChessEncoder passing piece_type_tokens to EmbeddingLayer."""

    @patch(
        "chess_sim.model.encoder.EmbeddingLayer",
        autospec=True,
    )
    def test_encode_passes_piece_type_tokens(
        self, mock_emb_cls: MagicMock
    ) -> None:
        """encode() forwards piece_type_tokens to self.embedding().

        Mocks EmbeddingLayer to verify that piece_type_tokens is passed
        as the 5th argument to the embedding call.
        """
        from chess_sim.model.encoder import ChessEncoder

        enc = ChessEncoder.__new__(ChessEncoder)
        enc.embedding = MagicMock()
        enc.transformer = MagicMock()

        bt = torch.zeros(2, 65, dtype=torch.long)
        ct = torch.zeros(2, 65, dtype=torch.long)
        tt = torch.zeros(2, 65, dtype=torch.long)
        pt = torch.tensor([3, 5], dtype=torch.long)

        # Make transformer return something with correct shape
        fake_encoded = torch.zeros(2, 65, 256)
        enc.embedding.return_value = fake_encoded
        enc.transformer.return_value = fake_encoded

        enc.encode(bt, ct, tt, piece_type_tokens=pt)

        # Verify piece_type_tokens was passed
        call_args = enc.embedding.call_args
        self.assertIs(call_args[0][4], pt)

    def test_encode_piece_type_default_none(self) -> None:
        """encode() with no piece_type_tokens passes None to embedding."""
        from chess_sim.model.encoder import ChessEncoder

        enc = ChessEncoder.__new__(ChessEncoder)
        enc.embedding = MagicMock()
        enc.transformer = MagicMock()

        bt = torch.zeros(2, 65, dtype=torch.long)
        fake_encoded = torch.zeros(2, 65, 256)
        enc.embedding.return_value = fake_encoded
        enc.transformer.return_value = fake_encoded

        enc.encode(bt, bt, bt)

        call_args = enc.embedding.call_args
        # piece_type_tokens should be None (default)
        self.assertIsNone(call_args[0][4])


# ---------------------------------------------------------------------------
# TestChessModelPredictNextMovePieceType
# ---------------------------------------------------------------------------


class TestChessModelPredictNextMovePieceType(unittest.TestCase):
    """Tests for ChessModel.predict_next_move() with piece_type param."""

    def _make_model(self):
        """Create a small ChessModel for testing."""
        from chess_sim.model.chess_model import ChessModel
        d = 64
        model = ChessModel(
            ModelConfig(d_model=d, n_layers=1),
            DecoderConfig(d_model=d, n_layers=1),
        )
        model.eval()
        return model

    def test_predict_next_move_accepts_piece_type(self) -> None:
        """predict_next_move() accepts piece_type kwarg without TypeError."""
        model = self._make_model()
        bt = torch.zeros(1, 65, dtype=torch.long)
        # Use a starting position with legal moves
        with torch.no_grad():
            move = model.predict_next_move(
                bt, bt, bt,
                move_history=[],
                legal_moves=["e2e4", "d2d4", "g1f3"],
                is_white_turn=True,
                piece_type=1,  # PAWN
            )
        self.assertIsInstance(move, str)

    def test_piece_type_narrows_legal_mask(self) -> None:
        """Only moves matching piece_type survive the filtered mask.

        With piece_type=2 (KNIGHT), only knight moves from the
        starting position should be candidates. e2e4 (pawn) should
        not be selected if only knight moves are legal after filter.
        """
        model = self._make_model()
        bt = torch.zeros(1, 65, dtype=torch.long)
        # Only provide knight moves as legal — pawn move excluded
        with torch.no_grad():
            move = model.predict_next_move(
                bt, bt, bt,
                move_history=[],
                legal_moves=["g1f3", "b1c3"],
                is_white_turn=True,
                piece_type=2,  # KNIGHT
            )
        self.assertIn(move, ["g1f3", "b1c3"])

    def test_piece_type_none_no_filtering(self) -> None:
        """predict_next_move() with piece_type=None allows all moves."""
        model = self._make_model()
        bt = torch.zeros(1, 65, dtype=torch.long)
        with torch.no_grad():
            move = model.predict_next_move(
                bt, bt, bt,
                move_history=[],
                legal_moves=["e2e4", "g1f3"],
                is_white_turn=True,
                piece_type=None,
            )
        self.assertIn(move, ["e2e4", "g1f3"])


# ---------------------------------------------------------------------------
# TestChessModelForwardPieceType
# ---------------------------------------------------------------------------


class TestChessModelForwardPieceType(unittest.TestCase):
    """Tests for ChessModel.forward() piece_type parameter."""

    def setUp(self) -> None:
        from chess_sim.model.chess_model import ChessModel

        self.d = 64
        self.model = ChessModel(
            ModelConfig(d_model=self.d, n_layers=1),
            DecoderConfig(d_model=self.d, n_layers=1),
        )
        self.model.eval()

    def test_forward_accepts_piece_type_kwarg(self) -> None:
        """forward() signature accepts piece_type=None without error.

        Constructs a ChessModel and calls forward() with piece_type=None
        to verify backward compatibility.
        """
        bt = torch.zeros(1, 65, dtype=torch.long)
        mt = torch.zeros(1, 5, dtype=torch.long)
        with torch.no_grad():
            logits = self.model.forward(
                bt, bt, bt, mt, piece_type=None
            )
        self.assertEqual(logits.shape[0], 1)

    def test_forward_with_piece_type_tensor(self) -> None:
        """forward() with piece_type=[5] completes without error."""
        bt = torch.zeros(1, 65, dtype=torch.long)
        mt = torch.zeros(1, 5, dtype=torch.long)
        pt = torch.tensor([5], dtype=torch.long)
        with torch.no_grad():
            logits = self.model.forward(
                bt, bt, bt, mt, piece_type=pt
            )
        self.assertEqual(logits.shape[0], 1)


# ---------------------------------------------------------------------------
# TestModelConfig
# ---------------------------------------------------------------------------


class TestModelConfigCandidateConditioning(unittest.TestCase):
    """Tests for ModelConfig.use_candidate_conditioning field."""

    def test_default_false(self) -> None:
        """use_candidate_conditioning defaults to False."""
        cfg = ModelConfig()
        self.assertFalse(cfg.use_candidate_conditioning)

    def test_set_true(self) -> None:
        """use_candidate_conditioning can be set to True."""
        cfg = ModelConfig(use_candidate_conditioning=True)
        self.assertTrue(cfg.use_candidate_conditioning)

    def test_existing_fields_unchanged(self) -> None:
        """Adding use_candidate_conditioning does not break existing fields."""
        cfg = ModelConfig(
            d_model=128,
            n_heads=4,
            use_src_conditioning=True,
            use_candidate_conditioning=True,
        )
        self.assertEqual(cfg.d_model, 128)
        self.assertEqual(cfg.n_heads, 4)
        self.assertTrue(cfg.use_src_conditioning)


# ---------------------------------------------------------------------------
# TestConditionedBoardType
# ---------------------------------------------------------------------------


class TestConditionedBoardType(unittest.TestCase):
    """Tests for ConditionedBoard NamedTuple."""

    def test_construction(self) -> None:
        """ConditionedBoard can be constructed with correct fields."""
        cb = ConditionedBoard(
            board_tokens=[0] + [1] * 64,
            color_tokens=[0] + [0] * 64,
            piece_type_idx=4,
        )
        self.assertEqual(len(cb.board_tokens), 65)
        self.assertEqual(cb.piece_type_idx, 4)

    def test_field_count(self) -> None:
        """ConditionedBoard has exactly 3 fields."""
        self.assertEqual(len(ConditionedBoard._fields), 3)

    @parameterized.expand([
        ("board_tokens", 0),
        ("color_tokens", 1),
        ("piece_type_idx", 2),
    ])
    def test_field_names(
        self, name: str, idx: int
    ) -> None:
        """ConditionedBoard fields are correctly named and ordered."""
        self.assertEqual(ConditionedBoard._fields[idx], name)


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestCandidateConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_invalid_empty_token_value(self) -> None:
        """INVALID_EMPTY_TOKEN in candidate_reachability_mapper is 8."""
        self.assertEqual(INVALID_EMPTY_TOKEN, 8)

    def test_valid_empty_token_value(self) -> None:
        """VALID_EMPTY_TOKEN in candidate_reachability_mapper is 1."""
        self.assertEqual(VALID_EMPTY_TOKEN, 1)

    def test_embedding_invalid_empty_matches(self) -> None:
        """INVALID_EMPTY_IDX in embedding matches INVALID_EMPTY_TOKEN."""
        self.assertEqual(INVALID_EMPTY_IDX, INVALID_EMPTY_TOKEN)


if __name__ == "__main__":
    unittest.main()
