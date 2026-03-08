"""Protocol interfaces for the chess encoder system.

Each Protocol defines one capability. Concrete classes implement these
implicitly via structural subtyping — no explicit `implements` declaration needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

import chess
import chess.pgn
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from chess_sim.types import (
    ChessBatch,
    EncoderOutput,
    EpisodeRecord,
    PlyTuple,
    PredictionOutput,
    RawTurnRecord,
    TokenizedBoard,
)

if TYPE_CHECKING:
    from chess_sim.config import Phase2Config, PreprocessV2Config
    from chess_sim.data.streaming_types import ManifestInfo, PreprocessConfig


@runtime_checkable
class Tokenizable(Protocol):
    """Converts a chess board state into integer token sequences."""

    def tokenize(self, board: chess.Board, turn: chess.Color) -> TokenizedBoard:
        """Convert a board position to (board_tokens, color_tokens) integer lists.

        Args:
            board: The current chess board state.
            turn: The color of the player whose perspective to encode from.

        Returns:
            TokenizedBoard with board_tokens and color_tokens of length 65.

        Example:
            >>> tok = BoardTokenizer()
            >>> result = tok.tokenize(chess.Board(), chess.WHITE)
        """
        ...


@runtime_checkable
class Embeddable(Protocol):
    """Converts token index tensors into dense embedding vectors."""

    def embed(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        trajectory_tokens: Tensor,
    ) -> Tensor:
        """Compose piece, color, square, and trajectory embeddings.

        Args:
            board_tokens: Integer tensor [B, 65] indices 0-7.
            color_tokens: Integer tensor [B, 65] indices 0-2.
            trajectory_tokens: Integer tensor [B, 65] indices 0-4.

        Returns:
            Float tensor [B, 65, 256] of composed embeddings.

        Example:
            >>> emb = EmbeddingLayer()
            >>> out = emb.embed(bt, ct, tt)
        """
        ...


@runtime_checkable
class Encodable(Protocol):
    """Runs the full transformer forward pass, producing CLS and square embeddings."""

    def encode(
        self,
        board_tokens: Tensor,
        color_tokens: Tensor,
        trajectory_tokens: Tensor,
    ) -> EncoderOutput:
        """Encode a batch of board states into embeddings.

        Args:
            board_tokens: Integer tensor [B, 65] piece indices.
            color_tokens: Integer tensor [B, 65] color indices.
            trajectory_tokens: Integer tensor [B, 65] values 0-4.

        Returns:
            EncoderOutput with cls_embedding [B, 256]
            and square_embeddings [B, 64, 256].

        Example:
            >>> enc = ChessEncoder()
            >>> out = enc.encode(bt, ct, tt)
        """
        ...


@runtime_checkable
class Predictable(Protocol):
    """Maps a CLS embedding to two square-prediction logit tensors."""

    def predict(self, cls_embedding: Tensor) -> PredictionOutput:
        """Predict player src/tgt squares from CLS embedding.

        Args:
            cls_embedding: Float tensor [B, 256] — the global board representation.

        Returns:
            PredictionOutput with two logit tensors each of shape [B, 64].

        Example:
            >>> heads = PredictionHeads()
            >>> preds = heads.predict(cls)  # preds.src_sq_logits.shape == (B, 64)
        """
        ...


@runtime_checkable
class Trainable(Protocol):
    """Orchestrates a full training loop with forward, backward, and optimization."""

    def train_step(self, batch: ChessBatch) -> float:
        """Execute one gradient update step and return the scalar loss value.

        Args:
            batch: A ChessBatch namedtuple with token tensors and label tensors.

        Returns:
            The combined cross-entropy loss as a Python float.

        Example:
            >>> loss = trainer.train_step(batch)
        """
        ...

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one full epoch over the DataLoader and return the average loss.

        Args:
            loader: DataLoader yielding ChessBatch instances.

        Returns:
            Mean loss across all batches in the epoch.

        Example:
            >>> avg_loss = trainer.train_epoch(loader)
        """
        ...


@runtime_checkable
class Samplable(Protocol):
    """Selects N items uniformly at random from a stream of unknown length."""

    def sample(
        self, stream: Iterator[chess.pgn.Game], n: int
    ) -> list[chess.pgn.Game]:
        """Sample n games uniformly from a potentially infinite stream using O(n) memory.

        Args:
            stream: An iterator of chess.pgn.Game objects.
            n: The number of games to sample.

        Returns:
            A list of up to n games selected uniformly at random.

        Example:
            >>> sampler = ReservoirSampler()
            >>> games = sampler.sample(reader.stream(path), 1_000_000)
        """
        ...


# ---------------------------------------------------------------------------
# Streaming data pipeline protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Preprocessable(Protocol):
    """Defines the interface for a PGN preprocessing pipeline."""

    def preprocess(
        self, pgn_path: Path, output_dir: Path, config: "PreprocessConfig"
    ) -> "ManifestInfo":
        """Run preprocessing on a PGN file and produce shard files + manifest.

        Args:
            pgn_path: Path to the source PGN file (plain or .zst).
            output_dir: Directory for output shards and manifest.
            config: PreprocessConfig with chunk_size, winners_only, etc.

        Returns:
            ManifestInfo with shard paths, example counts, and checksum.

        Example:
            >>> info = preprocessor.preprocess(pgn, out, config)
        """
        ...


@runtime_checkable
class ShardWritable(Protocol):
    """Defines the interface for shard serialization to disk."""

    def flush(
        self, tensors: dict[str, Tensor], shard_idx: int, output_dir: Path
    ) -> Path:
        """Serialize a tensor dict to a numbered .pt shard file.

        Args:
            tensors: Dict mapping field names to torch.long tensors.
            shard_idx: Zero-based shard index for filename generation.
            output_dir: Directory where the shard file is written.

        Returns:
            Path to the written shard file.

        Example:
            >>> path = writer.flush(tensors, 0, Path("/tmp/shards"))
        """
        ...


@runtime_checkable
class Cacheable(Protocol):
    """Defines the interface for cache validation and checksum computation."""

    def checksum(self, path: Path) -> str:
        """Compute a fast checksum of a file for cache keying.

        Args:
            path: Path to the file to checksum.

        Returns:
            Hex-encoded digest string.

        Example:
            >>> cs = cache.checksum(Path("data.pgn"))
        """
        ...

    def is_cached(self, manifest_path: Path, source_checksum: str) -> bool:
        """Check if a valid cached manifest exists for the given checksum.

        Args:
            manifest_path: Path to the manifest.json file.
            source_checksum: Expected checksum of the source file.

        Returns:
            True if manifest exists and checksums match.

        Example:
            >>> cache.is_cached(Path("manifest.json"), "abc123")
            False
        """
        ...


# ---------------------------------------------------------------------------
# v2 Encoder-Decoder protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class MoveTokenizable(Protocol):
    """Converts UCI move strings to integer vocabulary indices."""

    def tokenize_move(self, uci: str) -> int:
        """Convert a single UCI move string to its vocabulary index.

        Args:
            uci: UCI move string, e.g. "e2e4" or "e7e8q".

        Returns:
            Integer vocabulary index for the move.

        Example:
            >>> tok = MoveTokenizer()
            >>> tok.tokenize_move("e2e4")
            42
        """
        ...

    def tokenize_game(self, moves: list[str]) -> Tensor:
        """Convert a list of UCI moves to a LongTensor with SOS/EOS tokens.

        Args:
            moves: List of UCI move strings in game order.

        Returns:
            LongTensor of shape [T+2] with SOS prepended and EOS appended.

        Example:
            >>> tok.tokenize_game(["e2e4", "e7e5"])
            tensor([1, 42, 57, 2])
        """
        ...

    def build_legal_mask(self, legal_moves: list[str]) -> Tensor:
        """Build a boolean mask over the move vocabulary for legal moves.

        Args:
            legal_moves: List of legal UCI move strings.

        Returns:
            BoolTensor of shape [VOCAB_SIZE]. True = legal, False = illegal.

        Example:
            >>> mask = tok.build_legal_mask(["e2e4", "d2d4"])
            >>> mask.shape
            torch.Size([1971])
        """
        ...


@runtime_checkable
class Decodable(Protocol):
    """Autoregressive move decoder with cross-attention to encoder memory."""

    def decode(
        self,
        move_tokens: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Tensor | None,
    ) -> "DecoderOutput":
        """Decode move tokens autoregressively using encoder memory.

        Args:
            move_tokens: LongTensor [B, T] of move token indices.
            memory: FloatTensor [B, 65, d_model] encoder output.
            tgt_key_padding_mask: Optional BoolTensor [B, T]. True = PAD.

        Returns:
            DecoderOutput with logits [B, T, MOVE_VOCAB_SIZE].

        Example:
            >>> out = decoder.decode(move_tok, memory, pad_mask)
        """
        ...


@runtime_checkable
class MoveEmbeddable(Protocol):
    """Embeds move token sequences with positional encoding."""

    def embed_moves(self, move_tokens: Tensor) -> Tensor:
        """Embed a batch of move token sequences.

        Args:
            move_tokens: LongTensor [B, T] of move vocabulary indices.

        Returns:
            FloatTensor [B, T, d_model] of embedded move tokens.

        Example:
            >>> emb = MoveEmbedding()
            >>> out = emb.embed_moves(tokens)  # [B, T, 256]
        """
        ...


# ---------------------------------------------------------------------------
# HDF5 preprocessing pipeline protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class HDF5Parseable(Protocol):
    """Parses a PGN game into RawTurnRecord list."""

    def parse_game(
        self, game: chess.pgn.Game, game_id: int
    ) -> list[RawTurnRecord]:
        """Parse one game into per-ply records.

        Args:
            game: A parsed PGN game object.
            game_id: Sequential index of this game.

        Returns:
            List of RawTurnRecord, one per ply. Empty if filtered.
        """
        ...


@runtime_checkable
class HDF5Writable(Protocol):
    """Writes RawTurnRecord batches to HDF5."""

    def open(self, path: Path, mode: str) -> None:
        """Open or create HDF5 file for writing."""
        ...

    def write_batch(
        self, records: list[RawTurnRecord], split: str
    ) -> None:
        """Buffer records and flush when chunk_size reached."""
        ...

    def close(self) -> None:
        """Flush remaining buffers and close the file."""
        ...


@runtime_checkable
class HDF5Readable(Protocol):
    """Reads individual samples from an HDF5 dataset."""

    def open(self, path: Path) -> None:
        """Open HDF5 file for reading."""
        ...

    def read_sample(
        self, split: str, idx: int
    ) -> RawTurnRecord:
        """Read a single sample by split and index."""
        ...

    def split_len(self, split: str) -> int:
        """Return the number of samples in a split."""
        ...

    def close(self) -> None:
        """Close the HDF5 file handle."""
        ...


@runtime_checkable
class HDF5Validatable(Protocol):
    """Validates HDF5 file schema and value ranges."""

    def validate(
        self, path: Path, config: "PreprocessV2Config"
    ) -> None:
        """Raise ValueError on any schema violation."""
        ...


@runtime_checkable
class HDF5Preprocessable(Protocol):
    """Orchestrates full PGN-to-HDF5 preprocessing."""

    def run(self, config: "PreprocessV2Config") -> None:
        """Execute the complete preprocessing pipeline."""
        ...


# -------------------------------------------------------------------
# Phase 2 Self-Play Protocols
# -------------------------------------------------------------------


@runtime_checkable
class Recordable(Protocol):
    """Accumulates per-ply data during a self-play episode."""

    def record(self, ply: PlyTuple) -> None:
        """Append a single ply (player or opponent) to the buffer."""
        ...

    def finalize(self, outcome: float) -> EpisodeRecord:
        """Seal the episode, apply entropy normalization."""
        ...

    def reset(self) -> None:
        """Clear internal buffer for a new episode."""
        ...


@runtime_checkable
class Computable(Protocol):
    """Converts an EpisodeRecord into a per-player-ply reward tensor."""

    def compute(
        self,
        record: EpisodeRecord,
        cfg: "Phase2Config",
    ) -> Tensor:
        """Return reward tensor of shape [T_player]."""
        ...


@runtime_checkable
class Updatable(Protocol):
    """Performs EMA parameter sync from player to opponent model."""

    def step(
        self,
        player: nn.Module,
        opponent: nn.Module,
    ) -> None:
        """Update opponent: theta_opp <- alpha*theta_opp + (1-alpha)*theta_player."""
        ...
