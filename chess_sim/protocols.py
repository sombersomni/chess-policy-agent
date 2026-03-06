"""Protocol interfaces for the chess encoder system.

Each Protocol defines one capability. Concrete classes implement these
implicitly via structural subtyping — no explicit `implements` declaration needed.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import chess
import chess.pgn
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from chess_sim.types import (
    ChessBatch,
    EncoderOutput,
    PredictionOutput,
    TokenizedBoard,
)


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
    """Maps a CLS embedding to four square-prediction logit tensors."""

    def predict(self, cls_embedding: Tensor) -> PredictionOutput:
        """Predict player src/tgt and opponent src/tgt squares from CLS embedding.

        Args:
            cls_embedding: Float tensor [B, 256] — the global board representation.

        Returns:
            PredictionOutput with four logit tensors each of shape [B, 64].

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
