"""ChunkProcessor: converts a list of PGN games into dense tensors.

Reuses BoardTokenizer and game_to_examples logic from train_real.py to
produce pre-tensorized shard data. Each call to process_chunk returns a
dict of torch.long tensors ready for ShardWriter.flush.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess.pgn
from torch import Tensor

from chess_sim.protocols import Tokenizable

if TYPE_CHECKING:
    pass


class ChunkProcessor:
    """Converts a chunk of PGN games into a dict of dense torch.long tensors.

    Uses the existing BoardTokenizer (via the Tokenizable protocol) and
    game_to_examples logic to walk each game's mainline and emit one row
    per ply. The output dict is suitable for direct serialization by ShardWriter.

    Attributes:
        _tokenizer: A Tokenizable instance for board encoding.
        _winners_only: If True, only include positions from the winning side.

    Example:
        >>> cp = ChunkProcessor(BoardTokenizer(), winners_only=False)
        >>> tensors = cp.process_chunk(games)
        >>> tensors["board_tokens"].shape
        torch.Size([120, 65])
    """

    def __init__(self, tokenizer: Tokenizable, winners_only: bool = False) -> None:
        """Initialize the chunk processor.

        Args:
            tokenizer: A Tokenizable instance (e.g. BoardTokenizer).
            winners_only: If True, only include winning-side positions. Draws yield 0 examples.

        Example:
            >>> cp = ChunkProcessor(BoardTokenizer())
        """
        raise NotImplementedError("To be implemented")

    def process_chunk(self, games: list[chess.pgn.Game]) -> dict[str, Tensor]:
        """Convert a list of games into a dict of dense torch.long tensors.

        Walks each game's mainline, tokenizes every position, builds trajectory
        tokens, and packs all examples into stacked tensors.

        Args:
            games: List of chess.pgn.Game objects to process.

        Returns:
            Dict with keys: "board_tokens" [N, 65], "color_tokens" [N, 65],
            "trajectory_tokens" [N, 65], "src_sq" [N], "tgt_sq" [N].
            N is the total number of examples across all games in the chunk.
            Returns empty tensors (N=0) if no examples are produced.

        Example:
            >>> tensors = cp.process_chunk([game1, game2])
            >>> tensors["src_sq"].shape[0] == tensors["board_tokens"].shape[0]
            True
        """
        raise NotImplementedError("To be implemented")
