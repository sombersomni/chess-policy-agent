"""ChunkProcessor: converts a list of PGN games into dense tensors.

Reuses BoardTokenizer and game_to_examples logic from train_real.py to
produce pre-tensorized shard data. Each call to process_chunk returns a
dict of torch.long tensors ready for ShardWriter.flush.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import chess.pgn
import torch
from torch import Tensor

from chess_sim.protocols import Tokenizable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """Converts a chunk of PGN games into a dict of dense torch.long tensors.

    Uses the existing BoardTokenizer (via the Tokenizable protocol) and
    game_to_examples logic to walk each game's mainline and emit one row
    per ply. The output dict is suitable for direct serialization by
    ShardWriter.

    Attributes:
        _tokenizer: A Tokenizable instance for board encoding.
        _winners_only: If True, only include positions from the
            winning side.

    Example:
        >>> cp = ChunkProcessor(BoardTokenizer(), winners_only=False)
        >>> tensors = cp.process_chunk(games)
        >>> tensors["board_tokens"].shape
        torch.Size([120, 65])
    """

    def __init__(
        self,
        tokenizer: Tokenizable,
        winners_only: bool = False,
    ) -> None:
        """Initialize the chunk processor.

        Args:
            tokenizer: A Tokenizable instance (e.g. BoardTokenizer).
            winners_only: If True, only include winning-side positions.
                Draws yield 0 examples.

        Example:
            >>> cp = ChunkProcessor(BoardTokenizer())
        """
        self._tokenizer = tokenizer
        self._winners_only = winners_only

    def process_chunk(
        self, games: list[chess.pgn.Game]
    ) -> dict[str, Tensor]:
        """Convert a list of games into dense torch.long tensors.

        Reuses game_to_examples from train_real to avoid duplicating
        tokenization logic.

        Args:
            games: List of chess.pgn.Game objects to process.

        Returns:
            Dict with keys: "board_tokens" [N, 65],
            "color_tokens" [N, 65],
            "trajectory_tokens" [N, 65], "src_sq" [N],
            "tgt_sq" [N].
            N is the total number of examples across all games.
            Returns empty tensors (N=0) if no examples are produced.

        Example:
            >>> tensors = cp.process_chunk([game1, game2])
            >>> tensors["src_sq"].shape[0]
            8
        """
        # Import here to avoid circular imports; reuse existing logic
        from scripts.train_real import game_to_examples

        all_board: list[list[int]] = []
        all_color: list[list[int]] = []
        all_traj: list[list[int]] = []
        all_src: list[int] = []
        all_tgt: list[int] = []

        for game in games:
            examples = game_to_examples(
                game,
                self._tokenizer,
                winners_only=self._winners_only,
            )
            for ex in examples:
                all_board.append(ex.board_tokens)
                all_color.append(ex.color_tokens)
                all_traj.append(ex.trajectory_tokens)
                all_src.append(ex.src_sq)
                all_tgt.append(ex.tgt_sq)

        n = len(all_src)
        if n == 0:
            logger.debug("Chunk produced 0 examples")
            return {
                "board_tokens": torch.zeros(
                    0, 65, dtype=torch.long
                ),
                "color_tokens": torch.zeros(
                    0, 65, dtype=torch.long
                ),
                "trajectory_tokens": torch.zeros(
                    0, 65, dtype=torch.long
                ),
                "src_sq": torch.zeros(0, dtype=torch.long),
                "tgt_sq": torch.zeros(0, dtype=torch.long),
            }

        logger.debug("Chunk produced %d examples from %d games", n, len(games))
        return {
            "board_tokens": torch.tensor(
                all_board, dtype=torch.long
            ),
            "color_tokens": torch.tensor(
                all_color, dtype=torch.long
            ),
            "trajectory_tokens": torch.tensor(
                all_traj, dtype=torch.long
            ),
            "src_sq": torch.tensor(
                all_src, dtype=torch.long
            ),
            "tgt_sq": torch.tensor(
                all_tgt, dtype=torch.long
            ),
        }
