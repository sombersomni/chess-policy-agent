"""AuxiliaryHeads: training-time auxiliary heads for denser encoder supervision.

Three linear heads provide auxiliary losses during joint encoder-decoder
training. Never called at inference time.

Heads:
  capture_target_head: Linear(D, 1) on square_embeddings [B,64,D] -> BCE
  move_category_head:  Linear(D, 7) on cls_embedding [B,D] -> 7-class CE
  phase_head:          Linear(D, 3) on cls_embedding [B,D] -> 3-class CE
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chess_sim.types import AuxLossOutput


class AuxiliaryHeads(nn.Module):
    """Training-time auxiliary heads for encoder supervision.

    All three heads flow gradients into the encoder: capture head
    via square embeddings, category and phase heads via CLS
    embedding (no detach). This maximises encoder signal from all
    aux tasks.

    Example:
        >>> heads = AuxiliaryHeads(d_model=128)
        >>> out = heads(sq, cls, cap_gt, cat_gt, ph_gt)
        >>> out.capture_loss.shape
        torch.Size([])
    """

    def __init__(self, d_model: int) -> None:
        """Initialize three linear projection heads.

        Args:
            d_model: Encoder hidden dimension.

        Example:
            >>> heads = AuxiliaryHeads(256)
        """
        super().__init__()
        self.capture_target_head = nn.Linear(d_model, 1)
        self.move_category_head = nn.Linear(d_model, 7)
        self.phase_head = nn.Linear(d_model, 3)

    def forward(
        self,
        square_emb: Tensor,
        cls_emb: Tensor,
        capture_gt: Tensor,
        category_gt: Tensor,
        phase_gt: Tensor,
    ) -> AuxLossOutput:
        """Compute auxiliary losses from encoder outputs.

        Args:
            square_emb: [B, 64, D] — NOT detached, grads
                flow into encoder via capture head.
            cls_emb: [B, D] — NOT detached, grads flow into
                encoder via category and phase heads.
            capture_gt: [B, 64] float32 binary ground truth.
            category_gt: [B] long, values 0-6.
            phase_gt: [B] long, values 0-2.

        Returns:
            AuxLossOutput with scalar losses for each head.

        Example:
            >>> out = heads(sq, cls, cap, cat, ph)
            >>> out.phase_loss.item()
            1.1
        """
        # Capture head: [B, 64, 1] -> squeeze -> BCE
        cap_logits = self.capture_target_head(
            square_emb
        ).squeeze(-1)  # [B, 64]
        capture_loss = F.binary_cross_entropy_with_logits(
            cap_logits, capture_gt
        )

        # CLS heads receive un-detached embeddings; grads
        # flow into the encoder via category and phase heads.

        # Move category head: [B, 7] -> CE
        cat_logits = self.move_category_head(cls_emb)
        category_loss = F.cross_entropy(
            cat_logits, category_gt
        )

        # Phase head: [B, 3] -> CE
        phase_logits = self.phase_head(cls_emb)
        phase_loss = F.cross_entropy(
            phase_logits, phase_gt
        )

        return AuxLossOutput(
            capture_loss=capture_loss,
            category_loss=category_loss,
            phase_loss=phase_loss,
        )
