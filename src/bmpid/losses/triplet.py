"""Triplet Loss (expertos unimodales). Embeddings L2-normalizados."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def pairwise_distance(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).norm(p=2, dim=-1)


def triplet_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 0.2,
) -> Tensor:
    """max(0, m + d(a, p) − d(a, n))."""
    d_pos = pairwise_distance(anchor, positive)
    d_neg = pairwise_distance(anchor, negative)
    return F.relu(margin + d_pos - d_neg).mean()


def semi_hard_negative(
    anchor: Tensor,
    positive: Tensor,
    candidates: Tensor,
    margin: float = 0.2,
) -> Tensor:
    """Por ancla, el negativo semi-hard del batch (otros anclas). Si no hay, el más lejano."""
    b = anchor.size(0)
    d_pos = pairwise_distance(anchor, positive).unsqueeze(1)
    d_neg = torch.cdist(anchor, candidates, p=2)
    self = torch.eye(b, device=anchor.device, dtype=torch.bool)
    d_neg = d_neg.masked_fill(self, 1e9)
    semi = (d_neg > d_pos) & (d_neg < d_pos + float(margin))
    chosen = d_neg.clone()
    chosen = chosen.masked_fill(~semi, 1e9)
    has = semi.any(dim=1)
    idx_semi = chosen.argmin(dim=1)
    idx_far = d_neg.argmax(dim=1)
    idx = torch.where(has, idx_semi, idx_far)
    return candidates[idx]
